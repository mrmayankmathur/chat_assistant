from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.models import (
    CustomerOrder,
    FailureReason,
    Shipment,
    ShipmentEvent,
    ShipmentStatus,
    StatusCopy,
    Tenant,
)
from repositories.storage import (
    InMemoryStatusCopyRepository,
    SqlAlchemyStatusCopyRepository,
    StatusCopyRepositoryProtocol,
)

NON_BREACH_STATUSES: frozenset[ShipmentStatus] = frozenset(
    {
        ShipmentStatus.DELIVERED,
        ShipmentStatus.CANCELLED,
        ShipmentStatus.RTO_DELIVERED,
    }
)

DEFAULT_GLOBAL_STATUS_COPY: dict[tuple[int, int, int], str] = {
    (0, int(ShipmentStatus.ORDER_PLACED), int(FailureReason.NONE)): (
        "Your order has been placed and is being prepared at our warehouse."
    ),
    (0, int(ShipmentStatus.MANIFESTED), int(FailureReason.NONE)): (
        "Your shipment has been manifested and is awaiting carrier pickup."
    ),
    (0, int(ShipmentStatus.PICKED_UP), int(FailureReason.NONE)): (
        "Your package has been picked up by our delivery partner."
    ),
    (0, int(ShipmentStatus.IN_TRANSIT), int(FailureReason.NONE)): (
        "Your package has been picked up and is in transit to your city."
    ),
    (0, int(ShipmentStatus.OUT_FOR_DELIVERY), int(FailureReason.NONE)): (
        "Your package is out for delivery today."
    ),
    (0, int(ShipmentStatus.DELIVERED), int(FailureReason.NONE)): (
        "Your package has been delivered."
    ),
    (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.NONE)): (
        "We could not complete your delivery attempt."
    ),
    (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.CUSTOMER_UNAVAILABLE)): (
        "Delivery attempt failed because no one was available to receive the package."
    ),
    (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.ADDRESS_ISSUE)): (
        "Delivery attempt failed due to an incomplete or unreachable address."
    ),
    (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.REFUSED)): (
        "Delivery attempt failed because the package was refused at the doorstep."
    ),
    (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.COD_NOT_READY)): (
        "Delivery attempt failed because cash on delivery payment was not ready."
    ),
    (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.RESCHEDULE)): (
        "Delivery was rescheduled as requested."
    ),
    (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.ENTRY_RESTRICTED)): (
        "Delivery attempt failed because entry to the premises was restricted."
    ),
    (0, int(ShipmentStatus.RTO_INITIATED), int(FailureReason.NONE)): (
        "Your package is being returned to our warehouse."
    ),
    (0, int(ShipmentStatus.RTO_DELIVERED), int(FailureReason.NONE)): (
        "Your package has been returned to our warehouse."
    ),
    (0, int(ShipmentStatus.CANCELLED), int(FailureReason.NONE)): (
        "This order has been cancelled."
    ),
    (0, int(ShipmentStatus.LOST), int(FailureReason.NONE)): (
        "Your shipment was delayed in transit and is under investigation."
    ),
    (0, int(ShipmentStatus.DAMAGED), int(FailureReason.NONE)): (
        "Your package sustained damage in transit."
    ),
    (0, int(ShipmentStatus.PICKUP_FAILED), int(FailureReason.NONE)): (
        "Carrier pickup from our warehouse was delayed and is being rescheduled."
    ),
}


@dataclass(frozen=True)
class TrackingView:
    order_ref: str
    status: ShipmentStatus
    failure_reason: FailureReason
    customer_text: str
    edd: date | None
    edd_breached: bool
    ofd_attempts: int
    last_events: list[dict[str, Any]]


def _resolve_latest_failure_reason(events: Sequence[ShipmentEvent]) -> FailureReason:
    failed_events = [
        e for e in events if ShipmentStatus(e.status) == ShipmentStatus.FAILED_DELIVERY
    ]
    if not failed_events:
        return FailureReason.NONE
    latest = max(failed_events, key=lambda e: e.event_at)
    return FailureReason(latest.failure_reason)


def _lookup_customer_text(
    copy_map: dict[tuple[int, int, int], str],
    tenant_id: int,
    status: ShipmentStatus,
    failure_reason: FailureReason,
) -> str:
    keys = [
        (int(tenant_id), int(status), int(failure_reason)),
        (0, int(status), int(failure_reason)),
        (0, int(status), int(FailureReason.NONE)),
    ]
    for key in keys:
        if key in copy_map:
            return copy_map[key]
    for key in keys:
        if key in DEFAULT_GLOBAL_STATUS_COPY:
            return DEFAULT_GLOBAL_STATUS_COPY[key]
    return f"Status: {ShipmentStatus(status).name}"


class _ViewMethodDescriptor:
    def __get__(self, instance: TrackingService | None, owner: type[TrackingService]) -> Any:
        target = instance if instance is not None else owner()
        return target._view


class TrackingService:
    def __init__(
        self,
        repo_or_session: (
            StatusCopyRepositoryProtocol
            | Session
            | Sequence[StatusCopy]
            | dict[tuple[int, int, int], str]
            | None
        ) = None,
    ) -> None:
        if isinstance(repo_or_session, Session):
            self.repo: StatusCopyRepositoryProtocol = SqlAlchemyStatusCopyRepository(repo_or_session)
        elif isinstance(repo_or_session, StatusCopyRepositoryProtocol):
            self.repo = repo_or_session
        elif repo_or_session is not None:
            self.repo = InMemoryStatusCopyRepository(repo_or_session)
        else:
            self.repo = InMemoryStatusCopyRepository(DEFAULT_GLOBAL_STATUS_COPY)

    def _view(
        self,
        order: CustomerOrder,
        shipment: Shipment,
        events: Sequence[ShipmentEvent],
        tenant: Tenant,
        *,
        now: datetime | None = None,
    ) -> TrackingView:
        """Build a deterministic TrackingView for an order's shipment and events."""
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)

        status = ShipmentStatus(shipment.status)
        failure_reason = _resolve_latest_failure_reason(events)

        copy_map = self.repo.get_copy_map(
            tenant_id=tenant.tenant_id,
            status=status,
            failure_reason=failure_reason,
        )
        customer_text = _lookup_customer_text(
            copy_map=copy_map,
            tenant_id=tenant.tenant_id,
            status=status,
            failure_reason=failure_reason,
        )

        tenant_tz = ZoneInfo(tenant.timezone)
        today_in_tenant_tz = current_now.astimezone(tenant_tz).date()

        edd_breached = (
            shipment.edd is not None
            and today_in_tenant_tz > shipment.edd
            and status not in NON_BREACH_STATUSES
        )

        ofd_attempts = sum(
            1 for e in events if ShipmentStatus(e.status) == ShipmentStatus.OUT_FOR_DELIVERY
        )

        sorted_events = sorted(events, key=lambda e: e.event_at)
        last_five = sorted_events[-5:]
        last_events = [
            {
                "at": e.event_at,
                "status": ShipmentStatus(e.status),
                "location": e.location,
            }
            for e in last_five
        ]

        return TrackingView(
            order_ref=order.order_ref,
            status=status,
            failure_reason=failure_reason,
            customer_text=customer_text,
            edd=shipment.edd,
            edd_breached=edd_breached,
            ofd_attempts=ofd_attempts,
            last_events=last_events,
        )

    view = _ViewMethodDescriptor()
