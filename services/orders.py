from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.models import CustomerOrder, Thread
from repositories.storage import (
    InMemoryOrderRepository,
    OrderRepositoryProtocol,
    SqlAlchemyOrderRepository,
)

MAX_OWNED_ORDERS = 10
DEFAULT_LOOKBACK_DAYS = 30
NOT_OWNED_MESSAGE = "Order not found"

Order = CustomerOrder


class NotOwned(LookupError):
    """Raised identically when an order does not exist or is not owned by the customer."""

    def __init__(self, message: str = NOT_OWNED_MESSAGE) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class OrderCard:
    order_pk: int
    order_ref: str
    placed_at: datetime
    payment_mode: str
    cancelled_at: datetime | None
    items: tuple[dict[str, Any], ...]


def _owns_order(thread: Thread, order: CustomerOrder) -> bool:
    if order.tenant_id != thread.tenant_id:
        return False
    if getattr(order, "soft_deleted", False):
        return False
    phone_match = (
        thread.phone_hash is not None
        and order.phone_hash is not None
        and thread.phone_hash == order.phone_hash
    )
    email_match = (
        thread.email is not None
        and order.email is not None
        and thread.email.lower() == order.email.lower()
    )
    return phone_match or email_match


class OrderService:
    def __init__(self, repo_or_session: OrderRepositoryProtocol | Session | None = None) -> None:
        if isinstance(repo_or_session, Session):
            self.repo: OrderRepositoryProtocol = SqlAlchemyOrderRepository(repo_or_session)
        elif repo_or_session is None:
            self.repo = InMemoryOrderRepository()
        else:
            self.repo = repo_or_session

    def list_owned_orders(
        self,
        thread: Thread,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        *,
        now: datetime | None = None,
    ) -> list[OrderCard]:
        """Return up to 10 most recent orders owned by the thread within lookback_days."""
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)
        since = current_now - timedelta(days=lookback_days)

        orders = self.repo.list_for_customer(
            tenant_id=thread.tenant_id,
            phone_hash=thread.phone_hash,
            email=thread.email,
            since=since,
            limit=MAX_OWNED_ORDERS,
        )
        return [
            OrderCard(
                order_pk=o.order_pk,
                order_ref=o.order_ref,
                placed_at=o.placed_at,
                payment_mode=o.payment_mode,
                cancelled_at=o.cancelled_at,
                items=tuple(o.items or ()),
            )
            for o in orders[:MAX_OWNED_ORDERS]
        ]

    def get_owned_order(
        self,
        thread: Thread,
        order_ref: str | None = None,
        awb: str | None = None,
    ) -> CustomerOrder:
        """Fetch an order by order_ref or awb and verify customer ownership.

        Raises an identical NotOwned error for both missing orders and orders owned by another customer.
        """
        cleaned_ref = order_ref.strip() if order_ref and order_ref.strip() else None
        cleaned_awb = awb.strip() if awb and awb.strip() else None

        if cleaned_ref is None and cleaned_awb is None:
            raise NotOwned()

        order_by_ref: CustomerOrder | None = None
        order_by_awb: CustomerOrder | None = None

        if cleaned_ref is not None:
            order_by_ref = self.repo.get_by_order_ref(thread.tenant_id, cleaned_ref)
            if order_by_ref is None:
                raise NotOwned()

        if cleaned_awb is not None:
            order_by_awb = self.repo.get_by_awb(thread.tenant_id, cleaned_awb)
            if order_by_awb is None:
                raise NotOwned()

        if (
            order_by_ref is not None
            and order_by_awb is not None
            and order_by_ref.order_pk != order_by_awb.order_pk
        ):
            raise NotOwned()

        target = order_by_ref if order_by_ref is not None else order_by_awb
        if target is None or not _owns_order(thread, target):
            raise NotOwned()

        return target
