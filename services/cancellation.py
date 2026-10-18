from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from app.models import CustomerOrder, Shipment, ShipmentStatus, Tenant


class CancelReasonCode(StrEnum):
    CANCEL_DISABLED = "CANCEL_DISABLED"
    ALREADY_CANCELLED = "ALREADY_CANCELLED"
    REVERSE_SHIPMENT = "REVERSE_SHIPMENT"
    ALREADY_PICKED_UP = "ALREADY_PICKED_UP"
    ELIGIBLE = "ELIGIBLE"


PRE_PICKUP_STATUSES: frozenset[ShipmentStatus] = frozenset(
    {
        ShipmentStatus.ORDER_PLACED,
        ShipmentStatus.MANIFESTED,
        ShipmentStatus.PICKUP_FAILED,
    }
)


def check_cancel(
    order: CustomerOrder,
    shipments: Sequence[Shipment],
    tenant: Tenant,
) -> tuple[bool, CancelReasonCode]:
    """Deterministically evaluate whether an order can be cancelled.

    Evaluation order:
    1. tenant.features.cancel_enabled is false -> (False, CANCEL_DISABLED)
    2. Already cancelled -> (True, ALREADY_CANCELLED)
    3. Any reverse shipment -> (False, REVERSE_SHIPMENT)
    4. Any shipment status not in {ORDER_PLACED, MANIFESTED, PICKUP_FAILED} -> (False, ALREADY_PICKED_UP)
    5. Otherwise -> (True, ELIGIBLE)
    """
    features = tenant.features or {}
    if not bool(features.get("cancel_enabled", False)):
        return (False, CancelReasonCode.CANCEL_DISABLED)

    is_already_cancelled = order.cancelled_at is not None or (
        bool(shipments)
        and all(ShipmentStatus(s.status) == ShipmentStatus.CANCELLED for s in shipments)
    )
    if is_already_cancelled:
        return (True, CancelReasonCode.ALREADY_CANCELLED)

    if any(bool(s.is_reverse) for s in shipments):
        return (False, CancelReasonCode.REVERSE_SHIPMENT)

    if any(ShipmentStatus(s.status) not in PRE_PICKUP_STATUSES for s in shipments):
        return (False, CancelReasonCode.ALREADY_PICKED_UP)

    return (True, CancelReasonCode.ELIGIBLE)
