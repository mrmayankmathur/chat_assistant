from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session

from app.core import DEFAULT_DATABASE_URL, hash_phone_e164, normalize_email
from app.models import (
    AgentConfig,
    BusinessHours,
    ChannelGuideline,
    CrmIntegration,
    CrmProvider,
    CustomerOrder,
    FailureReason,
    Policy,
    PolicyType,
    Shipment,
    ShipmentEvent,
    ShipmentStatus,
    StatusCopy,
    Tenant,
    Thread,
    Tonality,
)

TENANT_ID = 1

# Raw E.164 phones exist only in memory here; stored ONLY as HMAC-SHA256 hex digest.
CUSTOMER_A_PHONE_E164 = "+14155550101"
CUSTOMER_A_EMAIL = normalize_email("Alice.Customer@Example.com")

CUSTOMER_B_PHONE_E164 = "+14155550102"
CUSTOMER_B_EMAIL = normalize_email("Bob.Buyer@Example.com")

# Customer B's AWB that Customer A will later ask about
CUSTOMER_B_AWB = "AWB-CUSTB-9001"

SEEDED_SHIPMENT_IDS = (2001, 2002, 2003, 2004, 2005, 2006)

_PG_SERIAL_TABLES: tuple[tuple[str, str], ...] = (
    ("tenant", "tenant_id"),
    ("agent_config", "agent_id"),
    ("policy", "policy_id"),
    ("crm_integration", "id"),
    ("thread", "thread_id"),
    ("message", "message_id"),
    ("conversation", "conversation_id"),
    ("escalation", "id"),
    ("crm_ticket", "id"),
    ("customer_order", "order_pk"),
    ("shipment", "id"),
    ("llm_call", "id"),
)


def _sync_postgres_sequences(session: Session) -> None:
    bind = session.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        return
    for table_name, pk_col in _PG_SERIAL_TABLES:
        session.execute(
            text(
                f"SELECT setval("
                f"pg_get_serial_sequence('{table_name}', '{pk_col}'), "
                f"COALESCE((SELECT MAX({pk_col}) FROM {table_name}), 1), "
                f"(SELECT MAX({pk_col}) IS NOT NULL FROM {table_name})"
                f")"
            )
        )


def seed_database(session: Session) -> None:
    """Non-destructively upsert 1 tenant, 2 customers, and 6 orders covering all required states."""
    now = datetime.now(UTC).replace(microsecond=0)
    today = now.date()

    customer_a_phone_hash = hash_phone_e164(CUSTOMER_A_PHONE_E164)
    customer_b_phone_hash = hash_phone_e164(CUSTOMER_B_PHONE_E164)

    # 1. Tenant + Control Plane (non-destructive merge by primary key)
    session.merge(
        Tenant(
            tenant_id=TENANT_ID,
            name="Lumina Botanics",
            timezone="America/New_York",
            features={
                "whatsapp_enabled": True,
                "web_chat_enabled": True,
                "cancel_enabled": True,
                "cancel_before_pickup": True,
            },
        )
    )
    session.flush()

    session.merge(
        AgentConfig(
            agent_id=1,
            tenant_id=TENANT_ID,
            brand_prompt="You are the warm, clear support assistant for Lumina Botanics.",
            tonality=Tonality.FRIENDLY,
            is_active=True,
            version=1,
            soft_deleted=False,
        )
    )
    for guideline in (
        ChannelGuideline(
            tenant_id=TENANT_ID,
            channel="whatsapp",
            prompt="Keep replies concise and mobile-friendly for WhatsApp.",
            soft_deleted=False,
        ),
        ChannelGuideline(
            tenant_id=TENANT_ID,
            channel="web",
            prompt="Provide structured order details clearly for web chat.",
            soft_deleted=False,
        ),
    ):
        session.merge(guideline)

    for policy in (
        Policy(
            policy_id=1,
            tenant_id=TENANT_ID,
            policy_type=PolicyType.CANCELLATION,
            text="Orders can only be cancelled before carrier pickup.",
            windows={
                "allow_statuses": [
                    int(ShipmentStatus.ORDER_PLACED),
                    int(ShipmentStatus.MANIFESTED),
                    int(ShipmentStatus.PICKUP_FAILED),
                ]
            },
            soft_deleted=False,
        ),
        Policy(
            policy_id=2,
            tenant_id=TENANT_ID,
            policy_type=PolicyType.REFUND,
            text="Refunds to original payment method take 5-7 business days after cancellation.",
            windows={"refund_sla_business_days": 7},
            soft_deleted=False,
        ),
    ):
        session.merge(policy)

    session.merge(
        BusinessHours(
            tenant_id=TENANT_ID,
            weekly={
                "mon": [["09:00", "18:00"]],
                "tue": [["09:00", "18:00"]],
                "wed": [["09:00", "18:00"]],
                "thu": [["09:00", "18:00"]],
                "fri": [["09:00", "18:00"]],
            },
            overrides=[],
            soft_deleted=False,
        )
    )
    session.merge(
        CrmIntegration(
            id=1,
            tenant_id=TENANT_ID,
            provider=CrmProvider.ZENDESK,
            secret_ref="vault://secrets/lumina/zendesk",
            supports_ticket=True,
            supports_live_handover=True,
            webhook_uuid=uuid.UUID("11111111-2222-3333-4444-555555555555"),
            soft_deleted=False,
        )
    )

    # Status copy defaults (tenant_id=0)
    for copy_row in (
        StatusCopy(
            tenant_id=0,
            status=ShipmentStatus.ORDER_PLACED,
            failure_reason=FailureReason.NONE,
            customer_text="Your order has been placed and is being prepared at our warehouse.",
            soft_deleted=False,
        ),
        StatusCopy(
            tenant_id=0,
            status=ShipmentStatus.IN_TRANSIT,
            failure_reason=FailureReason.NONE,
            customer_text="Your package has been picked up and is in transit to your city.",
            soft_deleted=False,
        ),
        StatusCopy(
            tenant_id=0,
            status=ShipmentStatus.OUT_FOR_DELIVERY,
            failure_reason=FailureReason.NONE,
            customer_text="Your package is out for delivery today.",
            soft_deleted=False,
        ),
        StatusCopy(
            tenant_id=0,
            status=ShipmentStatus.FAILED_DELIVERY,
            failure_reason=FailureReason.CUSTOMER_UNAVAILABLE,
            customer_text="Delivery attempt failed because no one was available to receive the package.",
            soft_deleted=False,
        ),
        StatusCopy(
            tenant_id=0,
            status=ShipmentStatus.DELIVERED,
            failure_reason=FailureReason.NONE,
            customer_text="Your package has been delivered.",
            soft_deleted=False,
        ),
        StatusCopy(
            tenant_id=0,
            status=ShipmentStatus.CANCELLED,
            failure_reason=FailureReason.NONE,
            customer_text="This order has been cancelled.",
            soft_deleted=False,
        ),
    ):
        session.merge(copy_row)

    # 2. Two Customer Threads (Customer A and Customer B)
    for thread_row in (
        Thread(
            thread_id=101,
            tenant_id=TENANT_ID,
            phone_hash=customer_a_phone_hash,
            email=CUSTOMER_A_EMAIL,
            expires_at=now + timedelta(days=30),
            migrated_from_thread_id=None,
            soft_deleted=False,
        ),
        Thread(
            thread_id=102,
            tenant_id=TENANT_ID,
            phone_hash=customer_b_phone_hash,
            email=CUSTOMER_B_EMAIL,
            expires_at=now + timedelta(days=30),
            migrated_from_thread_id=None,
            soft_deleted=False,
        ),
    ):
        session.merge(thread_row)

    # 3. Six Orders covering:
    #   (1) not-shipped (ORDER_PLACED, before pickup) -> Customer A
    #   (2) in-transit (IN_TRANSIT, EDD breached) -> Customer A
    #   (3) out-for-delivery (OUT_FOR_DELIVERY) -> Customer A
    #   (4) failed-delivery (FAILED_DELIVERY + CUSTOMER_UNAVAILABLE) -> Customer A
    #   (5) cancelled (CANCELLED) -> Customer A
    #   (6) delivered (DELIVERED, owned by Customer B with AWB that Customer A will ask about) -> Customer B
    for order_row in (
        CustomerOrder(
            order_pk=1001,
            tenant_id=TENANT_ID,
            order_ref="LUM-1001",
            phone_hash=customer_a_phone_hash,
            email=CUSTOMER_A_EMAIL,
            payment_mode="PREPAID",
            placed_at=now - timedelta(hours=6),
            cancelled_at=None,
            items=[{"sku": "SERUM-VITC", "name": "Vitamin C Glow Serum", "qty": 1, "price_cents": 3800}],
            soft_deleted=False,
        ),
        CustomerOrder(
            order_pk=1002,
            tenant_id=TENANT_ID,
            order_ref="LUM-1002",
            phone_hash=customer_a_phone_hash,
            email=CUSTOMER_A_EMAIL,
            payment_mode="PREPAID",
            placed_at=now - timedelta(days=6),
            cancelled_at=None,
            items=[{"sku": "CREAM-HYDRA", "name": "Deep Hydra Barrier Cream", "qty": 2, "price_cents": 8400}],
            soft_deleted=False,
        ),
        CustomerOrder(
            order_pk=1003,
            tenant_id=TENANT_ID,
            order_ref="LUM-1003",
            phone_hash=customer_a_phone_hash,
            email=CUSTOMER_A_EMAIL,
            payment_mode="COD",
            placed_at=now - timedelta(days=3),
            cancelled_at=None,
            items=[{"sku": "CLEANSER-GENTLE", "name": "Oat Milk Gentle Cleanser", "qty": 1, "price_cents": 2600}],
            soft_deleted=False,
        ),
        CustomerOrder(
            order_pk=1004,
            tenant_id=TENANT_ID,
            order_ref="LUM-1004",
            phone_hash=customer_a_phone_hash,
            email=CUSTOMER_A_EMAIL,
            payment_mode="PREPAID",
            placed_at=now - timedelta(days=4),
            cancelled_at=None,
            items=[{"sku": "MASK-CLAY", "name": "Pink Kaolin Detox Mask", "qty": 1, "price_cents": 3200}],
            soft_deleted=False,
        ),
        CustomerOrder(
            order_pk=1005,
            tenant_id=TENANT_ID,
            order_ref="LUM-1005",
            phone_hash=customer_a_phone_hash,
            email=CUSTOMER_A_EMAIL,
            payment_mode="PREPAID",
            placed_at=now - timedelta(days=5),
            cancelled_at=now - timedelta(days=5) + timedelta(hours=2),
            items=[{"sku": "TONER-PHA", "name": "PHA Pore Refining Toner", "qty": 1, "price_cents": 2900}],
            soft_deleted=False,
        ),
        CustomerOrder(
            order_pk=1006,
            tenant_id=TENANT_ID,
            order_ref="LUM-1006",
            phone_hash=customer_b_phone_hash,
            email=CUSTOMER_B_EMAIL,
            payment_mode="PREPAID",
            placed_at=now - timedelta(days=7),
            cancelled_at=None,
            items=[{"sku": "OIL-ROSEHIP", "name": "Cold-Pressed Rosehip Oil", "qty": 1, "price_cents": 4500}],
            soft_deleted=False,
        ),
    ):
        session.merge(order_row)
    session.flush()

    for shipment_row in (
        Shipment(
            id=2001,
            order_pk=1001,
            awb="AWB-CUSTA-1001",
            carrier_code="DELHIVERY",
            is_reverse=False,
            status=ShipmentStatus.ORDER_PLACED,
            edd=today + timedelta(days=4),
            picked_up_at=None,
            delivered_at=None,
        ),
        Shipment(
            id=2002,
            order_pk=1002,
            awb="AWB-CUSTA-1002",
            carrier_code="BLUEDART",
            is_reverse=False,
            status=ShipmentStatus.IN_TRANSIT,
            edd=today - timedelta(days=2),
            picked_up_at=now - timedelta(days=5),
            delivered_at=None,
        ),
        Shipment(
            id=2003,
            order_pk=1003,
            awb="AWB-CUSTA-1003",
            carrier_code="DELHIVERY",
            is_reverse=False,
            status=ShipmentStatus.OUT_FOR_DELIVERY,
            edd=today,
            picked_up_at=now - timedelta(days=2),
            delivered_at=None,
        ),
        Shipment(
            id=2004,
            order_pk=1004,
            awb="AWB-CUSTA-1004",
            carrier_code="BLUEDART",
            is_reverse=False,
            status=ShipmentStatus.FAILED_DELIVERY,
            edd=today,
            picked_up_at=now - timedelta(days=3),
            delivered_at=None,
        ),
        Shipment(
            id=2005,
            order_pk=1005,
            awb="AWB-CUSTA-1005",
            carrier_code="DELHIVERY",
            is_reverse=False,
            status=ShipmentStatus.CANCELLED,
            edd=None,
            picked_up_at=None,
            delivered_at=None,
        ),
        Shipment(
            id=2006,
            order_pk=1006,
            awb=CUSTOMER_B_AWB,
            carrier_code="BLUEDART",
            is_reverse=False,
            status=ShipmentStatus.DELIVERED,
            edd=today - timedelta(days=3),
            picked_up_at=now - timedelta(days=6),
            delivered_at=now - timedelta(days=3),
        ),
    ):
        session.merge(shipment_row)
    session.flush()

    # Replace events only for the 6 seeded shipment IDs (never unscoped)
    session.execute(
        delete(ShipmentEvent).where(ShipmentEvent.shipment_id.in_(SEEDED_SHIPMENT_IDS))
    )

    events = [
        ShipmentEvent(
            shipment_id=2001,
            event_at=now - timedelta(hours=6),
            status=ShipmentStatus.ORDER_PLACED,
            failure_reason=FailureReason.NONE,
            location="Newark Warehouse",
            raw_remark="Order received at fulfillment center",
        ),
        ShipmentEvent(
            shipment_id=2002,
            event_at=now - timedelta(days=5),
            status=ShipmentStatus.PICKED_UP,
            failure_reason=FailureReason.NONE,
            location="Newark Hub",
            raw_remark="Picked up by carrier",
        ),
        ShipmentEvent(
            shipment_id=2002,
            event_at=now - timedelta(days=1),
            status=ShipmentStatus.IN_TRANSIT,
            failure_reason=FailureReason.NONE,
            location="Chicago Sort Facility",
            raw_remark="In transit to destination hub - weather delay",
        ),
        ShipmentEvent(
            shipment_id=2003,
            event_at=now - timedelta(days=2),
            status=ShipmentStatus.PICKED_UP,
            failure_reason=FailureReason.NONE,
            location="Newark Hub",
            raw_remark="Shipment picked up",
        ),
        ShipmentEvent(
            shipment_id=2003,
            event_at=now - timedelta(hours=3),
            status=ShipmentStatus.OUT_FOR_DELIVERY,
            failure_reason=FailureReason.NONE,
            location="Brooklyn Local Hub",
            raw_remark="Out for delivery with courier",
        ),
        ShipmentEvent(
            shipment_id=2004,
            event_at=now - timedelta(days=3),
            status=ShipmentStatus.PICKED_UP,
            failure_reason=FailureReason.NONE,
            location="Newark Hub",
            raw_remark="Shipment picked up",
        ),
        ShipmentEvent(
            shipment_id=2004,
            event_at=now - timedelta(hours=5),
            status=ShipmentStatus.OUT_FOR_DELIVERY,
            failure_reason=FailureReason.NONE,
            location="Brooklyn Local Hub",
            raw_remark="Out for delivery",
        ),
        ShipmentEvent(
            shipment_id=2004,
            event_at=now - timedelta(hours=2),
            status=ShipmentStatus.FAILED_DELIVERY,
            failure_reason=FailureReason.CUSTOMER_UNAVAILABLE,
            location="Brooklyn, NY",
            raw_remark="Consignee unavailable; rang bell twice",
        ),
        ShipmentEvent(
            shipment_id=2005,
            event_at=now - timedelta(days=5),
            status=ShipmentStatus.ORDER_PLACED,
            failure_reason=FailureReason.NONE,
            location="Newark Warehouse",
            raw_remark="Order created",
        ),
        ShipmentEvent(
            shipment_id=2005,
            event_at=now - timedelta(days=5) + timedelta(hours=2),
            status=ShipmentStatus.CANCELLED,
            failure_reason=FailureReason.NONE,
            location="Newark Warehouse",
            raw_remark="Cancelled by customer before carrier pickup",
        ),
        ShipmentEvent(
            shipment_id=2006,
            event_at=now - timedelta(days=6),
            status=ShipmentStatus.PICKED_UP,
            failure_reason=FailureReason.NONE,
            location="Newark Hub",
            raw_remark="Picked up",
        ),
        ShipmentEvent(
            shipment_id=2006,
            event_at=now - timedelta(days=3),
            status=ShipmentStatus.DELIVERED,
            failure_reason=FailureReason.NONE,
            location="Boston, MA",
            raw_remark="Delivered to front desk",
        ),
    ]
    for ev in events:
        session.merge(ev)

    session.flush()
    _sync_postgres_sequences(session)
    session.commit()


def main() -> None:
    engine = create_engine(DEFAULT_DATABASE_URL)
    with Session(engine) as session:
        seed_database(session)


if __name__ == "__main__":
    main()
