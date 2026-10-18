from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.core import hash_phone_e164, normalize_and_hash_phone, normalize_email
from app.models import (
    CustomerOrder,
    FailureReason,
    Shipment,
    ShipmentEvent,
    ShipmentStatus,
    StatusCopy,
    Tenant,
    Thread,
)
from app.seed import CUSTOMER_A_PHONE_E164, CUSTOMER_B_AWB, CUSTOMER_B_PHONE_E164, TENANT_ID
from app.settings import Settings
from repositories import (
    InMemoryOrderRepository,
    InMemoryStatusCopyRepository,
    InMemoryThreadRepository,
)
from services import (
    CancelReasonCode,
    IdentityService,
    NotOwned,
    OrderCard,
    OrderService,
    TrackingService,
    TrackingView,
    check_cancel,
)

FIXED_NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=UTC)
TEST_SETTINGS = Settings(
    phone_hmac_secret="test-secret-key-123",
    default_phone_region="IN",
)


# ---------------------------------------------------------------------------
# 1. IdentityService.resolve_thread tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_phone", "expected_e164"),
    [
        ("9876543210", "+919876543210"),
        ("09876543210", "+919876543210"),
        ("+91 98765-43210", "+919876543210"),
        ("+14155550101", "+14155550101"),
    ],
)
def test_identity_phone_normalization_and_hmac(raw_phone: str, expected_e164: str) -> None:
    repo = InMemoryThreadRepository()
    svc = IdentityService(repo, settings=TEST_SETTINGS)

    thread = svc.resolve_thread(tenant_id=1, phone=raw_phone, email=None, now=FIXED_NOW)

    expected_hash = hash_phone_e164(expected_e164, secret=TEST_SETTINGS.phone_hmac_secret)
    assert thread.phone_hash == expected_hash
    assert thread.expires_at == FIXED_NOW + timedelta(days=30)
    assert thread.migrated_from_thread_id is None


@pytest.mark.parametrize(
    (
        "case_name",
        "existing_phone",
        "existing_email",
        "existing_expired",
        "lookup_phone",
        "lookup_email",
        "expect_new_thread",
        "expect_migrated",
        "expected_final_phone",
        "expected_final_email",
    ),
    [
        (
            "new_thread_created_with_30_day_expiry",
            None,
            None,
            False,
            "9876543210",
            "Alice@Example.com",
            True,
            False,
            "+919876543210",
            "alice@example.com",
        ),
        (
            "match_on_email_first_and_latch_missing_phone",
            None,
            "alice@example.com",
            False,
            "9876543210",
            "ALICE@example.com",
            False,
            False,
            "+919876543210",
            "alice@example.com",
        ),
        (
            "match_on_phone_and_latch_missing_email",
            "+919876543210",
            None,
            False,
            "09876543210",
            "Alice@Example.com",
            False,
            False,
            "+919876543210",
            "alice@example.com",
        ),
        (
            "refuse_to_overwrite_different_phone_on_email_match",
            "+919876543210",
            "alice@example.com",
            False,
            "9123456789",
            "alice@example.com",
            False,
            False,
            "+919876543210",
            "alice@example.com",
        ),
        (
            "refuse_to_overwrite_different_email_on_phone_match",
            "+919876543210",
            "alice@example.com",
            False,
            "9876543210",
            "other@example.com",
            False,
            False,
            "+919876543210",
            "alice@example.com",
        ),
        (
            "expired_thread_creates_new_thread_with_migrated_from_thread_id",
            "+919876543210",
            "alice@example.com",
            True,
            "9876543210",
            "alice@example.com",
            True,
            True,
            "+919876543210",
            "alice@example.com",
        ),
    ],
)
def test_identity_resolve_thread_rules(
    case_name: str,
    existing_phone: str | None,
    existing_email: str | None,
    existing_expired: bool,
    lookup_phone: str | None,
    lookup_email: str | None,
    expect_new_thread: bool,
    expect_migrated: bool,
    expected_final_phone: str | None,
    expected_final_email: str | None,
) -> None:
    initial_threads: list[Thread] = []
    if existing_phone is not None or existing_email is not None:
        initial_threads.append(
            Thread(
                thread_id=50,
                tenant_id=1,
                phone_hash=(
                    hash_phone_e164(existing_phone, secret=TEST_SETTINGS.phone_hmac_secret)
                    if existing_phone
                    else None
                ),
                email=existing_email,
                expires_at=(
                    FIXED_NOW - timedelta(seconds=1)
                    if existing_expired
                    else FIXED_NOW + timedelta(days=10)
                ),
                migrated_from_thread_id=None,
                soft_deleted=False,
            )
        )

    repo = InMemoryThreadRepository(initial_threads)
    svc = IdentityService(repo, settings=TEST_SETTINGS)

    resolved = svc.resolve_thread(
        tenant_id=1,
        phone=lookup_phone,
        email=lookup_email,
        now=FIXED_NOW,
    )

    expected_hash = (
        hash_phone_e164(expected_final_phone, secret=TEST_SETTINGS.phone_hmac_secret)
        if expected_final_phone
        else None
    )
    assert resolved.phone_hash == expected_hash, case_name
    assert resolved.email == expected_final_email, case_name

    if expect_new_thread:
        assert resolved.expires_at == FIXED_NOW + timedelta(days=30), case_name
        if expect_migrated:
            assert resolved.thread_id != 50, case_name
            assert resolved.migrated_from_thread_id == 50, case_name
        else:
            assert resolved.migrated_from_thread_id is None, case_name
    else:
        assert resolved.thread_id == 50, case_name
        assert resolved.migrated_from_thread_id is None, case_name


def test_identity_email_matched_before_phone_when_both_exist_on_different_threads() -> None:
    phone_thread = Thread(
        thread_id=10,
        tenant_id=1,
        phone_hash=normalize_and_hash_phone("9876543210", secret=TEST_SETTINGS.phone_hmac_secret),
        email="phone_owner@example.com",
        expires_at=FIXED_NOW + timedelta(days=15),
    )
    email_thread = Thread(
        thread_id=20,
        tenant_id=1,
        phone_hash=None,
        email="email_owner@example.com",
        expires_at=FIXED_NOW + timedelta(days=15),
    )
    repo = InMemoryThreadRepository([phone_thread, email_thread])
    svc = IdentityService(repo, settings=TEST_SETTINGS)

    resolved = svc.resolve_thread(
        tenant_id=1,
        phone="9876543210",
        email="email_owner@example.com",
        now=FIXED_NOW,
    )
    # Email thread wins, and MUST NOT latch a phone_hash already claimed by another thread
    assert resolved.thread_id == 20
    assert resolved.phone_hash is None
    assert phone_thread.phone_hash == normalize_and_hash_phone(
        "9876543210", secret=TEST_SETTINGS.phone_hmac_secret
    )


def test_identity_tenant_isolation_allows_same_identifiers_across_tenants() -> None:
    t1_thread = Thread(
        thread_id=10,
        tenant_id=1,
        phone_hash=normalize_and_hash_phone("9876543210", secret=TEST_SETTINGS.phone_hmac_secret),
        email="shared@example.com",
        expires_at=FIXED_NOW + timedelta(days=15),
    )
    repo = InMemoryThreadRepository([t1_thread])
    svc = IdentityService(repo, settings=TEST_SETTINGS)

    t2_thread = svc.resolve_thread(
        tenant_id=2,
        phone="9876543210",
        email="shared@example.com",
        now=FIXED_NOW,
    )
    assert t2_thread.thread_id != t1_thread.thread_id
    assert t2_thread.tenant_id == 2


def test_latching_phone_onto_email_only_thread_and_refusing_to_overwrite() -> None:
    email_only_thread = Thread(
        thread_id=77,
        tenant_id=1,
        phone_hash=None,
        email="latch@example.com",
        expires_at=FIXED_NOW + timedelta(days=20),
    )
    repo = InMemoryThreadRepository([email_only_thread])
    svc = IdentityService(repo, settings=TEST_SETTINGS)

    # Step 1: Latch phone onto email-only thread
    latched = svc.resolve_thread(
        tenant_id=1,
        phone="9876543210",
        email="latch@example.com",
        now=FIXED_NOW,
    )
    first_phone_hash = normalize_and_hash_phone("9876543210", secret=TEST_SETTINGS.phone_hmac_secret)
    assert latched.thread_id == 77
    assert latched.phone_hash == first_phone_hash

    # Step 2: Refuse to overwrite existing phone with a different phone on subsequent match
    after_second_call = svc.resolve_thread(
        tenant_id=1,
        phone="9123456789",
        email="latch@example.com",
        now=FIXED_NOW,
    )
    assert after_second_call.thread_id == 77
    assert after_second_call.phone_hash == first_phone_hash


# ---------------------------------------------------------------------------
# 2. OrderService tests
# ---------------------------------------------------------------------------


def _build_order_fixture() -> tuple[Thread, Thread, OrderService]:
    cust_a_hash = hash_phone_e164(CUSTOMER_A_PHONE_E164, secret=TEST_SETTINGS.phone_hmac_secret)
    cust_b_hash = hash_phone_e164(CUSTOMER_B_PHONE_E164, secret=TEST_SETTINGS.phone_hmac_secret)

    thread_a = Thread(
        thread_id=101,
        tenant_id=TENANT_ID,
        phone_hash=cust_a_hash,
        email="alice@example.com",
        expires_at=FIXED_NOW + timedelta(days=30),
    )
    thread_b = Thread(
        thread_id=102,
        tenant_id=TENANT_ID,
        phone_hash=cust_b_hash,
        email="bob@example.com",
        expires_at=FIXED_NOW + timedelta(days=30),
    )

    orders: list[CustomerOrder] = []
    # 12 recent orders for Customer A (to test max 10 limit & newest first; mix of phone & email ownership)
    for idx in range(1, 13):
        orders.append(
            CustomerOrder(
                order_pk=1000 + idx,
                tenant_id=TENANT_ID,
                order_ref=f"ORD-A-{idx:02d}",
                phone_hash=cust_a_hash if idx % 2 == 1 else None,
                email="alice@example.com" if idx % 2 == 0 else None,
                payment_mode="PREPAID",
                placed_at=FIXED_NOW - timedelta(days=13 - idx),
                cancelled_at=None,
                items=[{"sku": f"SKU-{idx}", "qty": 1}],
            )
        )

    # 1 order older than 30 days for Customer A (should be excluded by lookback_days=30)
    orders.append(
        CustomerOrder(
            order_pk=1099,
            tenant_id=TENANT_ID,
            order_ref="ORD-A-OLD",
            phone_hash=cust_a_hash,
            email="alice@example.com",
            payment_mode="PREPAID",
            placed_at=FIXED_NOW - timedelta(days=31),
            cancelled_at=None,
            items=[],
        )
    )

    # Customer B's order with CUSTOMER_B_AWB
    orders.append(
        CustomerOrder(
            order_pk=2001,
            tenant_id=TENANT_ID,
            order_ref="ORD-B-01",
            phone_hash=cust_b_hash,
            email="bob@example.com",
            payment_mode="PREPAID",
            placed_at=FIXED_NOW - timedelta(days=2),
            cancelled_at=None,
            items=[{"sku": "SKU-B", "qty": 1}],
        )
    )

    shipments = [
        Shipment(
            id=3001,
            order_pk=1012,
            awb="AWB-CUSTA-1012",
            carrier_code="DELHIVERY",
            is_reverse=False,
            status=ShipmentStatus.IN_TRANSIT,
        ),
        Shipment(
            id=3002,
            order_pk=2001,
            awb=CUSTOMER_B_AWB,
            carrier_code="BLUEDART",
            is_reverse=False,
            status=ShipmentStatus.DELIVERED,
        ),
    ]

    repo = InMemoryOrderRepository(orders=orders, shipments=shipments)
    return thread_a, thread_b, OrderService(repo)


def test_list_owned_orders_max_10_newest_first_and_lookback() -> None:
    thread_a, _, order_svc = _build_order_fixture()

    cards = order_svc.list_owned_orders(thread_a, lookback_days=30, now=FIXED_NOW)

    assert len(cards) == 10
    assert all(isinstance(c, OrderCard) for c in cards)
    assert [c.order_ref for c in cards] == [f"ORD-A-{idx:02d}" for idx in range(12, 2, -1)]
    assert cards[0].placed_at > cards[-1].placed_at


@pytest.mark.parametrize(
    ("order_ref", "awb", "should_succeed", "expected_order_ref"),
    [
        ("ORD-A-12", None, True, "ORD-A-12"),
        (None, "AWB-CUSTA-1012", True, "ORD-A-12"),
        ("ORD-A-12", "AWB-CUSTA-1012", True, "ORD-A-12"),
        ("DOES-NOT-EXIST", None, False, None),
        (None, "AWB-DOES-NOT-EXIST", False, None),
        ("ORD-B-01", None, False, None),
        (None, CUSTOMER_B_AWB, False, None),
        ("ORD-A-12", CUSTOMER_B_AWB, False, None),
        (None, None, False, None),
    ],
)
def test_get_owned_order_and_customer_a_asking_for_customer_b_awb(
    order_ref: str | None,
    awb: str | None,
    should_succeed: bool,
    expected_order_ref: str | None,
) -> None:
    thread_a, _, order_svc = _build_order_fixture()

    if should_succeed:
        order = order_svc.get_owned_order(thread_a, order_ref=order_ref, awb=awb)
        assert order.order_ref == expected_order_ref
    else:
        with pytest.raises(NotOwned) as exc_info:
            order_svc.get_owned_order(thread_a, order_ref=order_ref, awb=awb)
        assert str(exc_info.value) == "Order not found"


def test_missing_and_not_yours_raise_identical_not_owned_response() -> None:
    thread_a, _, order_svc = _build_order_fixture()

    with pytest.raises(NotOwned) as missing_exc:
        order_svc.get_owned_order(thread_a, awb="AWB-NONEXISTENT-000")

    with pytest.raises(NotOwned) as other_customer_exc:
        order_svc.get_owned_order(thread_a, awb=CUSTOMER_B_AWB)

    assert type(missing_exc.value) is type(other_customer_exc.value)
    assert str(missing_exc.value) == str(other_customer_exc.value)


# ---------------------------------------------------------------------------
# 3. TrackingService.view tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("copies", "status", "events_reasons", "expected_text"),
    [
        # 1. Tenant-specific copy for (tenant_id, status, failure_reason) wins
        (
            {
                (1, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.CUSTOMER_UNAVAILABLE)): (
                    "Tenant custom: missed you at home."
                ),
                (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.CUSTOMER_UNAVAILABLE)): (
                    "Global specific: customer unavailable."
                ),
                (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.NONE)): (
                    "Global generic failed delivery."
                ),
            },
            ShipmentStatus.FAILED_DELIVERY,
            [
                (FIXED_NOW - timedelta(hours=5), ShipmentStatus.FAILED_DELIVERY, FailureReason.ADDRESS_ISSUE),
                (FIXED_NOW - timedelta(hours=1), ShipmentStatus.FAILED_DELIVERY, FailureReason.CUSTOMER_UNAVAILABLE),
            ],
            "Tenant custom: missed you at home.",
        ),
        # 2. Falls back to (0, status, failure_reason) when tenant has no override
        (
            {
                (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.CUSTOMER_UNAVAILABLE)): (
                    "Global specific: customer unavailable."
                ),
                (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.NONE)): (
                    "Global generic failed delivery."
                ),
            },
            ShipmentStatus.FAILED_DELIVERY,
            [
                (FIXED_NOW - timedelta(hours=2), ShipmentStatus.FAILED_DELIVERY, FailureReason.CUSTOMER_UNAVAILABLE),
            ],
            "Global specific: customer unavailable.",
        ),
        # 3. Falls back to (0, status, 0) when specific failure_reason copy does not exist
        (
            {
                (0, int(ShipmentStatus.FAILED_DELIVERY), int(FailureReason.NONE)): (
                    "Global generic failed delivery."
                ),
            },
            ShipmentStatus.FAILED_DELIVERY,
            [
                (FIXED_NOW - timedelta(hours=2), ShipmentStatus.FAILED_DELIVERY, FailureReason.ENTRY_RESTRICTED),
            ],
            "Global generic failed delivery.",
        ),
    ],
)
def test_tracking_view_status_copy_fallback_and_latest_failure_reason(
    copies: dict[tuple[int, int, int], str],
    status: ShipmentStatus,
    events_reasons: list[tuple[datetime, ShipmentStatus, FailureReason]],
    expected_text: str,
) -> None:
    tenant = Tenant(tenant_id=1, name="Brand", timezone="Asia/Kolkata", features={})
    order = CustomerOrder(
        order_pk=1,
        tenant_id=1,
        order_ref="ORD-1",
        payment_mode="PREPAID",
        placed_at=FIXED_NOW - timedelta(days=2),
    )
    shipment = Shipment(
        id=10,
        order_pk=1,
        awb="AWB-1",
        carrier_code="DELHIVERY",
        is_reverse=False,
        status=status,
        edd=FIXED_NOW.date(),
    )
    events = [
        ShipmentEvent(
            shipment_id=10,
            event_at=at,
            status=st,
            failure_reason=fr,
            location="Mumbai",
            raw_remark="Secret remark",
        )
        for at, st, fr in events_reasons
    ]

    svc = TrackingService(InMemoryStatusCopyRepository(copies))
    view = svc.view(order, shipment, events, tenant, now=FIXED_NOW)

    assert isinstance(view, TrackingView)
    assert view.customer_text == expected_text
    with pytest.raises(FrozenInstanceError):
        view.customer_text = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("now_ist_iso", "edd_date", "status", "expected_breached"),
    [
        # 23:30 IST on 2026-10-18 (18:00 UTC on 2026-10-18); in IST it is 2026-10-18, so EDD 2026-10-17 IS breached
        # even if a western timezone or UTC comparison on a next-day boundary would differ
        ("2026-10-18T23:30:00+05:30", date(2026, 10, 17), ShipmentStatus.IN_TRANSIT, True),
        # 23:30 IST on 2026-10-18 (18:00 UTC on 2026-10-18); EDD is 2026-10-18 -> today in IST == edd -> NOT breached
        ("2026-10-18T23:30:00+05:30", date(2026, 10, 18), ShipmentStatus.IN_TRANSIT, False),
        # 00:30 IST on 2026-10-19 (19:00 UTC on 2026-10-18: UTC is still the previous day 2026-10-18!),
        # EDD is 2026-10-18 -> today in IST is 2026-10-19 > 2026-10-18 -> BREACHED
        ("2026-10-19T00:30:00+05:30", date(2026, 10, 18), ShipmentStatus.IN_TRANSIT, True),
        # Terminal statuses never breach even when today > edd
        ("2026-10-18T23:30:00+05:30", date(2026, 10, 15), ShipmentStatus.DELIVERED, False),
        ("2026-10-18T23:30:00+05:30", date(2026, 10, 15), ShipmentStatus.CANCELLED, False),
        ("2026-10-18T23:30:00+05:30", date(2026, 10, 15), ShipmentStatus.RTO_DELIVERED, False),
        # EDD is None -> never breached
        ("2026-10-18T23:30:00+05:30", None, ShipmentStatus.IN_TRANSIT, False),
    ],
)
def test_tracking_view_edd_boundary_in_tenant_timezone(
    now_ist_iso: str,
    edd_date: date | None,
    status: ShipmentStatus,
    expected_breached: bool,
) -> None:
    """Test EDD breach calculation including when IST is past midnight or at 23:30 IST vs UTC."""
    now_utc = datetime.fromisoformat(now_ist_iso).astimezone(UTC)
    tenant = Tenant(tenant_id=1, name="Brand IN", timezone="Asia/Kolkata", features={})
    order = CustomerOrder(
        order_pk=1,
        tenant_id=1,
        order_ref="ORD-EDD",
        payment_mode="PREPAID",
        placed_at=now_utc - timedelta(days=3),
    )
    shipment = Shipment(
        id=11,
        order_pk=1,
        awb="AWB-EDD",
        carrier_code="BLUEDART",
        is_reverse=False,
        status=status,
        edd=edd_date,
    )

    view = TrackingService().view(order, shipment, [], tenant, now=now_utc)
    assert view.edd_breached is expected_breached


def test_tracking_view_edd_boundary_2330_ist_when_utc_is_previous_day_relative_to_tz() -> None:
    """Explicit test for 23:30 IST and the UTC day boundary."""
    # At 2026-10-18 23:30 IST, local date in Asia/Kolkata is 2026-10-18.
    # If we also test a timezone ahead of UTC where 23:30 local has UTC on previous day, or IST vs UTC:
    ist_tz = ZoneInfo("Asia/Kolkata")
    moment_2330_ist = datetime(2026, 10, 18, 23, 30, 0, tzinfo=ist_tz)
    moment_utc = moment_2330_ist.astimezone(UTC)

    tenant = Tenant(tenant_id=1, name="Brand IN", timezone="Asia/Kolkata", features={})
    order = CustomerOrder(
        order_pk=1,
        tenant_id=1,
        order_ref="ORD-IST",
        payment_mode="PREPAID",
        placed_at=moment_utc - timedelta(days=4),
    )

    # Same-day EDD (2026-10-18) at 23:30 IST is not yet breached
    shipment_same_day = Shipment(
        id=21,
        order_pk=1,
        awb="AWB-21",
        carrier_code="DELHIVERY",
        is_reverse=False,
        status=ShipmentStatus.OUT_FOR_DELIVERY,
        edd=date(2026, 10, 18),
    )
    assert TrackingService().view(order, shipment_same_day, [], tenant, now=moment_utc).edd_breached is False

    # 31 minutes later (00:01 IST on 2026-10-19), UTC is STILL 2026-10-18 (18:31 UTC),
    # but in tenant's IST timezone today is 2026-10-19 > 2026-10-18 -> breached!
    moment_0001_next_day_ist = (moment_2330_ist + timedelta(minutes=31)).astimezone(UTC)
    assert moment_0001_next_day_ist.date() == date(2026, 10, 18)  # UTC is still previous day
    assert (
        TrackingService().view(order, shipment_same_day, [], tenant, now=moment_0001_next_day_ist).edd_breached
        is True
    )


def test_tracking_view_ofd_attempts_and_last_5_events_sanitization() -> None:
    tenant = Tenant(tenant_id=1, name="Brand", timezone="Asia/Kolkata", features={})
    order = CustomerOrder(
        order_pk=1,
        tenant_id=1,
        order_ref="ORD-EVENTS",
        payment_mode="COD",
        placed_at=FIXED_NOW - timedelta(days=5),
    )
    shipment = Shipment(
        id=30,
        order_pk=1,
        awb="AWB-30",
        carrier_code="BLUEDART",
        is_reverse=False,
        status=ShipmentStatus.OUT_FOR_DELIVERY,
        edd=FIXED_NOW.date() + timedelta(days=1),
    )

    raw_Progression = [
        (FIXED_NOW - timedelta(days=5), ShipmentStatus.ORDER_PLACED, "Warehouse"),
        (FIXED_NOW - timedelta(days=4), ShipmentStatus.MANIFESTED, "Warehouse"),
        (FIXED_NOW - timedelta(days=3), ShipmentStatus.PICKED_UP, "Origin Hub"),
        (FIXED_NOW - timedelta(days=2), ShipmentStatus.IN_TRANSIT, "Transit Hub"),
        (FIXED_NOW - timedelta(days=1, hours=6), ShipmentStatus.OUT_FOR_DELIVERY, "Local Hub 1"),
        (FIXED_NOW - timedelta(days=1, hours=2), ShipmentStatus.FAILED_DELIVERY, "Customer Area"),
        (FIXED_NOW - timedelta(hours=3), ShipmentStatus.OUT_FOR_DELIVERY, "Local Hub 1"),
    ]
    events = [
        ShipmentEvent(
            shipment_id=30,
            event_at=at,
            status=st,
            failure_reason=(
                FailureReason.CUSTOMER_UNAVAILABLE
                if st == ShipmentStatus.FAILED_DELIVERY
                else FailureReason.NONE
            ),
            location=loc,
            raw_remark="Carrier Delhivery called customer at 221B Baker Street",
        )
        for at, st, loc in raw_Progression
    ]

    view = TrackingService().view(order, shipment, events, tenant, now=FIXED_NOW)

    assert view.ofd_attempts == 2
    assert len(view.last_events) == 5
    for ev in view.last_events:
        assert set(ev.keys()) == {"at", "status", "location"}
        assert "raw_remark" not in ev
        assert "carrier_code" not in ev
        assert "carrier" not in ev
        assert "address" not in ev

    assert [ev["status"] for ev in view.last_events] == [
        ShipmentStatus.PICKED_UP,
        ShipmentStatus.IN_TRANSIT,
        ShipmentStatus.OUT_FOR_DELIVERY,
        ShipmentStatus.FAILED_DELIVERY,
        ShipmentStatus.OUT_FOR_DELIVERY,
    ]


# ---------------------------------------------------------------------------
# 4. check_cancel(order, shipments, tenant) tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "cancel_enabled",
        "order_cancelled_at",
        "shipment_specs",
        "expected_ok",
        "expected_reason",
    ),
    [
        # Rule 1: tenant.features.cancel_enabled is false -> CANCEL_DISABLED (even if already cancelled)
        (
            False,
            FIXED_NOW,
            [(ShipmentStatus.ORDER_PLACED, False)],
            False,
            CancelReasonCode.CANCEL_DISABLED,
        ),
        (
            False,
            None,
            [(ShipmentStatus.ORDER_PLACED, False)],
            False,
            CancelReasonCode.CANCEL_DISABLED,
        ),
        # Rule 2: Already cancelled -> (True, ALREADY_CANCELLED)
        (
            True,
            FIXED_NOW - timedelta(hours=1),
            [(ShipmentStatus.CANCELLED, False)],
            True,
            CancelReasonCode.ALREADY_CANCELLED,
        ),
        (
            True,
            FIXED_NOW - timedelta(hours=1),
            [(ShipmentStatus.IN_TRANSIT, True)],
            True,
            CancelReasonCode.ALREADY_CANCELLED,
        ),
        # Rule 3: Any reverse shipment -> (False, REVERSE_SHIPMENT)
        (
            True,
            None,
            [(ShipmentStatus.ORDER_PLACED, False), (ShipmentStatus.ORDER_PLACED, True)],
            False,
            CancelReasonCode.REVERSE_SHIPMENT,
        ),
        # Rule 4: Any shipment status not in {ORDER_PLACED, MANIFESTED, PICKUP_FAILED} -> ALREADY_PICKED_UP
        (
            True,
            None,
            [(ShipmentStatus.PICKED_UP, False)],
            False,
            CancelReasonCode.ALREADY_PICKED_UP,
        ),
        (
            True,
            None,
            [(ShipmentStatus.IN_TRANSIT, False)],
            False,
            CancelReasonCode.ALREADY_PICKED_UP,
        ),
        (
            True,
            None,
            [(ShipmentStatus.OUT_FOR_DELIVERY, False)],
            False,
            CancelReasonCode.ALREADY_PICKED_UP,
        ),
        (
            True,
            None,
            [(ShipmentStatus.DELIVERED, False)],
            False,
            CancelReasonCode.ALREADY_PICKED_UP,
        ),
        (
            True,
            None,
            [(ShipmentStatus.ORDER_PLACED, False), (ShipmentStatus.IN_TRANSIT, False)],
            False,
            CancelReasonCode.ALREADY_PICKED_UP,
        ),
        # Rule 5: Otherwise -> (True, ELIGIBLE) for {ORDER_PLACED, MANIFESTED, PICKUP_FAILED}
        (
            True,
            None,
            [(ShipmentStatus.ORDER_PLACED, False)],
            True,
            CancelReasonCode.ELIGIBLE,
        ),
        (
            True,
            None,
            [(ShipmentStatus.MANIFESTED, False)],
            True,
            CancelReasonCode.ELIGIBLE,
        ),
        (
            True,
            None,
            [(ShipmentStatus.PICKUP_FAILED, False)],
            True,
            CancelReasonCode.ELIGIBLE,
        ),
        (
            True,
            None,
            [],
            True,
            CancelReasonCode.ELIGIBLE,
        ),
    ],
)
def test_check_cancel_rules_in_order(
    cancel_enabled: bool,
    order_cancelled_at: datetime | None,
    shipment_specs: list[tuple[ShipmentStatus, bool]],
    expected_ok: bool,
    expected_reason: CancelReasonCode,
) -> None:
    tenant = Tenant(
        tenant_id=1,
        name="Brand",
        timezone="Asia/Kolkata",
        features={"cancel_enabled": cancel_enabled},
    )
    order = CustomerOrder(
        order_pk=100,
        tenant_id=1,
        order_ref="ORD-CANCEL",
        payment_mode="PREPAID",
        placed_at=FIXED_NOW - timedelta(hours=3),
        cancelled_at=order_cancelled_at,
    )
    shipments = [
        Shipment(
            id=idx + 1,
            order_pk=100,
            awb=f"AWB-{idx + 1}",
            carrier_code="DELHIVERY",
            is_reverse=is_rev,
            status=st,
        )
        for idx, (st, is_rev) in enumerate(shipment_specs)
    ]

    ok, reason = check_cancel(order, shipments, tenant)
    assert ok is expected_ok
    assert reason == expected_reason
