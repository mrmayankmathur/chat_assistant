from __future__ import annotations

import importlib
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.core import DEFAULT_DATABASE_URL, hash_phone_e164
from app.models import (
    Base,
    CrmTicket,
    CrmTicketKind,
    CrmTicketStatus,
    CustomerOrder,
    Shipment,
    ShipmentStatus,
    Tenant,
    Thread,
)
from app.seed import (
    CUSTOMER_A_PHONE_E164,
    CUSTOMER_B_AWB,
    CUSTOMER_B_PHONE_E164,
    TENANT_ID,
    seed_database,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_alembic_package():
    """Import installed alembic library without shadowing from local ./alembic directory."""
    saved_path = list(sys.path)
    saved_mod = sys.modules.pop("alembic", None)
    try:
        root_variants = {str(PROJECT_ROOT), str(PROJECT_ROOT.resolve()), "", "."}
        sys.path = [p for p in sys.path if p not in root_variants]
        command_mod = importlib.import_module("alembic.command")
        config_mod = importlib.import_module("alembic.config")
        return command_mod, config_mod.Config
    except ImportError:
        if saved_mod is not None:
            sys.modules["alembic"] = saved_mod
        return None, None
    finally:
        sys.path = saved_path


@pytest.fixture(scope="session")
def db_engine():
    db_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except (ImportError, ModuleNotFoundError, OperationalError) as exc:
        pytest.skip(f"PostgreSQL database not available for live schema test ({exc})")

    command_mod, config_cls = _load_alembic_package()
    if command_mod is not None and config_cls is not None:
        alembic_cfg = config_cls(str(PROJECT_ROOT / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        # Non-destructive migration to head (never downgrade("base") on shared/configured databases)
        command_mod.upgrade(alembic_cfg, "head")
    else:
        Base.metadata.create_all(engine)

    with Session(engine) as session:
        seed_database(session)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    connection = db_engine.connect()
    outer_tx = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        outer_tx.rollback()
        connection.close()


def test_second_open_crm_ticket_same_kind_same_thread_raises_integrity_error(db_session: Session) -> None:
    """Assert partial unique (thread_id, kind) WHERE status = 1 on crm_ticket."""
    now = datetime.now(UTC)

    first_ticket = CrmTicket(
        id=9001,
        thread_id=101,
        crm_integration_id=1,
        kind=CrmTicketKind.TICKET,
        status=CrmTicketStatus.OPEN,
        external_ticket_id="ZD-1001",
        opened_at=now,
    )
    db_session.add(first_ticket)
    db_session.commit()

    duplicate_open_ticket = CrmTicket(
        id=9002,
        thread_id=101,
        crm_integration_id=1,
        kind=CrmTicketKind.TICKET,
        status=CrmTicketStatus.OPEN,
        external_ticket_id="ZD-1002",
        opened_at=now,
    )
    db_session.add(duplicate_open_ticket)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()

    # Verify a closed ticket of the same kind on the same thread is permitted
    closed_ticket = CrmTicket(
        id=9003,
        thread_id=101,
        crm_integration_id=1,
        kind=CrmTicketKind.TICKET,
        status=CrmTicketStatus.CLOSED,
        external_ticket_id="ZD-1000",
        opened_at=now,
        closed_at=now,
    )
    db_session.add(closed_ticket)
    db_session.commit()


def test_seed_data_invariants(db_session: Session) -> None:
    """Verify the seeded data satisfies all prompt invariants."""
    tenant = db_session.get(Tenant, TENANT_ID)
    assert tenant is not None

    threads = db_session.scalars(
        select(Thread).where(
            Thread.tenant_id == TENANT_ID,
            Thread.thread_id.in_([101, 102]),
        )
    ).all()
    assert len(threads) == 2

    # Phones are stored as 64-char hex HMAC-SHA256 digests, never raw E.164
    expected_hashes = {
        hash_phone_e164(CUSTOMER_A_PHONE_E164),
        hash_phone_e164(CUSTOMER_B_PHONE_E164),
    }
    assert {t.phone_hash for t in threads} == expected_hashes
    for t in threads:
        assert t.email == t.email.lower()
        assert not t.phone_hash.startswith("+")

    orders = db_session.scalars(
        select(CustomerOrder).where(
            CustomerOrder.tenant_id == TENANT_ID,
            CustomerOrder.order_pk.in_([1001, 1002, 1003, 1004, 1005, 1006]),
        )
    ).all()
    assert len(orders) == 6

    shipments = db_session.scalars(
        select(Shipment).where(Shipment.id.in_([2001, 2002, 2003, 2004, 2005, 2006]))
    ).all()
    assert len(shipments) == 6
    statuses = {s.status for s in shipments}
    assert statuses == {
        ShipmentStatus.ORDER_PLACED,
        ShipmentStatus.IN_TRANSIT,
        ShipmentStatus.OUT_FOR_DELIVERY,
        ShipmentStatus.FAILED_DELIVERY,
        ShipmentStatus.DELIVERED,
        ShipmentStatus.CANCELLED,
    }

    # One order must have its EDD breached (edd < today and not delivered/cancelled)
    today = datetime.now(UTC).date()
    breached = [
        s
        for s in shipments
        if s.edd is not None
        and s.edd < today
        and s.status not in (ShipmentStatus.DELIVERED, ShipmentStatus.CANCELLED)
    ]
    assert len(breached) == 1

    # Customer B owns CUSTOMER_B_AWB
    cust_b_shipment = db_session.scalar(
        select(Shipment).where(Shipment.awb == CUSTOMER_B_AWB)
    )
    assert cust_b_shipment is not None
    cust_b_order = db_session.get(CustomerOrder, cust_b_shipment.order_pk)
    assert cust_b_order is not None
    assert cust_b_order.phone_hash == hash_phone_e164(CUSTOMER_B_PHONE_E164)
