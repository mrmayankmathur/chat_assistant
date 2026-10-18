from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from api.chat import create_app
from app.models import (
    BusinessHours,
    Conversation,
    ConversationIntent,
    ConversationStatus,
    CrmIntegration,
    CrmProvider,
    CrmTicket,
    CrmTicketKind,
    CrmTicketStatus,
    Message,
    MessageRole,
    Tenant,
    Thread,
)
from repositories.chat_storage import OutboxEntry
from services.business_hours import BusinessHoursService
from services.handoff import (
    ALLOWED,
    CSAT_BUTTONS,
    CrmWebhookService,
    DelayedJobQueue,
    EscalationWorker,
    EvaluatorOutput,
    FakeChannelPublisher,
    FakeCrm,
    InactivityWorker,
    ResolutionEvaluator,
    can_transition_conversation,
    change_ticket_kind,
    close_crm_ticket,
    compute_crm_webhook_signature,
    format_unassigned_messages_for_evaluator,
    transition_conversation_status,
    validate_evaluator_output,
)

# ===========================================================================
# 1. BusinessHoursService tests
# ===========================================================================


def test_business_hours_fri_2200_to_sat_0200_checked_at_sat_0130() -> None:
    """Fri 22:00 -> Sat 02:00 overnight shift checked at Sat 01:30 in tenant TZ."""
    tz = "Asia/Kolkata"
    tz_info = ZoneInfo(tz)
    cfg = {
        "weekly": {
            # Friday is 2026-10-16; Saturday is 2026-10-17
            "fri": [{"open": "22:00", "close": "02:00"}],
        },
        "overrides": [],
    }
    sat_0130_local = datetime(2026, 10, 17, 1, 30, tzinfo=tz_info)
    at_utc = sat_0130_local.astimezone(UTC)

    is_open, shift_end_utc, msg = BusinessHoursService.is_open(cfg, tz, at_utc)

    expected_end_utc = datetime(2026, 10, 17, 2, 0, tzinfo=tz_info).astimezone(UTC)
    assert is_open is True
    assert shift_end_utc == expected_end_utc
    assert msg is None


def test_business_hours_sun_2300_to_mon_0100_across_week_wrap() -> None:
    """Sun 23:00 -> Mon 01:00 overnight shift across the Sunday->Monday week wrap."""
    tz = "America/New_York"
    tz_info = ZoneInfo(tz)
    cfg = {
        "weekly": {
            # Sunday is 2026-10-18; Monday is 2026-10-19
            "sun": [{"open": "23:00", "close": "01:00"}],
            "mon": [{"open": "09:00", "close": "17:00"}],
        },
        "overrides": [],
    }
    # Validate on save passes because Sun 23:00->Mon 01:00 wraps to [0, 60) which doesn't overlap Mon 09:00
    BusinessHoursService.validate(
        cfg,
        tz=tz,
        now=datetime(2026, 10, 18, 12, 0, tzinfo=UTC),
    )

    # Checked on Sunday at 23:30 local
    sun_2330_utc = datetime(2026, 10, 18, 23, 30, tzinfo=tz_info).astimezone(UTC)
    expected_end_utc = datetime(2026, 10, 19, 1, 0, tzinfo=tz_info).astimezone(UTC)

    open_sun, end_sun, _ = BusinessHoursService.is_open(cfg, tz, sun_2330_utc)
    assert open_sun is True
    assert end_sun == expected_end_utc

    # Checked on Monday at 00:30 local (tail of yesterday's Sunday overnight shift)
    mon_0030_utc = datetime(2026, 10, 19, 0, 30, tzinfo=tz_info).astimezone(UTC)
    open_mon, end_mon, _ = BusinessHoursService.is_open(cfg, tz, mon_0030_utc)
    assert open_mon is True
    assert end_mon == expected_end_utc


def test_business_hours_closed_override_on_weekday_and_last_override_wins() -> None:
    """A CLOSED override on a weekday replaces that day's weekly shifts and returns its message;
    if several overrides cover the day, the last matching one wins.
    """
    tz = "UTC"
    # Wednesday 2026-10-21
    wed_1100_utc = datetime(2026, 10, 21, 11, 0, tzinfo=UTC)

    cfg = {
        "weekly": {
            "wed": [{"open": "09:00", "close": "18:00"}],
        },
        "overrides": [
            {
                "start_date": "2026-10-20",
                "end_date": "2026-10-22",
                "kind": "CUSTOM",
                "shifts": [{"open": "10:00", "close": "14:00"}],
                "message": "Reduced hours",
            },
            {
                "start_date": "2026-10-21",
                "end_date": "2026-10-21",
                "kind": "CLOSED",
                "shifts": [],
                "message": "Closed for company holiday",
            },
        ],
    }

    is_open, shift_end_utc, msg = BusinessHoursService.is_open(cfg, tz, wed_1100_utc)
    assert is_open is False
    assert shift_end_utc is None
    assert msg == "Closed for company holiday"


def test_business_hours_picks_containing_interval_with_latest_end() -> None:
    """When multiple candidate intervals contain `at_utc`, pick the one with the latest end."""
    tz = "UTC"
    sat_0100_utc = datetime(2026, 10, 17, 1, 0, tzinfo=UTC)
    cfg = {
        "weekly": {
            "fri": [{"open": "22:00", "close": "02:00"}],
            "sat": [{"open": "00:30", "close": "04:00"}],
        },
        "overrides": [],
    }
    is_open, shift_end_utc, _ = BusinessHoursService.is_open(cfg, tz, sat_0100_utc)
    assert is_open is True
    assert shift_end_utc == datetime(2026, 10, 17, 4, 0, tzinfo=UTC)


def test_business_hours_validation_rejects_overlaps_and_invalid_overrides() -> None:
    now = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    # 1. Same-day overlap on weekly axis
    overlapping_weekly = {
        "weekly": {
            "mon": [
                {"open": "09:00", "close": "13:00"},
                {"open": "12:30", "close": "17:00"},
            ]
        },
        "overrides": [],
    }
    with pytest.raises(ValueError, match="Overlapping weekly shifts"):
        BusinessHoursService.save(overlapping_weekly, tz="UTC", now=now)

    # 2. Overnight wrap overlap across Sunday -> Monday boundary
    overnight_wrap_overlap = {
        "weekly": {
            "sun": [{"open": "22:00", "close": "02:00"}],
            "mon": [{"open": "01:30", "close": "09:00"}],
        },
        "overrides": [],
    }
    with pytest.raises(ValueError, match="Overlapping weekly shifts"):
        BusinessHoursService.save(overnight_wrap_overlap, tz="UTC", now=now)

    # 3. open == close rejected
    equal_open_close = {
        "weekly": {
            "tue": [{"open": "10:00", "close": "10:00"}],
        },
        "overrides": [],
    }
    with pytest.raises(ValueError, match="cannot be equal"):
        BusinessHoursService.save(equal_open_close, tz="UTC", now=now)

    # 4. CUSTOM override overnight shift rejected
    overnight_custom_override = {
        "weekly": {},
        "overrides": [
            {
                "start_date": "2026-10-20",
                "end_date": "2026-10-20",
                "kind": "CUSTOM",
                "shifts": [{"open": "22:00", "close": "02:00"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="CUSTOM override shifts cannot be overnight"):
        BusinessHoursService.save(overnight_custom_override, tz="UTC", now=now)

    # 5. Override end_date in the past rejected
    past_override = {
        "weekly": {},
        "overrides": [
            {
                "start_date": "2026-10-10",
                "end_date": "2026-10-17",
                "kind": "CLOSED",
            }
        ],
    }
    with pytest.raises(ValueError, match="cannot be in the past"):
        BusinessHoursService.save(past_override, tz="UTC", now=now)


# ===========================================================================
# 2. Escalation Worker & CRM Ticket Rules tests
# ===========================================================================


def _build_worker_fixture(*, supports_live_handover: bool = True) -> tuple[EscalationWorker, FakeCrm]:
    fake_crm = FakeCrm()
    worker = EscalationWorker(crm_client=fake_crm)
    worker.tenants[1] = Tenant(
        tenant_id=1,
        name="Acme D2C",
        timezone="UTC",
        features={"cancel_enabled": True},
    )
    worker.business_hours[1] = BusinessHours(
        tenant_id=1,
        weekly={"mon": [{"open": "09:00", "close": "18:00"}]},
        overrides=[],
        soft_deleted=False,
    )
    worker.crm_integrations[1] = CrmIntegration(
        id=10,
        tenant_id=1,
        provider=CrmProvider.CUSTOM_WEBHOOK,
        webhook_uuid=uuid.uuid4(),
        secret_ref="test-secret",
        supports_live_handover=supports_live_handover,
        soft_deleted=False,
    )
    worker.threads[100] = Thread(
        thread_id=100,
        tenant_id=1,
        soft_deleted=False,
    )
    worker.messages.extend(
        [
            Message(
                message_id=1,
                tenant_id=1,
                thread_id=100,
                channel="whatsapp",
                role=MessageRole.CUSTOMER,
                content="Where is my order?",
                soft_deleted=False,
            ),
            Message(
                message_id=2,
                tenant_id=1,
                thread_id=100,
                channel="whatsapp",
                role=MessageRole.ASSISTANT,
                content="Let me connect you with an agent.",
                soft_deleted=False,
            ),
        ]
    )
    return worker, fake_crm


def test_escalation_worker_live_vs_offline_and_existing_open_ticket_linking() -> None:
    worker, fake_crm = _build_worker_fixture(supports_live_handover=True)

    # Monday 10:00 UTC -> within business hours AND supports_live_handover => live=True => kind=2
    mon_1000 = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
    outbox = [
        OutboxEntry(
            event_type="escalation.requested",
            tenant_id=1,
            thread_id=100,
            payload={"reason": "Damaged package", "subject": "Damaged item"},
            created_at=mon_1000,
        )
    ]
    created_tickets = worker.consume_outbox(outbox, now=mon_1000)
    assert len(created_tickets) == 1
    assert outbox == []

    ticket = created_tickets[0]
    assert CrmTicketKind(int(ticket.kind)) == CrmTicketKind.LIVE
    assert CrmTicketStatus(int(ticket.status)) == CrmTicketStatus.OPEN
    assert ticket.external_conversation_id is not None
    assert len(fake_crm.replayed_transcripts[ticket.external_conversation_id]) == 2

    # Second escalation on same thread while kind=2 ticket is still OPEN -> links instead of creating new
    second_entry = OutboxEntry(
        event_type="escalation.requested",
        tenant_id=1,
        thread_id=100,
        payload={"reason": "Follow up on damaged package"},
        created_at=mon_1000 + timedelta(minutes=5),
    )
    linked_ticket = worker.process_escalation_event(second_entry, now=mon_1000 + timedelta(minutes=5))
    assert linked_ticket.id == ticket.id
    assert len(worker.tickets) == 1
    assert len(worker.ticket_escalations) == 2

    # Outside business hours (Monday 21:00 UTC) -> live=False -> kind=1 (TICKET)
    mon_2100 = datetime(2026, 10, 19, 21, 0, tzinfo=UTC)
    offline_entry = OutboxEntry(
        event_type="escalation.requested",
        tenant_id=1,
        thread_id=100,
        payload={"reason": "Late night inquiry"},
        created_at=mon_2100,
    )
    offline_ticket = worker.process_escalation_event(offline_entry, now=mon_2100)
    assert CrmTicketKind(int(offline_ticket.kind)) == CrmTicketKind.TICKET
    assert offline_ticket.external_ticket_id is not None


def test_escalation_worker_integrity_error_fallback_links_to_existing_ticket() -> None:
    worker, _ = _build_worker_fixture(supports_live_handover=False)
    now = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)

    # Pre-insert an open kind=1 ticket and simulate race condition where initial lookup misses
    # and _create_new_ticket raises IntegrityError
    concurrent_ticket = CrmTicket(
        id=777,
        thread_id=100,
        crm_integration_id=10,
        kind=CrmTicketKind.TICKET,
        status=CrmTicketStatus.OPEN,
        external_ticket_id="ext-777",
        opened_at=now,
    )
    call_count = 0
    orig_find = worker._find_open_ticket

    def _find_after_race(*, thread_id: int, kind: CrmTicketKind) -> CrmTicket | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None
        worker.tickets.append(concurrent_ticket)
        return orig_find(thread_id=thread_id, kind=kind)

    worker._find_open_ticket = _find_after_race  # type: ignore[method-assign]
    worker._simulate_integrity_error_once = True

    entry = OutboxEntry(
        event_type="escalation.requested",
        tenant_id=1,
        thread_id=100,
        payload={"reason": "Concurrent escalation"},
        created_at=now,
    )
    resolved_ticket = worker.process_escalation_event(entry, now=now)
    assert resolved_ticket.id == 777
    assert any(te.crm_ticket_id == 777 for te in worker.ticket_escalations)


def test_crm_ticket_lifecycle_rules() -> None:
    """Ticket rules:
    - CLOSED is terminal.
    - The only allowed kind change is 2 -> 3.
    - closed_at is set once.
    - Closing a kind=2 ticket moves its conversation to RESOLVED_BY_HUMAN (5).
    """
    worker, _ = _build_worker_fixture(supports_live_handover=True)
    now = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)

    ticket = worker.process_escalation_event(
        OutboxEntry(
            event_type="escalation.requested",
            tenant_id=1,
            thread_id=100,
            payload={"reason": "Live agent needed"},
            created_at=now,
        ),
        now=now,
    )
    assert CrmTicketKind(int(ticket.kind)) == CrmTicketKind.LIVE

    # Illegal kind transitions: 2 -> 1, 1 -> 2, 1 -> 3, 3 -> 2
    with pytest.raises(ValueError, match="only 2 -> 3 is allowed"):
        change_ticket_kind(ticket, CrmTicketKind.TICKET)

    ticket_kind_1 = CrmTicket(
        id=201,
        thread_id=100,
        crm_integration_id=10,
        kind=CrmTicketKind.TICKET,
        status=CrmTicketStatus.OPEN,
        opened_at=now,
    )
    with pytest.raises(ValueError, match="only 2 -> 3 is allowed"):
        change_ticket_kind(ticket_kind_1, CrmTicketKind.LIVE)
    with pytest.raises(ValueError, match="only 2 -> 3 is allowed"):
        change_ticket_kind(ticket_kind_1, CrmTicketKind.LIVE_TURNED_TICKET)

    # Closing kind=2 ticket sets closed_at once and moves conversation to RESOLVED_BY_HUMAN (5)
    close_time = now + timedelta(minutes=20)
    worker.close_ticket(ticket, now=close_time)
    assert CrmTicketStatus(int(ticket.status)) == CrmTicketStatus.CLOSED
    assert ticket.closed_at == close_time
    assert ConversationStatus(int(worker.conversations[0].status)) == ConversationStatus.RESOLVED_BY_HUMAN

    # CLOSED is terminal: cannot close again (closed_at set once) and cannot change kind
    with pytest.raises(ValueError, match="CLOSED is terminal"):
        worker.close_ticket(ticket, now=close_time + timedelta(minutes=5))
    assert ticket.closed_at == close_time

    with pytest.raises(ValueError, match="CLOSED is terminal"):
        change_ticket_kind(ticket, CrmTicketKind.LIVE_TURNED_TICKET)

    # Allowed kind transition on an OPEN kind=2 ticket: 2 -> 3
    open_live_ticket = CrmTicket(
        id=202,
        thread_id=100,
        crm_integration_id=10,
        kind=CrmTicketKind.LIVE,
        status=CrmTicketStatus.OPEN,
        opened_at=now,
    )
    change_ticket_kind(open_live_ticket, CrmTicketKind.LIVE_TURNED_TICKET)
    assert CrmTicketKind(int(open_live_ticket.kind)) == CrmTicketKind.LIVE_TURNED_TICKET

    # Closing kind=3 ticket does NOT move conversation to RESOLVED_BY_HUMAN
    conv_for_k3 = Conversation(
        conversation_id=303,
        thread_id=100,
        status=ConversationStatus.ESCALATED,
        intents=[int(ConversationIntent.HUMAN_HANDOFF)],
        summary="Escalated",
    )
    close_crm_ticket(open_live_ticket, conversations=[conv_for_k3], now=close_time)
    assert ConversationStatus(int(conv_for_k3.status)) == ConversationStatus.ESCALATED


# ===========================================================================
# 3. CRM Webhook POST /v1/crm/{webhook_uuid} tests
# ===========================================================================


def test_crm_webhook_verifies_signature_persists_role_3_and_publishes() -> None:
    publisher = FakeChannelPublisher()
    webhook_service = CrmWebhookService(channel_publisher=publisher)
    webhook_uuid = uuid.uuid4()
    secret = "whsec_super_secret_key"

    integration = CrmIntegration(
        id=10,
        tenant_id=1,
        provider="zendesk",
        webhook_uuid=webhook_uuid,
        secret_ref=secret,
        supports_live_handover=True,
        soft_deleted=False,
    )
    webhook_service.register_integration(integration)
    webhook_service.tickets.append(
        CrmTicket(
            id=501,
            thread_id=100,
            crm_integration_id=10,
            kind=CrmTicketKind.LIVE,
            status=CrmTicketStatus.OPEN,
            external_conversation_id="conv-abc",
            opened_at=datetime(2026, 10, 19, 10, 0, tzinfo=UTC),
        )
    )

    app = create_app(crm_webhook_service=webhook_service)
    client = TestClient(app)

    body_dict = {
        "event": "agent_reply",
        "external_conversation_id": "conv-abc",
        "external_message_id": "crm-msg-1",
        "channel": "whatsapp",
        "text": "Hi! I am a human support specialist and I have reshipped your order.",
    }
    raw_body = json.dumps(body_dict).encode("utf-8")
    valid_sig = compute_crm_webhook_signature(secret, raw_body)

    # 1. Invalid signature -> 401
    bad_resp = client.post(
        f"/v1/crm/{webhook_uuid}",
        content=raw_body,
        headers={"X-Crm-Signature": "sha256=deadbeef", "Content-Type": "application/json"},
    )
    assert bad_resp.status_code == 401
    assert len(webhook_service.messages) == 0
    assert len(publisher.published) == 0

    # 2. Valid signature -> 200, persisted as role=3 (HUMAN_AGENT), published to channel adapter
    ok_resp = client.post(
        f"/v1/crm/{webhook_uuid}",
        content=raw_body,
        headers={"X-Crm-Signature": f"sha256={valid_sig}", "Content-Type": "application/json"},
    )
    assert ok_resp.status_code == 200
    data = ok_resp.json()
    assert data["role"] == int(MessageRole.HUMAN_AGENT) == 3

    assert len(webhook_service.messages) == 1
    saved_msg = webhook_service.messages[0]
    assert int(saved_msg.role) == 3
    assert saved_msg.role == MessageRole.HUMAN_AGENT
    assert saved_msg.thread_id == 100
    assert saved_msg.content == body_dict["text"]

    assert len(publisher.published) == 1
    pub = publisher.published[0]
    assert pub.thread_id == 100
    assert pub.role == MessageRole.HUMAN_AGENT
    assert pub.content == body_dict["text"]


# ===========================================================================
# 4. Resolution Evaluator & State-Transition Table tests
# ===========================================================================


ALL_ALLOWED_PAIRS: list[tuple[ConversationStatus, ConversationStatus]] = [
    (src, dst)
    for src, targets in ALLOWED.items()
    for dst in sorted(targets, key=lambda s: int(s))
]

ILLEGAL_PAIRS: list[tuple[ConversationStatus, ConversationStatus]] = [
    (ConversationStatus.RESOLVED, ConversationStatus.OPEN),
    (ConversationStatus.RESOLVED, ConversationStatus.ESCALATED),
    (ConversationStatus.ABANDONED, ConversationStatus.OPEN),
    (ConversationStatus.RESOLVED_BY_HUMAN, ConversationStatus.OPEN),
    (ConversationStatus.ESCALATED, ConversationStatus.OPEN),
    (ConversationStatus.ESCALATED, ConversationStatus.RESOLVED),
]


@pytest.mark.parametrize(("from_status", "to_status"), ALL_ALLOWED_PAIRS)
def test_conversation_state_transition_table_allowed_pairs(
    from_status: ConversationStatus,
    to_status: ConversationStatus,
) -> None:
    """Every (from_status, to_status) pair in ALLOWED must succeed."""
    assert can_transition_conversation(from_status, to_status) is True
    conv = Conversation(
        conversation_id=1,
        thread_id=100,
        status=from_status,
        intents=[],
        summary="initial",
    )
    applied = transition_conversation_status(conv, to_status)
    assert applied is True
    assert ConversationStatus(int(conv.status)) == to_status


@pytest.mark.parametrize(("from_status", "to_status"), ILLEGAL_PAIRS)
def test_conversation_state_transition_table_illegal_pairs_rejected(
    from_status: ConversationStatus,
    to_status: ConversationStatus,
) -> None:
    """Illegal transitions must return False, be logged, and leave conversation.status unchanged."""
    assert len(ILLEGAL_PAIRS) >= 3
    assert can_transition_conversation(from_status, to_status) is False
    conv = Conversation(
        conversation_id=1,
        thread_id=100,
        status=from_status,
        intents=[],
        summary="initial",
    )
    applied = transition_conversation_status(conv, to_status)
    assert applied is False
    assert ConversationStatus(int(conv.status)) == from_status


def test_evaluator_message_formatting_and_output_validation() -> None:
    now = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
    msgs = [
        Message(
            message_id=11,
            tenant_id=1,
            thread_id=100,
            channel="whatsapp",
            role=MessageRole.CUSTOMER,
            content="Where is my order #1001?",
            created_at=now - timedelta(minutes=18),
            soft_deleted=False,
        ),
        Message(
            message_id=12,
            tenant_id=1,
            thread_id=100,
            channel="whatsapp",
            role=MessageRole.ASSISTANT,
            content="Your order #1001 is out for delivery today.",
            created_at=now - timedelta(minutes=17),
            soft_deleted=False,
        ),
    ]

    rendered = format_unassigned_messages_for_evaluator(msgs, now=now)
    assert rendered == (
        "[id=11 | 18 min ago] Customer: Where is my order #1001?\n"
        "[id=12 | 17 min ago] Assistant: Your order #1001 is out for delivery today."
    )

    # 1. Valid output with exact multiset match
    valid_out = validate_evaluator_output(
        {
            "conversations": [
                {
                    "message_ids": [11, 12],
                    "status": int(ConversationStatus.RESOLVED),
                    "intents": [int(ConversationIntent.TRACK_ORDER)],
                    "summary": "Customer asked for tracking; assistant provided out-for-delivery status.",
                    "escalation_subject": None,
                    "escalation_summary": None,
                }
            ]
        },
        expected_message_ids=[11, 12],
    )
    assert isinstance(valid_out, EvaluatorOutput)

    # 2. Reject if multiset of returned IDs != input IDs (missing ID, extra ID, or duplicate ID)
    with pytest.raises(ValueError, match="multiset"):
        validate_evaluator_output(
            {
                "conversations": [
                    {
                        "message_ids": [11],
                        "status": int(ConversationStatus.RESOLVED),
                        "intents": [1],
                        "summary": "Missing 12",
                        "escalation_subject": None,
                        "escalation_summary": None,
                    }
                ]
            },
            expected_message_ids=[11, 12],
        )

    with pytest.raises(ValueError, match="multiset"):
        validate_evaluator_output(
            {
                "conversations": [
                    {
                        "message_ids": [11, 11, 12],
                        "status": int(ConversationStatus.RESOLVED),
                        "intents": [1],
                        "summary": "Duplicate 11",
                        "escalation_subject": None,
                        "escalation_summary": None,
                    }
                ]
            },
            expected_message_ids=[11, 12],
        )

    # 3. Reject if ESCALATED (status=3) has no escalation_subject
    with pytest.raises(ValueError, match="escalation_subject"):
        validate_evaluator_output(
            {
                "conversations": [
                    {
                        "message_ids": [11, 12],
                        "status": int(ConversationStatus.ESCALATED),
                        "intents": [int(ConversationIntent.HUMAN_HANDOFF)],
                        "summary": "Escalated without subject",
                        "escalation_subject": None,
                        "escalation_summary": "Needs human",
                    }
                ]
            },
            expected_message_ids=[11, 12],
        )

    # 4. Reject if status == 5 (RESOLVED_BY_HUMAN is never produced by LLM)
    with pytest.raises(ValueError, match="Status 5"):
        validate_evaluator_output(
            {
                "conversations": [
                    {
                        "message_ids": [11, 12],
                        "status": int(ConversationStatus.RESOLVED_BY_HUMAN),
                        "intents": [1],
                        "summary": "Illegal LLM status 5",
                        "escalation_subject": None,
                        "escalation_summary": None,
                    }
                ]
            },
            expected_message_ids=[11, 12],
        )


class _StubAnthropicMessages:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self.payloads) - 1)
        payload = self.payloads[idx]
        return SimpleNamespace(
            content=[{"type": "text", "text": json.dumps(payload)}],
            stop_reason="end_turn",
        )


def test_resolution_evaluator_retries_once_alerts_on_double_failure_and_never_writes_partially() -> None:
    now = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)

    # Case A: Both attempts fail validation -> alerts after 2 attempts and writes nothing
    bad_payload = {
        "conversations": [
            {
                "message_ids": [101],  # missing 102!
                "status": int(ConversationStatus.ESCALATED),
                "intents": [5],
                "summary": "Escalated without subject and missing ID",
                "escalation_subject": None,
                "escalation_summary": None,
            }
        ]
    }
    stub_messages = _StubAnthropicMessages([bad_payload, bad_payload])
    client = SimpleNamespace(messages=stub_messages)

    evaluator = ResolutionEvaluator(anthropic_client=client)
    evaluator.messages.extend(
        [
            Message(
                message_id=101,
                tenant_id=1,
                thread_id=100,
                channel="whatsapp",
                role=MessageRole.CUSTOMER,
                content="Help me!",
                created_at=now - timedelta(minutes=25),
                soft_deleted=False,
            ),
            Message(
                message_id=102,
                tenant_id=1,
                thread_id=100,
                channel="whatsapp",
                role=MessageRole.ASSISTANT,
                content="Escalating now.",
                created_at=now - timedelta(minutes=24),
                soft_deleted=False,
            ),
            # Thread 200 is only idle for 3 minutes (< 10 min threshold) -> must be ignored by run_cron
            Message(
                message_id=201,
                tenant_id=1,
                thread_id=200,
                channel="whatsapp",
                role=MessageRole.CUSTOMER,
                content="Active chat",
                created_at=now - timedelta(minutes=3),
                soft_deleted=False,
            ),
        ]
    )

    convs = asyncio.run(evaluator.run_cron(now=now))
    assert convs == []
    assert len(stub_messages.calls) == 2
    assert len(evaluator.alerts) == 1
    assert evaluator.alerts[0].thread_id == 100
    assert evaluator.alerts[0].attempts == 2
    # Never write partially
    assert evaluator.conversations == []
    assert evaluator.conversation_messages == []
    assert evaluator.escalations == []


def test_resolution_evaluator_succeeds_on_retry_upserts_idempotently_and_skips_illegal_transitions() -> None:
    now = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)

    first_bad = {
        "conversations": [
            {
                "message_ids": [101, 102],
                "status": int(ConversationStatus.ESCALATED),
                "intents": [5],
                "summary": "Missing subject on first try",
                "escalation_subject": None,
                "escalation_summary": None,
            }
        ]
    }
    second_good = {
        "conversations": [
            {
                "message_ids": [101, 102],
                "status": int(ConversationStatus.ESCALATED),
                "intents": [int(ConversationIntent.HUMAN_HANDOFF)],
                "summary": "Customer requested human agent for damaged item.",
                "escalation_subject": "Damaged item replacement",
                "escalation_summary": "Box arrived crushed.",
            }
        ]
    }
    stub_messages = _StubAnthropicMessages([first_bad, second_good])
    client = SimpleNamespace(messages=stub_messages)

    evaluator = ResolutionEvaluator(anthropic_client=client)
    evaluator.messages.extend(
        [
            Message(
                message_id=101,
                tenant_id=1,
                thread_id=100,
                channel="whatsapp",
                role=MessageRole.CUSTOMER,
                content="My box arrived crushed",
                created_at=now - timedelta(minutes=20),
                soft_deleted=False,
            ),
            Message(
                message_id=102,
                tenant_id=1,
                thread_id=100,
                channel="whatsapp",
                role=MessageRole.ASSISTANT,
                content="I am escalating this to our team.",
                created_at=now - timedelta(minutes=19),
                soft_deleted=False,
            ),
        ]
    )

    convs = asyncio.run(evaluator.run_cron(now=now))
    assert len(convs) == 1
    assert len(evaluator.alerts) == 0
    assert ConversationStatus(int(convs[0].status)) == ConversationStatus.ESCALATED
    assert len(evaluator.conversation_messages) == 2
    assert len(evaluator.escalations) == 1
    assert evaluator.escalations[0].subject == "Damaged item replacement"

    # Now set the conversation to RESOLVED_BY_HUMAN (5) and evaluate a new message that tries to move it back to OPEN (illegal)
    convs[0].status = ConversationStatus.RESOLVED_BY_HUMAN
    evaluator.messages.append(
        Message(
            message_id=103,
            tenant_id=1,
            thread_id=100,
            channel="whatsapp",
            role=MessageRole.CUSTOMER,
            content="Thanks!",
            created_at=now - timedelta(minutes=15),
            soft_deleted=False,
        )
    )
    stub_messages.payloads = [
        {
            "conversations": [
                {
                    "message_ids": [103],
                    "status": int(ConversationStatus.OPEN),
                    "intents": [int(ConversationIntent.OTHER)],
                    "summary": "Follow-up thank you",
                    "escalation_subject": None,
                    "escalation_summary": None,
                }
            ]
        }
    ]
    stub_messages.calls.clear()

    updated = asyncio.run(evaluator.run_cron(now=now))
    assert len(updated) == 1
    # Illegal move RESOLVED_BY_HUMAN -> OPEN was logged and skipped; status stays RESOLVED_BY_HUMAN
    assert ConversationStatus(int(updated[0].status)) == ConversationStatus.RESOLVED_BY_HUMAN
    assert len(evaluator.skipped_transitions) == 1
    assert evaluator.skipped_transitions[0][1:] == (
        ConversationStatus.RESOLVED_BY_HUMAN,
        ConversationStatus.OPEN,
    )


# ===========================================================================
# 5. Inactivity Delayed Job tests (15 min ETA, no open ticket, 1 per 24h)
# ===========================================================================


def test_inactivity_delayed_job_enqueues_with_15m_eta_and_enforces_rules() -> None:
    queue = DelayedJobQueue()
    publisher = FakeChannelPublisher()
    worker = InactivityWorker(job_queue=queue, channel_publisher=publisher)

    t0 = datetime(2026, 10, 19, 14, 0, tzinfo=UTC)
    worker.messages.append(
        Message(
            message_id=50,
            tenant_id=1,
            thread_id=100,
            channel="whatsapp",
            role=MessageRole.ASSISTANT,
            content="Your package will arrive tomorrow.",
            created_at=t0,
            soft_deleted=False,
        )
    )

    job = worker.enqueue_for_assistant_reply(
        tenant_id=1,
        thread_id=100,
        assistant_message_id=50,
        replied_at=t0,
        channel="whatsapp",
    )
    assert job.eta == t0 + timedelta(minutes=15)

    # Before 15 minutes (at +14m) -> job is not due yet
    assert worker.run_due_jobs(now=t0 + timedelta(minutes=14)) == []
    assert len(publisher.published) == 0

    # At +15m with no open ticket -> sends close message + CSAT buttons
    sent = worker.run_due_jobs(now=t0 + timedelta(minutes=15))
    assert len(sent) == 1
    assert sent[0].payload is not None
    assert sent[0].payload["csat_buttons"] == CSAT_BUTTONS
    assert len(publisher.published) == 1
    assert publisher.published[0].payload == sent[0].payload

    # Another assistant reply 2 hours later (within 24h cooldown) -> suppressed (at most once per 24h per thread)
    t1 = t0 + timedelta(hours=2)
    worker.messages.append(
        Message(
            message_id=51,
            tenant_id=1,
            thread_id=100,
            channel="whatsapp",
            role=MessageRole.ASSISTANT,
            content="Anything else I can help with?",
            created_at=t1,
            soft_deleted=False,
        )
    )
    worker.enqueue_for_assistant_reply(
        tenant_id=1,
        thread_id=100,
        assistant_message_id=51,
        replied_at=t1,
        channel="whatsapp",
    )
    assert worker.run_due_jobs(now=t1 + timedelta(minutes=15)) == []
    assert len(publisher.published) == 1

    # After 25 hours (> 24h) -> allowed again unless an open ticket exists
    t2 = t0 + timedelta(hours=25)
    worker.messages.append(
        Message(
            message_id=52,
            tenant_id=1,
            thread_id=100,
            channel="whatsapp",
            role=MessageRole.ASSISTANT,
            content="Checking in on your new order.",
            created_at=t2,
            soft_deleted=False,
        )
    )
    # Add an OPEN ticket -> must skip sending close/CSAT
    open_ticket = CrmTicket(
        id=900,
        thread_id=100,
        crm_integration_id=10,
        kind=CrmTicketKind.TICKET,
        status=CrmTicketStatus.OPEN,
        opened_at=t2,
    )
    worker.tickets.append(open_ticket)
    worker.enqueue_for_assistant_reply(
        tenant_id=1,
        thread_id=100,
        assistant_message_id=52,
        replied_at=t2,
        channel="whatsapp",
    )
    assert worker.run_due_jobs(now=t2 + timedelta(minutes=15)) == []

    # Close the ticket and reply again after >24h -> sends CSAT
    open_ticket.status = CrmTicketStatus.CLOSED
    t3 = t2 + timedelta(hours=1)
    worker.messages.append(
        Message(
            message_id=53,
            tenant_id=1,
            thread_id=100,
            channel="whatsapp",
            role=MessageRole.ASSISTANT,
            content="All resolved!",
            created_at=t3,
            soft_deleted=False,
        )
    )
    worker.enqueue_for_assistant_reply(
        tenant_id=1,
        thread_id=100,
        assistant_message_id=53,
        replied_at=t3,
        channel="whatsapp",
    )
    sent_after_24h = worker.run_due_jobs(now=t3 + timedelta(minutes=15))
    assert len(sent_after_24h) == 1
    assert len(publisher.published) == 2
