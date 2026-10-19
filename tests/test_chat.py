from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.chat import create_app
from app.core import hash_phone_e164
from app.models import (
    AgentConfig,
    BusinessHours,
    CrmTicket,
    CrmTicketKind,
    CrmTicketStatus,
    CustomerOrder,
    FailureReason,
    Message,
    MessageRole,
    Policy,
    PolicyType,
    Shipment,
    ShipmentEvent,
    ShipmentStatus,
    Tenant,
    Tonality,
)
from app.seed import CUSTOMER_A_PHONE_E164, CUSTOMER_B_AWB, CUSTOMER_B_PHONE_E164, TENANT_ID
from app.settings import Settings
from repositories import (
    InMemoryChatRepository,
    InMemoryOrderRepository,
    InMemoryThreadRepository,
    clear_tenant_config_cache,
)
from services import (
    FALLBACK_REPLY_TEXT,
    OUTBOX_ESCALATION_EVENT,
    REPLY_JSON_SCHEMA,
    TOOLS_DEFINITIONS,
    ChatRequest,
    ChatService,
    IdentityService,
    OrderService,
    Reply,
    TrackingService,
)

FIXED_NOW = datetime(2026, 10, 18, 10, 0, 0, tzinfo=UTC)
TEST_SETTINGS = Settings(
    phone_hmac_secret="test-secret-key-123",
    default_phone_region="IN",
    llm_model="gpt-6-luna",
)


@dataclass
class FakeUsage:
    input_tokens: int = 120
    output_tokens: int = 45
    cache_read_input_tokens: int = 90
    cache_creation_input_tokens: int = 30


@dataclass
class FakeMessageResponse:
    content: list[dict[str, Any]]
    stop_reason: str = "end_turn"
    model: str = "gpt-6-luna"
    usage: FakeUsage = field(default_factory=FakeUsage)


class ScriptedMessagesAPI:
    def __init__(self, responses: list[FakeMessageResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeMessageResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedMessagesAPI received an unexpected messages.create call")
        return self._responses.pop(0)


class ScriptedAnthropicClient:
    def __init__(self, responses: list[FakeMessageResponse]) -> None:
        self.messages = ScriptedMessagesAPI(responses)


def _make_reply_json(
    *,
    reply_text: str,
    order_cards: list[dict[str, Any]] | None = None,
    ask_order_selection: bool = False,
    show_return_link: bool = False,
    escalate: bool = False,
) -> str:
    return json.dumps(
        {
            "reply_text": reply_text,
            "order_cards": order_cards or [],
            "ask_order_selection": ask_order_selection,
            "show_return_link": show_return_link,
            "escalate": escalate,
        }
    )


def _build_test_harness(
    scripted_responses: list[FakeMessageResponse],
    *,
    customer_a_order_count: int = 2,
    prior_messages: list[Message] | None = None,
    tickets: list[CrmTicket] | None = None,
) -> tuple[ChatService, ScriptedAnthropicClient, InMemoryChatRepository, TestClient]:
    clear_tenant_config_cache()

    cust_a_hash = hash_phone_e164(CUSTOMER_A_PHONE_E164, secret=TEST_SETTINGS.phone_hmac_secret)
    cust_b_hash = hash_phone_e164(CUSTOMER_B_PHONE_E164, secret=TEST_SETTINGS.phone_hmac_secret)

    tenant = Tenant(
        tenant_id=TENANT_ID,
        name="Lumina Botanics",
        timezone="Asia/Kolkata",
        features={"cancel_enabled": True},
    )
    agent_cfg = AgentConfig(
        agent_id=1,
        tenant_id=TENANT_ID,
        brand_prompt="You are the warm, clear support assistant for Lumina Botanics.",
        tonality=Tonality.FRIENDLY,
        is_active=True,
        version=1,
    )
    policies = [
        Policy(
            policy_id=1,
            tenant_id=TENANT_ID,
            policy_type=PolicyType.CANCELLATION,
            text="Orders can only be cancelled before carrier pickup.",
            windows={},
        )
    ]
    bh = BusinessHours(
        tenant_id=TENANT_ID,
        weekly={
            "mon": [["09:00", "21:00"]],
            "tue": [["09:00", "21:00"]],
            "wed": [["09:00", "21:00"]],
            "thu": [["09:00", "21:00"]],
            "fri": [["09:00", "21:00"]],
            "sat": [["09:00", "21:00"]],
            "sun": [["09:00", "21:00"]],
        },
        overrides=[],
    )

    orders: list[CustomerOrder] = []
    shipments: list[Shipment] = []
    events: list[ShipmentEvent] = []

    if customer_a_order_count >= 1:
        orders.append(
            CustomerOrder(
                order_pk=1001,
                tenant_id=TENANT_ID,
                order_ref="LUM-1001",
                phone_hash=cust_a_hash,
                email="alice@example.com",
                payment_mode="PREPAID",
                placed_at=FIXED_NOW - timedelta(days=1),
                cancelled_at=None,
                items=[{"sku": "SERUM-VITC", "name": "Vitamin C Glow Serum", "qty": 1}],
            )
        )
        shipments.append(
            Shipment(
                id=2001,
                order_pk=1001,
                awb="AWB-CUSTA-1001",
                carrier_code="DELHIVERY",
                is_reverse=False,
                status=ShipmentStatus.ORDER_PLACED,
                edd=date(2026, 10, 22),
            )
        )
        events.append(
            ShipmentEvent(
                shipment_id=2001,
                event_at=FIXED_NOW - timedelta(hours=6),
                status=ShipmentStatus.ORDER_PLACED,
                failure_reason=FailureReason.NONE,
                location="Mumbai Warehouse",
                raw_remark="Order created",
            )
        )

    if customer_a_order_count >= 2:
        orders.append(
            CustomerOrder(
                order_pk=1002,
                tenant_id=TENANT_ID,
                order_ref="LUM-1002",
                phone_hash=cust_a_hash,
                email="alice@example.com",
                payment_mode="COD",
                placed_at=FIXED_NOW - timedelta(days=3),
                cancelled_at=None,
                items=[{"sku": "CREAM-HYDRA", "name": "Deep Hydra Barrier Cream", "qty": 1}],
            )
        )
        shipments.append(
            Shipment(
                id=2002,
                order_pk=1002,
                awb="AWB-CUSTA-1002",
                carrier_code="BLUEDART",
                is_reverse=False,
                status=ShipmentStatus.IN_TRANSIT,
                edd=date(2026, 10, 16),
                picked_up_at=FIXED_NOW - timedelta(days=2),
            )
        )

    # Customer B's order & AWB
    orders.append(
        CustomerOrder(
            order_pk=1006,
            tenant_id=TENANT_ID,
            order_ref="LUM-1006",
            phone_hash=cust_b_hash,
            email="bob@example.com",
            payment_mode="PREPAID",
            placed_at=FIXED_NOW - timedelta(days=5),
            cancelled_at=None,
            items=[{"sku": "OIL-ROSEHIP", "name": "Cold-Pressed Rosehip Oil", "qty": 1}],
        )
    )
    shipments.append(
        Shipment(
            id=2006,
            order_pk=1006,
            awb=CUSTOMER_B_AWB,
            carrier_code="BLUEDART",
            is_reverse=False,
            status=ShipmentStatus.DELIVERED,
            edd=date(2026, 10, 15),
        )
    )

    thread_repo = InMemoryThreadRepository()
    order_repo = InMemoryOrderRepository(orders=orders, shipments=shipments)
    chat_repo = InMemoryChatRepository(
        tenant=tenant,
        agent_config=agent_cfg,
        channel_guidelines={"whatsapp": "Keep replies short and friendly."},
        policies=policies,
        business_hours=bh,
        shipments=shipments,
        events=events,
        tickets=tickets or [],
        messages=prior_messages or [],
    )

    identity_svc = IdentityService(thread_repo, settings=TEST_SETTINGS)
    order_svc = OrderService(order_repo)
    tracking_svc = TrackingService()
    client = ScriptedAnthropicClient(scripted_responses)

    chat_svc = ChatService(
        identity_service=identity_svc,
        order_service=order_svc,
        tracking_service=tracking_svc,
        chat_repo=chat_repo,
        anthropic_client=client,
        settings=TEST_SETTINGS,
    )
    app = create_app(chat_svc)
    test_client = TestClient(app)
    return chat_svc, client, chat_repo, test_client


# ---------------------------------------------------------------------------
# (a) duplicate external_message_id -> zero LLM calls
# ---------------------------------------------------------------------------


def test_a_duplicate_external_message_id_makes_zero_llm_calls() -> None:
    first_reply_json = _make_reply_json(
        reply_text="Which of your recent orders would you like to track?",
        order_cards=[
            {
                "order_ref": "LUM-1001",
                "title": "Vitamin C Glow Serum",
                "placed_at": "2026-10-17T10:00:00+00:00",
            },
            {
                "order_ref": "LUM-1002",
                "title": "Deep Hydra Barrier Cream",
                "placed_at": "2026-10-15T10:00:00+00:00",
            },
        ],
        ask_order_selection=True,
    )
    _, anthropic_client, _, http_client = _build_test_harness(
        [FakeMessageResponse(content=[{"type": "text", "text": first_reply_json}])]
    )

    req_body = {
        "tenant_id": TENANT_ID,
        "channel": "whatsapp",
        "external_message_id": "wamid.dup.001",
        "phone": CUSTOMER_A_PHONE_E164,
        "text": "Where is my order?",
    }

    # First call hits the LLM once
    resp1 = http_client.post("/v1/chat", json=req_body, headers={"X-Tenant-Id": str(TENANT_ID)})
    assert resp1.status_code == 200
    assert len(anthropic_client.messages.calls) == 1

    # Duplicate call with same (tenant_id, channel, external_message_id) makes ZERO additional LLM calls
    calls_before = len(anthropic_client.messages.calls)
    resp2 = http_client.post("/v1/chat", json=req_body, headers={"X-Tenant-Id": str(TENANT_ID)})
    assert resp2.status_code == 200
    assert len(anthropic_client.messages.calls) == calls_before
    assert resp2.json() == resp1.json()


# ---------------------------------------------------------------------------
# (b) two orders and no pick -> cards returned, zero tool calls
# ---------------------------------------------------------------------------


def test_b_two_orders_and_no_pick_returns_cards_with_zero_tool_calls() -> None:
    expected_cards = [
        {
            "order_ref": "LUM-1001",
            "title": "Vitamin C Glow Serum",
            "placed_at": (FIXED_NOW - timedelta(days=1)).isoformat(),
        },
        {
            "order_ref": "LUM-1002",
            "title": "Deep Hydra Barrier Cream",
            "placed_at": (FIXED_NOW - timedelta(days=3)).isoformat(),
        },
    ]
    scripted_reply = _make_reply_json(
        reply_text="You have two recent orders. Which one would you like me to check?",
        order_cards=expected_cards,
        ask_order_selection=True,
    )
    chat_svc, anthropic_client, chat_repo, _ = _build_test_harness(
        [FakeMessageResponse(content=[{"type": "text", "text": scripted_reply}])],
        customer_a_order_count=2,
    )

    reply = pytest.MarkGenerator  # keep lint happy if unused
    import asyncio

    result = asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.two_orders.001",
                phone=CUSTOMER_A_PHONE_E164,
                text="Where is my order?",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW,
        )
    )

    assert result.ask_order_selection is True
    assert len(result.order_cards) == 2
    assert [c.order_ref for c in result.order_cards] == ["LUM-1001", "LUM-1002"]
    assert len(anthropic_client.messages.calls) == 1
    assert len(chat_repo.llm_calls) == 1
    assert chat_repo.llm_calls[0].tool_calls == []


# ---------------------------------------------------------------------------
# (c) a foreign AWB -> NOT_FOUND, and the reply doesn't contain that AWB
# ---------------------------------------------------------------------------


def test_c_foreign_awb_returns_not_found_and_reply_excludes_foreign_awb() -> None:
    step1 = FakeMessageResponse(
        stop_reason="tool_use",
        content=[
            {
                "type": "tool_use",
                "id": "toolu_foreign_1",
                "name": "get_tracking",
                "input": {"order_ref": None, "awb": CUSTOMER_B_AWB},
            }
        ],
    )
    # Even if the model tries to echo the foreign AWB in its text, the service strips it
    step2 = FakeMessageResponse(
        stop_reason="end_turn",
        content=[
            {
                "type": "text",
                "text": _make_reply_json(
                    reply_text=f"I couldn't find any order matching {CUSTOMER_B_AWB} under your account. Could you double-check the order number?"
                ),
            }
        ],
    )

    chat_svc, anthropic_client, _, _ = _build_test_harness([step1, step2])
    import asyncio

    reply = asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.foreign_awb.001",
                phone=CUSTOMER_A_PHONE_E164,
                text=f"Track AWB {CUSTOMER_B_AWB}",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW,
        )
    )

    # Verify tool_result sent to model on step 2 was "NOT_FOUND" with is_error=True
    assert len(anthropic_client.messages.calls) == 2
    second_call_messages = anthropic_client.messages.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_foreign_1",
            "content": "NOT_FOUND",
            "is_error": True,
        }
    ]
    # And the final reply does not contain Customer B's AWB
    assert CUSTOMER_B_AWB not in reply.reply_text


# ---------------------------------------------------------------------------
# (d) cancel_order without a prior confirmation question -> refused
# ---------------------------------------------------------------------------


def test_d_cancel_order_without_prior_confirmation_question_is_refused() -> None:
    step1 = FakeMessageResponse(
        stop_reason="tool_use",
        content=[
            {
                "type": "tool_use",
                "id": "toolu_cancel_unconfirmed",
                "name": "cancel_order",
                "input": {"order_ref": "LUM-1001", "customer_confirmed": True},
            }
        ],
    )
    step2 = FakeMessageResponse(
        stop_reason="end_turn",
        content=[
            {
                "type": "text",
                "text": _make_reply_json(
                    reply_text="Would you like me to cancel order LUM-1001? Please confirm to proceed."
                ),
            }
        ],
    )

    chat_svc, anthropic_client, chat_repo, _ = _build_test_harness([step1, step2])
    import asyncio

    reply = asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.cancel.001",
                phone=CUSTOMER_A_PHONE_E164,
                text="Please cancel order LUM-1001",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW,
        )
    )

    # Verify the server refused the cancel_order tool call with is_error=True
    second_call_messages = anthropic_client.messages.calls[1]["messages"]
    tool_result_block = second_call_messages[-1]["content"][0]
    assert tool_result_block["tool_use_id"] == "toolu_cancel_unconfirmed"
    assert tool_result_block["is_error"] is True
    assert "CONFIRMATION_REQUIRED" in tool_result_block["content"]

    # Order LUM-1001 must remain uncancelled
    order_1001 = chat_svc.order_service.get_owned_order(
        chat_svc.identity_service.resolve_thread(TENANT_ID, CUSTOMER_A_PHONE_E164, None, now=FIXED_NOW),
        order_ref="LUM-1001",
    )
    assert order_1001.cancelled_at is None
    assert "LUM-1001" in reply.reply_text

    # Now on the next turn, after the assistant asked for confirmation for LUM-1001, cancel succeeds!
    anthropic_client.messages._responses.extend(
        [
            FakeMessageResponse(
                stop_reason="tool_use",
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_cancel_confirmed",
                        "name": "cancel_order",
                        "input": {"order_ref": "LUM-1001", "customer_confirmed": True},
                    }
                ],
            ),
            FakeMessageResponse(
                stop_reason="end_turn",
                content=[
                    {
                        "type": "text",
                        "text": _make_reply_json(
                            reply_text="Your order LUM-1001 has been cancelled."
                        ),
                    }
                ],
            ),
        ]
    )

    reply2 = asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.cancel.002",
                phone=CUSTOMER_A_PHONE_E164,
                text="Yes, confirm cancellation for LUM-1001",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW + timedelta(minutes=1),
        )
    )
    assert order_1001.cancelled_at is not None
    assert "cancel:1001:1001" in chat_repo.cancel_idempotency_keys
    assert "cancelled" in reply2.reply_text.lower()


# ---------------------------------------------------------------------------
# (e) 4 steps of tool_use -> 4th call made with tool_choice none, valid Reply
# ---------------------------------------------------------------------------


def test_e_four_steps_uses_tool_choice_none_on_fourth_call_and_returns_valid_reply() -> None:
    step1 = FakeMessageResponse(
        stop_reason="tool_use",
        content=[
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "get_tracking",
                "input": {"order_ref": "LUM-1001", "awb": None},
            }
        ],
    )
    step2 = FakeMessageResponse(
        stop_reason="tool_use",
        content=[
            {
                "type": "tool_use",
                "id": "tu_2",
                "name": "get_refund_status",
                "input": {"order_ref": "LUM-1002"},
            }
        ],
    )
    step3 = FakeMessageResponse(
        stop_reason="tool_use",
        content=[
            {
                "type": "tool_use",
                "id": "tu_3",
                "name": "search_older_orders",
                "input": {"lookback_days": 90},
            }
        ],
    )
    step4 = FakeMessageResponse(
        stop_reason="end_turn",
        content=[
            {
                "type": "text",
                "text": _make_reply_json(
                    reply_text="Your order LUM-1001 is being prepared at our warehouse and is scheduled for delivery by 2026-10-22."
                ),
            }
        ],
    )

    chat_svc, anthropic_client, chat_repo, _ = _build_test_harness([step1, step2, step3, step4])
    import asyncio

    reply = asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.four_steps.001",
                phone=CUSTOMER_A_PHONE_E164,
                text="Check LUM-1001, refund on LUM-1002, and my older orders",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW,
        )
    )

    assert isinstance(reply, Reply)
    assert "LUM-1001" in reply.reply_text
    assert len(anthropic_client.messages.calls) == 4

    # Steps 1..3 do not force tool_choice, Step 4 sets tool_choice={"type": "none"}
    for idx in range(3):
        assert "tool_choice" not in anthropic_client.messages.calls[idx]
    assert anthropic_client.messages.calls[3]["tool_choice"] == {"type": "none"}

    # Every call (including step 4) receives the identical constant tools list without mutation
    expected_tools = list(TOOLS_DEFINITIONS)
    assert len(expected_tools) == 5
    for idx, call in enumerate(anthropic_client.messages.calls):
        assert call["tools"] == expected_tools, f"Step {idx + 1} tools list differed"
        assert len(call["tools"]) == 5

    # 4 llm_call rows logged with cache tokens and latency
    assert len(chat_repo.llm_calls) == 4
    assert [c.step for c in chat_repo.llm_calls] == [1, 2, 3, 4]
    assert all(c.cache_read_tokens == 90 for c in chat_repo.llm_calls)


# ---------------------------------------------------------------------------
# (f) invalid JSON on the final step -> FALLBACK
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("final_response"),
    [
        FakeMessageResponse(
            stop_reason="end_turn",
            content=[{"type": "text", "text": "{not valid json at all"}],
        ),
        FakeMessageResponse(
            stop_reason="end_turn",
            content=[{"type": "text", "text": '{"wrong_field": 123}'}],
        ),
        FakeMessageResponse(
            stop_reason="refusal",
            content=[{"type": "text", "text": ""}],
        ),
        FakeMessageResponse(
            stop_reason="max_tokens",
            content=[{"type": "text", "text": '{"reply_text": "truncated'}],
        ),
    ],
)
def test_f_invalid_json_or_refusal_or_max_tokens_returns_fallback(
    final_response: FakeMessageResponse,
) -> None:
    chat_svc, _, _, _ = _build_test_harness([final_response])
    import asyncio

    reply = asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.invalid_json.001",
                phone=CUSTOMER_A_PHONE_E164,
                text="Where is my order?",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW,
        )
    )

    assert reply.reply_text == FALLBACK_REPLY_TEXT
    assert reply.order_cards == []
    assert reply.ask_order_selection is False
    assert reply.show_return_link is False
    assert reply.escalate is False


# ---------------------------------------------------------------------------
# Additional constraint checks: live-handover, outbox, prompt & tool contracts
# ---------------------------------------------------------------------------


def test_open_live_handover_ticket_relays_without_llm_and_returns_escalate_true() -> None:
    live_ticket = CrmTicket(
        id=501,
        thread_id=1001,
        crm_integration_id=1,
        kind=CrmTicketKind.LIVE,
        status=CrmTicketStatus.OPEN,
        opened_at=FIXED_NOW - timedelta(minutes=10),
    )
    chat_svc, anthropic_client, chat_repo, _ = _build_test_harness(
        [],
        tickets=[live_ticket],
    )
    import asyncio

    reply = asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.live.001",
                phone=CUSTOMER_A_PHONE_E164,
                text="Hello human agent, are you there?",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW,
        )
    )

    assert reply.reply_text == ""
    assert reply.escalate is True
    assert len(anthropic_client.messages.calls) == 0
    assert len(chat_repo.relayed_messages) == 1
    assert chat_repo.relayed_messages[0].role == MessageRole.CUSTOMER


def test_escalation_writes_outbox_row_and_prompt_has_ephemeral_cache_and_no_runtime_data() -> None:
    step1 = FakeMessageResponse(
        stop_reason="tool_use",
        content=[
            {
                "type": "tool_use",
                "id": "tu_esc_1",
                "name": "escalate_to_human",
                "input": {"reason": "Customer requested human support", "category": "HUMAN_REQUESTED"},
            }
        ],
    )
    step2 = FakeMessageResponse(
        stop_reason="end_turn",
        content=[
            {
                "type": "text",
                "text": _make_reply_json(
                    reply_text="I am connecting you with a support specialist now.",
                    escalate=True,
                ),
            }
        ],
    )

    chat_svc, anthropic_client, chat_repo, _ = _build_test_harness([step1, step2])
    import asyncio

    reply = asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.esc.001",
                phone=CUSTOMER_A_PHONE_E164,
                text="Connect me to a human please",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW,
        )
    )

    assert reply.escalate is True
    assert len(chat_repo.outbox) == 1
    assert chat_repo.outbox[0].event_type == OUTBOX_ESCALATION_EVENT

    # Check system prompt layout & SDK parameters
    call_kwargs = anthropic_client.messages.calls[0]
    assert "temperature" not in call_kwargs
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": REPLY_JSON_SCHEMA},
    }
    system_blocks = call_kwargs["system"]
    assert len(system_blocks) == 1
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "<runtime_context>" not in system_blocks[0]["text"]
    assert "LUM-1001" not in system_blocks[0]["text"]
    assert "2026-10-18" not in system_blocks[0]["text"]

    # Verify runtime_context is in the final user turn of messages
    sent_messages = call_kwargs["messages"]
    assert sent_messages[-1]["role"] == "user"
    assert sent_messages[-1]["content"].startswith("<runtime_context>\n")
    assert sent_messages[-1]["content"].endswith("\nConnect me to a human please")

    # Check all tools have strict:true and additionalProperties:false
    for tool_def in TOOLS_DEFINITIONS:
        assert tool_def["strict"] is True
        assert tool_def["input_schema"]["additionalProperties"] is False


def test_runtime_context_placed_in_last_user_turn_after_history_for_cache_stability() -> None:
    """Verify system has only 1 static block and runtime_context is placed in the last user turn after history."""
    turn1_reply = _make_reply_json(
        reply_text="You have two recent orders. Which one would you like to track?",
        ask_order_selection=True,
    )
    turn2_reply = _make_reply_json(
        reply_text="Order LUM-1001 is placed and scheduled for delivery by 2026-10-22.",
    )
    chat_svc, anthropic_client, _, _ = _build_test_harness(
        [
            FakeMessageResponse(content=[{"type": "text", "text": turn1_reply}]),
            FakeMessageResponse(content=[{"type": "text", "text": turn2_reply}]),
        ]
    )
    import asyncio

    # Turn 1
    asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.cache.001",
                phone=CUSTOMER_A_PHONE_E164,
                text="Where is my order?",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW,
        )
    )

    # Turn 2 (1 minute later — runtime_context.now changes, but system + history prefix is unchanged)
    asyncio.run(
        chat_svc.handle_chat(
            ChatRequest(
                tenant_id=TENANT_ID,
                channel="whatsapp",
                external_message_id="wamid.cache.002",
                phone=CUSTOMER_A_PHONE_E164,
                text="Track LUM-1001 please",
            ),
            principal_tenant_id=TENANT_ID,
            now=FIXED_NOW + timedelta(minutes=1),
        )
    )

    call1 = anthropic_client.messages.calls[0]
    call2 = anthropic_client.messages.calls[1]

    # system is a single static block and identical across turns despite timestamp change
    assert len(call1["system"]) == 1
    assert len(call2["system"]) == 1
    assert call1["system"] == call2["system"]
    assert "<runtime_context>" not in call2["system"][0]["text"]

    # Turn 2 messages: history comes first, <runtime_context> + current customer text is in the last user turn
    msgs2 = call2["messages"]
    assert len(msgs2) == 3
    assert msgs2[0] == {"role": "user", "content": "Where is my order?"}
    assert msgs2[1] == {
        "role": "assistant",
        "content": "You have two recent orders. Which one would you like to track?",
    }
    assert msgs2[2]["role"] == "user"
    assert msgs2[2]["content"].startswith("<runtime_context>\n")
    assert msgs2[2]["content"].endswith("</runtime_context>\nTrack LUM-1001 please")


def test_execute_tool_returns_distinct_error_codes() -> None:
    """Verify execute_tool returns NOT_FOUND, DUPLICATE_CALL, BUDGET_EXCEEDED, and TIMEOUT distinctly."""
    import asyncio
    from services.tools import ToolExecutor, TurnBudget

    chat_svc, _, chat_repo, _ = _build_test_harness([])
    thread = chat_svc.identity_service.resolve_thread(
        TENANT_ID,
        CUSTOMER_A_PHONE_E164,
        None,
        now=FIXED_NOW,
    )
    tenant_bundle = chat_repo.load_tenant_config(TENANT_ID)
    executor = ToolExecutor(
        thread=thread,
        tenant=tenant_bundle.tenant,
        order_service=chat_svc.order_service,
        tracking_service=chat_svc.tracking_service,
        chat_repo=chat_repo,
        budget=TurnBudget(),
        now=FIXED_NOW,
        timeout_seconds=0.01,
    )

    # 1. NOT_FOUND (foreign or missing order)
    res_not_found = asyncio.run(
        executor.execute_tool(
            {
                "id": "tu_nf",
                "name": "get_tracking",
                "input": {"order_ref": None, "awb": CUSTOMER_B_AWB},
            }
        )
    )
    assert res_not_found.is_error is True
    assert res_not_found.content == "NOT_FOUND"

    # 2. DUPLICATE_CALL (identical tool name + canonical args)
    res_dup = asyncio.run(
        executor.execute_tool(
            {
                "id": "tu_dup",
                "name": "get_tracking",
                "input": {"awb": CUSTOMER_B_AWB, "order_ref": None},
            }
        )
    )
    assert res_dup.is_error is True
    assert res_dup.content == "DUPLICATE_CALL"

    # 3. BUDGET_EXCEEDED (get_tracking allowed at most 2 times per turn; 1st call used above, 2nd ok, 3rd -> BUDGET_EXCEEDED)
    res_second = asyncio.run(
        executor.execute_tool(
            {
                "id": "tu_ok2",
                "name": "get_tracking",
                "input": {"order_ref": "LUM-1001", "awb": None},
            }
        )
    )
    assert res_second.is_error is False

    res_budget = asyncio.run(
        executor.execute_tool(
            {
                "id": "tu_budget",
                "name": "get_tracking",
                "input": {"order_ref": "LUM-1002", "awb": None},
            }
        )
    )
    assert res_budget.is_error is True
    assert res_budget.content == "BUDGET_EXCEEDED"

    # 4. TIMEOUT (slow tool dispatch exceeds timeout_seconds)
    async def _slow_dispatch(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        await asyncio.sleep(0.05)
        return ("{}", False)

    executor._dispatch = _slow_dispatch  # type: ignore[method-assign]
    res_timeout = asyncio.run(
        executor.execute_tool(
            {
                "id": "tu_timeout",
                "name": "get_refund_status",
                "input": {"order_ref": "LUM-1001"},
            }
        )
    )
    assert res_timeout.is_error is True
    assert res_timeout.content == "TIMEOUT"


def test_tenant_id_strictly_from_principal_never_trusted_from_body() -> None:
    scripted_reply = _make_reply_json(reply_text="Hello!")
    _, anthropic_client, _, http_client = _build_test_harness(
        [FakeMessageResponse(content=[{"type": "text", "text": scripted_reply}])]
    )

    # 1. Missing X-Tenant-Id header -> 401 Unauthorized even if tenant_id is in the body
    resp_unauth = http_client.post(
        "/v1/chat",
        json={
            "tenant_id": TENANT_ID,
            "channel": "whatsapp",
            "external_message_id": "wamid.sec.001",
            "phone": CUSTOMER_A_PHONE_E164,
            "text": "Where is my order?",
        },
    )
    assert resp_unauth.status_code == 401
    assert len(anthropic_client.messages.calls) == 0

    # 2. Mismatched X-Tenant-Id header vs body tenant_id -> 403 Forbidden
    resp_mismatch = http_client.post(
        "/v1/chat",
        headers={"X-Tenant-Id": str(TENANT_ID)},
        json={
            "tenant_id": 999,
            "channel": "whatsapp",
            "external_message_id": "wamid.sec.002",
            "phone": CUSTOMER_A_PHONE_E164,
            "text": "Where is my order?",
        },
    )
    assert resp_mismatch.status_code == 403
    assert len(anthropic_client.messages.calls) == 0


def test_interactive_frontend_and_sandbox_playground_end_to_end() -> None:
    from app.main import build_playground_app

    playground_app, _ = build_playground_app()
    client = TestClient(playground_app)

    # 1. GET / serves the interactive frontend HTML
    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "WISMO Support Assistant" in index_resp.text

    # 2. GET /v1/sandbox/state returns initial seeded state
    state_resp = client.get("/v1/sandbox/state")
    assert state_resp.status_code == 200
    state_data = state_resp.json()
    assert state_data["tenant"]["is_business_hours"] is True
    assert "GPT 6 Luna" in state_data["tenant"]["llm_mode"]
    assert len(state_data["orders"]) == 6

    # 3. Ambiguous order query returns order cards
    chat_resp_1 = client.post(
        "/v1/chat",
        headers={"X-Tenant-Id": "1"},
        json={
            "channel": "whatsapp",
            "external_message_id": "ui-1",
            "phone": CUSTOMER_A_PHONE_E164,
            "text": "Where is my order?",
        },
    )
    assert chat_resp_1.status_code == 200
    reply_1 = chat_resp_1.json()
    assert reply_1["ask_order_selection"] is True
    assert len(reply_1["order_cards"]) >= 4

    # 4. Two-step pre-pickup cancellation via natural "Yes" confirmation
    chat_resp_2 = client.post(
        "/v1/chat",
        headers={"X-Tenant-Id": "1"},
        json={
            "channel": "whatsapp",
            "external_message_id": "ui-2",
            "phone": CUSTOMER_A_PHONE_E164,
            "text": "Please cancel my order LUM-1001",
        },
    )
    assert chat_resp_2.status_code == 200
    assert "confirm" in chat_resp_2.json()["reply_text"].lower()

    chat_resp_3 = client.post(
        "/v1/chat",
        headers={"X-Tenant-Id": "1"},
        json={
            "channel": "whatsapp",
            "external_message_id": "ui-3",
            "phone": CUSTOMER_A_PHONE_E164,
            "text": "Yes",
        },
    )
    assert chat_resp_3.status_code == 200
    assert "cancelled" in chat_resp_3.json()["reply_text"].lower()

    state_after_cancel = client.get("/v1/sandbox/state").json()
    lum_1001 = next(o for o in state_after_cancel["orders"] if o["order_ref"] == "LUM-1001")
    assert lum_1001["status"] == "CANCELLED"

    # 5. Toggle business hours and reset sandbox
    bh_resp = client.post(
        "/v1/sandbox/business-hours",
        json={"inside_business_hours": False},
    )
    assert bh_resp.status_code == 200
    assert bh_resp.json()["tenant"]["is_business_hours"] is False

    reset_resp = client.post("/v1/sandbox/reset")
    assert reset_resp.status_code == 200
    lum_1001_reset = next(o for o in reset_resp.json()["orders"] if o["order_ref"] == "LUM-1001")
    assert lum_1001_reset["status"] == "ORDER_PLACED"

    # 6. Place an instant demo order from the storefront catalog and track it in chat
    create_resp = client.post(
        "/v1/sandbox/create-order",
        json={
            "persona_id": "alice",
            "sku": "OIL-ROSEHIP",
            "name": "Cold-Pressed Rosehip Facial Oil",
            "price_cents": 4400,
            "payment_mode": "PREPAID",
        },
    )
    assert create_resp.status_code == 200
    created_data = create_resp.json()
    new_ref = created_data["created_order_ref"]
    assert new_ref == "LUM-1007"
    assert len(created_data["state"]["orders"]) == 7

    chat_resp_new = client.post(
        "/v1/chat",
        headers={"X-Tenant-Id": "1"},
        json={
            "channel": "web",
            "external_message_id": "ui-4",
            "phone": CUSTOMER_A_PHONE_E164,
            "text": f"Where is my order {new_ref}?",
        },
    )
    assert chat_resp_new.status_code == 200
    assert new_ref in chat_resp_new.json()["reply_text"]

    # 7. Verify all recorded LLM calls in the sandbox are strictly powered by gpt-6-luna
    final_state = client.get("/v1/sandbox/state").json()
    assert final_state["tenant"]["llm_mode"] in (
        "GPT 6 Luna (gpt-6-luna)",
        "GPT 6 Luna · Live (gpt-6-luna)",
    )
    assert len(final_state["llm_calls"]) >= 2
    assert all(call["model"] == "gpt-6-luna" for call in final_state["llm_calls"])
    assert Settings(llm_model="any-other-model").llm_model == "gpt-6-luna"




