from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.models import Message, Tenant, Thread
from repositories.chat_storage import ChatRepositoryProtocol
from services.cancellation import check_cancel
from services.chat_schemas import to_reply_order_card
from services.orders import NotOwned, OrderService
from services.tracking import TrackingService

TOOL_TIMEOUT_SECONDS = 5.0
MAX_GET_TRACKING_CALLS = 2
MAX_SEARCH_OLDER_ORDERS_CALLS = 1

CONFIRMATION_PATTERN = re.compile(
    r"(\?|confirm|sure|proceed|would you like to cancel|want to cancel|wish to cancel)",
    re.IGNORECASE,
)

TOOLS_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "get_tracking",
        "description": "Fetch tracking details and status explanation for an owned order by order_ref or awb.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["order_ref", "awb"],
            "properties": {
                "order_ref": {"type": ["string", "null"]},
                "awb": {"type": ["string", "null"]},
            },
        },
    },
    {
        "name": "search_older_orders",
        "description": "Search owned orders over a longer lookback window (90 or 180 days). Allowed at most once per turn.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["lookback_days"],
            "properties": {
                "lookback_days": {
                    "type": "integer",
                    "enum": [90, 180],
                },
            },
        },
    },
    {
        "name": "get_refund_status",
        "description": "Get deterministic refund status for an owned order. Returns 'not applicable' for COD orders.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["order_ref"],
            "properties": {
                "order_ref": {"type": "string"},
            },
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel an owned order before carrier pickup after explicit customer confirmation.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["order_ref", "customer_confirmed"],
            "properties": {
                "order_ref": {"type": "string"},
                "customer_confirmed": {"type": "boolean"},
            },
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Request escalation to a human support agent.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["reason", "category"],
            "properties": {
                "reason": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": [
                        "HUMAN_REQUESTED",
                        "COMPLAINT",
                        "DAMAGED_OR_LOST",
                        "OUT_OF_SCOPE",
                    ],
                },
            },
        },
    },
)


def _canonical_args_key(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


class TurnBudgetExceeded(Exception):
    def __init__(self, code: str = "BUDGET_EXCEEDED") -> None:
        super().__init__(code)
        self.code = code


class DuplicateToolCall(TurnBudgetExceeded):
    def __init__(self) -> None:
        super().__init__("DUPLICATE_CALL")


@dataclass
class TurnBudget:
    seen_calls: set[tuple[str, str]] = field(default_factory=set)
    tool_counts: dict[str, int] = field(default_factory=dict)

    def check_and_record(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        sig = (tool_name, _canonical_args_key(tool_input))
        if sig in self.seen_calls:
            raise DuplicateToolCall()

        current_count = self.tool_counts.get(tool_name, 0)
        if tool_name == "get_tracking" and current_count >= MAX_GET_TRACKING_CALLS:
            raise TurnBudgetExceeded("BUDGET_EXCEEDED")
        if tool_name == "search_older_orders" and current_count >= MAX_SEARCH_OLDER_ORDERS_CALLS:
            raise TurnBudgetExceeded("BUDGET_EXCEEDED")

        self.seen_calls.add(sig)
        self.tool_counts[tool_name] = current_count + 1


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


def _assistant_message_asked_cancel_confirmation(
    message: Message | None,
    order_ref: str,
) -> bool:
    if message is None:
        return False
    text = (message.content or "").strip()
    if not text:
        return False
    if order_ref.lower() not in text.lower():
        return False
    if "cancel" not in text.lower():
        return False
    return bool(CONFIRMATION_PATTERN.search(text))


class ToolExecutor:
    def __init__(
        self,
        *,
        thread: Thread,
        tenant: Tenant,
        order_service: OrderService,
        tracking_service: TrackingService,
        chat_repo: ChatRepositoryProtocol,
        budget: TurnBudget,
        now: datetime | None = None,
        timeout_seconds: float = TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self.thread = thread
        self.tenant = tenant
        self.order_service = order_service
        self.tracking_service = tracking_service
        self.chat_repo = chat_repo
        self.budget = budget
        self.now = now if now is not None else datetime.now(UTC)
        self.timeout_seconds = timeout_seconds

        self.escalated: bool = False
        self.escalation_reason: str | None = None
        self.escalation_category: str | None = None
        self.rejected_identifiers: set[str] = set()

    async def execute_all(
        self,
        tool_uses: list[dict[str, Any]],
    ) -> list[ToolResultBlock]:
        """Run all tool_use blocks concurrently with asyncio.gather and per-tool timeout."""
        tasks = [self.execute_tool(tu) for tu in tool_uses]
        return list(await asyncio.gather(*tasks))

    async def execute_tool(self, tool_use: dict[str, Any]) -> ToolResultBlock:
        tool_use_id = str(tool_use.get("id", ""))
        name = str(tool_use.get("name", ""))
        raw_input = tool_use.get("input")
        tool_input = raw_input if isinstance(raw_input, dict) else {}

        try:
            self.budget.check_and_record(name, tool_input)
        except DuplicateToolCall:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content="DUPLICATE_CALL",
                is_error=True,
            )
        except TurnBudgetExceeded as exc:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=exc.code,
                is_error=True,
            )

        try:
            content, is_error = await asyncio.wait_for(
                self._dispatch(name, tool_input),
                timeout=self.timeout_seconds,
            )
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=content,
                is_error=is_error,
            )
        except TimeoutError:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content="TIMEOUT",
                is_error=True,
            )
        except NotOwned:
            self._record_rejected_identifiers(tool_input)
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content="NOT_FOUND",
                is_error=True,
            )
        except TurnBudgetExceeded as exc:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=exc.code,
                is_error=True,
            )

    _execute_one_safe = execute_tool

    def _record_rejected_identifiers(self, tool_input: dict[str, Any]) -> None:
        for key in ("awb", "order_ref"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                self.rejected_identifiers.add(val.strip())

    async def _dispatch(
        self,
        name: str,
        tool_input: dict[str, Any],
    ) -> tuple[str, bool]:
        if name == "get_tracking":
            return self._handle_get_tracking(tool_input)
        if name == "search_older_orders":
            return self._handle_search_older_orders(tool_input)
        if name == "get_refund_status":
            return self._handle_get_refund_status(tool_input)
        if name == "cancel_order":
            return self._handle_cancel_order(tool_input)
        if name == "escalate_to_human":
            return self._handle_escalate_to_human(tool_input)
        return (f"UNKNOWN_TOOL:{name}", True)

    def _handle_get_tracking(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        order_ref = tool_input.get("order_ref")
        awb = tool_input.get("awb")
        order = self.order_service.get_owned_order(
            self.thread,
            order_ref=order_ref if isinstance(order_ref, str) else None,
            awb=awb if isinstance(awb, str) else None,
        )
        shipments = self.chat_repo.get_shipments_for_order(
            self.tenant.tenant_id,
            order.order_pk,
        )
        if not shipments:
            return (
                json.dumps(
                    {
                        "order_ref": order.order_ref,
                        "status": "ORDER_PLACED",
                        "customer_text": "Your order has been placed and is being prepared at our warehouse.",
                        "edd": None,
                        "edd_breached": False,
                        "ofd_attempts": 0,
                        "last_events": [],
                    }
                ),
                False,
            )

        selected_shipment = shipments[0]
        if isinstance(awb, str) and awb.strip():
            for s in shipments:
                if s.awb == awb.strip():
                    selected_shipment = s
                    break

        events = self.chat_repo.get_events_for_shipment(
            self.tenant.tenant_id,
            selected_shipment.id,
        )
        view = self.tracking_service.view(
            order,
            selected_shipment,
            events,
            self.tenant,
            now=self.now,
        )
        payload = {
            "order_ref": view.order_ref,
            "status": view.status.name,
            "failure_reason": view.failure_reason.name,
            "customer_text": view.customer_text,
            "edd": view.edd.isoformat() if view.edd else None,
            "edd_breached": view.edd_breached,
            "ofd_attempts": view.ofd_attempts,
            "last_events": [
                {
                    "at": ev["at"].isoformat() if isinstance(ev["at"], datetime) else str(ev["at"]),
                    "status": ev["status"].name if hasattr(ev["status"], "name") else str(ev["status"]),
                    "location": ev["location"],
                }
                for ev in view.last_events
            ],
        }
        return (json.dumps(payload), False)

    def _handle_search_older_orders(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        lookback_days = int(tool_input.get("lookback_days", 90))
        if lookback_days not in (90, 180):
            return ("INVALID_LOOKBACK_DAYS", True)
        cards = self.order_service.list_owned_orders(
            self.thread,
            lookback_days=lookback_days,
            now=self.now,
        )
        # Re-verify each returned card via get_owned_order
        verified = []
        for card in cards:
            _ = self.order_service.get_owned_order(self.thread, order_ref=card.order_ref)
            verified.append(to_reply_order_card(card).model_dump())
        return (json.dumps({"orders": verified}), False)

    def _handle_get_refund_status(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        order_ref = tool_input.get("order_ref")
        order = self.order_service.get_owned_order(
            self.thread,
            order_ref=order_ref if isinstance(order_ref, str) else None,
        )
        if order.payment_mode.strip().upper() == "COD":
            return ("not applicable", False)

        if order.cancelled_at is not None:
            return (
                json.dumps(
                    {
                        "order_ref": order.order_ref,
                        "refund_status": "INITIATED",
                        "cancelled_at": order.cancelled_at.isoformat(),
                    }
                ),
                False,
            )
        return (
            json.dumps(
                {
                    "order_ref": order.order_ref,
                    "refund_status": "NO_REFUND_PENDING",
                }
            ),
            False,
        )

    def _handle_cancel_order(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        order_ref = tool_input.get("order_ref")
        customer_confirmed = bool(tool_input.get("customer_confirmed", False))

        order = self.order_service.get_owned_order(
            self.thread,
            order_ref=order_ref if isinstance(order_ref, str) else None,
        )

        if not customer_confirmed:
            return ("CONFIRMATION_REQUIRED", True)

        last_asst_msg = self.chat_repo.get_last_assistant_message(
            self.tenant.tenant_id,
            self.thread.thread_id,
        )
        if not _assistant_message_asked_cancel_confirmation(last_asst_msg, order.order_ref):
            return ("CONFIRMATION_REQUIRED: prior confirmation prompt missing for order", True)

        shipments = self.chat_repo.get_shipments_for_order(
            self.tenant.tenant_id,
            order.order_pk,
        )
        eligible, reason_code = check_cancel(order, shipments, self.tenant)
        if not eligible:
            return (f"CANCEL_REFUSED:{reason_code}", True)

        idempotency_key = f"cancel:{self.thread.thread_id}:{order.order_pk}"
        self.chat_repo.execute_cancel_order(
            self.tenant.tenant_id,
            order,
            idempotency_key=idempotency_key,
            now=self.now,
        )
        return (
            json.dumps(
                {
                    "order_ref": order.order_ref,
                    "cancelled": True,
                    "reason_code": str(reason_code),
                    "idempotency_key": idempotency_key,
                }
            ),
            False,
        )

    def _handle_escalate_to_human(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        order_ref = tool_input.get("order_ref")
        if isinstance(order_ref, str) and order_ref.strip():
            _ = self.order_service.get_owned_order(self.thread, order_ref=order_ref)

        reason = str(tool_input.get("reason", "")).strip()
        category = str(tool_input.get("category", "HUMAN_REQUESTED")).strip()
        if category not in {
            "HUMAN_REQUESTED",
            "COMPLAINT",
            "DAMAGED_OR_LOST",
            "OUT_OF_SCOPE",
        }:
            return ("INVALID_ESCALATION_CATEGORY", True)

        self.escalated = True
        self.escalation_reason = reason
        self.escalation_category = category
        return (
            json.dumps(
                {
                    "escalation_queued": True,
                    "category": category,
                }
            ),
            False,
        )
