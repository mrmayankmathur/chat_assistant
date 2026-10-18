from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from app.settings import Settings, get_settings
from repositories.chat_storage import (
    ChatRepositoryProtocol,
    LlmCallRecord,
    OutboxEntry,
    TenantConfigBundle,
)
from services.business_hours import is_within_business_hours
from services.chat_schemas import (
    REPLY_JSON_SCHEMA,
    ChatRequest,
    Reply,
    build_fallback_reply,
    build_live_handover_reply,
)
from services.identity import IdentityService
from services.orders import OrderCard, OrderService
from services.prompt import (
    build_initial_messages,
    build_runtime_context_block,
    build_system_blocks,
)
from services.tools import TOOLS_DEFINITIONS, ToolExecutor, TurnBudget
from services.tracking import TrackingService

MAX_TURN_STEPS = 4
MAX_TOKENS = 4096
OUTBOX_ESCALATION_EVENT = "escalation.requested"
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _extract_block_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}) or {},
        }
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return {"type": str(block_type or "unknown")}


def _extract_usage_int(usage: Any, *attr_names: str) -> int:
    if usage is None:
        return 0
    for name in attr_names:
        if isinstance(usage, dict) and name in usage and usage[name] is not None:
            return int(usage[name])
        val = getattr(usage, name, None)
        if val is not None:
            return int(val)
    return 0


class ChatService:
    def __init__(
        self,
        *,
        identity_service: IdentityService,
        order_service: OrderService,
        tracking_service: TrackingService,
        chat_repo: ChatRepositoryProtocol,
        anthropic_client: Any,
        settings: Settings | None = None,
        tool_timeout_seconds: float = 5.0,
    ) -> None:
        self.identity_service = identity_service
        self.order_service = order_service
        self.tracking_service = tracking_service
        self.chat_repo = chat_repo
        self.anthropic_client = anthropic_client
        self.settings = settings or get_settings()
        self.tool_timeout_seconds = tool_timeout_seconds

    async def handle_chat(
        self,
        request: ChatRequest,
        *,
        principal_tenant_id: int | None = None,
        now: datetime | None = None,
    ) -> Reply:
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)

        if principal_tenant_id is None:
            raise ValueError("tenant_id is required from authenticated principal")
        if request.tenant_id is not None and request.tenant_id != principal_tenant_id:
            raise PermissionError(
                "tenant_id in request body does not match authenticated principal"
            )
        tenant_id = principal_tenant_id

        # 1. Idempotency: if (tenant, channel, external_message_id) already exists -> return stored reply. No LLM call.
        stored_reply = self.chat_repo.find_reply_by_external_message_id(
            tenant_id=tenant_id,
            channel=request.channel,
            external_message_id=request.external_message_id,
        )
        if stored_reply is not None:
            return Reply.model_validate(stored_reply)

        # 2. resolve_thread
        thread = self.identity_service.resolve_thread(
            tenant_id=tenant_id,
            phone=request.phone,
            email=request.email,
            now=current_now,
        )

        # 3. If the thread has an OPEN live-handover ticket -> save message as role=customer,
        #    relay it to the human, and return an empty reply with escalate=true.
        if self.chat_repo.has_open_live_handover(tenant_id, thread.thread_id):
            handover_reply = build_live_handover_reply()
            self.chat_repo.relay_live_handover_message(
                tenant_id=tenant_id,
                thread_id=thread.thread_id,
                channel=request.channel,
                external_message_id=request.external_message_id,
                content=request.text,
                reply_payload=handover_reply.model_dump(mode="json"),
                now=current_now,
            )
            return handover_reply

        # 4. Build the context in parallel: owned orders (30 days), business-hours check, and tenant config
        async def _load_owned_orders() -> list[OrderCard]:
            return self.order_service.list_owned_orders(
                thread,
                lookback_days=30,
                now=current_now,
            )

        async def _load_tenant_config() -> TenantConfigBundle:
            return self.chat_repo.load_tenant_config(tenant_id)

        async def _load_business_hours() -> bool:
            cfg = self.chat_repo.load_tenant_config(tenant_id)
            return is_within_business_hours(
                cfg.tenant,
                cfg.business_hours,
                now=current_now,
            )

        owned_orders, tenant_bundle, in_bh = await asyncio.gather(
            _load_owned_orders(),
            _load_tenant_config(),
            _load_business_hours(),
        )
        has_open_ticket = self.chat_repo.has_open_ticket(tenant_id, thread.thread_id)

        # 5. Run the turn loop
        reply, llm_calls, escalation_outbox = await self._run_turn_loop(
            thread=thread,
            tenant_bundle=tenant_bundle,
            owned_orders=owned_orders,
            is_business_hours=in_bh,
            has_open_ticket=has_open_ticket,
            channel=request.channel,
            customer_text=request.text,
            now=current_now,
        )

        # 6. In ONE transaction: save customer message, assistant message (payload = Reply JSON),
        #    plus an outbox row "escalation.requested" if the turn escalated.
        self.chat_repo.save_turn_transaction(
            tenant_id=tenant_id,
            thread_id=thread.thread_id,
            channel=request.channel,
            external_message_id=request.external_message_id,
            customer_text=request.text,
            reply_text=reply.reply_text,
            reply_payload=reply.model_dump(mode="json"),
            llm_calls=llm_calls,
            escalation_outbox=escalation_outbox,
            now=current_now,
        )

        return reply

    async def _run_turn_loop(
        self,
        *,
        thread: Any,
        tenant_bundle: TenantConfigBundle,
        owned_orders: list[OrderCard],
        is_business_hours: bool,
        has_open_ticket: bool,
        channel: str,
        customer_text: str,
        now: datetime,
    ) -> tuple[Reply, list[LlmCallRecord], OutboxEntry | None]:
        system_blocks = build_system_blocks(tenant_bundle)
        tenant_id = tenant_bundle.tenant.tenant_id
        earlier_summaries = self.chat_repo.get_earlier_conversation_summaries(
            tenant_id,
            thread.thread_id,
        )
        open_messages = self.chat_repo.get_open_conversation_messages(
            tenant_id,
            thread.thread_id,
        )
        runtime_context_block = build_runtime_context_block(
            now=now,
            tenant_timezone=tenant_bundle.tenant.timezone,
            is_business_hours=is_business_hours,
            channel=channel,
            owned_orders=owned_orders,
            has_open_ticket=has_open_ticket,
        )
        messages = build_initial_messages(
            earlier_summaries=earlier_summaries,
            open_messages=open_messages,
            runtime_context_block=runtime_context_block,
            customer_text=customer_text,
        )

        budget = TurnBudget()
        executor = ToolExecutor(
            thread=thread,
            tenant=tenant_bundle.tenant,
            order_service=self.order_service,
            tracking_service=self.tracking_service,
            chat_repo=self.chat_repo,
            budget=budget,
            now=now,
            timeout_seconds=self.tool_timeout_seconds,
        )

        tools_list = list(TOOLS_DEFINITIONS)
        output_config = {
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": REPLY_JSON_SCHEMA,
            },
        }

        llm_calls: list[LlmCallRecord] = []
        final_response: Any = None

        for step in range(1, MAX_TURN_STEPS + 1):
            create_kwargs: dict[str, Any] = {
                "model": self.settings.llm_model,
                "max_tokens": MAX_TOKENS,
                "system": system_blocks,
                "messages": list(messages),
                "tools": tools_list,
                "output_config": output_config,
            }
            if step == MAX_TURN_STEPS:
                create_kwargs["tool_choice"] = {"type": "none"}

            t0 = time.perf_counter()
            response = await self.anthropic_client.messages.create(**create_kwargs)
            latency_ms = max(0, int((time.perf_counter() - t0) * 1000))

            stop_reason = getattr(response, "stop_reason", None)
            raw_content = getattr(response, "content", []) or []
            content_blocks = [_extract_block_dict(b) for b in raw_content]
            tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

            usage = getattr(response, "usage", None)
            llm_calls.append(
                LlmCallRecord(
                    step=step,
                    model=getattr(response, "model", None) or self.settings.llm_model,
                    stop_reason=stop_reason,
                    input_tokens=_extract_usage_int(usage, "input_tokens"),
                    output_tokens=_extract_usage_int(usage, "output_tokens"),
                    cache_read_tokens=_extract_usage_int(
                        usage, "cache_read_input_tokens", "cache_read_tokens"
                    ),
                    cache_write_tokens=_extract_usage_int(
                        usage, "cache_creation_input_tokens", "cache_write_tokens"
                    ),
                    latency_ms=latency_ms,
                    tool_calls=[
                        {"id": tu.get("id"), "name": tu.get("name"), "input": tu.get("input")}
                        for tu in tool_uses
                    ],
                )
            )

            final_response = response

            if stop_reason in ("refusal", "max_tokens"):
                break

            if not tool_uses or step == MAX_TURN_STEPS:
                break

            # Run tool_use blocks concurrently with 5s timeout each and return ALL tool_result blocks in ONE user message
            tool_results = await executor.execute_all(tool_uses)
            if llm_calls:
                result_by_id = {tr.tool_use_id: tr for tr in tool_results}
                for tc in llm_calls[-1].tool_calls:
                    tr_block = result_by_id.get(str(tc.get("id", "")))
                    if tr_block is not None:
                        tc["result"] = tr_block.content
                        tc["is_error"] = tr_block.is_error
            messages.append({"role": "assistant", "content": content_blocks})
            messages.append(
                {
                    "role": "user",
                    "content": [tr.to_api_dict() for tr in tool_results],
                }
            )

        reply = self._parse_final_reply(final_response, executor)
        escalation_outbox: OutboxEntry | None = None
        if reply.escalate or executor.escalated:
            if not reply.escalate:
                reply = reply.model_copy(update={"escalate": True})
            escalation_outbox = OutboxEntry(
                event_type=OUTBOX_ESCALATION_EVENT,
                tenant_id=tenant_bundle.tenant.tenant_id,
                thread_id=thread.thread_id,
                payload={
                    "tenant_id": tenant_bundle.tenant.tenant_id,
                    "thread_id": thread.thread_id,
                    "reason": executor.escalation_reason or "Customer escalation requested",
                    "category": executor.escalation_category or "HUMAN_REQUESTED",
                },
                created_at=now,
            )

        return reply, llm_calls, escalation_outbox

    def _parse_final_reply(
        self,
        response: Any,
        executor: ToolExecutor,
    ) -> Reply:
        if response is None:
            return build_fallback_reply()

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason in ("refusal", "max_tokens"):
            return build_fallback_reply()

        raw_content = getattr(response, "content", []) or []
        text_parts: list[str] = []
        for block in raw_content:
            b = _extract_block_dict(block)
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                text_parts.append(b["text"])

        combined_text = "\n".join(text_parts).strip()
        if not combined_text:
            return build_fallback_reply()

        try:
            reply = Reply.model_validate_json(combined_text)
        except ValidationError:
            return build_fallback_reply()

        sanitized_text = _URL_RE.sub("", reply.reply_text).strip()
        for rejected in executor.rejected_identifiers:
            if rejected and rejected in sanitized_text:
                sanitized_text = sanitized_text.replace(rejected, "[order not found]")

        if sanitized_text != reply.reply_text:
            reply = reply.model_copy(update={"reply_text": sanitized_text})

        return reply
