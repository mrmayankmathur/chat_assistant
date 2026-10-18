from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AgentConfig,
    BusinessHours,
    ChannelGuideline,
    Conversation,
    ConversationMessage,
    ConversationStatus,
    CrmTicket,
    CrmTicketKind,
    CrmTicketStatus,
    CustomerOrder,
    LlmCall,
    Message,
    MessageRole,
    Policy,
    Shipment,
    ShipmentEvent,
    ShipmentStatus,
    Tenant,
    Thread,
    Tonality,
)


@dataclass(frozen=True)
class TenantConfigBundle:
    tenant: Tenant
    agent_config: AgentConfig
    channel_guidelines: dict[str, str]
    policies: tuple[Policy, ...]
    business_hours: BusinessHours | None


@dataclass
class OutboxEntry:
    event_type: str
    tenant_id: int
    thread_id: int
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class LlmCallRecord:
    step: int
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: int
    tool_calls: list[dict[str, Any]]


# In-process cache keyed strictly by (tenant_id, agent_config.version)
_TENANT_CONFIG_CACHE: dict[tuple[int, int], TenantConfigBundle] = {}


def clear_tenant_config_cache() -> None:
    _TENANT_CONFIG_CACHE.clear()


def get_cached_tenant_config(tenant_id: int, version: int) -> TenantConfigBundle | None:
    return _TENANT_CONFIG_CACHE.get((tenant_id, version))


def set_cached_tenant_config(bundle: TenantConfigBundle) -> TenantConfigBundle:
    key = (bundle.tenant.tenant_id, bundle.agent_config.version)
    _TENANT_CONFIG_CACHE[key] = bundle
    return bundle


@runtime_checkable
class ChatRepositoryProtocol(Protocol):
    def find_reply_by_external_message_id(
        self,
        tenant_id: int,
        channel: str,
        external_message_id: str,
    ) -> dict[str, Any] | None: ...

    def has_open_live_handover(self, tenant_id: int, thread_id: int) -> bool: ...

    def has_open_ticket(self, tenant_id: int, thread_id: int) -> bool: ...

    def load_tenant_config(self, tenant_id: int) -> TenantConfigBundle: ...

    def get_earlier_conversation_summaries(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> list[str]: ...

    def get_open_conversation_messages(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> list[Message]: ...

    def get_last_assistant_message(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> Message | None: ...

    def get_shipments_for_order(
        self,
        tenant_id: int,
        order_pk: int,
    ) -> list[Shipment]: ...

    def get_events_for_shipment(
        self,
        tenant_id: int,
        shipment_id: int,
    ) -> list[ShipmentEvent]: ...

    def execute_cancel_order(
        self,
        tenant_id: int,
        order: CustomerOrder,
        *,
        idempotency_key: str,
        now: datetime,
    ) -> None: ...

    def relay_live_handover_message(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        channel: str,
        external_message_id: str | None,
        content: str,
        reply_payload: dict[str, Any],
        now: datetime,
    ) -> Message: ...

    def save_turn_transaction(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        channel: str,
        external_message_id: str | None,
        customer_text: str,
        reply_text: str,
        reply_payload: dict[str, Any],
        llm_calls: Sequence[LlmCallRecord],
        escalation_outbox: OutboxEntry | None,
        now: datetime,
    ) -> tuple[Message, Message]: ...


class SqlAlchemyChatRepository:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.outbox: list[OutboxEntry] = []
        self.relayed_messages: list[Message] = []
        self._cancel_idempotency_keys: set[str] = set()

    def find_reply_by_external_message_id(
        self,
        tenant_id: int,
        channel: str,
        external_message_id: str,
    ) -> dict[str, Any] | None:
        cust_msg = self.session.scalar(
            select(Message).where(
                Message.tenant_id == tenant_id,
                Message.channel == channel,
                Message.external_message_id == external_message_id,
                Message.soft_deleted.is_(False),
            )
        )
        if cust_msg is None:
            return None
        if isinstance(cust_msg.payload, dict) and "reply_text" in cust_msg.payload:
            return cust_msg.payload

        assistant_msg = self.session.scalar(
            select(Message)
            .where(
                Message.tenant_id == tenant_id,
                Message.thread_id == cust_msg.thread_id,
                Message.role == MessageRole.ASSISTANT,
                Message.message_id > cust_msg.message_id,
                Message.soft_deleted.is_(False),
            )
            .order_by(Message.message_id.asc())
            .limit(1)
        )
        if assistant_msg is not None and isinstance(assistant_msg.payload, dict):
            return assistant_msg.payload
        return None

    def has_open_live_handover(self, tenant_id: int, thread_id: int) -> bool:
        ticket = self.session.scalar(
            select(CrmTicket)
            .join(Thread, Thread.thread_id == CrmTicket.thread_id)
            .where(
                Thread.tenant_id == tenant_id,
                Thread.soft_deleted.is_(False),
                CrmTicket.thread_id == thread_id,
                CrmTicket.status == int(CrmTicketStatus.OPEN),
                CrmTicket.kind == int(CrmTicketKind.LIVE),
            )
        )
        return ticket is not None

    def has_open_ticket(self, tenant_id: int, thread_id: int) -> bool:
        ticket = self.session.scalar(
            select(CrmTicket)
            .join(Thread, Thread.thread_id == CrmTicket.thread_id)
            .where(
                Thread.tenant_id == tenant_id,
                Thread.soft_deleted.is_(False),
                CrmTicket.thread_id == thread_id,
                CrmTicket.status == int(CrmTicketStatus.OPEN),
            )
        )
        return ticket is not None

    def load_tenant_config(self, tenant_id: int) -> TenantConfigBundle:
        agent_cfg = self.session.scalar(
            select(AgentConfig).where(
                AgentConfig.tenant_id == tenant_id,
                AgentConfig.is_active.is_(True),
                AgentConfig.soft_deleted.is_(False),
            )
        )
        if agent_cfg is not None:
            cached = get_cached_tenant_config(tenant_id, agent_cfg.version)
            if cached is not None:
                return cached

        tenant = self.session.get(Tenant, tenant_id)
        if tenant is None:
            raise ValueError(f"Unknown tenant_id={tenant_id}")

        if agent_cfg is None:
            agent_cfg = AgentConfig(
                agent_id=0,
                tenant_id=tenant_id,
                brand_prompt=f"You are the support assistant for {tenant.name}.",
                tonality=Tonality.FRIENDLY,
                is_active=True,
                version=1,
            )

        guidelines_rows = self.session.scalars(
            select(ChannelGuideline).where(
                ChannelGuideline.tenant_id == tenant_id,
                ChannelGuideline.soft_deleted.is_(False),
            )
        ).all()
        guidelines = {g.channel: g.prompt for g in guidelines_rows}

        policies_rows = self.session.scalars(
            select(Policy)
            .where(
                Policy.tenant_id == tenant_id,
                Policy.soft_deleted.is_(False),
            )
            .order_by(Policy.policy_type.asc())
        ).all()

        bh = self.session.scalar(
            select(BusinessHours).where(
                BusinessHours.tenant_id == tenant_id,
                BusinessHours.soft_deleted.is_(False),
            )
        )

        bundle = TenantConfigBundle(
            tenant=tenant,
            agent_config=agent_cfg,
            channel_guidelines=guidelines,
            policies=tuple(policies_rows),
            business_hours=bh,
        )
        return set_cached_tenant_config(bundle)

    def get_earlier_conversation_summaries(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> list[str]:
        convs = self.session.scalars(
            select(Conversation)
            .join(Thread, Thread.thread_id == Conversation.thread_id)
            .where(
                Thread.tenant_id == tenant_id,
                Thread.soft_deleted.is_(False),
                Conversation.thread_id == thread_id,
                Conversation.status != int(ConversationStatus.OPEN),
            )
            .order_by(Conversation.conversation_id.asc())
        ).all()
        return [
            " ".join(c.summary.strip().splitlines())
            for c in convs
            if c.summary and c.summary.strip()
        ]

    def get_open_conversation_messages(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> list[Message]:
        open_conv = self.session.scalar(
            select(Conversation)
            .join(Thread, Thread.thread_id == Conversation.thread_id)
            .where(
                Thread.tenant_id == tenant_id,
                Thread.soft_deleted.is_(False),
                Conversation.thread_id == thread_id,
                Conversation.status == int(ConversationStatus.OPEN),
            )
            .order_by(Conversation.conversation_id.desc())
            .limit(1)
        )
        if open_conv is not None:
            msgs = self.session.scalars(
                select(Message)
                .join(ConversationMessage, ConversationMessage.message_id == Message.message_id)
                .where(
                    ConversationMessage.conversation_id == open_conv.conversation_id,
                    Message.tenant_id == tenant_id,
                    Message.soft_deleted.is_(False),
                )
                .order_by(Message.message_id.asc())
            ).all()
            if msgs:
                return list(msgs)

        # Fallback: messages on the thread scoped by tenant_id
        msgs = self.session.scalars(
            select(Message)
            .where(
                Message.tenant_id == tenant_id,
                Message.thread_id == thread_id,
                Message.soft_deleted.is_(False),
            )
            .order_by(Message.message_id.asc())
        ).all()
        return list(msgs)

    def get_last_assistant_message(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> Message | None:
        return self.session.scalar(
            select(Message)
            .where(
                Message.tenant_id == tenant_id,
                Message.thread_id == thread_id,
                Message.role == int(MessageRole.ASSISTANT),
                Message.soft_deleted.is_(False),
            )
            .order_by(Message.message_id.desc())
            .limit(1)
        )

    def get_shipments_for_order(
        self,
        tenant_id: int,
        order_pk: int,
    ) -> list[Shipment]:
        return list(
            self.session.scalars(
                select(Shipment)
                .join(CustomerOrder, CustomerOrder.order_pk == Shipment.order_pk)
                .where(
                    CustomerOrder.tenant_id == tenant_id,
                    CustomerOrder.soft_deleted.is_(False),
                    Shipment.order_pk == order_pk,
                )
                .order_by(Shipment.id.asc())
            ).all()
        )

    def get_events_for_shipment(
        self,
        tenant_id: int,
        shipment_id: int,
    ) -> list[ShipmentEvent]:
        return list(
            self.session.scalars(
                select(ShipmentEvent)
                .join(Shipment, Shipment.id == ShipmentEvent.shipment_id)
                .join(CustomerOrder, CustomerOrder.order_pk == Shipment.order_pk)
                .where(
                    CustomerOrder.tenant_id == tenant_id,
                    CustomerOrder.soft_deleted.is_(False),
                    ShipmentEvent.shipment_id == shipment_id,
                )
                .order_by(ShipmentEvent.event_at.asc())
            ).all()
        )

    def execute_cancel_order(
        self,
        tenant_id: int,
        order: CustomerOrder,
        *,
        idempotency_key: str,
        now: datetime,
    ) -> None:
        if order.tenant_id != tenant_id:
            raise ValueError("Cross-tenant cancel refused")
        if idempotency_key in self._cancel_idempotency_keys or order.cancelled_at is not None:
            self._cancel_idempotency_keys.add(idempotency_key)
            return
        order.cancelled_at = now
        shipments = self.get_shipments_for_order(tenant_id, order.order_pk)
        for s in shipments:
            s.status = ShipmentStatus.CANCELLED
        self._cancel_idempotency_keys.add(idempotency_key)
        self.session.flush()

    def relay_live_handover_message(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        channel: str,
        external_message_id: str | None,
        content: str,
        reply_payload: dict[str, Any],
        now: datetime,
    ) -> Message:
        msg = Message(
            tenant_id=tenant_id,
            thread_id=thread_id,
            channel=channel,
            external_message_id=external_message_id,
            role=MessageRole.CUSTOMER,
            content=content,
            payload=reply_payload,
            attachments=None,
            soft_deleted=False,
        )
        self.session.add(msg)
        self.session.flush()
        self.relayed_messages.append(msg)
        return msg

    def save_turn_transaction(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        channel: str,
        external_message_id: str | None,
        customer_text: str,
        reply_text: str,
        reply_payload: dict[str, Any],
        llm_calls: Sequence[LlmCallRecord],
        escalation_outbox: OutboxEntry | None,
        now: datetime,
    ) -> tuple[Message, Message]:
        with self.session.begin_nested():
            cust_msg = Message(
                tenant_id=tenant_id,
                thread_id=thread_id,
                channel=channel,
                external_message_id=external_message_id,
                role=MessageRole.CUSTOMER,
                content=customer_text,
                payload=reply_payload,
                attachments=None,
                soft_deleted=False,
            )
            self.session.add(cust_msg)
            self.session.flush()

            asst_msg = Message(
                tenant_id=tenant_id,
                thread_id=thread_id,
                channel=channel,
                external_message_id=None,
                role=MessageRole.ASSISTANT,
                content=reply_text,
                payload=reply_payload,
                attachments=None,
                soft_deleted=False,
            )
            self.session.add(asst_msg)
            self.session.flush()

            for rec in llm_calls:
                self.session.add(
                    LlmCall(
                        message_id=asst_msg.message_id,
                        step=rec.step,
                        model=rec.model,
                        stop_reason=rec.stop_reason,
                        input_tokens=rec.input_tokens,
                        output_tokens=rec.output_tokens,
                        cache_read_tokens=rec.cache_read_tokens,
                        cache_write_tokens=rec.cache_write_tokens,
                        latency_ms=rec.latency_ms,
                        tool_calls=rec.tool_calls,
                    )
                )

            if escalation_outbox is not None:
                self.outbox.append(escalation_outbox)

            self.session.flush()
        return cust_msg, asst_msg


class InMemoryChatRepository:
    def __init__(
        self,
        *,
        tenant: Tenant,
        agent_config: AgentConfig | None = None,
        channel_guidelines: dict[str, str] | None = None,
        policies: Sequence[Policy] = (),
        business_hours: BusinessHours | None = None,
        shipments: Sequence[Shipment] = (),
        events: Sequence[ShipmentEvent] = (),
        tickets: Sequence[CrmTicket] = (),
        messages: Sequence[Message] = (),
        conversations: Sequence[Conversation] = (),
    ) -> None:
        self.tenant = tenant
        self.agent_config = agent_config or AgentConfig(
            agent_id=1,
            tenant_id=tenant.tenant_id,
            brand_prompt=f"You are the helpful support assistant for {tenant.name}.",
            tonality=Tonality.FRIENDLY,
            is_active=True,
            version=1,
        )
        self.channel_guidelines = dict(channel_guidelines or {})
        self.policies = tuple(policies)
        self.business_hours = business_hours
        self.shipments: list[Shipment] = list(shipments)
        self.events: list[ShipmentEvent] = list(events)
        self.tickets: list[CrmTicket] = list(tickets)
        self.messages: list[Message] = list(messages)
        self.conversations: list[Conversation] = list(conversations)
        self.llm_calls: list[LlmCall] = []
        self.outbox: list[OutboxEntry] = []
        self.relayed_messages: list[Message] = []
        self.cancel_idempotency_keys: set[str] = set()
        self.load_tenant_config_db_hits: int = 0
        self._next_msg_id: int = (
            max((m.message_id for m in self.messages if m.message_id is not None), default=5000) + 1
        )
        self._next_llm_id: int = 1

    def find_reply_by_external_message_id(
        self,
        tenant_id: int,
        channel: str,
        external_message_id: str,
    ) -> dict[str, Any] | None:
        for msg in self.messages:
            if (
                msg.tenant_id == tenant_id
                and msg.channel == channel
                and msg.external_message_id == external_message_id
                and not getattr(msg, "soft_deleted", False)
            ):
                if isinstance(msg.payload, dict) and "reply_text" in msg.payload:
                    return msg.payload
                for candidate in self.messages:
                    if (
                        candidate.tenant_id == tenant_id
                        and candidate.thread_id == msg.thread_id
                        and candidate.role == MessageRole.ASSISTANT
                        and (candidate.message_id or 0) > (msg.message_id or 0)
                        and isinstance(candidate.payload, dict)
                    ):
                        return candidate.payload
        return None

    def has_open_live_handover(self, tenant_id: int, thread_id: int) -> bool:
        if tenant_id != self.tenant.tenant_id:
            return False
        return any(
            t.thread_id == thread_id
            and CrmTicketStatus(t.status) == CrmTicketStatus.OPEN
            and CrmTicketKind(t.kind) == CrmTicketKind.LIVE
            for t in self.tickets
        )

    def has_open_ticket(self, tenant_id: int, thread_id: int) -> bool:
        if tenant_id != self.tenant.tenant_id:
            return False
        return any(
            t.thread_id == thread_id and CrmTicketStatus(t.status) == CrmTicketStatus.OPEN
            for t in self.tickets
        )

    def load_tenant_config(self, tenant_id: int) -> TenantConfigBundle:
        if tenant_id != self.tenant.tenant_id:
            raise ValueError(f"Unknown tenant_id={tenant_id}")
        cached = get_cached_tenant_config(tenant_id, self.agent_config.version)
        if cached is not None:
            return cached
        self.load_tenant_config_db_hits += 1
        bundle = TenantConfigBundle(
            tenant=self.tenant,
            agent_config=self.agent_config,
            channel_guidelines=dict(self.channel_guidelines),
            policies=self.policies,
            business_hours=self.business_hours,
        )
        return set_cached_tenant_config(bundle)

    def get_earlier_conversation_summaries(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> list[str]:
        if tenant_id != self.tenant.tenant_id:
            return []
        return [
            " ".join(c.summary.strip().splitlines())
            for c in self.conversations
            if c.thread_id == thread_id
            and ConversationStatus(c.status) != ConversationStatus.OPEN
            and c.summary
            and c.summary.strip()
        ]

    def get_open_conversation_messages(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> list[Message]:
        return [
            m
            for m in self.messages
            if m.tenant_id == tenant_id
            and m.thread_id == thread_id
            and not getattr(m, "soft_deleted", False)
        ]

    def get_last_assistant_message(
        self,
        tenant_id: int,
        thread_id: int,
    ) -> Message | None:
        asst_msgs = [
            m
            for m in self.messages
            if m.tenant_id == tenant_id
            and m.thread_id == thread_id
            and MessageRole(m.role) == MessageRole.ASSISTANT
            and not getattr(m, "soft_deleted", False)
        ]
        if not asst_msgs:
            return None
        return asst_msgs[-1]

    def get_shipments_for_order(
        self,
        tenant_id: int,
        order_pk: int,
    ) -> list[Shipment]:
        if tenant_id != self.tenant.tenant_id:
            return []
        return [s for s in self.shipments if s.order_pk == order_pk]

    def get_events_for_shipment(
        self,
        tenant_id: int,
        shipment_id: int,
    ) -> list[ShipmentEvent]:
        if tenant_id != self.tenant.tenant_id:
            return []
        return [e for e in self.events if e.shipment_id == shipment_id]

    def execute_cancel_order(
        self,
        tenant_id: int,
        order: CustomerOrder,
        *,
        idempotency_key: str,
        now: datetime,
    ) -> None:
        if order.tenant_id != tenant_id or tenant_id != self.tenant.tenant_id:
            raise ValueError("Cross-tenant cancel refused")
        if idempotency_key in self.cancel_idempotency_keys or order.cancelled_at is not None:
            self.cancel_idempotency_keys.add(idempotency_key)
            return
        order.cancelled_at = now
        for s in self.get_shipments_for_order(tenant_id, order.order_pk):
            s.status = ShipmentStatus.CANCELLED
        self.cancel_idempotency_keys.add(idempotency_key)

    def relay_live_handover_message(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        channel: str,
        external_message_id: str | None,
        content: str,
        reply_payload: dict[str, Any],
        now: datetime,
    ) -> Message:
        msg = Message(
            message_id=self._next_msg_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
            channel=channel,
            external_message_id=external_message_id,
            role=MessageRole.CUSTOMER,
            content=content,
            payload=reply_payload,
            attachments=None,
            soft_deleted=False,
        )
        self._next_msg_id += 1
        self.messages.append(msg)
        self.relayed_messages.append(msg)
        return msg

    def save_turn_transaction(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        channel: str,
        external_message_id: str | None,
        customer_text: str,
        reply_text: str,
        reply_payload: dict[str, Any],
        llm_calls: Sequence[LlmCallRecord],
        escalation_outbox: OutboxEntry | None,
        now: datetime,
    ) -> tuple[Message, Message]:
        cust_msg = Message(
            message_id=self._next_msg_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
            channel=channel,
            external_message_id=external_message_id,
            role=MessageRole.CUSTOMER,
            content=customer_text,
            payload=reply_payload,
            attachments=None,
            soft_deleted=False,
        )
        self._next_msg_id += 1

        asst_msg = Message(
            message_id=self._next_msg_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
            channel=channel,
            external_message_id=None,
            role=MessageRole.ASSISTANT,
            content=reply_text,
            payload=reply_payload,
            attachments=None,
            soft_deleted=False,
        )
        self._next_msg_id += 1

        self.messages.append(cust_msg)
        self.messages.append(asst_msg)

        for rec in llm_calls:
            self.llm_calls.append(
                LlmCall(
                    id=self._next_llm_id,
                    message_id=asst_msg.message_id,
                    step=rec.step,
                    model=rec.model,
                    stop_reason=rec.stop_reason,
                    input_tokens=rec.input_tokens,
                    output_tokens=rec.output_tokens,
                    cache_read_tokens=rec.cache_read_tokens,
                    cache_write_tokens=rec.cache_write_tokens,
                    latency_ms=rec.latency_ms,
                    tool_calls=rec.tool_calls,
                )
            )
            self._next_llm_id += 1

        if escalation_outbox is not None:
            self.outbox.append(escalation_outbox)

        return cust_msg, asst_msg
