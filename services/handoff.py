from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    BusinessHours,
    Conversation,
    ConversationIntent,
    ConversationMessage,
    ConversationStatus,
    CrmIntegration,
    CrmTicket,
    CrmTicketEscalation,
    CrmTicketKind,
    CrmTicketStatus,
    Escalation,
    EscalationMode,
    Message,
    MessageRole,
    Tenant,
    Thread,
)
from repositories.chat_storage import OutboxEntry
from services.business_hours import BusinessHoursService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation state transition table
# ---------------------------------------------------------------------------

ALLOWED: dict[ConversationStatus, frozenset[ConversationStatus]] = {
    ConversationStatus.OPEN: frozenset(
        {
            ConversationStatus.RESOLVED,
            ConversationStatus.ESCALATED,
            ConversationStatus.ABANDONED,
            ConversationStatus.STALE,
        }
    ),
    ConversationStatus.ESCALATED: frozenset(
        {
            ConversationStatus.RESOLVED_BY_HUMAN,
            ConversationStatus.ABANDONED,
            ConversationStatus.STALE,
        }
    ),
    ConversationStatus.STALE: frozenset(
        {
            ConversationStatus.OPEN,
            ConversationStatus.RESOLVED,
            ConversationStatus.ESCALATED,
            ConversationStatus.ABANDONED,
        }
    ),
    ConversationStatus.RESOLVED: frozenset(),
    ConversationStatus.ABANDONED: frozenset(),
    ConversationStatus.RESOLVED_BY_HUMAN: frozenset(),
}


def can_transition_conversation(
    from_status: ConversationStatus | int,
    to_status: ConversationStatus | int,
) -> bool:
    """Return True if transitioning from `from_status` to `to_status` is allowed."""
    src = ConversationStatus(int(from_status))
    dst = ConversationStatus(int(to_status))
    if src == dst:
        return True
    return dst in ALLOWED.get(src, frozenset())


def transition_conversation_status(
    conversation: Conversation,
    new_status: ConversationStatus | int,
) -> bool:
    """Apply `new_status` to `conversation` if allowed by `ALLOWED`; log and skip otherwise."""
    src = ConversationStatus(int(conversation.status))
    dst = ConversationStatus(int(new_status))
    if src == dst:
        return True
    if not can_transition_conversation(src, dst):
        logger.warning(
            "Skipping illegal conversation status transition %s -> %s for conversation_id=%s",
            src.name,
            dst.name,
            conversation.conversation_id,
        )
        return False
    conversation.status = dst
    return True


# ---------------------------------------------------------------------------
# CRM Ticket rules & helpers
# ---------------------------------------------------------------------------


def change_ticket_kind(
    ticket: CrmTicket,
    new_kind: CrmTicketKind | int,
) -> CrmTicket:
    """Enforce CrmTicket kind transition rules.

    Rules:
    - CLOSED is terminal.
    - The only allowed kind change is 2 (LIVE) -> 3 (LIVE_TURNED_TICKET).
    """
    current_status = CrmTicketStatus(int(ticket.status))
    if current_status == CrmTicketStatus.CLOSED:
        raise ValueError("Cannot change kind of a CLOSED ticket; CLOSED is terminal")

    current_kind = CrmTicketKind(int(ticket.kind))
    target_kind = CrmTicketKind(int(new_kind))
    if current_kind == target_kind:
        return ticket

    if current_kind == CrmTicketKind.LIVE and target_kind == CrmTicketKind.LIVE_TURNED_TICKET:
        ticket.kind = CrmTicketKind.LIVE_TURNED_TICKET
        return ticket

    raise ValueError(
        f"Illegal CrmTicket kind transition: {current_kind.name} ({int(current_kind)}) "
        f"-> {target_kind.name} ({int(target_kind)}); only 2 -> 3 is allowed"
    )


def close_crm_ticket(
    ticket: CrmTicket,
    *,
    conversations: Sequence[Conversation] = (),
    now: datetime | None = None,
) -> CrmTicket:
    """Close a CrmTicket and enforce terminal/closed_at/conversation rules.

    Rules:
    - CLOSED is terminal.
    - closed_at is set once.
    - Closing a kind=2 (LIVE) ticket moves its conversation to RESOLVED_BY_HUMAN (5).
    """
    current_status = CrmTicketStatus(int(ticket.status))
    if current_status == CrmTicketStatus.CLOSED or ticket.closed_at is not None:
        raise ValueError("CrmTicket is already CLOSED; CLOSED is terminal and closed_at is set once")

    close_time = now if now is not None else datetime.now(UTC)
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=UTC)

    ticket.status = CrmTicketStatus.CLOSED
    ticket.closed_at = close_time

    if CrmTicketKind(int(ticket.kind)) == CrmTicketKind.LIVE:
        for conv in conversations:
            conv.status = ConversationStatus.RESOLVED_BY_HUMAN

    return ticket


# ---------------------------------------------------------------------------
# CRM Client interface & FakeCrm implementation
# ---------------------------------------------------------------------------


@dataclass
class CrmTranscriptMessage:
    message_id: int
    role: MessageRole
    content: str
    created_at: datetime | None = None


@runtime_checkable
class CrmClient(Protocol):
    def open_conversation(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        subject: str,
        summary: str,
    ) -> str: ...

    def replay_transcript(
        self,
        *,
        external_conversation_id: str,
        messages: Sequence[CrmTranscriptMessage],
    ) -> None: ...

    def create_ticket(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        subject: str,
        summary: str,
    ) -> str: ...


class FakeCrm:
    """In-memory CRM provider implementation for testing and worker orchestration."""

    def __init__(self) -> None:
        self.conversations: dict[str, dict[str, Any]] = {}
        self.replayed_transcripts: dict[str, list[CrmTranscriptMessage]] = {}
        self.tickets: dict[str, dict[str, Any]] = {}
        self._seq: int = 1

    def open_conversation(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        subject: str,
        summary: str,
    ) -> str:
        ext_conv_id = f"fake-conv-{self._seq}"
        self._seq += 1
        self.conversations[ext_conv_id] = {
            "external_conversation_id": ext_conv_id,
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "subject": subject,
            "summary": summary,
        }
        self.replayed_transcripts.setdefault(ext_conv_id, [])
        return ext_conv_id

    def replay_transcript(
        self,
        *,
        external_conversation_id: str,
        messages: Sequence[CrmTranscriptMessage],
    ) -> None:
        self.replayed_transcripts.setdefault(external_conversation_id, []).extend(messages)

    def create_ticket(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        subject: str,
        summary: str,
    ) -> str:
        ext_ticket_id = f"fake-ticket-{self._seq}"
        self._seq += 1
        self.tickets[ext_ticket_id] = {
            "external_ticket_id": ext_ticket_id,
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "subject": subject,
            "summary": summary,
        }
        return ext_ticket_id


# ---------------------------------------------------------------------------
# Channel Adapter Publisher interface & Fake implementation
# ---------------------------------------------------------------------------


@dataclass
class PublishedChannelMessage:
    tenant_id: int
    thread_id: int
    channel: str
    message_id: int
    role: MessageRole
    content: str
    payload: dict[str, Any] | None = None


@runtime_checkable
class ChannelPublisher(Protocol):
    def publish(self, message: PublishedChannelMessage) -> None: ...


class FakeChannelPublisher:
    def __init__(self) -> None:
        self.published: list[PublishedChannelMessage] = []

    def publish(self, message: PublishedChannelMessage) -> None:
        self.published.append(message)


# ---------------------------------------------------------------------------
# 2. Escalation Worker (consumes "escalation.requested" outbox events)
# ---------------------------------------------------------------------------


class EscalationWorker:
    """Consumes `escalation.requested` outbox events and manages CRM tickets/conversations."""

    def __init__(
        self,
        *,
        crm_client: CrmClient | None = None,
        business_hours_service: BusinessHoursService | None = None,
        session: Session | None = None,
    ) -> None:
        self.crm_client: CrmClient = crm_client or FakeCrm()
        self.business_hours_service = business_hours_service or BusinessHoursService()
        self.session = session

        # In-memory state when running without a SQLAlchemy session
        self.tenants: dict[int, Tenant] = {}
        self.business_hours: dict[int, BusinessHours] = {}
        self.crm_integrations: dict[int, CrmIntegration] = {}
        self.threads: dict[int, Thread] = {}
        self.messages: list[Message] = []
        self.conversations: list[Conversation] = []
        self.escalations: list[Escalation] = []
        self.tickets: list[CrmTicket] = []
        self.ticket_escalations: list[CrmTicketEscalation] = []
        self._simulate_integrity_error_once: bool = False
        self._next_id: int = 1000

    def _alloc_id(self) -> int:
        val = self._next_id
        self._next_id += 1
        return val

    def consume_outbox(
        self,
        outbox_entries: list[OutboxEntry],
        *,
        now: datetime | None = None,
    ) -> list[CrmTicket]:
        """Process all `escalation.requested` entries in `outbox_entries`."""
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)

        results: list[CrmTicket] = []
        remaining: list[OutboxEntry] = []
        for entry in outbox_entries:
            if entry.event_type == "escalation.requested":
                ticket = self.process_escalation_event(entry, now=current_now)
                results.append(ticket)
            else:
                remaining.append(entry)
        outbox_entries[:] = remaining
        return results

    def process_escalation_event(
        self,
        entry: OutboxEntry | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> CrmTicket:
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)

        if isinstance(entry, OutboxEntry):
            tenant_id = entry.tenant_id
            thread_id = entry.thread_id
            payload = entry.payload
        else:
            tenant_id = int(entry["tenant_id"])
            thread_id = int(entry["thread_id"])
            payload = dict(entry.get("payload") or entry)

        reason = str(payload.get("reason") or "Customer requested support")
        subject = str(payload.get("subject") or f"Support escalation for thread #{thread_id}")
        summary = str(payload.get("summary") or reason)
        conversation_id = payload.get("conversation_id")

        tenant = self._get_tenant(tenant_id)
        crm_integration = self._get_crm_integration(tenant_id)
        bh = self._get_business_hours(tenant_id)

        if bh is not None:
            is_open_flag, _, _ = self.business_hours_service.is_open(
                bh,
                tenant.timezone if tenant is not None else "UTC",
                current_now,
            )
        else:
            is_open_flag = bool(payload.get("is_open", True))

        live = bool(crm_integration.supports_live_handover and is_open_flag)
        target_kind = CrmTicketKind.LIVE if live else CrmTicketKind.TICKET
        mode = EscalationMode.LIVE if live else EscalationMode.TICKET

        conv = self._ensure_conversation(
            thread_id=thread_id,
            conversation_id=int(conversation_id) if conversation_id is not None else None,
            now=current_now,
        )
        escalation = self._create_escalation(
            conversation_id=conv.conversation_id,
            subject=subject,
            summary=summary,
            mode=mode,
        )

        existing_ticket = self._find_open_ticket(thread_id=thread_id, kind=target_kind)
        if existing_ticket is not None:
            self._link_escalation(existing_ticket.id, escalation.id)
            return existing_ticket

        try:
            ticket = self._create_new_ticket(
                tenant_id=tenant_id,
                thread_id=thread_id,
                crm_integration=crm_integration,
                kind=target_kind,
                live=live,
                subject=subject,
                summary=summary,
                now=current_now,
            )
        except IntegrityError:
            if self.session is not None:
                self.session.rollback()
                # Re-persist conversation & escalation if rolled back
                self.session.add(conv)
                self.session.flush()
                escalation.conversation_id = conv.conversation_id
                self.session.add(escalation)
                self.session.flush()
            ticket_fallback = self._find_open_ticket(thread_id=thread_id, kind=target_kind)
            if ticket_fallback is None:
                raise
            ticket = ticket_fallback

        self._link_escalation(ticket.id, escalation.id)
        return ticket

    def change_kind(self, ticket: CrmTicket, new_kind: CrmTicketKind | int) -> CrmTicket:
        updated = change_ticket_kind(ticket, new_kind)
        if self.session is not None:
            self.session.flush()
        return updated

    def close_ticket(
        self,
        ticket: CrmTicket,
        *,
        now: datetime | None = None,
    ) -> CrmTicket:
        linked_conversations = self._get_conversations_for_ticket(ticket)
        closed = close_crm_ticket(ticket, conversations=linked_conversations, now=now)
        if self.session is not None:
            self.session.flush()
        return closed

    def _get_tenant(self, tenant_id: int) -> Tenant | None:
        if self.session is not None:
            return self.session.get(Tenant, tenant_id)
        return self.tenants.get(tenant_id)

    def _get_business_hours(self, tenant_id: int) -> BusinessHours | None:
        if self.session is not None:
            return self.session.scalar(
                select(BusinessHours).where(
                    BusinessHours.tenant_id == tenant_id,
                    BusinessHours.soft_deleted.is_(False),
                )
            )
        return self.business_hours.get(tenant_id)

    def _get_crm_integration(self, tenant_id: int) -> CrmIntegration:
        if self.session is not None:
            crm = self.session.scalar(
                select(CrmIntegration).where(
                    CrmIntegration.tenant_id == tenant_id,
                    CrmIntegration.soft_deleted.is_(False),
                )
            )
            if crm is None:
                raise ValueError(f"No active CrmIntegration for tenant_id={tenant_id}")
            return crm
        crm = self.crm_integrations.get(tenant_id)
        if crm is None:
            raise ValueError(f"No active CrmIntegration for tenant_id={tenant_id}")
        return crm

    def _ensure_conversation(
        self,
        *,
        thread_id: int,
        conversation_id: int | None,
        now: datetime,
    ) -> Conversation:
        if self.session is not None:
            if conversation_id is not None:
                conv = self.session.get(Conversation, conversation_id)
                if conv is not None:
                    transition_conversation_status(conv, ConversationStatus.ESCALATED)
                    self.session.flush()
                    return conv
            conv = self.session.scalar(
                select(Conversation)
                .where(Conversation.thread_id == thread_id)
                .order_by(Conversation.conversation_id.desc())
                .limit(1)
            )
            if conv is not None and ConversationStatus(int(conv.status)) in (
                ConversationStatus.OPEN,
                ConversationStatus.ESCALATED,
                ConversationStatus.STALE,
            ):
                transition_conversation_status(conv, ConversationStatus.ESCALATED)
                self.session.flush()
                return conv
            new_conv = Conversation(
                thread_id=thread_id,
                status=ConversationStatus.ESCALATED,
                intents=[ int(ConversationIntent.HUMAN_HANDOFF) ],
                summary="Escalated to support",
                evaluated_at=now,
            )
            self.session.add(new_conv)
            self.session.flush()
            return new_conv

        if conversation_id is not None:
            for c in self.conversations:
                if c.conversation_id == conversation_id:
                    transition_conversation_status(c, ConversationStatus.ESCALATED)
                    return c
        for c in reversed(self.conversations):
            if c.thread_id == thread_id and ConversationStatus(int(c.status)) in (
                ConversationStatus.OPEN,
                ConversationStatus.ESCALATED,
                ConversationStatus.STALE,
            ):
                transition_conversation_status(c, ConversationStatus.ESCALATED)
                return c

        new_conv = Conversation(
            conversation_id=self._alloc_id(),
            thread_id=thread_id,
            status=ConversationStatus.ESCALATED,
            intents=[int(ConversationIntent.HUMAN_HANDOFF)],
            summary="Escalated to support",
            evaluated_at=now,
        )
        self.conversations.append(new_conv)
        return new_conv

    def _create_escalation(
        self,
        *,
        conversation_id: int,
        subject: str,
        summary: str,
        mode: EscalationMode,
    ) -> Escalation:
        if self.session is not None:
            esc = Escalation(
                conversation_id=conversation_id,
                subject=subject,
                summary=summary,
                mode=mode,
            )
            self.session.add(esc)
            self.session.flush()
            return esc

        esc = Escalation(
            id=self._alloc_id(),
            conversation_id=conversation_id,
            subject=subject,
            summary=summary,
            mode=mode,
        )
        self.escalations.append(esc)
        return esc

    def _find_open_ticket(self, *, thread_id: int, kind: CrmTicketKind) -> CrmTicket | None:
        if self.session is not None:
            return self.session.scalar(
                select(CrmTicket).where(
                    CrmTicket.thread_id == thread_id,
                    CrmTicket.kind == int(kind),
                    CrmTicket.status == int(CrmTicketStatus.OPEN),
                )
            )
        for t in self.tickets:
            if (
                t.thread_id == thread_id
                and CrmTicketKind(int(t.kind)) == kind
                and CrmTicketStatus(int(t.status)) == CrmTicketStatus.OPEN
            ):
                return t
        return None

    def _get_thread_transcript(self, tenant_id: int, thread_id: int) -> list[CrmTranscriptMessage]:
        if self.session is not None:
            msgs = self.session.scalars(
                select(Message)
                .where(
                    Message.tenant_id == tenant_id,
                    Message.thread_id == thread_id,
                    Message.soft_deleted.is_(False),
                )
                .order_by(Message.message_id.asc())
            ).all()
        else:
            msgs = [
                m
                for m in self.messages
                if m.tenant_id == tenant_id
                and m.thread_id == thread_id
                and not getattr(m, "soft_deleted", False)
            ]
        return [
            CrmTranscriptMessage(
                message_id=m.message_id,
                role=MessageRole(int(m.role)),
                content=m.content,
                created_at=getattr(m, "created_at", None),
            )
            for m in msgs
        ]

    def _create_new_ticket(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        crm_integration: CrmIntegration,
        kind: CrmTicketKind,
        live: bool,
        subject: str,
        summary: str,
        now: datetime,
    ) -> CrmTicket:
        if self._simulate_integrity_error_once:
            self._simulate_integrity_error_once = False
            raise IntegrityError("INSERT INTO crm_ticket", {}, Exception("uq_crm_ticket_open_thread_kind"))

        ext_conv_id: str | None = None
        ext_ticket_id: str | None = None

        if live:
            ext_conv_id = self.crm_client.open_conversation(
                tenant_id=tenant_id,
                thread_id=thread_id,
                subject=subject,
                summary=summary,
            )
            transcript = self._get_thread_transcript(tenant_id, thread_id)
            self.crm_client.replay_transcript(
                external_conversation_id=ext_conv_id,
                messages=transcript,
            )
        else:
            ext_ticket_id = self.crm_client.create_ticket(
                tenant_id=tenant_id,
                thread_id=thread_id,
                subject=subject,
                summary=summary,
            )

        if self.session is not None:
            with self.session.begin_nested():
                ticket = CrmTicket(
                    thread_id=thread_id,
                    crm_integration_id=crm_integration.id,
                    kind=kind,
                    status=CrmTicketStatus.OPEN,
                    external_ticket_id=ext_ticket_id,
                    external_conversation_id=ext_conv_id,
                    opened_at=now,
                )
                self.session.add(ticket)
                self.session.flush()
                return ticket

        ticket = CrmTicket(
            id=self._alloc_id(),
            thread_id=thread_id,
            crm_integration_id=crm_integration.id,
            kind=kind,
            status=CrmTicketStatus.OPEN,
            external_ticket_id=ext_ticket_id,
            external_conversation_id=ext_conv_id,
            opened_at=now,
        )
        self.tickets.append(ticket)
        return ticket

    def _link_escalation(self, crm_ticket_id: int, escalation_id: int) -> CrmTicketEscalation:
        if self.session is not None:
            existing = self.session.scalar(
                select(CrmTicketEscalation).where(
                    CrmTicketEscalation.crm_ticket_id == crm_ticket_id,
                    CrmTicketEscalation.escalation_id == escalation_id,
                )
            )
            if existing is not None:
                return existing
            link = CrmTicketEscalation(
                crm_ticket_id=crm_ticket_id,
                escalation_id=escalation_id,
            )
            self.session.add(link)
            self.session.flush()
            return link

        for link in self.ticket_escalations:
            if link.crm_ticket_id == crm_ticket_id and link.escalation_id == escalation_id:
                return link
        new_link = CrmTicketEscalation(
            crm_ticket_id=crm_ticket_id,
            escalation_id=escalation_id,
        )
        self.ticket_escalations.append(new_link)
        return new_link

    def _get_conversations_for_ticket(self, ticket: CrmTicket) -> list[Conversation]:
        if self.session is not None:
            convs = list(
                self.session.scalars(
                    select(Conversation)
                    .join(Escalation, Escalation.conversation_id == Conversation.conversation_id)
                    .join(
                        CrmTicketEscalation,
                        CrmTicketEscalation.escalation_id == Escalation.id,
                    )
                    .where(CrmTicketEscalation.crm_ticket_id == ticket.id)
                ).all()
            )
            if not convs:
                convs = list(
                    self.session.scalars(
                        select(Conversation).where(Conversation.thread_id == ticket.thread_id)
                    ).all()
                )
            return convs

        esc_ids = {
            te.escalation_id
            for te in self.ticket_escalations
            if te.crm_ticket_id == ticket.id
        }
        conv_ids = {
            e.conversation_id
            for e in self.escalations
            if e.id in esc_ids
        }
        convs = [c for c in self.conversations if c.conversation_id in conv_ids]
        if not convs:
            convs = [c for c in self.conversations if c.thread_id == ticket.thread_id]
        return convs


# ---------------------------------------------------------------------------
# 3. CRM Webhook Signature Verification & Processing
# ---------------------------------------------------------------------------


def compute_crm_webhook_signature(secret: str, raw_body: bytes) -> str:
    """Compute HMAC-SHA256 hex digest for a CRM webhook payload."""
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def verify_crm_webhook_signature(
    secret: str,
    raw_body: bytes,
    signature_header: str | None,
) -> bool:
    """Constant-time verification of the CRM webhook HMAC-SHA256 signature."""
    if not signature_header or not signature_header.strip():
        return False
    provided = signature_header.strip()
    if provided.lower().startswith("sha256="):
        provided = provided.split("=", 1)[1].strip()
    expected = compute_crm_webhook_signature(secret, raw_body)
    return hmac.compare_digest(provided.lower(), expected.lower())


class CrmWebhookPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: str = "agent_reply"
    thread_id: int | None = None
    external_conversation_id: str | None = None
    external_ticket_id: str | None = None
    external_message_id: str | None = None
    channel: str = "whatsapp"
    text: str | None = None
    content: str | None = None
    attachments: list[dict[str, Any]] | None = None


class CrmWebhookService:
    """Verifies CRM webhooks, persists human agent replies (role=3), and publishes to channel adapter."""

    def __init__(
        self,
        *,
        channel_publisher: ChannelPublisher | None = None,
        session: Session | None = None,
        secret_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self.channel_publisher: ChannelPublisher = channel_publisher or FakeChannelPublisher()
        self.session = session
        self.secret_resolver = secret_resolver or (lambda ref: ref)

        # In-memory collections when running without a SQLAlchemy session
        self.integrations_by_uuid: dict[uuid.UUID, CrmIntegration] = {}
        self.tickets: list[CrmTicket] = []
        self.conversations: list[Conversation] = []
        self.messages: list[Message] = []
        self._next_message_id: int = 9000

    def register_integration(self, integration: CrmIntegration) -> None:
        self.integrations_by_uuid[integration.webhook_uuid] = integration

    def handle_webhook(
        self,
        *,
        webhook_uuid: uuid.UUID,
        raw_body: bytes,
        signature: str | None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        integration = self._find_integration(webhook_uuid)
        if integration is None:
            raise LookupError(f"Unknown CRM webhook_uuid: {webhook_uuid}")

        secret = self.secret_resolver(integration.secret_ref)
        if not verify_crm_webhook_signature(secret, raw_body, signature):
            raise PermissionError("Invalid CRM webhook signature")

        data = json.loads(raw_body.decode("utf-8"))
        payload = CrmWebhookPayload.model_validate(data)

        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)

        ticket = self._resolve_ticket(integration.id, payload)
        thread_id = payload.thread_id or (ticket.thread_id if ticket is not None else None)
        if thread_id is None:
            raise ValueError("Unable to resolve thread_id for CRM webhook payload")

        if payload.event in ("ticket.closed", "conversation.closed", "close"):
            if ticket is None:
                raise ValueError("No matching CrmTicket found to close")
            convs = self._get_conversations_for_thread(thread_id)
            close_crm_ticket(ticket, conversations=convs, now=current_now)
            if self.session is not None:
                self.session.flush()
            return {"status": "closed", "ticket_id": ticket.id}

        reply_text = payload.text if payload.text is not None else (payload.content or "")
        msg = self._persist_human_reply(
            tenant_id=integration.tenant_id,
            thread_id=thread_id,
            channel=payload.channel,
            external_message_id=payload.external_message_id,
            content=reply_text,
            attachments=payload.attachments,
        )

        self.channel_publisher.publish(
            PublishedChannelMessage(
                tenant_id=integration.tenant_id,
                thread_id=thread_id,
                channel=msg.channel,
                message_id=msg.message_id,
                role=MessageRole(int(msg.role)),
                content=msg.content,
                payload=msg.payload,
            )
        )
        return {
            "status": "ok",
            "message_id": msg.message_id,
            "role": int(msg.role),
        }

    def _find_integration(self, webhook_uuid: uuid.UUID) -> CrmIntegration | None:
        if self.session is not None:
            return self.session.scalar(
                select(CrmIntegration).where(
                    CrmIntegration.webhook_uuid == webhook_uuid,
                    CrmIntegration.soft_deleted.is_(False),
                )
            )
        return self.integrations_by_uuid.get(webhook_uuid)

    def _resolve_ticket(
        self,
        crm_integration_id: int,
        payload: CrmWebhookPayload,
    ) -> CrmTicket | None:
        if self.session is not None:
            if payload.external_conversation_id:
                t = self.session.scalar(
                    select(CrmTicket).where(
                        CrmTicket.crm_integration_id == crm_integration_id,
                        CrmTicket.external_conversation_id == payload.external_conversation_id,
                    )
                )
                if t is not None:
                    return t
            if payload.external_ticket_id:
                t = self.session.scalar(
                    select(CrmTicket).where(
                        CrmTicket.crm_integration_id == crm_integration_id,
                        CrmTicket.external_ticket_id == payload.external_ticket_id,
                    )
                )
                if t is not None:
                    return t
            if payload.thread_id is not None:
                return self.session.scalar(
                    select(CrmTicket)
                    .where(
                        CrmTicket.crm_integration_id == crm_integration_id,
                        CrmTicket.thread_id == payload.thread_id,
                    )
                    .order_by(CrmTicket.id.desc())
                    .limit(1)
                )
            return None

        for t in reversed(self.tickets):
            if t.crm_integration_id != crm_integration_id:
                continue
            if (
                payload.external_conversation_id
                and t.external_conversation_id == payload.external_conversation_id
            ):
                return t
            if payload.external_ticket_id and t.external_ticket_id == payload.external_ticket_id:
                return t
            if payload.thread_id is not None and t.thread_id == payload.thread_id:
                return t
        return None

    def _get_conversations_for_thread(self, thread_id: int) -> list[Conversation]:
        if self.session is not None:
            return list(
                self.session.scalars(
                    select(Conversation).where(Conversation.thread_id == thread_id)
                ).all()
            )
        return [c for c in self.conversations if c.thread_id == thread_id]

    def _persist_human_reply(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        channel: str,
        external_message_id: str | None,
        content: str,
        attachments: list[dict[str, Any]] | None,
    ) -> Message:
        if self.session is not None:
            msg = Message(
                tenant_id=tenant_id,
                thread_id=thread_id,
                channel=channel,
                external_message_id=external_message_id,
                role=MessageRole.HUMAN_AGENT,
                content=content,
                payload={"role": int(MessageRole.HUMAN_AGENT), "content": content},
                attachments=attachments,
                soft_deleted=False,
            )
            self.session.add(msg)
            self.session.flush()
            return msg

        msg = Message(
            message_id=self._next_message_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
            channel=channel,
            external_message_id=external_message_id,
            role=MessageRole.HUMAN_AGENT,
            content=content,
            payload={"role": int(MessageRole.HUMAN_AGENT), "content": content},
            attachments=attachments,
            soft_deleted=False,
        )
        self._next_message_id += 1
        self.messages.append(msg)
        return msg


# ---------------------------------------------------------------------------
# 4. Resolution Evaluator (cron every 10 min, threads idle > 10 min)
# ---------------------------------------------------------------------------

IDLE_THRESHOLD_MINUTES = 10
ALLOWED_LLM_CONVERSATION_STATUSES: frozenset[int] = frozenset(
    {
        int(ConversationStatus.OPEN),
        int(ConversationStatus.RESOLVED),
        int(ConversationStatus.ESCALATED),
        int(ConversationStatus.ABANDONED),
        int(ConversationStatus.STALE),
    }
)

EVALUATOR_OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["conversations"],
    "properties": {
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "message_ids",
                    "status",
                    "intents",
                    "summary",
                    "escalation_subject",
                    "escalation_summary",
                ],
                "properties": {
                    "message_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "status": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 6],
                    },
                    "intents": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "summary": {"type": "string"},
                    "escalation_subject": {"type": ["string", "null"]},
                    "escalation_summary": {"type": ["string", "null"]},
                },
            },
        }
    },
}


class EvaluatedConversationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_ids: list[int]
    status: int
    intents: list[int] = Field(default_factory=list)
    summary: str
    escalation_subject: str | None = None
    escalation_summary: str | None = None


class EvaluatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversations: list[EvaluatedConversationItem]


def format_unassigned_messages_for_evaluator(
    messages: Sequence[Message],
    *,
    now: datetime,
) -> str:
    """Render messages not yet in conversation_message as:
    `[id=N | M min ago] Customer: …` / `Assistant: …`
    """
    base_now = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    lines: list[str] = []
    for msg in sorted(messages, key=lambda m: m.message_id):
        msg_time = getattr(msg, "created_at", None) or base_now
        if msg_time.tzinfo is None:
            msg_time = msg_time.replace(tzinfo=UTC)
        minutes_ago = max(0, int((base_now - msg_time).total_seconds() // 60))
        role_enum = MessageRole(int(msg.role))
        if role_enum == MessageRole.CUSTOMER:
            speaker = "Customer"
        elif role_enum == MessageRole.ASSISTANT:
            speaker = "Assistant"
        else:
            speaker = "Human Agent"
        lines.append(f"[id={msg.message_id} | {minutes_ago} min ago] {speaker}: {msg.content}")
    return "\n".join(lines)


def validate_evaluator_output(
    raw_output: str | Mapping[str, Any] | EvaluatorOutput,
    *,
    expected_message_ids: Sequence[int],
) -> EvaluatorOutput:
    """Validate LLM structured output for the resolution evaluator.

    Rejects unless:
    - Schema is valid and status is in {1, 2, 3, 4, 6} (never 5 / RESOLVED_BY_HUMAN).
    - Multiset of returned message_ids == multiset of expected_message_ids exactly.
    - Every ESCALATED (status=3) conversation has a non-empty escalation_subject.
    """
    if isinstance(raw_output, EvaluatorOutput):
        parsed = raw_output
    elif isinstance(raw_output, str):
        parsed = EvaluatorOutput.model_validate_json(raw_output)
    elif isinstance(raw_output, Mapping):
        parsed = EvaluatorOutput.model_validate(raw_output)
    else:
        raise ValueError(f"Unsupported evaluator output type: {type(raw_output)!r}")

    returned_ids: list[int] = []
    for item in parsed.conversations:
        if item.status == int(ConversationStatus.RESOLVED_BY_HUMAN):
            raise ValueError("Status 5 (RESOLVED_BY_HUMAN) is never allowed from the LLM")
        if item.status not in ALLOWED_LLM_CONVERSATION_STATUSES:
            raise ValueError(f"Invalid conversation status from LLM: {item.status}")
        if not item.message_ids:
            raise ValueError("Conversation item cannot have empty message_ids")
        if item.status == int(ConversationStatus.ESCALATED):
            if not item.escalation_subject or not item.escalation_subject.strip():
                raise ValueError("ESCALATED conversation requires a non-empty escalation_subject")
        returned_ids.extend(item.message_ids)

    if Counter(returned_ids) != Counter(expected_message_ids):
        raise ValueError(
            f"Returned message_ids multiset {Counter(returned_ids)} does not match "
            f"input message_ids multiset {Counter(expected_message_ids)}"
        )

    return parsed


@dataclass
class EvaluatorAlert:
    thread_id: int
    reason: str
    attempts: int
    created_at: datetime


class ResolutionEvaluator:
    """Evaluates unassigned messages on threads idle for more than 10 minutes."""

    def __init__(
        self,
        *,
        anthropic_client: Any,
        session: Session | None = None,
        model: str = "claude-opus-5-5",
        idle_minutes: int = IDLE_THRESHOLD_MINUTES,
    ) -> None:
        self.anthropic_client = anthropic_client
        self.session = session
        self.model = model
        self.idle_minutes = idle_minutes

        # In-memory storage when running without SQLAlchemy session
        self.threads: list[Thread] = []
        self.messages: list[Message] = []
        self.conversations: list[Conversation] = []
        self.conversation_messages: list[ConversationMessage] = []
        self.escalations: list[Escalation] = []
        self.alerts: list[EvaluatorAlert] = []
        self.skipped_transitions: list[tuple[int, ConversationStatus, ConversationStatus]] = []
        self._next_conv_id: int = 5000
        self._next_esc_id: int = 7000

    async def run_cron(self, *, now: datetime | None = None) -> list[Conversation]:
        """Evaluate all threads idle for more than `self.idle_minutes` minutes."""
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)

        idle_thread_ids = self._find_idle_threads_with_unassigned_messages(current_now)
        updated_conversations: list[Conversation] = []
        for tid in idle_thread_ids:
            convs = await self.evaluate_thread(tid, now=current_now)
            updated_conversations.extend(convs)
        return updated_conversations

    async def evaluate_thread(
        self,
        thread_id: int,
        *,
        now: datetime | None = None,
    ) -> list[Conversation]:
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)

        unassigned = self._get_unassigned_messages(thread_id, current_now)
        if not unassigned:
            return []

        input_ids = [m.message_id for m in unassigned]
        formatted_prompt = format_unassigned_messages_for_evaluator(unassigned, now=current_now)

        validated: EvaluatorOutput | None = None
        last_error: str = ""

        # Try once, and on rejection retry once (max 2 attempts), then alert and never write partially
        for attempt in (1, 2):
            try:
                raw_text = await self._call_llm(formatted_prompt)
                validated = validate_evaluator_output(
                    raw_text,
                    expected_message_ids=input_ids,
                )
                break
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                logger.warning(
                    "ResolutionEvaluator validation failed on attempt %d for thread_id=%d: %s",
                    attempt,
                    thread_id,
                    last_error,
                )

        if validated is None:
            self.alerts.append(
                EvaluatorAlert(
                    thread_id=thread_id,
                    reason=last_error or "Evaluator output rejected",
                    attempts=2,
                    created_at=current_now,
                )
            )
            return []

        return self._upsert_evaluated_conversations(
            thread_id=thread_id,
            validated=validated,
            now=current_now,
        )

    async def _call_llm(self, formatted_messages: str) -> str:
        response = await self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": (
                        "Segment the unassigned support messages into conversations and classify "
                        "their status (1=OPEN, 2=RESOLVED, 3=ESCALATED, 4=ABANDONED, 6=STALE)."
                    ),
                }
            ],
            messages=[{"role": "user", "content": formatted_messages}],
            output_config={
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": EVALUATOR_OUTPUT_JSON_SCHEMA,
                },
            },
        )
        raw_content = getattr(response, "content", []) or []
        parts: list[str] = []
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "")))
        return "\n".join(parts).strip()

    def _find_idle_threads_with_unassigned_messages(self, now: datetime) -> list[int]:
        cutoff = now - timedelta(minutes=self.idle_minutes)
        if self.session is not None:
            assigned_subq = select(ConversationMessage.message_id)
            unassigned_msgs = list(
                self.session.scalars(
                    select(Message)
                    .where(
                        Message.soft_deleted.is_(False),
                        Message.message_id.not_in(assigned_subq),
                    )
                    .order_by(Message.message_id.asc())
                ).all()
            )
            candidate_thread_ids = sorted({m.thread_id for m in unassigned_msgs})
            idle_threads: list[int] = []
            for tid in candidate_thread_ids:
                all_thread_msgs = list(
                    self.session.scalars(
                        select(Message).where(
                            Message.thread_id == tid,
                            Message.soft_deleted.is_(False),
                        )
                    ).all()
                )
                latest_at = max(
                    (
                        (m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=UTC))
                        for m in all_thread_msgs
                        if getattr(m, "created_at", None) is not None
                    ),
                    default=None,
                )
                if latest_at is not None and latest_at <= cutoff:
                    idle_threads.append(tid)
            return idle_threads

        assigned_ids = {cm.message_id for cm in self.conversation_messages}
        by_thread: dict[int, list[Message]] = {}
        for m in self.messages:
            if not getattr(m, "soft_deleted", False):
                by_thread.setdefault(m.thread_id, []).append(m)

        idle_threads = []
        for tid, msgs in sorted(by_thread.items()):
            has_unassigned = any(m.message_id not in assigned_ids for m in msgs)
            if not has_unassigned:
                continue
            latest_at = max(
                (
                    (m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=UTC))
                    for m in msgs
                    if getattr(m, "created_at", None) is not None
                ),
                default=None,
            )
            if latest_at is not None and latest_at <= cutoff:
                idle_threads.append(tid)
        return idle_threads

    def _get_unassigned_messages(self, thread_id: int, now: datetime) -> list[Message]:
        cutoff = now - timedelta(minutes=self.idle_minutes)
        if self.session is not None:
            all_thread_msgs = list(
                self.session.scalars(
                    select(Message)
                    .where(
                        Message.thread_id == thread_id,
                        Message.soft_deleted.is_(False),
                    )
                    .order_by(Message.message_id.asc())
                ).all()
            )
            if not all_thread_msgs:
                return []
            latest_at = max(
                (
                    (m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=UTC))
                    for m in all_thread_msgs
                    if getattr(m, "created_at", None) is not None
                ),
                default=None,
            )
            if latest_at is not None and latest_at > cutoff:
                return []
            assigned_ids = set(
                self.session.scalars(
                    select(ConversationMessage.message_id).join(
                        Message, Message.message_id == ConversationMessage.message_id
                    ).where(Message.thread_id == thread_id)
                ).all()
            )
            return [m for m in all_thread_msgs if m.message_id not in assigned_ids]

        thread_msgs = [
            m
            for m in self.messages
            if m.thread_id == thread_id and not getattr(m, "soft_deleted", False)
        ]
        if not thread_msgs:
            return []
        latest_at = max(
            (
                (m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=UTC))
                for m in thread_msgs
                if getattr(m, "created_at", None) is not None
            ),
            default=None,
        )
        if latest_at is not None and latest_at > cutoff:
            return []
        assigned_ids = {cm.message_id for cm in self.conversation_messages}
        return sorted(
            [m for m in thread_msgs if m.message_id not in assigned_ids],
            key=lambda m: m.message_id,
        )

    def _upsert_evaluated_conversations(
        self,
        *,
        thread_id: int,
        validated: EvaluatorOutput,
        now: datetime,
    ) -> list[Conversation]:
        result_convs: list[Conversation] = []

        if self.session is not None:
            with self.session.begin_nested():
                for item in validated.conversations:
                    conv = self._upsert_single_conversation_db(
                        thread_id=thread_id,
                        item=item,
                        now=now,
                    )
                    result_convs.append(conv)
            return result_convs

        for item in validated.conversations:
            conv = self._upsert_single_conversation_mem(
                thread_id=thread_id,
                item=item,
                now=now,
            )
            result_convs.append(conv)
        return result_convs

    def _upsert_single_conversation_db(
        self,
        *,
        thread_id: int,
        item: EvaluatedConversationItem,
        now: datetime,
    ) -> Conversation:
        assert self.session is not None
        target_status = ConversationStatus(item.status)

        # Check if any message_id is already linked or if an open conversation exists on the thread
        existing_cm = self.session.scalar(
            select(ConversationMessage).where(
                ConversationMessage.message_id.in_(item.message_ids)
            )
        )
        conv: Conversation | None = None
        if existing_cm is not None:
            conv = self.session.get(Conversation, existing_cm.conversation_id)
        else:
            conv = self.session.scalar(
                select(Conversation)
                .where(Conversation.thread_id == thread_id)
                .order_by(Conversation.conversation_id.desc())
                .limit(1)
            )

        if conv is not None:
            old_status = ConversationStatus(int(conv.status))
            if not transition_conversation_status(conv, target_status):
                self.skipped_transitions.append((conv.conversation_id, old_status, target_status))
            conv.intents = [int(i) for i in item.intents]
            conv.summary = item.summary
            conv.evaluated_at = now
        else:
            conv = Conversation(
                thread_id=thread_id,
                status=target_status,
                intents=[int(i) for i in item.intents],
                summary=item.summary,
                evaluated_at=now,
            )
            self.session.add(conv)
            self.session.flush()

        for mid in item.message_ids:
            cm_exists = self.session.scalar(
                select(ConversationMessage).where(ConversationMessage.message_id == mid)
            )
            if cm_exists is None:
                self.session.add(
                    ConversationMessage(
                        message_id=mid,
                        conversation_id=conv.conversation_id,
                    )
                )
            else:
                cm_exists.conversation_id = conv.conversation_id

        if (
            ConversationStatus(int(conv.status)) == ConversationStatus.ESCALATED
            and item.escalation_subject
        ):
            existing_esc = self.session.scalar(
                select(Escalation).where(Escalation.conversation_id == conv.conversation_id)
            )
            if existing_esc is None:
                self.session.add(
                    Escalation(
                        conversation_id=conv.conversation_id,
                        subject=item.escalation_subject.strip(),
                        summary=(item.escalation_summary or item.summary).strip(),
                        mode=EscalationMode.TICKET,
                    )
                )
            else:
                existing_esc.subject = item.escalation_subject.strip()
                existing_esc.summary = (item.escalation_summary or item.summary).strip()

        self.session.flush()
        return conv

    def _upsert_single_conversation_mem(
        self,
        *,
        thread_id: int,
        item: EvaluatedConversationItem,
        now: datetime,
    ) -> Conversation:
        target_status = ConversationStatus(item.status)
        existing_conv_id: int | None = None
        for cm in self.conversation_messages:
            if cm.message_id in item.message_ids:
                existing_conv_id = cm.conversation_id
                break

        conv: Conversation | None = None
        if existing_conv_id is not None:
            for c in self.conversations:
                if c.conversation_id == existing_conv_id:
                    conv = c
                    break
        else:
            for c in reversed(self.conversations):
                if c.thread_id == thread_id:
                    conv = c
                    break

        if conv is not None:
            old_status = ConversationStatus(int(conv.status))
            if not transition_conversation_status(conv, target_status):
                self.skipped_transitions.append((conv.conversation_id, old_status, target_status))
            conv.intents = [int(i) for i in item.intents]
            conv.summary = item.summary
            conv.evaluated_at = now
        else:
            conv = Conversation(
                conversation_id=self._next_conv_id,
                thread_id=thread_id,
                status=target_status,
                intents=[int(i) for i in item.intents],
                summary=item.summary,
                evaluated_at=now,
            )
            self._next_conv_id += 1
            self.conversations.append(conv)

        for mid in item.message_ids:
            existing_cm = next(
                (cm for cm in self.conversation_messages if cm.message_id == mid),
                None,
            )
            if existing_cm is None:
                self.conversation_messages.append(
                    ConversationMessage(
                        message_id=mid,
                        conversation_id=conv.conversation_id,
                    )
                )
            else:
                existing_cm.conversation_id = conv.conversation_id

        if (
            ConversationStatus(int(conv.status)) == ConversationStatus.ESCALATED
            and item.escalation_subject
        ):
            existing_esc = next(
                (e for e in self.escalations if e.conversation_id == conv.conversation_id),
                None,
            )
            if existing_esc is None:
                self.escalations.append(
                    Escalation(
                        id=self._next_esc_id,
                        conversation_id=conv.conversation_id,
                        subject=item.escalation_subject.strip(),
                        summary=(item.escalation_summary or item.summary).strip(),
                        mode=EscalationMode.TICKET,
                    )
                )
                self._next_esc_id += 1
            else:
                existing_esc.subject = item.escalation_subject.strip()
                existing_esc.summary = (item.escalation_summary or item.summary).strip()

        return conv


# ---------------------------------------------------------------------------
# 5. Inactivity Delayed Job (15 min ETA after last assistant reply, 1 per 24h)
# ---------------------------------------------------------------------------

INACTIVITY_DELAY_MINUTES = 15
INACTIVITY_COOLDOWN_HOURS = 24
INACTIVITY_CLOSE_TEXT = (
    "Since we haven't heard back from you, we're closing this chat for now. "
    "How would you rate your support experience today?"
)
CSAT_BUTTONS: list[dict[str, Any]] = [
    {"id": "csat_good", "label": "Great", "score": 5},
    {"id": "csat_ok", "label": "Okay", "score": 3},
    {"id": "csat_bad", "label": "Poor", "score": 1},
]


@dataclass
class DelayedJob:
    job_id: str
    task_name: str
    tenant_id: int
    thread_id: int
    trigger_message_id: int
    channel: str
    enqueued_at: datetime
    eta: datetime
    executed_at: datetime | None = None


class DelayedJobQueue:
    """ETA-based delayed job queue (never a polling loop)."""

    def __init__(self) -> None:
        self.jobs: list[DelayedJob] = []

    def enqueue(
        self,
        *,
        task_name: str,
        tenant_id: int,
        thread_id: int,
        trigger_message_id: int,
        channel: str = "whatsapp",
        eta: datetime,
        now: datetime | None = None,
    ) -> DelayedJob:
        enqueued_at = now if now is not None else datetime.now(UTC)
        if enqueued_at.tzinfo is None:
            enqueued_at = enqueued_at.replace(tzinfo=UTC)
        eta_utc = eta if eta.tzinfo is not None else eta.replace(tzinfo=UTC)

        job = DelayedJob(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            task_name=task_name,
            tenant_id=tenant_id,
            thread_id=thread_id,
            trigger_message_id=trigger_message_id,
            channel=channel,
            enqueued_at=enqueued_at,
            eta=eta_utc,
        )
        self.jobs.append(job)
        return job

    def pop_ready(self, *, now: datetime) -> list[DelayedJob]:
        now_utc = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        ready = [
            j
            for j in self.jobs
            if j.executed_at is None and j.eta <= now_utc
        ]
        ready.sort(key=lambda j: j.eta)
        return ready


class InactivityWorker:
    """Enqueues and executes 15-minute ETA delayed jobs for thread inactivity + CSAT."""

    def __init__(
        self,
        *,
        job_queue: DelayedJobQueue | None = None,
        channel_publisher: ChannelPublisher | None = None,
        session: Session | None = None,
    ) -> None:
        self.job_queue = job_queue or DelayedJobQueue()
        self.channel_publisher: ChannelPublisher = channel_publisher or FakeChannelPublisher()
        self.session = session

        # In-memory collections when running without a SQLAlchemy session
        self.messages: list[Message] = []
        self.tickets: list[CrmTicket] = []
        self.last_csat_sent_at: dict[int, datetime] = {}
        self._next_message_id: int = 12000

    def enqueue_for_assistant_reply(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        assistant_message_id: int,
        replied_at: datetime,
        channel: str = "whatsapp",
    ) -> DelayedJob:
        """Enqueue a delayed job with ETA = replied_at + 15 minutes."""
        base_time = replied_at if replied_at.tzinfo is not None else replied_at.replace(tzinfo=UTC)
        eta = base_time + timedelta(minutes=INACTIVITY_DELAY_MINUTES)
        return self.job_queue.enqueue(
            task_name="inactivity_close_csat",
            tenant_id=tenant_id,
            thread_id=thread_id,
            trigger_message_id=assistant_message_id,
            channel=channel,
            eta=eta,
            now=base_time,
        )

    def run_due_jobs(self, *, now: datetime) -> list[Message]:
        now_utc = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        sent: list[Message] = []
        for job in self.job_queue.pop_ready(now=now_utc):
            msg = self.execute_job(job, now=now_utc)
            if msg is not None:
                sent.append(msg)
        return sent

    def execute_job(self, job: DelayedJob, *, now: datetime | None = None) -> Message | None:
        now_utc = now if now is not None else job.eta
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=UTC)

        if now_utc < job.eta:
            return None

        job.executed_at = now_utc

        # 1. If there is an open ticket on the thread -> skip
        if self._has_open_ticket(job.thread_id):
            return None

        # 2. If newer messages arrived after trigger_message_id -> skip
        latest_msg = self._get_latest_non_csat_message(job.thread_id)
        if latest_msg is not None and latest_msg.message_id != job.trigger_message_id:
            return None

        # 3. At most once per 24h per thread
        if self._sent_csat_within_24h(job.thread_id, now_utc):
            return None

        payload = {
            "type": "inactivity_close",
            "csat_buttons": list(CSAT_BUTTONS),
            "sent_at": now_utc.isoformat(),
        }
        close_msg = self._persist_csat_message(
            tenant_id=job.tenant_id,
            thread_id=job.thread_id,
            channel=job.channel,
            content=INACTIVITY_CLOSE_TEXT,
            payload=payload,
            now=now_utc,
        )
        self.last_csat_sent_at[job.thread_id] = now_utc

        self.channel_publisher.publish(
            PublishedChannelMessage(
                tenant_id=job.tenant_id,
                thread_id=job.thread_id,
                channel=job.channel,
                message_id=close_msg.message_id,
                role=MessageRole.ASSISTANT,
                content=close_msg.content,
                payload=payload,
            )
        )
        return close_msg

    def _has_open_ticket(self, thread_id: int) -> bool:
        if self.session is not None:
            t = self.session.scalar(
                select(CrmTicket).where(
                    CrmTicket.thread_id == thread_id,
                    CrmTicket.status == int(CrmTicketStatus.OPEN),
                )
            )
            return t is not None
        return any(
            t.thread_id == thread_id and CrmTicketStatus(int(t.status)) == CrmTicketStatus.OPEN
            for t in self.tickets
        )

    def _get_latest_non_csat_message(self, thread_id: int) -> Message | None:
        if self.session is not None:
            msgs = list(
                self.session.scalars(
                    select(Message)
                    .where(
                        Message.thread_id == thread_id,
                        Message.soft_deleted.is_(False),
                    )
                    .order_by(Message.message_id.asc())
                ).all()
            )
        else:
            msgs = [
                m
                for m in self.messages
                if m.thread_id == thread_id and not getattr(m, "soft_deleted", False)
            ]

        non_csat = [
            m
            for m in msgs
            if not (isinstance(m.payload, dict) and m.payload.get("type") == "inactivity_close")
        ]
        return non_csat[-1] if non_csat else None

    def _sent_csat_within_24h(self, thread_id: int, now: datetime) -> bool:
        cutoff = now - timedelta(hours=INACTIVITY_COOLDOWN_HOURS)
        last_mem = self.last_csat_sent_at.get(thread_id)
        if last_mem is not None and last_mem > cutoff:
            return True

        if self.session is not None:
            msgs = list(
                self.session.scalars(
                    select(Message).where(
                        Message.thread_id == thread_id,
                        Message.soft_deleted.is_(False),
                    )
                ).all()
            )
        else:
            msgs = [
                m
                for m in self.messages
                if m.thread_id == thread_id and not getattr(m, "soft_deleted", False)
            ]

        for m in msgs:
            if isinstance(m.payload, dict) and m.payload.get("type") == "inactivity_close":
                sent_at_raw = m.payload.get("sent_at")
                sent_at = (
                    datetime.fromisoformat(sent_at_raw)
                    if isinstance(sent_at_raw, str)
                    else getattr(m, "created_at", None)
                )
                if sent_at is not None:
                    if sent_at.tzinfo is None:
                        sent_at = sent_at.replace(tzinfo=UTC)
                    if sent_at > cutoff:
                        return True
        return False

    def _persist_csat_message(
        self,
        *,
        tenant_id: int,
        thread_id: int,
        channel: str,
        content: str,
        payload: dict[str, Any],
        now: datetime,
    ) -> Message:
        if self.session is not None:
            msg = Message(
                tenant_id=tenant_id,
                thread_id=thread_id,
                channel=channel,
                external_message_id=None,
                role=MessageRole.ASSISTANT,
                content=content,
                payload=payload,
                attachments=None,
                soft_deleted=False,
                created_at=now,
                updated_at=now,
            )
            self.session.add(msg)
            self.session.flush()
            return msg

        msg = Message(
            message_id=self._next_message_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
            channel=channel,
            external_message_id=None,
            role=MessageRole.ASSISTANT,
            content=content,
            payload=payload,
            attachments=None,
            soft_deleted=False,
            created_at=now,
            updated_at=now,
        )
        self._next_message_id += 1
        self.messages.append(msg)
        return msg
