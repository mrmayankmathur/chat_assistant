from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import IntEnum
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from app.models.enums import (
    ConversationIntent,
    ConversationStatus,
    CrmProvider,
    CrmTicketKind,
    CrmTicketStatus,
    EscalationMode,
    FailureReason,
    MessageRole,
    PolicyType,
    ShipmentStatus,
    Tonality,
)


class IntEnumSmallInt(TypeDecorator[IntEnum]):
    """Stores a Python IntEnum as a PostgreSQL SMALLINT."""

    impl = SmallInteger
    cache_ok = True

    def __init__(self, enum_cls: type[IntEnum], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enum_cls = enum_cls

    def process_bind_param(self, value: IntEnum | int | None, dialect: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    def process_result_value(self, value: int | None, dialect: Any) -> IntEnum | None:
        if value is None:
            return None
        return self.enum_cls(value)


class LowercaseEmailType(TypeDecorator[str]):
    """Ensures email strings are always stripped and lowercased on bind."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect: Any) -> str | None:
        if value is None:
            return None
        return value.strip().lower()

    def process_result_value(self, value: str | None, dialect: Any) -> str | None:
        if value is None:
            return None
        return value.lower()


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class TenantSoftDeleteMixin:
    soft_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------


class Tenant(TimestampMixin, Base):
    __tablename__ = "tenant"

    tenant_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    timezone: Mapped[str] = mapped_column(String, nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )


class AgentConfig(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "agent_config"
    __table_args__ = (
        Index(
            "uq_agent_config_active_tenant",
            "tenant_id",
            unique=True,
            postgresql_where=text("is_active AND NOT soft_deleted"),
        ),
    )

    agent_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tenant.tenant_id"),
        nullable=False,
    )
    brand_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    tonality: Mapped[Tonality] = mapped_column(
        IntEnumSmallInt(Tonality),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class ChannelGuideline(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "channel_guideline"

    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tenant.tenant_id"),
        primary_key=True,
    )
    channel: Mapped[str] = mapped_column(String, primary_key=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)


class Policy(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "policy"
    __table_args__ = (
        UniqueConstraint("tenant_id", "policy_type", name="uq_policy_tenant_type"),
    )

    policy_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tenant.tenant_id"),
        nullable=False,
    )
    policy_type: Mapped[PolicyType] = mapped_column(
        IntEnumSmallInt(PolicyType),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    windows: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )


class BusinessHours(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "business_hours"

    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tenant.tenant_id"),
        primary_key=True,
    )
    weekly: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    overrides: Mapped[list[dict[str, Any]] | dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )


class CrmIntegration(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "crm_integration"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tenant.tenant_id"),
        nullable=False,
    )
    provider: Mapped[CrmProvider] = mapped_column(
        IntEnumSmallInt(CrmProvider),
        nullable=False,
    )
    secret_ref: Mapped[str] = mapped_column(String, nullable=False)
    supports_ticket: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    supports_live_handover: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    webhook_uuid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
        default=uuid.uuid4,
    )


# ---------------------------------------------------------------------------
# Identity + messages
# ---------------------------------------------------------------------------


class Thread(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "thread"
    __table_args__ = (
        CheckConstraint(
            "phone_hash IS NOT NULL OR email IS NOT NULL",
            name="ck_thread_phone_hash_or_email",
        ),
        Index(
            "ix_thread_tenant_phone_hash",
            "tenant_id",
            "phone_hash",
            postgresql_where=text("NOT soft_deleted"),
        ),
        Index(
            "ix_thread_tenant_email",
            "tenant_id",
            "email",
            postgresql_where=text("NOT soft_deleted"),
        ),
    )

    thread_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tenant.tenant_id"),
        nullable=False,
    )
    phone_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(LowercaseEmailType(), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    migrated_from_thread_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("thread.thread_id"),
        nullable=True,
    )


class Message(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "message"
    __table_args__ = (
        Index(
            "uq_message_tenant_channel_external_id",
            "tenant_id",
            "channel",
            "external_message_id",
            unique=True,
            postgresql_where=text("external_message_id IS NOT NULL"),
        ),
        Index("ix_message_thread_message_id", "thread_id", "message_id"),
    )

    message_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tenant.tenant_id"),
        nullable=False,
    )
    thread_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("thread.thread_id"),
        nullable=False,
    )
    channel: Mapped[str] = mapped_column(String, nullable=False)
    external_message_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    role: Mapped[MessageRole] = mapped_column(
        IntEnumSmallInt(MessageRole),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    attachments: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSONB, nullable=True)


# ---------------------------------------------------------------------------
# Conversations (filled asynchronously by an evaluator, never in request path)
# ---------------------------------------------------------------------------


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversation"

    conversation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    thread_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("thread.thread_id"),
        nullable=False,
    )
    status: Mapped[ConversationStatus] = mapped_column(
        IntEnumSmallInt(ConversationStatus),
        nullable=False,
    )
    intents: Mapped[list[ConversationIntent | int]] = mapped_column(
        ARRAY(SmallInteger),
        nullable=False,
        default=list,
        server_default=text("'{}'::smallint[]"),
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    follow_up_of: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("conversation.conversation_id"),
        nullable=True,
    )
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ConversationMessage(TimestampMixin, Base):
    __tablename__ = "conversation_message"

    message_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("message.message_id"),
        primary_key=True,
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversation.conversation_id"),
        nullable=False,
    )


class Escalation(TimestampMixin, Base):
    __tablename__ = "escalation"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversation.conversation_id"),
        nullable=False,
    )
    subject: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[EscalationMode] = mapped_column(
        IntEnumSmallInt(EscalationMode),
        nullable=False,
    )


class CrmTicket(TimestampMixin, Base):
    __tablename__ = "crm_ticket"
    __table_args__ = (
        Index(
            "uq_crm_ticket_open_thread_kind",
            "thread_id",
            "kind",
            unique=True,
            postgresql_where=text("status = 1"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    thread_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("thread.thread_id"),
        nullable=False,
    )
    crm_integration_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("crm_integration.id"),
        nullable=False,
    )
    kind: Mapped[CrmTicketKind] = mapped_column(
        IntEnumSmallInt(CrmTicketKind),
        nullable=False,
    )
    status: Mapped[CrmTicketStatus] = mapped_column(
        IntEnumSmallInt(CrmTicketStatus),
        nullable=False,
    )
    external_ticket_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    external_conversation_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class CrmTicketEscalation(TimestampMixin, Base):
    __tablename__ = "crm_ticket_escalation"

    crm_ticket_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("crm_ticket.id"),
        primary_key=True,
    )
    escalation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("escalation.id"),
        primary_key=True,
    )


# ---------------------------------------------------------------------------
# Orders (a read model, seeded for now)
# ---------------------------------------------------------------------------


class CustomerOrder(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "customer_order"
    __table_args__ = (
        UniqueConstraint("tenant_id", "order_ref", name="uq_customer_order_tenant_ref"),
        Index(
            "ix_customer_order_tenant_phone_placed_desc",
            "tenant_id",
            "phone_hash",
            text("placed_at DESC"),
        ),
        Index(
            "ix_customer_order_tenant_email_placed_desc",
            "tenant_id",
            "email",
            text("placed_at DESC"),
        ),
    )

    order_pk: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tenant.tenant_id"),
        nullable=False,
    )
    order_ref: Mapped[str] = mapped_column(String, nullable=False)
    phone_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(LowercaseEmailType(), nullable=True)
    payment_mode: Mapped[str] = mapped_column(String, nullable=False)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    items: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )


class Shipment(TimestampMixin, Base):
    __tablename__ = "shipment"
    __table_args__ = (
        UniqueConstraint("carrier_code", "awb", name="uq_shipment_carrier_awb"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    order_pk: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("customer_order.order_pk"),
        nullable=False,
    )
    awb: Mapped[str] = mapped_column(String, nullable=False)
    carrier_code: Mapped[str] = mapped_column(String, nullable=False)
    is_reverse: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    status: Mapped[ShipmentStatus] = mapped_column(
        IntEnumSmallInt(ShipmentStatus),
        nullable=False,
    )
    edd: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    picked_up_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    delivered_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ShipmentEvent(TimestampMixin, Base):
    __tablename__ = "shipment_event"

    shipment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("shipment.id"),
        primary_key=True,
    )
    event_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
    )
    status: Mapped[ShipmentStatus] = mapped_column(
        IntEnumSmallInt(ShipmentStatus),
        primary_key=True,
    )
    failure_reason: Mapped[FailureReason] = mapped_column(
        IntEnumSmallInt(FailureReason),
        nullable=False,
        default=FailureReason.NONE,
        server_default=text("0"),
    )
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    raw_remark: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class StatusCopy(TimestampMixin, TenantSoftDeleteMixin, Base):
    __tablename__ = "status_copy"

    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        default=0,
        server_default=text("0"),
    )
    status: Mapped[ShipmentStatus] = mapped_column(
        IntEnumSmallInt(ShipmentStatus),
        primary_key=True,
    )
    failure_reason: Mapped[FailureReason] = mapped_column(
        IntEnumSmallInt(FailureReason),
        primary_key=True,
        default=FailureReason.NONE,
        server_default=text("0"),
    )
    customer_text: Mapped[str] = mapped_column(Text, nullable=False)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class LlmCall(TimestampMixin, Base):
    __tablename__ = "llm_call"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("message.message_id"),
        nullable=False,
    )
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    stop_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    cache_write_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
