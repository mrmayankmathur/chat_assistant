"""Initial WISMO schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-10-18 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Control plane
    # ------------------------------------------------------------------
    op.create_table(
        "tenant",
        sa.Column("tenant_id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("timezone", sa.String(), nullable=False),
        sa.Column(
            "features",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "agent_config",
        sa.Column("agent_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            sa.ForeignKey("tenant.tenant_id"),
            nullable=False,
        ),
        sa.Column("brand_prompt", sa.Text(), nullable=False),
        sa.Column("tonality", sa.SmallInteger(), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "uq_agent_config_active_tenant",
        "agent_config",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text("is_active AND NOT soft_deleted"),
    )

    op.create_table(
        "channel_guideline",
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            sa.ForeignKey("tenant.tenant_id"),
            primary_key=True,
        ),
        sa.Column("channel", sa.String(), primary_key=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "policy",
        sa.Column("policy_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            sa.ForeignKey("tenant.tenant_id"),
            nullable=False,
        ),
        sa.Column("policy_type", sa.SmallInteger(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "windows",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("tenant_id", "policy_type", name="uq_policy_tenant_type"),
    )

    op.create_table(
        "business_hours",
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            sa.ForeignKey("tenant.tenant_id"),
            primary_key=True,
        ),
        sa.Column(
            "weekly",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "overrides",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "crm_integration",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            sa.ForeignKey("tenant.tenant_id"),
            nullable=False,
        ),
        sa.Column("provider", sa.SmallInteger(), nullable=False),
        sa.Column("secret_ref", sa.String(), nullable=False),
        sa.Column(
            "supports_ticket",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "supports_live_handover",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("webhook_uuid", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ------------------------------------------------------------------
    # Identity + messages
    # ------------------------------------------------------------------
    op.create_table(
        "thread",
        sa.Column("thread_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            sa.ForeignKey("tenant.tenant_id"),
            nullable=False,
        ),
        sa.Column("phone_hash", sa.String(length=64), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "migrated_from_thread_id",
            sa.BigInteger(),
            sa.ForeignKey("thread.thread_id"),
            nullable=True,
        ),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "phone_hash IS NOT NULL OR email IS NOT NULL",
            name="ck_thread_phone_hash_or_email",
        ),
    )
    op.create_index(
        "ix_thread_tenant_phone_hash",
        "thread",
        ["tenant_id", "phone_hash"],
        unique=False,
        postgresql_where=sa.text("NOT soft_deleted"),
    )
    op.create_index(
        "ix_thread_tenant_email",
        "thread",
        ["tenant_id", "email"],
        unique=False,
        postgresql_where=sa.text("NOT soft_deleted"),
    )

    op.create_table(
        "message",
        sa.Column("message_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            sa.ForeignKey("tenant.tenant_id"),
            nullable=False,
        ),
        sa.Column(
            "thread_id",
            sa.BigInteger(),
            sa.ForeignKey("thread.thread_id"),
            nullable=False,
        ),
        sa.Column("channel", sa.String(), nullable=False),
        sa.Column("external_message_id", sa.String(), nullable=True),
        sa.Column("role", sa.SmallInteger(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("attachments", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "uq_message_tenant_channel_external_id",
        "message",
        ["tenant_id", "channel", "external_message_id"],
        unique=True,
        postgresql_where=sa.text("external_message_id IS NOT NULL"),
    )
    op.create_index(
        "ix_message_thread_message_id",
        "message",
        ["thread_id", "message_id"],
        unique=False,
    )

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------
    op.create_table(
        "conversation",
        sa.Column("conversation_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "thread_id",
            sa.BigInteger(),
            sa.ForeignKey("thread.thread_id"),
            nullable=False,
        ),
        sa.Column("status", sa.SmallInteger(), nullable=False),
        sa.Column(
            "intents",
            postgresql.ARRAY(sa.SmallInteger()),
            nullable=False,
            server_default=sa.text("'{}'::smallint[]"),
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column(
            "follow_up_of",
            sa.BigInteger(),
            sa.ForeignKey("conversation.conversation_id"),
            nullable=True,
        ),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "conversation_message",
        sa.Column(
            "message_id",
            sa.BigInteger(),
            sa.ForeignKey("message.message_id"),
            primary_key=True,
        ),
        sa.Column(
            "conversation_id",
            sa.BigInteger(),
            sa.ForeignKey("conversation.conversation_id"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "escalation",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.BigInteger(),
            sa.ForeignKey("conversation.conversation_id"),
            nullable=False,
        ),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("mode", sa.SmallInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "crm_ticket",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "thread_id",
            sa.BigInteger(),
            sa.ForeignKey("thread.thread_id"),
            nullable=False,
        ),
        sa.Column(
            "crm_integration_id",
            sa.BigInteger(),
            sa.ForeignKey("crm_integration.id"),
            nullable=False,
        ),
        sa.Column("kind", sa.SmallInteger(), nullable=False),
        sa.Column("status", sa.SmallInteger(), nullable=False),
        sa.Column("external_ticket_id", sa.String(), nullable=True),
        sa.Column("external_conversation_id", sa.String(), nullable=True),
        sa.Column(
            "opened_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "uq_crm_ticket_open_thread_kind",
        "crm_ticket",
        ["thread_id", "kind"],
        unique=True,
        postgresql_where=sa.text("status = 1"),
    )

    op.create_table(
        "crm_ticket_escalation",
        sa.Column(
            "crm_ticket_id",
            sa.BigInteger(),
            sa.ForeignKey("crm_ticket.id"),
            primary_key=True,
        ),
        sa.Column(
            "escalation_id",
            sa.BigInteger(),
            sa.ForeignKey("escalation.id"),
            primary_key=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    op.create_table(
        "customer_order",
        sa.Column("order_pk", sa.BigInteger(), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            sa.ForeignKey("tenant.tenant_id"),
            nullable=False,
        ),
        sa.Column("order_ref", sa.String(), nullable=False),
        sa.Column("phone_hash", sa.String(length=64), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("payment_mode", sa.String(), nullable=False),
        sa.Column("placed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "items",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("tenant_id", "order_ref", name="uq_customer_order_tenant_ref"),
    )
    op.create_index(
        "ix_customer_order_tenant_phone_placed_desc",
        "customer_order",
        ["tenant_id", "phone_hash", sa.text("placed_at DESC")],
        unique=False,
    )
    op.create_index(
        "ix_customer_order_tenant_email_placed_desc",
        "customer_order",
        ["tenant_id", "email", sa.text("placed_at DESC")],
        unique=False,
    )

    op.create_table(
        "shipment",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "order_pk",
            sa.BigInteger(),
            sa.ForeignKey("customer_order.order_pk"),
            nullable=False,
        ),
        sa.Column("awb", sa.String(), nullable=False),
        sa.Column("carrier_code", sa.String(), nullable=False),
        sa.Column(
            "is_reverse",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("status", sa.SmallInteger(), nullable=False),
        sa.Column("edd", sa.Date(), nullable=True),
        sa.Column("picked_up_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("carrier_code", "awb", name="uq_shipment_carrier_awb"),
    )

    op.create_table(
        "shipment_event",
        sa.Column(
            "shipment_id",
            sa.BigInteger(),
            sa.ForeignKey("shipment.id"),
            primary_key=True,
        ),
        sa.Column("event_at", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("status", sa.SmallInteger(), primary_key=True),
        sa.Column(
            "failure_reason",
            sa.SmallInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("location", sa.String(), nullable=True),
        sa.Column("raw_remark", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "status_copy",
        sa.Column(
            "tenant_id",
            sa.BigInteger(),
            primary_key=True,
            server_default=sa.text("0"),
        ),
        sa.Column("status", sa.SmallInteger(), primary_key=True),
        sa.Column(
            "failure_reason",
            sa.SmallInteger(),
            primary_key=True,
            server_default=sa.text("0"),
        ),
        sa.Column("customer_text", sa.Text(), nullable=False),
        sa.Column(
            "soft_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    op.create_table(
        "llm_call",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "message_id",
            sa.BigInteger(),
            sa.ForeignKey("message.message_id"),
            nullable=False,
        ),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("stop_reason", sa.String(), nullable=True),
        sa.Column(
            "input_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "output_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "cache_read_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "cache_write_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column(
            "tool_calls",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("llm_call")
    op.drop_table("status_copy")
    op.drop_table("shipment_event")
    op.drop_table("shipment")
    op.drop_index("ix_customer_order_tenant_email_placed_desc", table_name="customer_order")
    op.drop_index("ix_customer_order_tenant_phone_placed_desc", table_name="customer_order")
    op.drop_table("customer_order")
    op.drop_table("crm_ticket_escalation")
    op.drop_index("uq_crm_ticket_open_thread_kind", table_name="crm_ticket")
    op.drop_table("crm_ticket")
    op.drop_table("escalation")
    op.drop_table("conversation_message")
    op.drop_table("conversation")
    op.drop_index("ix_message_thread_message_id", table_name="message")
    op.drop_index("uq_message_tenant_channel_external_id", table_name="message")
    op.drop_table("message")
    op.drop_index("ix_thread_tenant_email", table_name="thread")
    op.drop_index("ix_thread_tenant_phone_hash", table_name="thread")
    op.drop_table("thread")
    op.drop_table("crm_integration")
    op.drop_table("business_hours")
    op.drop_table("policy")
    op.drop_table("channel_guideline")
    op.drop_index("uq_agent_config_active_tenant", table_name="agent_config")
    op.drop_table("agent_config")
    op.drop_table("tenant")
