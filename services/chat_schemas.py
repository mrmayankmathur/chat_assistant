from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.orders import OrderCard

FALLBACK_REPLY_TEXT = "couldn't fetch right now, want me to connect you to support?"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: int | None = None
    channel: str
    external_message_id: str
    phone: str | None = None
    email: str | None = None
    text: str

    @model_validator(mode="after")
    def _validate_identifier(self) -> ChatRequest:
        has_phone = bool(self.phone and self.phone.strip())
        has_email = bool(self.email and self.email.strip())
        if not has_phone and not has_email:
            raise ValueError("At least one of phone or email must be provided")
        return self


class ReplyOrderCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_ref: str
    title: str
    placed_at: str


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply_text: str
    order_cards: list[ReplyOrderCard] = Field(default_factory=list)
    ask_order_selection: bool = False
    show_return_link: bool = False
    escalate: bool = False


REPLY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "reply_text",
        "order_cards",
        "ask_order_selection",
        "show_return_link",
        "escalate",
    ],
    "properties": {
        "reply_text": {"type": "string"},
        "order_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["order_ref", "title", "placed_at"],
                "properties": {
                    "order_ref": {"type": "string"},
                    "title": {"type": "string"},
                    "placed_at": {"type": "string"},
                },
            },
        },
        "ask_order_selection": {"type": "boolean"},
        "show_return_link": {"type": "boolean"},
        "escalate": {"type": "boolean"},
    },
}


def format_order_card_title(items: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> str:
    if not items:
        return "Order"
    first = items[0]
    name = str(first.get("name") or first.get("title") or first.get("sku") or "Item")
    if len(items) == 1:
        return name
    return f"{name} +{len(items) - 1} more"


def to_reply_order_card(card: OrderCard) -> ReplyOrderCard:
    placed_iso = (
        card.placed_at.isoformat()
        if isinstance(card.placed_at, datetime)
        else str(card.placed_at)
    )
    return ReplyOrderCard(
        order_ref=card.order_ref,
        title=format_order_card_title(card.items),
        placed_at=placed_iso,
    )


def build_fallback_reply() -> Reply:
    return Reply(
        reply_text=FALLBACK_REPLY_TEXT,
        order_cards=[],
        ask_order_selection=False,
        show_return_link=False,
        escalate=False,
    )


def build_live_handover_reply() -> Reply:
    return Reply(
        reply_text="",
        order_cards=[],
        ask_order_selection=False,
        show_return_link=False,
        escalate=True,
    )
