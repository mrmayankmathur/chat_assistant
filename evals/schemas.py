from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GLOBAL_DENYLIST: tuple[str, ...] = (
    # Carrier names
    "delhivery",
    "bluedart",
    "blue dart",
    "ecom express",
    "xpressbees",
    "dtdc",
    "shadowfax",
    "ekart",
    "fedex",
    "dhl",
    # Forbidden filler phrases
    "let me check",
    "please wait",
)

JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reasoning", "failing_turn"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["PASS", "FAIL"],
        },
        "reasoning": {
            "type": "string",
        },
        "failing_turn": {
            "type": ["integer", "null"],
        },
    },
}


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"]
    reasoning: str
    failing_turn: int | None = None


class ToolCallBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: int = 0
    max: int | None = None


class DeterministicAssertionsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: dict[str, ToolCallBounds] = Field(default_factory=dict)
    tool_results: dict[str, str] = Field(default_factory=dict)
    reply_flags: dict[str, bool] = Field(default_factory=dict)
    reply_not_contains: list[str] = Field(default_factory=list)


class TurnAssertionsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: dict[str, ToolCallBounds] = Field(default_factory=dict)
    tool_results: dict[str, str] = Field(default_factory=dict)
    reply_flags: dict[str, bool] = Field(default_factory=dict)
    reply_not_contains: list[str] = Field(default_factory=list)


class AssertionsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deterministic: DeterministicAssertionsSpec = Field(
        default_factory=DeterministicAssertionsSpec
    )
    judge: str


class CustomerIdentityFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phone: str | None = None
    email: str | None = None
    channel: str = "whatsapp"


class TenantFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: int = 1
    name: str = "Lumina Botanics"
    timezone: str = "Asia/Kolkata"
    cancel_enabled: bool = True
    edd_breach_days: int = 2
    lookback_days: int = 30
    business_hours: dict[str, Any] | None = None
    policies: list[dict[str, Any]] = Field(default_factory=list)


class OrderFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_pk: int
    order_ref: str
    owner: Literal["customer", "foreign"] = "customer"
    phone: str | None = None
    email: str | None = None
    payment_mode: str = "PREPAID"
    placed_at: str
    cancelled_at: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)


class ShipmentFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    order_pk: int
    awb: str
    carrier_code: str = "DELHIVERY"
    is_reverse: bool = False
    status: str
    edd: str | None = None


class ShipmentEventFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shipment_id: int
    event_at: str
    status: str
    failure_reason: str = "NONE"
    location: str = "Mumbai Hub"
    raw_remark: str | None = None


class ScenarioFixtures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    now: str = "2026-10-18T10:00:00Z"
    tenant: TenantFixture = Field(default_factory=TenantFixture)
    orders: list[OrderFixture] = Field(default_factory=list)
    shipments: list[ShipmentFixture] = Field(default_factory=list)
    events: list[ShipmentEventFixture] = Field(default_factory=list)


class ScriptedTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user: str
    assertions: TurnAssertionsSpec | None = None


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    customer: CustomerIdentityFixture
    fixtures: ScenarioFixtures
    turns: list[ScriptedTurn]
    assertions: AssertionsSpec


class TurnExecutionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_index: int
    user_text: str
    reply_text: str
    reply_payload: dict[str, Any]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_ms: int = 0


class ScenarioEvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    passed: bool
    deterministic_passed: bool
    deterministic_failures: list[str] = Field(default_factory=list)
    judge_result: JudgeResult
    tool_calls: list[str] = Field(default_factory=list)
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_ms: int = 0
    turns: list[TurnExecutionTrace] = Field(default_factory=list)
    evaluated_at: datetime
