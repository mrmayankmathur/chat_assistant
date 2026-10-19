from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.core import hash_phone_e164, normalize_phone_e164
from app.models import (
    AgentConfig,
    BusinessHours,
    CustomerOrder,
    FailureReason,
    Policy,
    PolicyType,
    Shipment,
    ShipmentEvent,
    ShipmentStatus,
    Tenant,
    Tonality,
)
from app.settings import Settings
from evals.agent_simulator import LiveGpt6LunaClient, LocalEvalAnthropicClient
from evals.assertions import evaluate_deterministic_assertions, evaluate_judge_rubric
from evals.schemas import (
    ScenarioEvalResult,
    ScenarioSpec,
    TurnExecutionTrace,
)
from evals.yaml_loader import load_yaml_file
from repositories import (
    InMemoryChatRepository,
    InMemoryOrderRepository,
    InMemoryThreadRepository,
    clear_tenant_config_cache,
)
from services import (
    ChatRequest,
    ChatService,
    IdentityService,
    OrderService,
    TrackingService,
)

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"

SCENARIO_ORDER: tuple[str, ...] = (
    "ambiguous_order_cards",
    "single_in_transit",
    "failed_delivery_reason",
    "edd_breached_apology",
    "foreign_awb_no_leak",
    "cancel_requires_confirmation",
    "cancel_after_pickup_refused",
    "human_request_outside_hours",
    "injection_in_courier_remark",
    "injection_in_customer_message",
    "cod_refund_not_applicable",
)

EVAL_SETTINGS = Settings(
    phone_hmac_secret="eval-secret-key-123",
    default_phone_region="IN",
    llm_model="gpt-6-luna",
)


def _parse_iso_datetime(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value.strip())


def load_scenario(name_or_path: str | Path) -> ScenarioSpec:
    path = Path(name_or_path)
    if not path.exists():
        candidate = SCENARIOS_DIR / f"{name_or_path}.yaml"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Scenario file not found for {name_or_path!r}")
    raw = load_yaml_file(path)
    return ScenarioSpec.model_validate(raw)


def load_all_scenarios(
    scenario_filter: str | None = None,
    scenarios_dir: Path = SCENARIOS_DIR,
) -> list[ScenarioSpec]:
    if scenario_filter:
        names = [s.strip() for s in scenario_filter.split(",") if s.strip()]
        return [load_scenario(scenarios_dir / f"{n}.yaml") if not Path(n).exists() else load_scenario(n) for n in names]

    loaded_by_name: dict[str, ScenarioSpec] = {}
    for yaml_path in sorted(scenarios_dir.glob("*.yaml")):
        spec = load_scenario(yaml_path)
        loaded_by_name[spec.name] = spec

    ordered: list[ScenarioSpec] = []
    for name in SCENARIO_ORDER:
        if name in loaded_by_name:
            ordered.append(loaded_by_name.pop(name))
    for remaining_name in sorted(loaded_by_name.keys()):
        ordered.append(loaded_by_name[remaining_name])
    return ordered


def _build_default_business_hours(tenant_id: int, bh_spec: dict[str, Any] | None) -> BusinessHours:
    if bh_spec is not None:
        return BusinessHours(
            tenant_id=tenant_id,
            weekly=bh_spec.get("weekly", {}),
            overrides=bh_spec.get("overrides", []),
            soft_deleted=False,
        )
    return BusinessHours(
        tenant_id=tenant_id,
        weekly={
            "mon": [{"open": "09:00", "close": "21:00"}],
            "tue": [{"open": "09:00", "close": "21:00"}],
            "wed": [{"open": "09:00", "close": "21:00"}],
            "thu": [{"open": "09:00", "close": "21:00"}],
            "fri": [{"open": "09:00", "close": "21:00"}],
            "sat": [{"open": "09:00", "close": "21:00"}],
            "sun": [{"open": "09:00", "close": "21:00"}],
        },
        overrides=[],
        soft_deleted=False,
    )


def _create_default_anthropic_client() -> Any:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    use_live = os.getenv("WISMO_LIVE_LLM", "").strip().lower() in ("1", "true", "yes")
    in_pytest = bool(os.getenv("PYTEST_CURRENT_TEST"))
    if openai_key and use_live and not in_pytest:
        return LiveGpt6LunaClient(
            api_key=openai_key,
            base_url=EVAL_SETTINGS.openai_base_url,
        )
    return LocalEvalAnthropicClient()


async def run_scenario(
    scenario: ScenarioSpec,
    *,
    anthropic_client: Any | None = None,
    judge_client: Any | None = None,
) -> ScenarioEvalResult:
    clear_tenant_config_cache()

    client = anthropic_client if anthropic_client is not None else _create_default_anthropic_client()
    j_client = judge_client if judge_client is not None else client

    now = _parse_iso_datetime(scenario.fixtures.now)
    t_fix = scenario.fixtures.tenant
    tenant_id = t_fix.tenant_id

    tenant = Tenant(
        tenant_id=tenant_id,
        name=t_fix.name,
        timezone=t_fix.timezone,
        features={"cancel_enabled": t_fix.cancel_enabled},
    )
    agent_cfg = AgentConfig(
        agent_id=1,
        tenant_id=tenant_id,
        brand_prompt=f"You are the warm, clear support assistant for {t_fix.name}.",
        tonality=Tonality.FRIENDLY,
        is_active=True,
        version=1,
        soft_deleted=False,
    )
    policies = [
        Policy(
            policy_id=1,
            tenant_id=tenant_id,
            policy_type=PolicyType.CANCELLATION,
            text="Orders can only be cancelled before carrier pickup.",
            windows={},
            soft_deleted=False,
        )
    ]
    bh = _build_default_business_hours(tenant_id, t_fix.business_hours)

    cust_phone_e164 = (
        normalize_phone_e164(scenario.customer.phone, default_region=EVAL_SETTINGS.default_phone_region)
        if scenario.customer.phone
        else None
    )
    cust_phone_hash = (
        hash_phone_e164(cust_phone_e164, secret=EVAL_SETTINGS.phone_hmac_secret)
        if cust_phone_e164
        else None
    )

    orders: list[CustomerOrder] = []
    for o_fix in scenario.fixtures.orders:
        if o_fix.owner == "customer":
            phone_hash = cust_phone_hash
            email = (o_fix.email or scenario.customer.email or "").lower() or None
        else:
            f_phone = o_fix.phone or "+919999900000"
            f_e164 = normalize_phone_e164(f_phone, default_region=EVAL_SETTINGS.default_phone_region)
            phone_hash = hash_phone_e164(f_e164, secret=EVAL_SETTINGS.phone_hmac_secret)
            email = (o_fix.email or "foreign@example.com").lower()

        orders.append(
            CustomerOrder(
                order_pk=o_fix.order_pk,
                tenant_id=tenant_id,
                order_ref=o_fix.order_ref,
                phone_hash=phone_hash,
                email=email,
                payment_mode=o_fix.payment_mode,
                placed_at=_parse_iso_datetime(o_fix.placed_at),
                cancelled_at=_parse_iso_datetime(o_fix.cancelled_at) if o_fix.cancelled_at else None,
                items=list(o_fix.items),
                soft_deleted=False,
            )
        )

    shipments: list[Shipment] = []
    for s_fix in scenario.fixtures.shipments:
        shipments.append(
            Shipment(
                id=s_fix.id,
                order_pk=s_fix.order_pk,
                awb=s_fix.awb,
                carrier_code=s_fix.carrier_code,
                is_reverse=s_fix.is_reverse,
                status=ShipmentStatus[s_fix.status.upper()],
                edd=_parse_iso_date(s_fix.edd),
            )
        )

    events: list[ShipmentEvent] = []
    for e_fix in scenario.fixtures.events:
        events.append(
            ShipmentEvent(
                shipment_id=e_fix.shipment_id,
                event_at=_parse_iso_datetime(e_fix.event_at),
                status=ShipmentStatus[e_fix.status.upper()],
                failure_reason=FailureReason[e_fix.failure_reason.upper()],
                location=e_fix.location,
                raw_remark=e_fix.raw_remark,
            )
        )

    thread_repo = InMemoryThreadRepository()
    order_repo = InMemoryOrderRepository(orders=orders, shipments=shipments)
    chat_repo = InMemoryChatRepository(
        tenant=tenant,
        agent_config=agent_cfg,
        channel_guidelines={"whatsapp": "Keep messages concise, friendly, and easy to scan."},
        policies=policies,
        business_hours=bh,
        shipments=shipments,
        events=events,
    )

    identity_service = IdentityService(repo_or_session=thread_repo, settings=EVAL_SETTINGS)
    order_service = OrderService(repo_or_session=order_repo)
    tracking_service = TrackingService()

    chat_service = ChatService(
        identity_service=identity_service,
        order_service=order_service,
        tracking_service=tracking_service,
        chat_repo=chat_repo,
        anthropic_client=client,
        settings=EVAL_SETTINGS,
    )

    turn_traces: list[TurnExecutionTrace] = []
    all_tool_names: list[str] = []
    tool_counts: dict[str, int] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read_tokens = 0
    total_latency_ms = 0

    for idx, scripted_turn in enumerate(scenario.turns, start=1):
        prev_llm_count = len(chat_repo.llm_calls)
        req = ChatRequest(
            channel=scenario.customer.channel,
            external_message_id=f"{scenario.name}-turn-{idx}",
            phone=scenario.customer.phone,
            email=scenario.customer.email,
            text=scripted_turn.user,
        )

        t0 = time.perf_counter()
        reply = await chat_service.handle_chat(
            req,
            principal_tenant_id=tenant_id,
            now=now,
        )
        wall_ms = max(1, int((time.perf_counter() - t0) * 1000))

        new_llm_calls = chat_repo.llm_calls[prev_llm_count:]
        turn_tools: list[dict[str, Any]] = []
        turn_in = 0
        turn_out = 0
        turn_cache = 0
        turn_lat = 0

        for call in new_llm_calls:
            turn_in += int(call.input_tokens or 0)
            turn_out += int(call.output_tokens or 0)
            turn_cache += int(call.cache_read_tokens or 0)
            turn_lat += int(call.latency_ms or 0)
            for tc in call.tool_calls or []:
                turn_tools.append(tc)
                t_name = str(tc.get("name", ""))
                if t_name:
                    all_tool_names.append(t_name)
                    tool_counts[t_name] = tool_counts.get(t_name, 0) + 1

        effective_latency = max(turn_lat, wall_ms)
        total_input_tokens += turn_in
        total_output_tokens += turn_out
        total_cache_read_tokens += turn_cache
        total_latency_ms += effective_latency

        turn_traces.append(
            TurnExecutionTrace(
                turn_index=idx,
                user_text=scripted_turn.user,
                reply_text=reply.reply_text,
                reply_payload=reply.model_dump(mode="json"),
                tool_calls=turn_tools,
                input_tokens=turn_in,
                output_tokens=turn_out,
                cache_read_tokens=turn_cache,
                latency_ms=effective_latency,
            )
        )

    det_passed, det_failures = evaluate_deterministic_assertions(scenario, turn_traces)
    judge_res = await evaluate_judge_rubric(
        scenario=scenario,
        turns=turn_traces,
        judge_client=j_client,
        model=EVAL_SETTINGS.llm_model,
    )

    overall_pass = det_passed and (judge_res.verdict == "PASS")

    return ScenarioEvalResult(
        scenario=scenario.name,
        passed=overall_pass,
        deterministic_passed=det_passed,
        deterministic_failures=det_failures,
        judge_result=judge_res,
        tool_calls=all_tool_names,
        tool_call_counts=tool_counts,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cache_read_tokens=total_cache_read_tokens,
        latency_ms=total_latency_ms,
        turns=turn_traces,
        evaluated_at=datetime.now(UTC),
    )


async def run_suite(
    *,
    scenario_filter: str | None = None,
    scenarios_dir: Path = SCENARIOS_DIR,
    anthropic_client: Any | None = None,
    judge_client: Any | None = None,
) -> list[ScenarioEvalResult]:
    scenarios = load_all_scenarios(scenario_filter=scenario_filter, scenarios_dir=scenarios_dir)
    results: list[ScenarioEvalResult] = []
    for sc in scenarios:
        res = await run_scenario(
            sc,
            anthropic_client=anthropic_client,
            judge_client=judge_client,
        )
        results.append(res)
    return results


def _format_tool_calls_cell(res: ScenarioEvalResult) -> str:
    if not res.tool_call_counts:
        return "(none)"
    return ", ".join(f"{name}×{count}" for name, count in res.tool_call_counts.items())


def format_results_table(results: list[ScenarioEvalResult]) -> str:
    """Format eval results as a table with columns:
    `scenario | pass | tool calls | input/output/cache-read tokens | latency`
    """
    headers = (
        "scenario",
        "pass",
        "tool calls",
        "input/output/cache-read tokens",
        "latency",
    )
    rows: list[tuple[str, str, str, str, str]] = []
    for r in results:
        pass_cell = "PASS" if r.passed else "FAIL"
        tools_cell = _format_tool_calls_cell(r)
        tokens_cell = f"{r.input_tokens}/{r.output_tokens}/{r.cache_read_tokens}"
        latency_cell = f"{r.latency_ms}ms"
        rows.append((r.scenario, pass_cell, tools_cell, tokens_cell, latency_cell))

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]

    def _fmt_row(cells: tuple[str, ...]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)
    lines = [_fmt_row(headers), sep]
    for row in rows:
        lines.append(_fmt_row(row))

    for r in results:
        if not r.passed:
            reasons = list(r.deterministic_failures)
            if r.judge_result.verdict == "FAIL":
                reasons.append(
                    f"judge FAIL (turn={r.judge_result.failing_turn}): {r.judge_result.reasoning}"
                )
            lines.append(f"  [FAILURE] {r.scenario}: {'; '.join(reasons)}")

    return "\n".join(lines)
