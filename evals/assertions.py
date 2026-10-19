from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from evals.schemas import (
    GLOBAL_DENYLIST,
    JUDGE_JSON_SCHEMA,
    DeterministicAssertionsSpec,
    JudgeResult,
    ScenarioSpec,
    ToolCallBounds,
    TurnAssertionsSpec,
    TurnExecutionTrace,
)

ORDER_REF_TOKEN_RE = re.compile(r"\b(?:(?:LUM|ORD)-[A-Z0-9-]+|B\d+)\b", re.IGNORECASE)


def _check_tool_results(
    expected_results: Mapping[str, str],
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
) -> list[str]:
    failures: list[str] = []
    for tool_name, expected_substr in expected_results.items():
        matching_calls = [
            tc for tc in tool_calls if str(tc.get("name", "")) == tool_name
        ]
        if not matching_calls:
            failures.append(
                f"{scope_label}: tool '{tool_name}' was not called, expected result {expected_substr!r}"
            )
            continue
        matched = any(
            expected_substr in str(tc.get("result", ""))
            for tc in matching_calls
        )
        if not matched:
            actual_results = [tc.get("result") for tc in matching_calls]
            failures.append(
                f"{scope_label}: tool '{tool_name}' returned {actual_results!r}, expected {expected_substr!r}"
            )
    return failures


def _check_tool_call_bounds(
    bounds_map: Mapping[str, ToolCallBounds],
    actual_counts: Mapping[str, int],
    *,
    scope_label: str,
) -> list[str]:
    failures: list[str] = []
    for tool_name, bounds in bounds_map.items():
        count = actual_counts.get(tool_name, 0)
        if count < bounds.min:
            failures.append(
                f"{scope_label}: tool '{tool_name}' called {count} time(s), expected min={bounds.min}"
            )
        if bounds.max is not None and count > bounds.max:
            failures.append(
                f"{scope_label}: tool '{tool_name}' called {count} time(s), expected max={bounds.max}"
            )
    return failures


def _check_reply_flags(
    expected_flags: Mapping[str, bool],
    reply_payload: Mapping[str, Any],
    *,
    scope_label: str,
) -> list[str]:
    failures: list[str] = []
    for flag_name, expected_val in expected_flags.items():
        actual_val = reply_payload.get(flag_name)
        if bool(actual_val) is not bool(expected_val):
            failures.append(
                f"{scope_label}: reply flag '{flag_name}' was {actual_val!r}, expected {expected_val!r}"
            )
    return failures


def _check_reply_not_contains(
    forbidden_phrases: Sequence[str],
    reply_text: str,
    *,
    scope_label: str,
) -> list[str]:
    failures: list[str] = []
    lower_reply = reply_text.lower()

    # Combine scenario-specific forbidden phrases with the global denylist
    combined: list[str] = []
    seen: set[str] = set()
    for phrase in [*GLOBAL_DENYLIST, *forbidden_phrases]:
        norm = phrase.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            combined.append(phrase.strip())

    for phrase in combined:
        if phrase.lower() in lower_reply:
            failures.append(
                f"{scope_label}: reply_text contains forbidden phrase {phrase!r}"
            )
    return failures


def _check_no_unowned_order_refs(
    scenario: ScenarioSpec,
    turns: Sequence[TurnExecutionTrace],
) -> list[str]:
    failures: list[str] = []
    customer_order_refs = {
        o.order_ref.upper()
        for o in scenario.fixtures.orders
        if o.owner == "customer"
    }
    foreign_order_refs = {
        o.order_ref.upper()
        for o in scenario.fixtures.orders
        if o.owner != "customer"
    }
    customer_order_pks = {
        o.order_pk
        for o in scenario.fixtures.orders
        if o.owner == "customer"
    }
    foreign_awbs = {
        s.awb.upper()
        for s in scenario.fixtures.shipments
        if s.order_pk not in customer_order_pks
    }

    for turn in turns:
        label = f"turn {turn.turn_index}"
        cards = turn.reply_payload.get("order_cards") or []
        for card in cards:
            ref = str(card.get("order_ref", "")).strip()
            if ref.upper() not in customer_order_refs:
                failures.append(
                    f"{label}: order_card contains order_ref {ref!r} not in customer's fixture"
                )

        for match in ORDER_REF_TOKEN_RE.findall(turn.reply_text):
            if match.upper() not in customer_order_refs:
                failures.append(
                    f"{label}: reply_text mentions order_ref {match!r} not in customer's fixture"
                )

        upper_text = turn.reply_text.upper()
        for f_ref in sorted(foreign_order_refs):
            if f_ref in upper_text:
                failures.append(
                    f"{label}: reply_text leaks foreign order_ref {f_ref!r}"
                )
        for f_awb in sorted(foreign_awbs):
            if f_awb in upper_text:
                failures.append(
                    f"{label}: reply_text leaks foreign awb {f_awb!r}"
                )

    return failures


def evaluate_deterministic_assertions(
    scenario: ScenarioSpec,
    turns: Sequence[TurnExecutionTrace],
) -> tuple[bool, list[str]]:
    """Run all deterministic assertions for `scenario` against `turns`."""
    failures: list[str] = []
    if not turns:
        return (False, ["no turns were executed"])

    det: DeterministicAssertionsSpec = scenario.assertions.deterministic

    # 1. Per-turn checks (global denylist on every turn + optional per-turn assertions)
    total_tool_counts: dict[str, int] = {}
    all_tool_calls: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        turn_label = f"turn {turn.turn_index}"
        turn_counts: dict[str, int] = {}
        for tc in turn.tool_calls:
            all_tool_calls.append(tc)
            name = str(tc.get("name", ""))
            if name:
                turn_counts[name] = turn_counts.get(name, 0) + 1
                total_tool_counts[name] = total_tool_counts.get(name, 0) + 1

        scripted_turn = scenario.turns[idx] if idx < len(scenario.turns) else None
        turn_spec: TurnAssertionsSpec | None = (
            scripted_turn.assertions if scripted_turn is not None else None
        )

        if turn_spec is not None:
            failures.extend(
                _check_tool_call_bounds(
                    turn_spec.tool_calls,
                    turn_counts,
                    scope_label=turn_label,
                )
            )
            failures.extend(
                _check_tool_results(
                    turn_spec.tool_results,
                    turn.tool_calls,
                    scope_label=turn_label,
                )
            )
            failures.extend(
                _check_reply_flags(
                    turn_spec.reply_flags,
                    turn.reply_payload,
                    scope_label=turn_label,
                )
            )
            failures.extend(
                _check_reply_not_contains(
                    [*det.reply_not_contains, *turn_spec.reply_not_contains],
                    turn.reply_text,
                    scope_label=turn_label,
                )
            )
        else:
            failures.extend(
                _check_reply_not_contains(
                    det.reply_not_contains,
                    turn.reply_text,
                    scope_label=turn_label,
                )
            )

    # 2. Scenario-level tool_calls bounds and tool_results checks
    failures.extend(
        _check_tool_call_bounds(
            det.tool_calls,
            total_tool_counts,
            scope_label="scenario",
        )
    )
    failures.extend(
        _check_tool_results(
            det.tool_results,
            all_tool_calls,
            scope_label="scenario",
        )
    )

    # 3. Scenario-level reply_flags on final reply
    final_turn = turns[-1]
    failures.extend(
        _check_reply_flags(
            det.reply_flags,
            final_turn.reply_payload,
            scope_label="final_turn",
        )
    )

    # 4. No order_ref that isn't in the customer's fixture
    failures.extend(_check_no_unowned_order_refs(scenario, turns))

    # Deduplicate while preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for f in failures:
        if f not in seen:
            seen.add(f)
            deduped.append(f)

    return (len(deduped) == 0, deduped)


async def evaluate_judge_rubric(
    *,
    scenario: ScenarioSpec,
    turns: Sequence[TurnExecutionTrace],
    judge_client: Any,
    model: str = "gpt-6-luna",
) -> JudgeResult:
    """Evaluate the scenario conversation against `scenario.assertions.judge` rubric via structured output."""
    customer_orders = [
        {
            "order_ref": o.order_ref,
            "payment_mode": o.payment_mode,
            "placed_at": o.placed_at,
            "cancelled_at": o.cancelled_at,
            "items": o.items,
        }
        for o in scenario.fixtures.orders
        if o.owner == "customer"
    ]
    transcript_payload = [
        {
            "turn": t.turn_index,
            "user": t.user_text,
            "tool_calls": t.tool_calls,
            "assistant_reply": t.reply_payload,
        }
        for t in turns
    ]

    judge_input = {
        "scenario": scenario.name,
        "rubric": scenario.assertions.judge,
        "customer_owned_orders": customer_orders,
        "turns": transcript_payload,
    }

    response = await judge_client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": (
                    "You are a strict WISMO support evaluation judge. Evaluate whether the "
                    "assistant's turns satisfy the rubric based only on the provided fixture "
                    "and transcript. Return JSON matching the schema with verdict PASS or FAIL, "
                    "reasoning, and failing_turn (1-indexed turn number if FAIL, or null if PASS)."
                ),
            }
        ],
        messages=[
            {
                "type": "user",
                "content": json.dumps(judge_input, indent=2),
            }
        ],
        output_config={
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": JUDGE_JSON_SCHEMA,
            },
        },
    )

    raw_content = getattr(response, "content", []) or []
    text_parts: list[str] = []
    for block in raw_content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        elif getattr(block, "type", None) == "text":
            text_parts.append(str(getattr(block, "text", "")))

    raw_json = "\n".join(text_parts).strip()
    return JudgeResult.model_validate_json(raw_json)
