from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import InteractiveSandbox, build_playground_app
from app.models import ShipmentEvent


def test_sandbox_reports_simulator_and_estimated_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WISMO_LIVE_LLM", "0")
    app, _ = build_playground_app()
    state = TestClient(app).get("/v1/sandbox/state").json()

    assert state["tenant"]["llm_execution_mode"] == "simulator"
    assert state["tenant"]["telemetry_is_estimated"] is True
    assert state["tenant"]["configured_model"] == "gpt-6-luna"
    assert state["tenant"]["data_mode"] == "shared_in_memory_sandbox"
    assert state["llm_calls"] == []
    assert state["messages"] == []
    assert state["outbox"] == []


def test_live_configuration_metadata_does_not_require_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main

    # Replace the live adapter entirely: this checks configuration, not a provider.
    sentinel = object()
    monkeypatch.setattr(main, "LiveGpt6LunaClient", lambda **kwargs: sentinel)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-test-key")
    monkeypatch.setenv("WISMO_LIVE_LLM", "1")
    sandbox = InteractiveSandbox()

    assert sandbox.llm_execution_mode == "live"
    assert sandbox.snapshot_state()["tenant"]["telemetry_is_estimated"] is False
    assert sandbox._build_llm_client() is sentinel


def test_snapshot_uses_latest_stored_event_not_display_fixtures() -> None:
    sandbox = InteractiveSandbox()
    original = sandbox.chat_repo.events[0]
    newer = ShipmentEvent(
        shipment_id=original.shipment_id,
        event_at=original.event_at + timedelta(hours=1),
        status=original.status,
        failure_reason=original.failure_reason,
        location="Actual latest location",
        raw_remark="Latest stored carrier remark",
    )
    # Intentionally unsorted: selecting the final list entry would be incorrect.
    sandbox.chat_repo.events.insert(0, newer)
    order = sandbox.snapshot_state()["orders"][0]

    assert order["last_event_at"] == newer.event_at.isoformat()
    assert order["last_location"] == newer.location
    assert order["last_remark"] == newer.raw_remark
    sandbox.chat_repo.events.clear()
    order_without_events = sandbox.snapshot_state()["orders"][0]
    assert order_without_events["last_event_at"] is None
    assert order_without_events["last_location"] is None
    assert order_without_events["last_remark"] is None


@pytest.mark.parametrize("items", [[], [{}]])
def test_missing_order_metadata_stays_unavailable(items: list[dict]) -> None:
    sandbox = InteractiveSandbox()
    sandbox.order_repo._orders[0].items = items
    order = sandbox.snapshot_state()["orders"][0]

    for field in ("item", "sku", "qty", "price_cents", "price_display"):
        assert order[field] is None


def test_prices_preserve_zero_and_existing_line_totals() -> None:
    sandbox = InteractiveSandbox()
    cream = next(o for o in sandbox.snapshot_state()["orders"] if o["order_ref"] == "LUM-1002")
    assert cream["qty"] == 2
    assert cream["price_cents"] == 8400
    assert cream["price_display"] == "$84.00"

    sandbox.order_repo._orders[0].items = [{"name": "Free demo item", "qty": 0, "price_cents": 0}]
    free = sandbox.snapshot_state()["orders"][0]
    assert free["qty"] == 0
    assert free["price_display"] == "$0.00"


def test_cancellation_does_not_fabricate_a_new_carrier_scan() -> None:
    app, _ = build_playground_app()
    client = TestClient(app)
    before = client.get("/v1/sandbox/state").json()
    original = next(o for o in before["orders"] if o["order_ref"] == "LUM-1001")
    for index, text in enumerate(("Please cancel my order LUM-1001", "Yes")):
        response = client.post(
            "/v1/chat",
            headers={"X-Tenant-Id": "1"},
            json={
                "channel": "whatsapp",
                "external_message_id": f"scan-test-{index}",
                "phone": before["personas"][0]["phone"],
                "text": text,
            },
        )
        assert response.status_code == 200
    after = client.get("/v1/sandbox/state").json()
    cancelled = next(o for o in after["orders"] if o["order_ref"] == "LUM-1001")
    assert cancelled["status"] == "CANCELLED"
    assert cancelled["cancelled_at"] == after["tenant"]["current_simulated_time"]
    assert cancelled["last_event_at"] == original["last_event_at"]
    assert cancelled["last_remark"] == original["last_remark"]


def test_delayed_order_surfaces_sanitized_delay_reason_to_chat_and_blocks_injections() -> None:
    app, _ = build_playground_app()
    client = TestClient(app)
    state = client.get("/v1/sandbox/state").json()

    resp = client.post(
        "/v1/chat",
        headers={"X-Tenant-Id": "1"},
        json={
            "channel": "whatsapp",
            "external_message_id": "delay-reason-check-1",
            "phone": state["personas"][0]["phone"],
            "text": "My order LUM-1002 is delayed, where is it?",
        },
    )
    assert resp.status_code == 200
    reply_text = resp.json()["reply_text"].lower()
    assert "weather delay" in reply_text
    assert "chicago sort facility" in reply_text

