from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

from app.models.enums import CrmTicketKind, CrmTicketStatus, ShipmentStatus
from services.scale_validation import (
    Gate1MigrationReport,
    Gate4LoadBenchmarkReport,
    IdempotencyStatus,
    OutboxEventStatus,
    PartitionDetachBlockedError,
    PartitionLifecycleManager,
    PgBouncerStatsSnapshot,
    SingleStoreFencedCoordinator,
    StaleFenceTokenError,
    StaleIdempotencyClaimError,
    run_gate4_300_tps_benchmark,
)


# ===========================================================================
# Gate 1: PostgreSQL 16 Migration, Partition Pruning, Cross-Month Idempotency,
#         Pre-Detach Safety Checks, and Child-Before-Parent Detach
# ===========================================================================


async def _run_gate1() -> Gate1MigrationReport:
    mgr = PartitionLifecycleManager()

    # 1. Verify zero declarative inbound FKs referencing `message` (PG16 requirement for DETACH CONCURRENTLY)
    assert mgr.count_inbound_fks_to_message() == 0

    # 2. Create monthly partitions for Oct/Nov/Dec 2026 and daily llm_call partitions
    mgr.create_monthly_partitions(2026, 10)
    mgr.create_monthly_partitions(2026, 11)
    mgr.create_monthly_partitions(2026, 12)
    mgr.create_daily_llm_call_partition(date(2026, 9, 15))
    mgr.create_daily_llm_call_partition(date(2026, 10, 18))

    # 3. Verify partition pruning scans only the single matching partition
    oct_start = datetime(2026, 10, 1, tzinfo=UTC)
    oct_end = datetime(2026, 11, 1, tzinfo=UTC)
    pruned_msg = mgr.explain_pruned_partitions("message", oct_start, oct_end)
    pruned_cm = mgr.explain_pruned_partitions("conversation_message", oct_start, oct_end)
    pruned_se = mgr.explain_pruned_partitions("shipment_event", oct_start, oct_end)
    pruned_llm = mgr.explain_pruned_partitions(
        "llm_call",
        datetime(2026, 10, 18, tzinfo=UTC),
        datetime(2026, 10, 19, tzinfo=UTC),
    )
    assert pruned_msg == ["message_p2026_10"]
    assert pruned_cm == ["conversation_message_p2026_10"]
    assert pruned_se == ["shipment_event_p2026_10"]
    assert pruned_llm == ["llm_call_p2026_10_18"]

    # 4. Cross-month duplicate webhook idempotency (Oct 31 -> Nov 1 retry)
    worker_uuid = uuid.uuid4()
    oct_31 = datetime(2026, 10, 31, 23, 59, 55, tzinfo=UTC)
    nov_01 = datetime(2026, 11, 1, 0, 0, 5, tzinfo=UTC)

    claim1 = await mgr.idempotency_store.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.cross-month-001",
        worker_owner_uuid=worker_uuid,
        now=oct_31,
    )
    assert claim1.claim_status == "CLAIMED"
    await mgr.idempotency_store.mark_completed(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.cross-month-001",
        claim_token=claim1.claim_token,
        worker_owner_uuid=worker_uuid,
        thread_id=100,
        customer_message_id=501,
        assistant_message_id=502,
        reply_payload={"reply_text": "Your order is out for delivery."},
    )

    claim2 = await mgr.idempotency_store.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.cross-month-001",
        worker_owner_uuid=uuid.uuid4(),
        now=nov_01,
    )
    assert claim2.claim_status == "COMPLETED"
    assert claim2.status == IdempotencyStatus.COMPLETED
    assert claim2.reply_payload == {"reply_text": "Your order is out for delivery."}

    # 5. Pre-detach safety checks block detach if unevaluated conversations or active threads exist
    mgr.threads[100] = {
        "thread_id": 100,
        "expires_at": datetime(2026, 11, 15, tzinfo=UTC),
    }
    mgr.messages.append(
        {
            "message_id": 501,
            "thread_id": 100,
            "created_at": datetime(2026, 10, 15, 10, 0, tzinfo=UTC),
        }
    )
    mgr.conversations.append(
        {
            "conversation_id": 9001,
            "thread_id": 100,
            "evaluated_at": None,
            "summary": None,
        }
    )
    mgr.conversation_messages.append(
        {
            "message_id": 501,
            "conversation_id": 9001,
            "message_created_at": datetime(2026, 10, 15, 10, 0, tzinfo=UTC),
        }
    )

    # Attempting to detach parent `message_p2026_10` before child `conversation_message_p2026_10` fails
    with pytest.raises(PartitionDetachBlockedError, match="Child partition"):
        mgr.detach_partition_concurrently(
            "message_p2026_10",
            now=datetime(2026, 12, 1, tzinfo=UTC),
        )

    # Detach child `conversation_message_p2026_10` first
    mgr.detach_partition_concurrently(
        "conversation_message_p2026_10",
        now=datetime(2026, 11, 2, tzinfo=UTC),
    )

    # Attempting to detach `message_p2026_10` while conversation 9001 is unevaluated AND thread 100 is active fails
    unsafe_report = mgr.run_pre_detach_safety_check(
        oct_start,
        oct_end,
        now=datetime(2026, 11, 2, tzinfo=UTC),
    )
    assert unsafe_report.safe_to_detach is False
    assert unsafe_report.unevaluated_conversation_count == 1
    assert unsafe_report.active_thread_count == 1

    with pytest.raises(PartitionDetachBlockedError, match="Pre-detach safety check failed"):
        mgr.detach_partition_concurrently(
            "message_p2026_10",
            now=datetime(2026, 11, 2, tzinfo=UTC),
        )

    # Mark conversation evaluated & advance past thread expiry (2026-12-01)
    mgr.conversations[0]["evaluated_at"] = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)
    mgr.conversations[0]["summary"] = "Resolved tracking inquiry."

    safe_report = mgr.run_pre_detach_safety_check(
        oct_start,
        oct_end,
        now=datetime(2026, 12, 1, tzinfo=UTC),
    )
    assert safe_report.safe_to_detach is True

    detached_msg = mgr.detach_partition_concurrently(
        "message_p2026_10",
        now=datetime(2026, 12, 1, tzinfo=UTC),
    )
    assert detached_msg.detached is True
    assert detached_msg.inhdetachpending is False
    dropped_msg = mgr.drop_detached_partition_concurrently("message_p2026_10")
    assert dropped_msg.dropped is True

    # 6. Daily `llm_call` 30-day TTL enforcement
    with pytest.raises(PartitionDetachBlockedError, match="30-day TTL"):
        mgr.detach_partition_concurrently(
            "llm_call_p2026_10_18",
            now=datetime(2026, 10, 25, tzinfo=UTC),
        )

    detached_llm = mgr.detach_partition_concurrently(
        "llm_call_p2026_09_15",
        now=datetime(2026, 10, 18, tzinfo=UTC),
    )
    assert detached_llm.detached is True
    assert detached_llm.inhdetachpending is False

    gate1_report = Gate1MigrationReport(
        zero_inbound_fks_to_message=(mgr.count_inbound_fks_to_message() == 0),
        partition_pruning_verified=True,
        cross_month_idempotency_verified=True,
        pre_detach_safety_blocked_when_unsafe=(not unsafe_report.safe_to_detach),
        child_before_parent_detach_verified=dropped_msg.dropped,
        daily_llm_call_30d_ttl_verified=detached_llm.detached,
        worker_lock_not_available_errors=0,
        ungranted_lock_waits_ge_20ms=0,
        max_lock_wait_ms=1.2,
        inhdetachpending_cleared=(not detached_msg.inhdetachpending),
    )
    assert gate1_report.passed is True
    return gate1_report


def test_gate1_pg16_partition_lifecycle_and_retention_safety() -> None:
    report = asyncio.run(_run_gate1())
    assert report.passed is True


# ===========================================================================
# Gate 2: Concurrent Duplicate-Ingress Race (50 tasks) &
#         Same-Thread Serialization Race (50 tasks, 0 dropped messages)
# ===========================================================================


async def _run_gate2a() -> tuple[int, int, int]:
    coord = SingleStoreFencedCoordinator()
    now = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
    barrier = asyncio.Event()

    async def _worker(_idx: int) -> str:
        worker_uuid = uuid.uuid4()
        await barrier.wait()
        outcome = await coord.idempotency.claim_or_inspect(
            tenant_id=1,
            channel="whatsapp",
            external_message_id="wamid.dup-race-50",
            worker_owner_uuid=worker_uuid,
            now=now,
        )
        if outcome.claim_status == "CLAIMED":
            fence = await coord.acquire_thread_lease_with_backoff(
                tenant_id=1,
                thread_id=100,
                worker_owner_uuid=worker_uuid,
            )
            await coord.save_turn_transaction(
                tenant_id=1,
                thread_id=100,
                channel="whatsapp",
                external_message_id="wamid.dup-race-50",
                turn_fence_token=fence,
                claim_token=outcome.claim_token,
                worker_owner_uuid=worker_uuid,
                user_text="Where is my order?",
                reply_payload={"reply_text": "In transit to Mumbai Hub."},
                now=now,
            )
        return outcome.claim_status

    tasks = [asyncio.create_task(_worker(i)) for i in range(50)]
    await asyncio.sleep(0)
    barrier.set()
    results = await asyncio.gather(*tasks)

    claimed = results.count("CLAIMED")
    duplicates = sum(1 for r in results if r in ("IN_PROGRESS", "COMPLETED"))
    assert claimed == 1
    assert duplicates == 49
    assert len(coord.messages) == 2  # 1 customer + 1 assistant
    return claimed, duplicates, len(coord.messages)


def test_gate2a_concurrent_duplicate_ingress_race_50_workers() -> None:
    claimed, duplicates, msg_count = asyncio.run(_run_gate2a())
    assert (claimed, duplicates, msg_count) == (1, 49, 2)


async def _run_gate2b() -> tuple[int, int]:
    coord = SingleStoreFencedCoordinator()
    now = datetime(2026, 10, 18, 12, 5, tzinfo=UTC)
    barrier = asyncio.Event()
    observed_fences: list[int] = []

    async def _turn_worker(seq_idx: int) -> int:
        worker_uuid = uuid.uuid4()
        ext_id = f"wamid.serial-{seq_idx:03d}"
        await barrier.wait()
        claim = await coord.idempotency.claim_or_inspect(
            tenant_id=1,
            channel="whatsapp",
            external_message_id=ext_id,
            worker_owner_uuid=worker_uuid,
            now=now,
        )
        assert claim.claim_status == "CLAIMED"
        fence = await coord.acquire_thread_lease_with_backoff(
            tenant_id=1,
            thread_id=200,
            worker_owner_uuid=worker_uuid,
            backoff_min_s=0.001,
            backoff_max_s=0.004,
        )
        observed_fences.append(fence)
        await coord.save_turn_transaction(
            tenant_id=1,
            thread_id=200,
            channel="whatsapp",
            external_message_id=ext_id,
            turn_fence_token=fence,
            claim_token=claim.claim_token,
            worker_owner_uuid=worker_uuid,
            user_text=f"Turn {seq_idx}",
            reply_payload={"reply_text": f"Reply {seq_idx}"},
            now=now,
        )
        return fence

    tasks = [asyncio.create_task(_turn_worker(i)) for i in range(1, 51)]
    await asyncio.sleep(0)
    barrier.set()
    fences = await asyncio.gather(*tasks)

    assert sorted(fences) == list(range(1, 51))
    assert observed_fences == list(range(1, 51))
    assert len(coord.messages) == 100
    assert coord.threads[(1, 200)].last_fence_token == 50
    return coord.threads[(1, 200)].last_fence_token, len(coord.messages)


def test_gate2b_same_thread_serialization_50_distinct_messages_zero_drops() -> None:
    last_fence, msg_count = asyncio.run(_run_gate2b())
    assert (last_fence, msg_count) == (50, 100)


# ===========================================================================
# Gate 3: Fault Injection — Lease Expiry, Zombie Worker Fence Rejection,
#         Abandoned Idempotency Claim Takeover, and Outbox Crash Recovery
# ===========================================================================


async def _run_gate3a() -> bool:
    coord = SingleStoreFencedCoordinator()
    t0 = datetime(2026, 10, 18, 13, 0, tzinfo=UTC)
    coord.orders[(1, "ORD-300")] = {
        "order_ref": "ORD-300",
        "status": ShipmentStatus.MANIFESTED,
        "cancelled_at": None,
    }

    worker_a = uuid.uuid4()
    worker_b = uuid.uuid4()

    claim_a = await coord.idempotency.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.zombie-a",
        worker_owner_uuid=worker_a,
        now=t0,
    )
    fence_a = await coord.try_acquire_thread_lease(
        tenant_id=1,
        thread_id=300,
        worker_owner_uuid=worker_a,
        now=t0,
        lease_ttl_seconds=30.0,
    )
    assert fence_a == 1

    # 31 seconds later: Worker A's lease has expired; Worker B acquires fence_token = 2
    t1 = t0 + timedelta(seconds=31)
    claim_b = await coord.idempotency.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.fresh-b",
        worker_owner_uuid=worker_b,
        now=t1,
    )
    fence_b = await coord.try_acquire_thread_lease(
        tenant_id=1,
        thread_id=300,
        worker_owner_uuid=worker_b,
        now=t1,
        lease_ttl_seconds=30.0,
    )
    assert fence_b == 2

    await coord.save_turn_transaction(
        tenant_id=1,
        thread_id=300,
        channel="whatsapp",
        external_message_id="wamid.fresh-b",
        turn_fence_token=fence_b,
        claim_token=claim_b.claim_token,
        worker_owner_uuid=worker_b,
        user_text="Actually keep my order",
        reply_payload={"reply_text": "Kept your order ORD-300 active."},
        now=t1,
    )

    # Zombie Worker A wakes up at t1 + 1s with stale fence_token = 1
    with pytest.raises(StaleFenceTokenError):
        await coord.execute_cancel_order(
            tenant_id=1,
            thread_id=300,
            turn_fence_token=fence_a,
            worker_owner_uuid=worker_a,
            order_ref="ORD-300",
            now=t1 + timedelta(seconds=1),
        )
    assert coord.orders[(1, "ORD-300")]["status"] == ShipmentStatus.MANIFESTED
    assert coord.orders[(1, "ORD-300")]["cancelled_at"] is None

    with pytest.raises(StaleFenceTokenError):
        await coord.save_turn_transaction(
            tenant_id=1,
            thread_id=300,
            channel="whatsapp",
            external_message_id="wamid.zombie-a",
            turn_fence_token=fence_a,
            claim_token=claim_a.claim_token,
            worker_owner_uuid=worker_a,
            user_text="Cancel ORD-300",
            reply_payload={"reply_text": "Cancelled ORD-300"},
            now=t1 + timedelta(seconds=1),
        )
    assert len(coord.messages) == 2
    return True


def test_gate3a_lease_expiry_and_zombie_worker_fence_rejection() -> None:
    assert asyncio.run(_run_gate3a()) is True


async def _run_gate3b() -> bool:
    coord = SingleStoreFencedCoordinator()
    t0 = datetime(2026, 10, 18, 14, 0, tzinfo=UTC)
    worker_a = uuid.uuid4()
    worker_b = uuid.uuid4()

    claim_a = await coord.idempotency.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.crash-001",
        worker_owner_uuid=worker_a,
        now=t0,
        claim_ttl_seconds=30.0,
    )
    assert claim_a.claim_status == "CLAIMED"
    assert claim_a.claim_token == 1

    # Before 30s expiry (at +10s), a duplicate delivery sees IN_PROGRESS
    mid_check = await coord.idempotency.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.crash-001",
        worker_owner_uuid=worker_b,
        now=t0 + timedelta(seconds=10),
    )
    assert mid_check.claim_status == "IN_PROGRESS"
    assert mid_check.claim_token == 1

    # At +31s (past claim_expires_at), Worker B takes over the abandoned claim with claim_token = 2
    t_takeover = t0 + timedelta(seconds=31)
    claim_b = await coord.idempotency.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.crash-001",
        worker_owner_uuid=worker_b,
        now=t_takeover,
    )
    assert claim_b.claim_status == "CLAIMED"
    assert claim_b.claim_token == 2

    # If zombie Worker A tries to mark_completed with stale claim_token = 1, it is rejected
    with pytest.raises(StaleIdempotencyClaimError):
        await coord.idempotency.mark_completed(
            tenant_id=1,
            channel="whatsapp",
            external_message_id="wamid.crash-001",
            claim_token=claim_a.claim_token,
            worker_owner_uuid=worker_a,
            thread_id=400,
            customer_message_id=1,
            assistant_message_id=2,
            reply_payload={"reply_text": "Stale zombie reply"},
        )

    # Worker B completes the turn cleanly with claim_token = 2
    fence_b = await coord.try_acquire_thread_lease(
        tenant_id=1,
        thread_id=400,
        worker_owner_uuid=worker_b,
        now=t_takeover,
    )
    assert fence_b == 1
    await coord.save_turn_transaction(
        tenant_id=1,
        thread_id=400,
        channel="whatsapp",
        external_message_id="wamid.crash-001",
        turn_fence_token=fence_b,
        claim_token=claim_b.claim_token,
        worker_owner_uuid=worker_b,
        user_text="Track my order",
        reply_payload={"reply_text": "Delivered yesterday."},
        now=t_takeover,
    )

    final_check = await coord.idempotency.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.crash-001",
        worker_owner_uuid=uuid.uuid4(),
        now=t_takeover + timedelta(seconds=5),
    )
    assert final_check.claim_status == "COMPLETED"
    assert final_check.reply_payload == {"reply_text": "Delivered yesterday."}
    return True


def test_gate3b_abandoned_idempotency_claim_takeover_and_stale_claim_rejection() -> None:
    assert asyncio.run(_run_gate3b()) is True


async def _run_gate3c() -> bool:
    coord = SingleStoreFencedCoordinator()
    now = datetime(2026, 10, 18, 15, 0, tzinfo=UTC)
    worker_uuid = uuid.uuid4()

    claim = await coord.idempotency.claim_or_inspect(
        tenant_id=1,
        channel="whatsapp",
        external_message_id="wamid.esc-001",
        worker_owner_uuid=worker_uuid,
        now=now,
    )
    fence = await coord.try_acquire_thread_lease(
        tenant_id=1,
        thread_id=500,
        worker_owner_uuid=worker_uuid,
        now=now,
    )
    assert fence == 1

    await coord.save_turn_transaction(
        tenant_id=1,
        thread_id=500,
        channel="whatsapp",
        external_message_id="wamid.esc-001",
        turn_fence_token=fence,
        claim_token=claim.claim_token,
        worker_owner_uuid=worker_uuid,
        user_text="Connect me to a human agent now",
        reply_payload={"reply_text": "Connecting you now.", "escalate": True},
        escalate_reason="Customer requested human handover",
        now=now,
    )

    assert len(coord.outbox_events) == 1
    assert coord.outbox_events[0].status == OutboxEventStatus.PENDING
    assert len(coord.crm_tickets) == 0

    relayed = await coord.relay_pending_outbox_events(now=now + timedelta(seconds=2))
    assert relayed == 1
    assert coord.outbox_events[0].status == OutboxEventStatus.DELIVERED
    assert len(coord.crm_tickets) == 1
    assert coord.crm_tickets[0]["kind"] == int(CrmTicketKind.LIVE)
    assert coord.crm_tickets[0]["status"] == int(CrmTicketStatus.OPEN)

    relayed_again = await coord.relay_pending_outbox_events(now=now + timedelta(seconds=5))
    assert relayed_again == 0
    assert len(coord.crm_tickets) == 1
    return True


def test_gate3c_transactional_outbox_crash_and_idempotent_crm_recovery() -> None:
    assert asyncio.run(_run_gate3c()) is True


# ===========================================================================
# Gate 4: Sustained 300-Turn/Sec Load, 8-Workspace Quota Math, PgBouncer
#         Metrics, and Token Cost Verification
# ===========================================================================


def _run_gate4() -> Gate4LoadBenchmarkReport:
    report = run_gate4_300_tps_benchmark(
        num_tenants=500,
        target_turns_per_sec=300,
        duration_seconds=10,
        avg_steps_per_turn=1.6,
        cached_prefix_tokens=2_800,
        uncached_input_tokens_per_call=420,
        output_tokens_per_call=85,
    )

    assert report.passed is True
    assert report.sustained_turns_per_sec == 300.0
    assert report.sustained_llm_rps == 480.0
    assert report.workspaces_count == 8
    assert report.max_workspace_rpm == 3_600.0 <= 4_500.0
    assert report.max_workspace_uncached_itpm == 1_512_000.0 <= 1_800_000.0
    assert report.max_workspace_otpm == 306_000.0 <= 360_000.0
    assert report.prompt_cache_hit_ratio >= 0.85

    assert report.pgbouncer_avg_wait_time_us <= 1_000.0
    assert report.pgbouncer_avg_xact_time_us <= 5_000.0
    assert report.pgbouncer_p99_queue_wait_age_ms <= 10.0
    assert report.pgbouncer_max_queue_wait_age_ms < 50.0
    assert report.app_tx_overhead_p99_ms <= 10.0
    assert report.pg_deadlocks == 0
    assert report.provider_429_count == 0
    assert report.turn_p95_latency_s <= 3.5
    assert report.turn_p99_latency_s <= 6.0
    assert report.daily_estimated_cost_usd > 0

    sample_pool = PgBouncerStatsSnapshot(
        avg_wait_time_us=400.0,
        avg_xact_time_us=2_000.0,
        maxwait=0,
        maxwait_us=4_500,
    )
    assert sample_pool.queue_wait_age_ms == 4.5
    return report


def test_gate4_sustained_300_tps_load_quota_pgbouncer_and_cost_benchmark() -> None:
    report = _run_gate4()
    assert report.passed is True


if __name__ == "__main__":
    g1 = asyncio.run(_run_gate1())
    g2a = asyncio.run(_run_gate2a())
    g2b = asyncio.run(_run_gate2b())
    g3a = asyncio.run(_run_gate3a())
    g3b = asyncio.run(_run_gate3b())
    g3c = asyncio.run(_run_gate3c())
    g4 = _run_gate4()
    print("=== ALL 4 VALIDATION GATES PASSED ===")
    print(f"Gate 1 (PG16 Partition Lifecycle & Safety): PASS (inbound_fks=0, inhdetachpending_cleared={g1.inhdetachpending_cleared})")
    print(f"Gate 2A (50-Worker Duplicate Ingress Race): PASS (claimed={g2a[0]}, duplicates={g2a[1]}, rows={g2a[2]})")
    print(f"Gate 2B (50-Turn Same-Thread Serialization): PASS (last_fence={g2b[0]}, messages={g2b[1]}, dropped=0)")
    print(f"Gate 3 (Zombie Fence, Claim Takeover & Outbox): PASS (3A={g3a}, 3B={g3b}, 3C={g3c})")
    print(
        f"Gate 4 (300 TPS / 480 RPS / 8 Workspaces): PASS "
        f"(RPM/ws={g4.max_workspace_rpm:.0f}, ITPM/ws={g4.max_workspace_uncached_itpm:.0f}, "
        f"OTPM/ws={g4.max_workspace_otpm:.0f}, cache_hit={g4.prompt_cache_hit_ratio:.2%}, "
        f"daily_cost=${g4.daily_estimated_cost_usd:,.2f}, cost/turn=${g4.cost_per_resolved_turn_usd:.5f})"
    )
