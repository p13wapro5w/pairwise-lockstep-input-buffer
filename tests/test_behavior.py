"""Behavioural tests for the lockstep input buffer.

Each test targets one of the failure modes from the bug report rather than
merely exercising a function signature.
"""

import random

import pytest

import gameplay.lockstep_input_buffer as m


def inp(player, frame, seq, *actions):
    return m.FrameInput(player, frame, seq, actions)


# ---------------------------------------------------------------------------
# 1. Inputs queue by server frame, not by local frame
# ---------------------------------------------------------------------------


def test_queue_orders_by_server_frame_regardless_of_arrival_order():
    q = m.FrameQueue(start_frame=0, players=("a", "b"))
    # Jitter delivers frame 2 before frame 1, and one player's frame 0 late.
    assert q.push(inp("a", 2, 3, "jump"))
    assert q.push(inp("b", 2, 3, "move"))
    assert q.push(inp("a", 1, 2, "move"))
    assert q.push(inp("b", 1, 2, "move"))
    assert q.push(inp("a", 0, 1, "idle"))
    assert q.push(inp("b", 0, 1, "idle"))

    assert q.queued_frames == (0, 1, 2)
    # Frame 0 is consumable first even though it arrived last.
    assert q.ready(0)
    consumed = q.pop_frame(0)
    assert [i.server_frame for i in consumed] == [0, 0]
    assert q.floor_frame == 1


def test_queue_never_partially_consumes_a_frame():
    q = m.FrameQueue(start_frame=0, players=("a", "b"))
    q.push(inp("a", 0, 1, "move"))
    assert not q.ready(0)
    assert q.pop_frame(0) is None
    assert q.ready(0) is False
    assert q.backlog == 1  # the one input is still queued, not dropped

    q.push(inp("b", 0, 1, "move"))
    consumed = q.pop_frame(0)
    assert {i.player for i in consumed} == {"a", "b"}
    assert consumed[0].player == "a"  # required-player order, deterministic


def test_queue_rejects_duplicate_player_frame():
    q = m.FrameQueue(start_frame=0, capacity=8)
    assert q.push(inp("a", 1, 1, "jump"))
    assert not q.push(inp("a", 1, 1, "jump"))
    assert q.stats.duplicates == 1
    assert q.stats.accepted == 1


def test_queue_rejects_stale_and_out_of_window():
    q = m.FrameQueue(start_frame=5, capacity=4)
    assert not q.push(inp("a", 4, 1, "old"))
    assert q.stats.stale == 1
    assert not q.push(inp("a", 20, 2, "far"))
    assert q.stats.out_of_window == 1
    assert q.push(inp("a", 5, 3, "ok"))
    assert q.push(inp("a", 8, 4, "edge"))


def test_queue_advance_drops_inputs_below_floor():
    q = m.FrameQueue(start_frame=0, capacity=16)
    q.push(inp("a", 0, 1))
    q.push(inp("a", 3, 2))
    dropped = q.advance_to(2)
    assert dropped == 1
    assert q.floor_frame == 2
    assert q.queued_frames == (3,)
    assert q.stats.dropped_on_advance == 1


def test_frame_queue_factory_accepts_input_iterable():
    q = m.frame_queue([inp("a", 3, 1), inp("a", 4, 2), inp("a", 4, 3, "x")])
    assert q.floor_frame == 3
    assert q.queued_frames == (3, 4)
    assert len(q) == 2


# ---------------------------------------------------------------------------
# 2. Catch-up and compensation caps
# ---------------------------------------------------------------------------


def test_catchup_is_capped_per_tick():
    plan = m.catchup_cap(20, max_catchup=5, resync_after=50)
    assert plan.steps == 5
    assert plan.remaining == 15
    assert plan.capped
    assert not plan.resync_required


def test_catchup_requests_resync_past_threshold():
    plan = m.catchup_cap(100, max_catchup=5, resync_after=30)
    assert plan.steps == 5
    assert plan.resync_required
    assert m.fallback(plan).request_snapshot


def test_catchup_caught_up():
    plan = m.catchup_cap(3, max_catchup=5)
    assert plan.steps == 3
    assert plan.caught_up
    assert not plan.capped


def test_catchup_controller_counts_across_ticks():
    controller = m.CatchupController(max_catchup=4, resync_after=100)
    for _ in range(3):
        plan = controller.plan(10)
        assert plan.steps == 4
    assert controller.total_steps == 12


def test_compensation_cap_refuses_excessive_rewind():
    plan = m.compensation_cap(12, max_compensation=4)
    assert plan.allowed == 4
    assert plan.capped
    assert plan.resync_required
    assert m.fallback(plan).resync_required


# ---------------------------------------------------------------------------
# 3. RNG state rolls back with the snapshot
# ---------------------------------------------------------------------------


def test_rng_snapshot_restores_identical_sequence():
    rng = random.Random(1234)
    state = m.rng_snapshot(rng)
    expected = [rng.random() for _ in range(5)]

    replay = m.rng_restore(state)
    actual = [replay.random() for _ in range(5)]
    assert actual == expected


def test_rng_restore_rejects_tampered_state():
    state = m.rng_snapshot(random.Random(7))
    tampered = m.RNGState(state=state.state, digest="0" * 64)
    with pytest.raises(m.LockstepError):
        m.rng_restore(tampered)


def test_snapshot_captures_state_frame_and_rng_together():
    rng = random.Random(99)
    snap = m.Snapshot.capture(10, {"hp": 5}, rng)
    after_capture = rng.random()

    rng.random()  # advance away from the snapshot
    restored = snap.restore_rng(rng)
    assert restored.random() == after_capture
    assert snap.logical_frame == 10
    assert snap.restore_state() == {"hp": 5}
    assert snap.verify()


def test_rollback_restores_rng_so_replay_matches_original():
    rng = random.Random(42)
    buffer = m.RollbackBuffer(max_frames=32, max_compensation=16)

    # Original timeline: capture at frame 5, then simulate frames 5..10.
    buffer.capture(5, {"hp": 10}, rng)
    original_events = []
    for frame in range(5, 11):
        original_events.append((frame, rng.randint(1, 6), rng.random()))
        buffer.current_frame = frame

    # Rewind to frame 6 and replay the same frames with the same RNG.
    result = buffer.rollback(6, rng=rng)
    assert result.success
    assert result.restored_from == 5
    assert result.state == {"hp": 10}

    replayed = []
    for frame in range(5, 11):
        event = (frame, rng.randint(1, 6), rng.random())
        if frame > 6:
            replayed.append(event)
    assert replayed == [e for e in original_events if e[0] > 6]


def test_rollback_without_rng_restore_would_diverge():
    """Guard against a regression where only simulation state is restored."""
    rng = random.Random(5)
    buffer = m.RollbackBuffer(max_compensation=16)
    buffer.capture(1, {"v": 0}, rng)
    original_next = rng.randint(1, 1000)

    rng2 = random.Random(5)
    buffer2 = m.RollbackBuffer(max_compensation=16)
    buffer2.capture(1, {"v": 0}, rng2)
    rng2.random()  # simulate consuming randomness on the discarded timeline
    buffer2.rollback(1, rng=rng2)
    assert rng2.randint(1, 1000) == original_next


def test_rollback_refuses_replay_beyond_compensation_cap():
    rng = random.Random(0)
    buffer = m.RollbackBuffer(max_frames=64, max_compensation=2)
    for frame in range(0, 40):
        buffer.capture(frame, {"frame": frame}, rng)
        buffer.current_frame = frame

    # Restoring frame 0 from current frame 39 means 39 frames of re-simulation,
    # far beyond a cap of 2.
    result = buffer.rollback(0, rng=rng)
    assert not result.success
    assert result.resync_required
    assert result.reason == "exceeds_compensation_cap"


def test_rollback_drops_speculative_snapshots_after_restore_point():
    rng = random.Random(1)
    buffer = m.RollbackBuffer(max_compensation=8)
    for frame in range(0, 6):
        buffer.capture(frame, {"f": frame}, rng)
        buffer.current_frame = frame
    result = buffer.rollback(2, rng=rng)
    assert result.success
    assert buffer.frames == (0, 1, 2)


def test_rollback_to_future_is_rejected():
    rng = random.Random(1)
    buffer = m.RollbackBuffer()
    buffer.capture(1, {"f": 1}, rng)
    buffer.current_frame = 4
    result = buffer.rollback(5, rng=rng)
    assert not result.success
    assert result.reason == "target_frame_is_in_the_future"


# ---------------------------------------------------------------------------
# 4. Cooldowns use logical frames
# ---------------------------------------------------------------------------


def test_logical_frame_uses_input_delay():
    assert m.logical_frame(10, input_delay=2) == 8
    assert m.logical_frame(10, input_delay=0) == 10
    # Never simulate ahead of the server's confirmed frame.
    assert m.logical_frame(10, input_delay=2, confirmed_frame=7) == 7
    assert m.logical_frame(0, input_delay=5) == 0


def test_cooldown_uses_logical_frame_not_render_frame():
    tracker = m.CooldownTracker(input_delay=2)
    cooldown = tracker.start("fire", local_frame=10, duration=3)
    assert cooldown.started_frame == 8
    assert cooldown.ready_frame == 11
    # At render frame 12 the logical frame is 10: still cooling.
    assert not cooldown.ready(m.logical_frame(12, input_delay=2))
    assert cooldown.remaining(m.logical_frame(12, input_delay=2)) == 1


def test_two_clients_with_different_render_frames_agree_on_cooldown():
    # Different render frame + different input delay can still land on the same
    # logical frame; the cooldown must anchor there.
    a = m.CooldownTracker(input_delay=2)
    b = m.CooldownTracker(input_delay=4)
    cd_a = a.start("fire", local_frame=10, duration=4)  # logical 8
    cd_b = b.start("fire", local_frame=12, duration=4)  # logical 8
    assert cd_a.started_frame == cd_b.started_frame == 8
    assert cd_a.ready(12) == cd_b.ready(12)


def test_validation_flags_render_frame_cooldown():
    bad = m.Cooldown("fire", 3, 10, frame_basis="render")
    report = m.validation({"cooldowns": [bad]})
    assert not report.ok
    assert not report.checks["cooldowns_use_logical_frames"]


# ---------------------------------------------------------------------------
# 5. Reconnect de-duplication and watermarks
# ---------------------------------------------------------------------------


def test_reconnect_replay_of_old_inputs_is_rejected():
    replay = [
        inp("a", 5, 1, "move"),
        inp("a", 6, 2, "move"),
        inp("a", 7, 3, "jump"),
    ]
    # Watermark 3 means sequences 1..3 were already processed.  The first two
    # resends are lower than the watermark (regressed_seq) and the last is the
    # exactly-seen sequence (duplicate); all three must be dropped.
    batch = m.reconnect_dedupe(replay, floor_frame=0, watermarks={"a": 3})
    assert batch.accepted_count == 0
    assert batch.already_processed_count == 3


def test_reconnect_replay_of_exactly_seen_sequences_is_duplicate():
    replay = [inp("a", 5, 1), inp("a", 6, 2), inp("a", 7, 3)]
    batch = m.reconnect_dedupe(replay, watermarks={"a": 3})
    assert batch.accepted_count == 0
    assert batch.duplicate_count == 1
    assert batch.regressed_count == 2


def test_reconnect_resumes_after_watermark():
    replay = [inp("a", 8, 3, "x"), inp("a", 9, 4, "y")]
    batch = m.reconnect_dedupe(replay, watermarks={"a": 2})
    assert [i.seq for i in batch.accepted] == [3, 4]
    assert batch.watermarks["a"] == 4


def test_watermark_drops_inputs_for_consumed_frames():
    replay = [inp("a", 2, 10, "old"), inp("a", 9, 11, "new")]
    batch = m.reconnect_dedupe(replay, floor_frame=5)
    assert batch.accepted_count == 1
    assert batch.accepted[0].server_frame == 9
    assert batch.stale_count == 1


def test_watermark_summary_and_resume_point():
    processed = [inp("a", 1, 1), inp("a", 2, 2), inp("b", 2, 1)]
    wm = m.watermark(processed)
    assert wm.per_player == {"a": 2, "b": 1}
    assert wm.resume_from == {"a": 3, "b": 2}
    assert wm.frame_floor == 1


def test_watermark_overflow_does_not_rewind():
    """A replay with a lower seq must not move a watermark backwards."""
    batch = m.reconnect_dedupe(
        [inp("a", 20, 7, "old"), inp("a", 21, 9, "new")], watermarks={"a": 8}
    )
    assert batch.watermarks["a"] == 9
    assert batch.accepted_count == 1
    assert batch.results[0].reason == "regressed_seq"


def test_repeated_duplicate_injection_is_idempotent():
    dedupe = m.InputDeduplicator(floor_frame=0)
    first = dedupe.accept_many([inp("a", 1, 1), inp("a", 2, 2)])
    second = dedupe.accept_many([inp("a", 1, 1), inp("a", 2, 2)])
    assert first.accepted_count == 2
    assert second.accepted_count == 0
    assert dedupe.watermarks == {"a": 2}


def test_audit_reports_duplicates_and_stale():
    log = m.audit(
        [inp("a", 1, 1), inp("a", 1, 1), inp("a", 2, 2), inp("a", 0, 3)],
        floor_frame=1,
    )
    assert not log.ok
    assert log.duplicates == 1
    assert log.stale >= 1
    assert log.counts["accepted"] == 2
    assert len(log.findings()) >= 2


# ---------------------------------------------------------------------------
# 6. Presentation rebuild after rollback
# ---------------------------------------------------------------------------


def test_presentation_discards_samples_after_rollback_point():
    buffer = m.PresentationBuffer()
    for frame in range(0, 8):
        buffer.add(frame, {"x": frame})
    rebuild = buffer.rebuild(4, authoritative_state={"x": 4})
    assert [s.logical_frame for s in rebuild.retained] == [0, 1, 2, 3, 4]
    assert [s.logical_frame for s in rebuild.discarded] == [5, 6, 7]
    assert rebuild.anchor_frame == 4
    assert rebuild.dirty
    assert len(buffer) == 5


def test_presentation_reanchor_without_snapshot_uses_last_sample():
    buffer = m.PresentationBuffer()
    buffer.add(0, "a")
    buffer.add(3, "b")
    buffer.add(6, "c")
    rebuild = buffer.rebuild(4)
    assert rebuild.anchor_frame == 3
    assert rebuild.needs_reanchor
    assert buffer.samples == (m.PresentSample(0, "a", "state", 1), m.PresentSample(3, "b", "state", 2))


def test_present_rebuild_function_accepts_iterable():
    samples = [m.PresentSample(f, {"f": f}) for f in range(5)]
    result = m.present_rebuild(samples, 2, authoritative_state={"f": 2})
    assert result.anchor_frame == 2
    assert result.discarded_count == 2
    assert len(result.retained) == 3


def test_presentation_after_rebuild_stops_extrapolating_old_timeline():
    buffer = m.PresentationBuffer()
    for frame in range(0, 5):
        buffer.add(frame, frame)
    buffer.rebuild(2, authoritative_state=2)
    buffer.add(3, "rewritten")
    assert buffer.samples[-1].value == "rewritten"
    assert buffer.later_than(2) == (buffer.samples[-1],)


# ---------------------------------------------------------------------------
# Cross-cutting: ordering, determinism, validation, metrics, fallback
# ---------------------------------------------------------------------------


def test_input_order_is_stable_for_unordered_arrival():
    items = [inp("b", 1, 2), inp("a", 1, 2), inp("b", 0, 1), inp("a", 0, 1)]
    order = m.input_order(items)
    assert [(i.server_frame, i.player) for i in order.ordered] == [
        (0, "a"),
        (0, "b"),
        (1, "a"),
        (1, "b"),
    ]
    # Re-running on the output produces the same digest.
    assert m.input_order(order.ordered).digest == order.digest


def test_input_order_does_not_mark_late_earlier_frames_stale():
    items = [inp("a", 3, 3), inp("a", 2, 2)]
    order = m.input_order(items)
    assert len(order.ordered) == 2
    assert order.stale == ()


def test_input_order_drops_replayed_duplicates():
    items = [inp("a", 1, 1), inp("a", 1, 1), inp("a", 2, 1)]
    order = m.input_order(items)
    assert len(order.ordered) == 1
    assert len(order.duplicates) == 2


def test_determinism_detects_state_and_rng_divergence():
    rng = random.Random(3)
    ref = random.Random(3)
    report = m.determinism({"hp": 10}, reference={"hp": 10}, rng=rng, reference_rng=ref)
    assert report.ok

    rng.random()
    ref.random()
    ref.random()
    report = m.determinism({"hp": 10}, reference={"hp": 10}, rng=rng, reference_rng=ref)
    assert not report.ok
    assert "rng_differs" in report.divergences


def test_determinism_reports_keyed_divergence():
    report = m.determinism({"hp": 10, "x": 1}, reference={"hp": 10, "x": 2})
    assert not report.ok
    assert report.divergences == ("differs:'x'",)


def test_state_digest_is_order_independent_for_mappings():
    assert m.state_digest({"a": 1, "b": 2}) == m.state_digest({"b": 2, "a": 1})
    assert m.state_digest({"a": 1}) != m.state_digest({"a": 2})


def test_validation_accepts_healthy_rollback_buffer():
    rng = random.Random(0)
    buffer = m.RollbackBuffer(max_compensation=4)
    buffer.capture(0, {"v": 0}, rng)
    buffer.capture(1, {"v": 1}, rng)
    report = m.validation(buffer)
    assert report.ok
    assert report.checks["snapshots_carry_rng"]


def test_validation_flags_snapshot_without_rng():
    class FakeBuffer:
        max_compensation = 4
        failed_rollbacks = 0
        _snapshots = [type("S", (), {"rng": None, "verify": lambda self: True})()]

    report = m.validation(FakeBuffer())
    assert not report.ok
    assert not report.checks["snapshots_carry_rng"]


def test_latency_comp_converts_ms_and_caps_delay():
    plan = m.latency_comp(100, tick_ms=50.0, unit="ms", max_delay=4)
    assert plan.latency_frames == 2
    assert plan.input_delay == 3
    assert plan.buffer_frames == 4

    capped = m.latency_comp(50, tick_ms=33.0, max_delay=3)
    assert capped.input_delay == 3


def test_metrics_merge_and_record():
    q = m.FrameQueue(start_frame=0)
    q.push(inp("a", 0, 1))
    q.push(inp("a", 0, 1))
    q.pop_frame(0)

    metrics = m.metrics(q.stats)
    metrics.record_catchup(m.catchup_cap(12, max_catchup=4))
    assert metrics.inputs_accepted == 1
    assert metrics.inputs_duplicate == 1
    assert metrics.frames_simulated == 1
    assert metrics.catchup_steps == 4
    assert metrics.max_backlog == 12

    metrics.merge(m.metrics({"inputs_stale": 2}))
    assert metrics.inputs_stale == 2


def test_fallback_plan_is_actionable():
    plan = m.fallback(m.catchup_cap(200, max_catchup=8, resync_after=16), frame_floor=120)
    assert plan.resync_required
    assert plan.drop_inputs_before == 120
    assert "clear_presentation_samples" in plan.steps
    assert "restore_rng_with_snapshot" in plan.steps


def test_end_to_end_jitter_rollback_and_reconnect():
    """The full failure scenario from the report, in miniature."""
    rng = random.Random(2024)
    players = ("p1", "p2")
    queue = m.FrameQueue(start_frame=0, players=players)
    buffer = m.RollbackBuffer(max_frames=16, max_compensation=8)
    present = m.PresentationBuffer()
    dedupe = m.InputDeduplicator(floor_frame=0)
    dedupe_states: dict[str, int] = {}

    simulated = {}
    for frame in range(0, 6):
        for index, player in enumerate(players):
            item = inp(player, frame, frame + 1, ("act", frame))
            dedupe_states[player] = frame + 1
            queue.push(item)
        ready = queue.pop_frame(frame)
        assert ready is not None
        buffer.capture(frame, {"frame": frame, "score": rng.randint(0, 9)}, rng)
        simulated[frame] = buffer.get(frame).state
        buffer.current_frame = frame
        present.add(m.logical_frame(frame * 30, input_delay=2), {"visual": frame})

    # Server corrects frame 3; client rewinds, restores RNG, rebuilds visuals.
    rollback = buffer.rollback(3, rng=rng)
    assert rollback.success
    rebuild = present.rebuild(3, authoritative_state=rollback.state)
    assert all(s.logical_frame > 3 for s in rebuild.discarded)

    # Reconnect: server replays frames 0..5 again; all must be dropped.
    replay = [inp(p, f, f + 1, ("act", f)) for f in range(0, 6) for p in players]
    batch = m.reconnect_dedupe(replay, watermarks=dedupe_states)
    assert batch.accepted_count == 0
    assert batch.already_processed_count == len(replay)

    # New inputs after the reconnect still get through.
    batch = m.reconnect_dedupe([inp("p1", 6, 7, "move")], watermarks=dedupe_states)
    assert batch.accepted_count == 1
