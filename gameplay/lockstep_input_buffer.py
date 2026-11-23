"""Frame-synchronised (lockstep) input buffering, rollback and reconnect support.

The module models the parts of a lockstep client that have to agree exactly
with every other client and with the server:

* :class:`FrameQueue` orders inputs by **server frame**, never by local render
  frame.  Duplicate ``(player, server_frame)`` inputs are dropped, and inputs
  that arrive below the consumed frame floor are considered stale.
* :func:`catchup_cap` / :func:`compensation_cap` bound how many server frames a
  single tick may chase and how far a rollback may rewind.  Exceeding the cap
  returns a plan that asks for an authoritative snapshot instead of letting the
  client fall into a catch-up death spiral.
* :func:`rng_snapshot` / :func:`rng_restore` capture and restore random-number
  generator state.  :class:`Snapshot` stores that state next to the simulation
  state so a rollback cannot leave the RNG ahead of the replayed frames.
* :func:`logical_frame` and :class:`Cooldown` use logical (confirmed) frames
  instead of the local render frame, removing the off-by-one tick between
  clients.
* :class:`InputDeduplicator` keeps a per-player sequence watermark.  Reconnect
  replays below the watermark (or below the frame floor) are rejected, so old
  inputs are not injected into the queue twice.
* :class:`PresentationBuffer` discards interpolation/effect samples from frames
  after the rollback point and re-anchors on the authoritative frame.

Every public entry point returns a non-``None`` object so callers can inspect
the result in diagnostics, audits and tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

__all__ = [
    "LockstepError",
    "FrameInput",
    "QueueStats",
    "FrameQueue",
    "frame_queue",
    "CatchupPlan",
    "CatchupController",
    "catchup_cap",
    "CompensationPlan",
    "compensation_cap",
    "RNGState",
    "rng_snapshot",
    "rng_restore",
    "logical_frame",
    "Cooldown",
    "cooldown_remaining",
    "DedupeResult",
    "DedupeBatch",
    "InputDeduplicator",
    "reconnect_dedupe",
    "watermark",
    "PresentSample",
    "PresentRebuild",
    "PresentationBuffer",
    "present_rebuild",
    "Snapshot",
    "RollbackResult",
    "RollbackBuffer",
    "rollback",
    "state_digest",
    "ValidationReport",
    "validation",
    "AuditEvent",
    "AuditLog",
    "audit",
    "FallbackPlan",
    "fallback",
    "LatencyPlan",
    "latency_comp",
    "Metrics",
    "metrics",
    "InputOrder",
    "input_order",
    "DeterminismReport",
    "determinism",
]

_MISSING = object()


class LockstepError(ValueError):
    """Raised when input, snapshot or frame data breaks a lockstep invariant."""


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LockstepError(f"{name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise LockstepError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LockstepError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise LockstepError(f"{name} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise LockstepError(f"{name} must be >= {minimum}, got {number}")
    return number


def _stable_key(value: Any) -> tuple[str, str]:
    """Ordering key that never depends on memory addresses or hash seeds."""
    return (type(value).__qualname__, repr(value))


# ---------------------------------------------------------------------------
# Inputs and their queue
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameInput:
    """One player's input for one *server* frame.

    ``seq`` is a per-player, strictly increasing sequence number assigned by
    the sender.  It is what reconnect de-duplication uses; ``server_frame`` is
    what the input queue orders on.
    """

    player: Hashable
    server_frame: int
    seq: int
    actions: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        _require_int(self.server_frame, "server_frame", minimum=0)
        _require_int(self.seq, "seq", minimum=1)
        try:
            hash(self.player)
        except TypeError:
            raise LockstepError("player must be hashable") from None
        try:
            actions = tuple(self.actions)
        except TypeError:
            raise LockstepError("actions must be an iterable") from None
        object.__setattr__(self, "actions", actions)

    @property
    def key(self) -> tuple[Hashable, int]:
        return (self.player, self.server_frame)


@dataclass(slots=True)
class QueueStats:
    accepted: int = 0
    duplicates: int = 0
    stale: int = 0
    out_of_window: int = 0
    dropped_on_advance: int = 0
    frames_consumed: int = 0

    @property
    def rejected(self) -> int:
        return self.duplicates + self.stale + self.out_of_window + self.dropped_on_advance

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "stale": self.stale,
            "out_of_window": self.out_of_window,
            "dropped_on_advance": self.dropped_on_advance,
            "frames_consumed": self.frames_consumed,
            "rejected": self.rejected,
        }


class FrameQueue:
    """Input buffer keyed by server frame.

    ``floor_frame`` is the first frame that may still be queued.  Consuming a
    frame or advancing moves the floor forward and discards anything older.
    ``capacity`` bounds how far ahead of the floor inputs may be buffered; a
    frame outside that window is rejected as ``out_of_window`` instead of
    growing memory without bound.
    """

    def __init__(
        self,
        start_frame: int = 0,
        *,
        players: Iterable[Hashable] = (),
        capacity: int = 64,
        duplicate_policy: str = "reject",
    ) -> None:
        self.floor_frame = _require_int(start_frame, "start_frame", minimum=0)
        self.players = tuple(players)
        self.capacity = _require_int(capacity, "capacity", minimum=1)
        if duplicate_policy not in {"reject", "latest"}:
            raise LockstepError("duplicate_policy must be 'reject' or 'latest'")
        self.duplicate_policy = duplicate_policy
        self._frames: dict[int, dict[Hashable, FrameInput]] = {}
        self._stats = QueueStats()

    @property
    def stats(self) -> QueueStats:
        return copy.copy(self._stats)

    @property
    def backlog(self) -> int:
        return len(self._frames)

    @property
    def queued_frames(self) -> tuple[int, ...]:
        return tuple(sorted(self._frames))

    def __len__(self) -> int:
        return len(self._frames)

    def __contains__(self, server_frame: object) -> bool:
        return server_frame in self._frames

    def push(self, item: FrameInput) -> bool:
        if not isinstance(item, FrameInput):
            raise LockstepError("FrameQueue.push expects a FrameInput")
        frame = item.server_frame
        if frame < self.floor_frame:
            self._stats.stale += 1
            return False
        if frame >= self.floor_frame + self.capacity:
            self._stats.out_of_window += 1
            return False

        bucket = self._frames.setdefault(frame, {})
        existing = bucket.get(item.player)
        if existing is not None:
            same_payload = existing.seq == item.seq and existing.actions == item.actions
            if same_payload:
                self._stats.duplicates += 1
                return False
            if self.duplicate_policy == "latest" and item.seq > existing.seq:
                bucket[item.player] = item
                self._stats.accepted += 1
                return True
            self._stats.duplicates += 1
            return False

        bucket[item.player] = item
        self._stats.accepted += 1
        return True

    def extend(self, items: Iterable[FrameInput]) -> tuple[bool, ...]:
        return tuple(self.push(item) for item in items)

    def ready(self, server_frame: int, *, required_players: Iterable[Hashable] | None = None) -> bool:
        frame = _require_int(server_frame, "server_frame", minimum=0)
        bucket = self._frames.get(frame)
        if not bucket:
            return False
        required = tuple(self.players if required_players is None else required_players)
        if required:
            return all(player in bucket for player in required)
        return True

    def pop_frame(
        self,
        server_frame: int,
        *,
        required_players: Iterable[Hashable] | None = None,
    ) -> tuple[FrameInput, ...] | None:
        """Consume one frame once every required player has supplied input.

        Returns ``None`` while the frame is incomplete, leaving it queued.  A
        frame is never partially consumed, which is what keeps all peers
        simulating the same inputs.
        """
        frame = _require_int(server_frame, "server_frame", minimum=0)
        if frame < self.floor_frame:
            return None
        bucket = self._frames.get(frame)
        if not bucket:
            return None

        required = tuple(self.players if required_players is None else required_players)
        if required:
            if any(player not in bucket for player in required):
                return None
            ordered = tuple(bucket[player] for player in required)
        else:
            ordered = tuple(bucket[player] for player in sorted(bucket, key=_stable_key))

        del self._frames[frame]
        self.floor_frame = max(self.floor_frame, frame + 1)
        self._stats.frames_consumed += 1
        self._drop_below_floor()
        return ordered

    def advance_to(self, server_frame: int) -> int:
        """Move the consumed-frame floor forward; return inputs discarded."""
        frame = _require_int(server_frame, "server_frame", minimum=0)
        if frame > self.floor_frame:
            self.floor_frame = frame
            return self._drop_below_floor()
        return 0

    def _drop_below_floor(self) -> int:
        dropped = 0
        for frame in tuple(self._frames):
            if frame >= self.floor_frame:
                continue
            bucket = self._frames.pop(frame)
            dropped += len(bucket)
        self._stats.dropped_on_advance += dropped
        return dropped

    def clear(self) -> None:
        self._frames.clear()


def frame_queue(
    start_frame: int | Iterable[FrameInput] = 0,
    *,
    players: Iterable[Hashable] = (),
    capacity: int = 64,
    duplicate_policy: str = "reject",
) -> FrameQueue:
    """Build a :class:`FrameQueue`.

    The first argument may also be an iterable of :class:`FrameInput`; in that
    case the floor is the lowest input frame and the inputs are queued.
    """
    if isinstance(start_frame, FrameInput) or not isinstance(start_frame, int):
        items = tuple(start_frame) if not isinstance(start_frame, FrameInput) else (start_frame,)
        floor = min((item.server_frame for item in items), default=0)
        queue = FrameQueue(
            floor,
            players=players,
            capacity=capacity,
            duplicate_policy=duplicate_policy,
        )
        queue.extend(items)
        return queue
    return FrameQueue(
        start_frame,
        players=players,
        capacity=capacity,
        duplicate_policy=duplicate_policy,
    )


# ---------------------------------------------------------------------------
# Catch-up and compensation limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CatchupPlan:
    """How many frames to advance this tick, and whether to resync."""

    requested: int
    steps: int
    remaining: int
    capped: bool
    resync_required: bool
    reason: str = "caught_up"

    @property
    def caught_up(self) -> bool:
        return self.remaining == 0


@dataclass(frozen=True, slots=True)
class CompensationPlan:
    """Whether a rollback of ``requested`` frames is allowed."""

    requested: int
    allowed: int
    capped: bool
    resync_required: bool
    reason: str = "within_cap"


def catchup_cap(
    pending_frames: int = 0,
    *,
    max_catchup: int = 8,
    resync_after: int = 32,
) -> CatchupPlan:
    """Cap one tick's catch-up work.

    ``max_catchup`` is the hard per-tick step limit.  ``resync_after`` is how
    much backlog may remain after that step before the client stops chasing and
    asks for an authoritative snapshot.  A backlog that large usually means the
    client cannot catch up at the current tick rate.
    """
    pending = _require_int(pending_frames, "pending_frames", minimum=0)
    limit = _require_int(max_catchup, "max_catchup", minimum=0)
    threshold = _require_int(resync_after, "resync_after", minimum=0)
    steps = min(pending, limit)
    remaining = pending - steps
    capped = remaining > 0
    resync = remaining > threshold
    if resync:
        reason = "backlog_exceeds_resync_threshold"
    elif capped:
        reason = "capped_continue_next_tick"
    else:
        reason = "caught_up"
    return CatchupPlan(
        requested=pending,
        steps=steps,
        remaining=remaining,
        capped=capped,
        resync_required=resync,
        reason=reason,
    )


def compensation_cap(
    requested_frames: int = 0,
    *,
    max_compensation: int = 8,
) -> CompensationPlan:
    """Cap how far a rollback/replay may rewind.

    Exceeding the cap is not handled by replaying anyway; the caller must
    request a full authoritative snapshot instead.
    """
    requested = _require_int(requested_frames, "requested_frames", minimum=0)
    limit = _require_int(max_compensation, "max_compensation", minimum=0)
    allowed = min(requested, limit)
    capped = requested > limit
    return CompensationPlan(
        requested=requested,
        allowed=allowed,
        capped=capped,
        resync_required=capped,
        reason="exceeds_compensation_cap" if capped else "within_cap",
    )


class CatchupController:
    """Stateful helper that applies :func:`catchup_cap` tick after tick."""

    def __init__(self, *, max_catchup: int = 8, resync_after: int = 32) -> None:
        self.max_catchup = _require_int(max_catchup, "max_catchup", minimum=0)
        self.resync_after = _require_int(resync_after, "resync_after", minimum=0)
        self.total_steps = 0
        self.resyncs = 0
        self.last_plan: CatchupPlan | None = None

    def plan(self, pending_frames: int) -> CatchupPlan:
        plan = catchup_cap(
            pending_frames,
            max_catchup=self.max_catchup,
            resync_after=self.resync_after,
        )
        self.total_steps += plan.steps
        if plan.resync_required:
            self.resyncs += 1
        self.last_plan = plan
        return plan


# ---------------------------------------------------------------------------
# RNG snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RNGState:
    """Immutable capture of a generator's state plus a digest for auditing."""

    state: Any
    digest: str
    source: str = "random.Random"


def _is_random_state(state: Any) -> bool:
    return isinstance(state, tuple) and len(state) == 3 and isinstance(state[1], tuple)


def rng_snapshot(source: Any = 0) -> RNGState:
    """Capture RNG state.

    ``source`` may be an existing :class:`RNGState`, an integer seed, a
    :class:`random.Random`, any object with ``getstate()``/``setstate()`` or a
    ``state`` attribute, or a raw state tuple returned by ``Random.getstate()``.
    """
    if isinstance(source, RNGState):
        return source

    if isinstance(source, int) and not isinstance(source, bool):
        rng: Any = random.Random(source)
        state = rng.getstate()
        kind = "random.Random"
    elif hasattr(source, "getstate"):
        state = source.getstate()
        kind = f"{type(source).__module__}.{type(source).__qualname__}"
    elif hasattr(source, "state"):
        state = copy.deepcopy(source.state)
        kind = f"{type(source).__module__}.{type(source).__qualname__}"
    elif _is_random_state(source):
        state = copy.deepcopy(source)
        kind = "random.Random"
    else:
        raise LockstepError(
            "rng_snapshot expects a seed, random.Random, state tuple or object "
            "with getstate()/setstate()"
        )

    return RNGState(state=state, digest=state_digest(state), source=kind)


def rng_restore(snapshot: Any, rng: Any = None) -> Any:
    """Restore and return an RNG from ``snapshot``.

    When ``rng`` is omitted a fresh :class:`random.Random` is created, which is
    only valid for ``random.Random`` snapshots.  The restored state is verified
    against the captured digest so a corrupted/edited snapshot fails loudly.
    """
    if not isinstance(snapshot, RNGState):
        snapshot = rng_snapshot(snapshot)
    if snapshot.digest != state_digest(snapshot.state):
        raise LockstepError("RNG snapshot digest mismatch")

    if rng is None:
        if not _is_random_state(snapshot.state):
            raise LockstepError(
                f"rng instance required to restore snapshot from {snapshot.source}"
            )
        rng = random.Random()

    if hasattr(rng, "setstate"):
        try:
            rng.setstate(copy.deepcopy(snapshot.state))
        except (TypeError, ValueError) as exc:
            raise LockstepError(f"cannot restore RNG state: {exc}") from exc
    elif hasattr(rng, "state"):
        rng.state = copy.deepcopy(snapshot.state)
    else:
        raise LockstepError("target RNG must provide setstate() or a state attribute")

    if hasattr(rng, "getstate"):
        if state_digest(rng.getstate()) != snapshot.digest:
            raise LockstepError("RNG state changed while restoring snapshot")
    return rng


# ---------------------------------------------------------------------------
# Logical frames and cooldowns
# ---------------------------------------------------------------------------


def logical_frame(
    local_frame: int,
    *,
    input_delay: int = 2,
    confirmed_frame: int | None = None,
) -> int:
    """Return the logical (confirmed) frame being simulated.

    ``local_frame`` is the client's render frame.  ``input_delay`` is the fixed
    number of frames between input collection and simulation.  Simulation code
    must use this value -- not ``local_frame`` -- for cooldowns, buff expiry and
    any other frame-based game logic.  Two clients with different render
    timelines then still agree on the logical frame.

    ``confirmed_frame`` optionally clamps the result to the newest frame the
    server has confirmed, so a client that stalls cannot simulate ahead of its
    inputs.
    """
    local = _require_int(local_frame, "local_frame", minimum=0)
    delay = _require_int(input_delay, "input_delay", minimum=0)
    frame = max(0, local - delay)
    if confirmed_frame is not None:
        confirmed = _require_int(confirmed_frame, "confirmed_frame", minimum=0)
        frame = min(frame, confirmed)
    return frame


@dataclass(frozen=True, slots=True)
class Cooldown:
    """A cooldown measured exclusively in logical frames.

    ``started_frame`` must come from :func:`logical_frame`; using the render
    frame here is the classic source of a one-tick disagreement between peers.
    ``frame_basis`` is recorded so :func:`validation` can reject cooldowns that
    were started on the render clock.
    """

    action: Hashable
    duration: int
    started_frame: int
    frame_basis: str = "logical"

    def __post_init__(self) -> None:
        _require_int(self.duration, "duration", minimum=1)
        _require_int(self.started_frame, "started_frame", minimum=0)
        if self.frame_basis not in {"logical", "render"}:
            raise LockstepError("frame_basis must be 'logical' or 'render'")
        try:
            hash(self.action)
        except TypeError:
            raise LockstepError("cooldown action must be hashable") from None

    @property
    def ready_frame(self) -> int:
        return self.started_frame + self.duration

    def remaining(self, current_logical_frame: int) -> int:
        current = _require_int(current_logical_frame, "current_logical_frame", minimum=0)
        return max(0, self.ready_frame - current)

    def ready(self, current_logical_frame: int) -> bool:
        return self.remaining(current_logical_frame) == 0

    def progress(self, current_logical_frame: int) -> float:
        current = _require_int(current_logical_frame, "current_logical_frame", minimum=0)
        elapsed = min(max(current - self.started_frame, 0), self.duration)
        return elapsed / self.duration


def cooldown_remaining(
    cooldown: Cooldown | int,
    current_logical_frame: int,
    *,
    duration: int = 1,
) -> int:
    """Convenience wrapper; an integer ``cooldown`` is treated as remaining."""
    if isinstance(cooldown, Cooldown):
        return cooldown.remaining(current_logical_frame)
    remaining = _require_int(cooldown, "cooldown", minimum=0)
    _require_int(duration, "duration", minimum=1)
    return remaining


class CooldownTracker:
    """Per-action cooldown bookkeeping driven by logical frames."""

    def __init__(self, *, input_delay: int = 2) -> None:
        self.input_delay = _require_int(input_delay, "input_delay", minimum=0)
        self._active: dict[Hashable, Cooldown] = {}

    def logical(self, local_frame: int, confirmed_frame: int | None = None) -> int:
        return logical_frame(
            local_frame, input_delay=self.input_delay, confirmed_frame=confirmed_frame
        )

    def start(
        self,
        action: Hashable,
        *,
        local_frame: int,
        duration: int,
        confirmed_frame: int | None = None,
    ) -> Cooldown:
        cooldown = Cooldown(
            action=action,
            duration=duration,
            started_frame=self.logical(local_frame, confirmed_frame),
        )
        self._active[action] = cooldown
        return cooldown

    def get(self, action: Hashable) -> Cooldown | None:
        return self._active.get(action)

    def remaining(self, action: Hashable, logical: int) -> int:
        cooldown = self._active.get(action)
        return 0 if cooldown is None else cooldown.remaining(logical)

    def ready(self, action: Hashable, logical: int) -> bool:
        return self.remaining(action, logical) == 0

    def prune(self, logical: int) -> tuple[Hashable, ...]:
        expired = tuple(a for a, cd in self._active.items() if cd.ready(logical))
        for action in expired:
            del self._active[action]
        return expired


# ---------------------------------------------------------------------------
# Reconnect de-duplication and watermarks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DedupeResult:
    player: Hashable
    seq: int
    server_frame: int
    accepted: bool
    reason: str
    watermark: int

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True, slots=True)
class DedupeBatch:
    results: tuple[DedupeResult, ...]
    accepted: tuple[FrameInput, ...]
    watermarks: dict[Hashable, int]
    floor_frame: int

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def duplicate_count(self) -> int:
        return sum(1 for r in self.results if r.reason == "duplicate")

    @property
    def regressed_count(self) -> int:
        return sum(1 for r in self.results if r.reason == "regressed_seq")

    @property
    def stale_count(self) -> int:
        return sum(1 for r in self.results if r.reason == "stale")

    @property
    def already_processed_count(self) -> int:
        """Everything rejected because it was seen before (any reason)."""
        return len(self.results) - self.accepted_count

    def __len__(self) -> int:
        return len(self.results)


class InputDeduplicator:
    """Per-player sequence watermark used across reconnects.

    The watermark is the highest *sequence* accepted for a player.  It is
    deliberately independent of the server-frame floor: a reconnect may resend
    a frame that is still queued (duplicate) or one already consumed (stale),
    and neither may be injected again.
    """

    def __init__(
        self,
        players: Iterable[Hashable] = (),
        *,
        start_seq: int = 0,
        floor_frame: int = 0,
        watermarks: Mapping[Hashable, int] | None = None,
    ) -> None:
        self._start_seq = _require_int(start_seq, "start_seq", minimum=0)
        self._watermarks: dict[Hashable, int] = {p: self._start_seq for p in players}
        for player, seq in (watermarks or {}).items():
            # Seeded progress is authoritative and may never be lowered, so a
            # reconnect replay below it is rejected rather than re-applied.
            self._watermarks[player] = max(self._watermarks.get(player, 0), _require_int(seq, "seq"))
        self.floor_frame = _require_int(floor_frame, "floor_frame", minimum=0)
        self.accepted = 0
        self.duplicates = 0
        self.stale = 0
        self.regressions = 0

    def watermark(self, player: Hashable) -> int:
        return self._watermarks.get(player, self._start_seq)

    @property
    def watermarks(self) -> dict[Hashable, int]:
        return dict(self._watermarks)

    def advance_floor(self, server_frame: int) -> None:
        frame = _require_int(server_frame, "server_frame", minimum=0)
        self.floor_frame = max(self.floor_frame, frame)

    def accept(self, item: FrameInput) -> DedupeResult:
        if not isinstance(item, FrameInput):
            raise LockstepError("InputDeduplicator.accept expects a FrameInput")

        current = self.watermark(item.player)
        if item.server_frame < self.floor_frame:
            self.stale += 1
            return DedupeResult(
                item.player, item.seq, item.server_frame, False, "stale", current
            )
        if item.seq <= current:
            reason = "duplicate" if item.seq == current else "regressed_seq"
            if reason == "duplicate":
                self.duplicates += 1
            else:
                self.regressions += 1
            return DedupeResult(
                item.player, item.seq, item.server_frame, False, reason, current
            )

        self._watermarks[item.player] = item.seq
        self.accepted += 1
        return DedupeResult(
            item.player, item.seq, item.server_frame, True, "accepted", item.seq
        )

    def accept_many(self, items: Iterable[FrameInput]) -> DedupeBatch:
        accepted: list[FrameInput] = []
        results: list[DedupeResult] = []
        for item in items:
            result = self.accept(item)
            results.append(result)
            if result.accepted:
                accepted.append(item)
        return DedupeBatch(
            results=tuple(results),
            accepted=tuple(accepted),
            watermarks=self.watermarks,
            floor_frame=self.floor_frame,
        )

    def resume_from(self, player: Hashable) -> int:
        """First sequence the player should resend after a reconnect."""
        return max(self._start_seq, self.watermark(player) + 1)


def reconnect_dedupe(
    items: Iterable[FrameInput] | int = (),
    *,
    floor_frame: int = 0,
    watermarks: Mapping[Hashable, int] | None = None,
    start_seq: int = 0,
) -> DedupeBatch:
    """Filter a reconnect replay through per-player watermarks.

    An integer argument is treated as the server-frame floor (useful when the
    caller knows how far the simulation has been confirmed), producing an empty
    batch.  ``watermarks`` seeds known progress without replaying the history.
    """
    if isinstance(items, int) and not isinstance(items, bool):
        return InputDeduplicator(floor_frame=items).accept_many(())

    dedupe = InputDeduplicator(
        start_seq=start_seq, floor_frame=floor_frame, watermarks=watermarks
    )
    return dedupe.accept_many(items)


@dataclass(frozen=True, slots=True)
class Watermark:
    per_player: dict[Hashable, int]
    frame_floor: int
    min_seq: int
    max_seq: int
    resume_from: dict[Hashable, int]
    frames_seen: int = 0

    @property
    def safe_frame(self) -> int:
        return self.frame_floor

    def __len__(self) -> int:
        return len(self.per_player)


def watermark(
    processed: Iterable[FrameInput] | Mapping[Hashable, int] | int = (),
    *,
    floor_frame: int | None = None,
    start_seq: int = 0,
) -> Watermark:
    """Summarise how far each player has been processed.

    ``processed`` may be an iterable of inputs (their sequence numbers are
    reduced with ``max``), a mapping of player to sequence, or an integer frame
    floor.  ``resume_from`` is the first sequence a reconnect should resend;
    anything below it has already been seen and must be dropped.
    """
    per_player: dict[Hashable, int] = {}
    frames: set[int] = set()

    if isinstance(processed, int) and not isinstance(processed, bool):
        return Watermark(
            per_player={},
            frame_floor=_require_int(processed, "frame_floor", minimum=0),
            min_seq=start_seq,
            max_seq=start_seq,
            resume_from={},
        )

    if isinstance(processed, Mapping):
        for player, seq in processed.items():
            per_player[player] = max(per_player.get(player, start_seq), _require_int(seq, "seq"))
    else:
        for item in processed:
            if not isinstance(item, FrameInput):
                raise LockstepError("watermark expects FrameInput items")
            per_player[item.player] = max(per_player.get(item.player, start_seq), item.seq)
            frames.add(item.server_frame)

    if floor_frame is None:
        floor = min(frames) if frames else 0
    else:
        floor = _require_int(floor_frame, "floor_frame", minimum=0)

    return Watermark(
        per_player=dict(per_player),
        frame_floor=floor,
        min_seq=min(per_player.values(), default=start_seq),
        max_seq=max(per_player.values(), default=start_seq),
        resume_from={p: s + 1 for p, s in per_player.items()},
        frames_seen=len(frames),
    )


# ---------------------------------------------------------------------------
# Presentation rebuild after rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PresentSample:
    logical_frame: int
    value: Any
    kind: str = "state"
    seq: int = 0

    def __post_init__(self) -> None:
        _require_int(self.logical_frame, "logical_frame", minimum=0)
        _require_int(self.seq, "seq", minimum=0)


@dataclass(frozen=True, slots=True)
class PresentRebuild:
    retained: tuple[PresentSample, ...]
    discarded: tuple[PresentSample, ...]
    anchor_frame: int | None
    anchor_state: Any
    dirty: bool
    reason: str

    @property
    def discarded_count(self) -> int:
        return len(self.discarded)

    @property
    def needs_reanchor(self) -> bool:
        return self.dirty or self.anchor_frame is None


class PresentationBuffer:
    """Interpolated/visual samples that must be rebuilt after a rollback.

    Samples are keyed by logical frame.  Data beyond the rollback point is
    invalid once the simulation rewinds, because it was produced from the
    now-discarded timeline.  :meth:`rebuild` drops it and re-anchors on the
    authoritative snapshot so interpolation starts from real state instead of
    extrapolating from a future that no longer exists.
    """

    def __init__(self, *, max_samples: int = 512) -> None:
        self.max_samples = _require_int(max_samples, "max_samples", minimum=1)
        self._samples: list[PresentSample] = []
        self._seq = 0
        self.rebuilds = 0

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> tuple[PresentSample, ...]:
        return tuple(self._samples)

    def add(self, logical_frame: int, value: Any, *, kind: str = "state") -> PresentSample:
        self._seq += 1
        sample = PresentSample(logical_frame, value, kind, self._seq)
        self._samples.append(sample)
        self._samples.sort(key=lambda s: (s.logical_frame, s.seq))
        if len(self._samples) > self.max_samples:
            del self._samples[: len(self._samples) - self.max_samples]
        return sample

    def extend(self, samples: Iterable[PresentSample]) -> None:
        for sample in samples:
            self._samples.append(sample)
        # Re-number so externally supplied samples sort after existing ones.
        max_seq = max((s.seq for s in self._samples), default=0)
        rebuilt: list[PresentSample] = []
        next_seq = max_seq
        for sample in self._samples:
            if sample.seq == 0:
                next_seq += 1
                sample = PresentSample(sample.logical_frame, sample.value, sample.kind, next_seq)
            rebuilt.append(sample)
        self._samples = sorted(rebuilt, key=lambda s: (s.logical_frame, s.seq))
        if len(self._samples) > self.max_samples:
            del self._samples[: len(self._samples) - self.max_samples]

    def later_than(self, logical_frame: int) -> tuple[PresentSample, ...]:
        return tuple(s for s in self._samples if s.logical_frame > logical_frame)

    def rebuild(
        self,
        rollback_frame: int,
        *,
        authoritative_state: Any = None,
        reason: str = "rollback",
    ) -> PresentRebuild:
        target = _require_int(rollback_frame, "rollback_frame", minimum=0)
        retained = tuple(s for s in self._samples if s.logical_frame <= target)
        discarded = tuple(s for s in self._samples if s.logical_frame > target)
        self._samples = list(retained)
        self.rebuilds += 1

        # The authoritative snapshot at ``target`` is the new interpolation
        # anchor.  Without one, fall back to the newest retained visual sample;
        # either way nothing after ``target`` survives, so the layer stops
        # extrapolating from a timeline that is about to be rewritten.
        if authoritative_state is not None:
            anchor_frame: int | None = target
        else:
            anchor_frame = max((s.logical_frame for s in retained), default=None)
        dirty = bool(discarded) or authoritative_state is not None
        return PresentRebuild(
            retained=retained,
            discarded=discarded,
            anchor_frame=anchor_frame,
            anchor_state=authoritative_state,
            dirty=dirty,
            reason=reason,
        )

    def clear(self) -> None:
        self._samples.clear()


def present_rebuild(
    samples: PresentationBuffer | Iterable[PresentSample] | int = (),
    rollback_frame: int | None = None,
    *,
    authoritative_state: Any = None,
    max_samples: int = 512,
) -> PresentRebuild:
    """Rebuild the presentation layer at ``rollback_frame``.

    Accepts a :class:`PresentationBuffer`, an iterable of samples, or an
    integer rollback frame (with no samples).  Returns the retained/discarded
    samples and the new anchor.
    """
    if isinstance(samples, PresentationBuffer):
        buffer = samples
        if rollback_frame is None:
            raise LockstepError("rollback_frame is required")
    else:
        buffer = PresentationBuffer(max_samples=max_samples)
        if isinstance(samples, int) and not isinstance(samples, bool):
            if rollback_frame is not None:
                raise LockstepError("rollback_frame given twice")
            rollback_frame = samples
        else:
            buffer.extend(samples)
        if rollback_frame is None:
            raise LockstepError("rollback_frame is required")

    return buffer.rebuild(
        rollback_frame,
        authoritative_state=authoritative_state,
        reason="present_rebuild",
    )


# ---------------------------------------------------------------------------
# Snapshots and rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Simulation state, logical frame and RNG state captured together.

    Capturing them together is the point: restoring the simulation without
    restoring the RNG is exactly how a rollback replay diverges from the
    original timeline.
    """

    logical_frame: int
    state: Any
    rng: RNGState
    digest: str = ""

    def __post_init__(self) -> None:
        _require_int(self.logical_frame, "logical_frame", minimum=0)
        if not isinstance(self.rng, RNGState):
            raise LockstepError("Snapshot.rng must be an RNGState")
        digest = state_digest({"frame": self.logical_frame, "state": self.state, "rng": self.rng.digest})
        if self.digest and self.digest != digest:
            raise LockstepError("snapshot digest mismatch")
        object.__setattr__(self, "digest", digest)

    @classmethod
    def capture(
        cls,
        logical_frame: int,
        state: Any,
        rng: Any,
        *,
        clone: bool = True,
    ) -> "Snapshot":
        rng_state = rng_snapshot(rng)
        captured = copy.deepcopy(state) if clone else state
        return cls(_require_int(logical_frame, "logical_frame", minimum=0), captured, rng_state)

    def verify(self) -> bool:
        return self.digest == state_digest(
            {"frame": self.logical_frame, "state": self.state, "rng": self.rng.digest}
        )

    def restore_state(self) -> Any:
        return copy.deepcopy(self.state)

    def restore_rng(self, rng: Any = None) -> Any:
        return rng_restore(self.rng, rng)

    @property
    def state_digest(self) -> str:
        return state_digest(self.state)


@dataclass(frozen=True, slots=True)
class RollbackResult:
    success: bool
    target_frame: int
    restored_from: int | None
    frames_rewound: int
    state: Any
    rng: Any
    digest: str
    resync_required: bool
    reason: str
    replayed: int = 0

    def __bool__(self) -> bool:
        return self.success


class RollbackBuffer:
    """Bounded snapshot history with capped rollback and RNG restore."""

    def __init__(self, *, max_frames: int = 64, max_compensation: int = 8) -> None:
        self.max_frames = _require_int(max_frames, "max_frames", minimum=1)
        self.max_compensation = _require_int(max_compensation, "max_compensation", minimum=0)
        self._snapshots: list[Snapshot] = []
        self.current_frame: int | None = None
        self.rollbacks = 0
        self.failed_rollbacks = 0

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(s.logical_frame for s in self._snapshots)

    def capture(self, logical_frame: int, state: Any, rng: Any) -> Snapshot:
        snapshot = Snapshot.capture(logical_frame, state, rng)
        self._snapshots = [s for s in self._snapshots if s.logical_frame <= logical_frame]
        if self._snapshots and self._snapshots[-1].logical_frame == logical_frame:
            self._snapshots[-1] = snapshot
        else:
            self._snapshots.append(snapshot)
            self._snapshots.sort(key=lambda s: s.logical_frame)
        if len(self._snapshots) > self.max_frames:
            del self._snapshots[: len(self._snapshots) - self.max_frames]
        self.current_frame = max(self.current_frame or 0, logical_frame)
        return snapshot

    def get(self, logical_frame: int) -> Snapshot | None:
        for snapshot in self._snapshots:
            if snapshot.logical_frame == logical_frame:
                return snapshot
        return None

    def nearest_before(self, logical_frame: int) -> Snapshot | None:
        candidates = [s for s in self._snapshots if s.logical_frame <= logical_frame]
        return candidates[-1] if candidates else None

    def latest(self) -> Snapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def rollback(
        self,
        target_frame: int,
        *,
        rng: Any = None,
        max_compensation: int | None = None,
    ) -> RollbackResult:
        target = _require_int(target_frame, "target_frame", minimum=0)
        cap = self.max_compensation if max_compensation is None else _require_int(
            max_compensation, "max_compensation", minimum=0
        )

        if self.current_frame is not None and target > self.current_frame:
            self.failed_rollbacks += 1
            return RollbackResult(
                success=False,
                target_frame=target,
                restored_from=None,
                frames_rewound=0,
                state=None,
                rng=None,
                digest="",
                resync_required=True,
                reason="target_frame_is_in_the_future",
            )

        snapshot = self.nearest_before(target)
        if snapshot is None:
            self.failed_rollbacks += 1
            return RollbackResult(
                success=False,
                target_frame=target,
                restored_from=None,
                frames_rewound=0,
                state=None,
                rng=None,
                digest="",
                resync_required=True,
                reason="no_snapshot_at_or_before_target",
            )

        current = self.current_frame if self.current_frame is not None else snapshot.logical_frame
        # Compensation work is everything that must be re-simulated after
        # restoring the snapshot to get back to the present: the replay gap
        # from the snapshot up to ``target`` plus the discarded prediction from
        # ``target`` to ``current``.  Equivalently ``current - snapshot``.
        replay = max(0, target - snapshot.logical_frame)
        cost = max(0, current - snapshot.logical_frame)
        plan = compensation_cap(cost, max_compensation=cap)
        if plan.capped:
            self.failed_rollbacks += 1
            return RollbackResult(
                success=False,
                target_frame=target,
                restored_from=snapshot.logical_frame,
                frames_rewound=max(0, current - target),
                state=None,
                rng=None,
                digest=snapshot.digest,
                resync_required=True,
                reason=plan.reason,
            )

        if not snapshot.verify():
            self.failed_rollbacks += 1
            return RollbackResult(
                success=False,
                target_frame=target,
                restored_from=snapshot.logical_frame,
                frames_rewound=max(0, current - target),
                state=None,
                rng=None,
                digest=snapshot.digest,
                resync_required=True,
                reason="snapshot_corrupt",
            )

        restored_state = snapshot.restore_state()
        try:
            restored_rng = snapshot.restore_rng(rng)
        except LockstepError as exc:
            self.failed_rollbacks += 1
            return RollbackResult(
                success=False,
                target_frame=target,
                restored_from=snapshot.logical_frame,
                frames_rewound=max(0, current - target),
                state=restored_state,
                rng=None,
                digest=snapshot.digest,
                resync_required=True,
                reason=f"rng_restore_failed: {exc}",
            )

        # Snapshots after the restore point describe a timeline that is about
        # to be rewritten; keeping them would let a later rollback jump to a
        # state that never happened.
        self._snapshots = [s for s in self._snapshots if s.logical_frame <= snapshot.logical_frame]
        # The caller re-simulates the deterministic frames in
        # ``(snapshot.logical_frame, target]``, so the timeline resumes at
        # ``target``; ``replayed`` says how many frames that costs.
        self.current_frame = target
        self.rollbacks += 1
        return RollbackResult(
            success=True,
            target_frame=target,
            restored_from=snapshot.logical_frame,
            frames_rewound=max(0, current - target),
            state=restored_state,
            rng=restored_rng,
            digest=snapshot.digest,
            resync_required=False,
            reason="restored_with_rng",
            replayed=replay,
        )


def rollback(
    buffer: RollbackBuffer | int = 0,
    target_frame: int | None = None,
    *,
    max_compensation: int = 8,
) -> RollbackResult:
    """Convenience wrapper around :meth:`RollbackBuffer.rollback`.

    An integer first argument is treated as the target frame for an empty
    buffer, which returns a resync request rather than raising.
    """
    if isinstance(buffer, RollbackBuffer):
        if target_frame is None:
            raise LockstepError("target_frame is required")
        return buffer.rollback(target_frame, max_compensation=max_compensation)

    if target_frame is None:
        target_frame = _require_int(buffer, "target_frame", minimum=0)
    return RollbackBuffer(max_compensation=max_compensation).rollback(
        target_frame, max_compensation=max_compensation
    )


# ---------------------------------------------------------------------------
# Digests, validation and auditing
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """Convert a value into a JSON-serialisable, ordering-stable structure."""
    if isinstance(value, RNGState):
        return {"$rng": value.digest}
    if isinstance(value, Snapshot):
        return {"$snapshot": value.digest}
    if isinstance(value, FrameInput):
        return {
            "$input": [repr(value.player), value.server_frame, value.seq, _canonical(value.actions)],
        }
    if isinstance(value, random.Random):
        # A live RNG object is state; fold in its exact state so two states that
        # consumed different amounts of randomness never collide.
        return {"$random": state_digest(value.getstate())}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "nan"}
        if math.isinf(value):
            return {"$float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (bytes, bytearray)):
        return {"$bytes": bytes(value).hex()}
    if isinstance(value, Mapping):
        items = [
            [_canonical(key), _canonical(item)]
            for key, item in sorted(value.items(), key=lambda kv: _stable_key(kv[0]))
        ]
        return {"$map": items}
    if isinstance(value, (set, frozenset)):
        return {"$set": sorted((_canonical(item) for item in value), key=_stable_key)}
    if isinstance(value, Sequence):
        return [_canonical(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$dataclass": type(value).__qualname__,
            "fields": [[f.name, _canonical(getattr(value, f.name))] for f in fields(value)],
        }
    if hasattr(value, "__dict__"):
        return {
            "$object": f"{type(value).__module__}.{type(value).__qualname__}",
            "attrs": _canonical(vars(value)),
        }
    return {"$repr": repr(value)}


def state_digest(state: Any) -> str:
    """Stable SHA-256 digest of arbitrary simulation state."""
    payload = json.dumps(_canonical(state), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)
    digest: str = ""

    def __bool__(self) -> bool:
        return self.ok

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
            "digest": self.digest,
        }


def _cooldown_basis_ok(cooldowns: Iterable[Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for cooldown in cooldowns:
        if isinstance(cooldown, Cooldown):
            basis = cooldown.frame_basis
            started = cooldown.started_frame
        elif isinstance(cooldown, Mapping):
            basis = cooldown.get("frame_basis", "logical")
            started = cooldown.get("started_frame", 0)
        else:
            continue
        if basis != "logical":
            errors.append(f"cooldown started on {basis} frame ({started})")
    return (not errors), errors


def validation(
    target: Any = None,
    *,
    input_delay: int = 2,
    confirmed_frame: int | None = None,
    required_players: Iterable[Hashable] = (),
    capacity: int = 64,
) -> ValidationReport:
    """Validate lockstep invariants, returning a structured report.

    Detection (not exception) keeps this usable from the reconnect handshake
    and from telemetry.  ``target`` may be a :class:`FrameQueue`,
    :class:`RollbackBuffer`, a mapping/object describing client state, an
    iterable of :class:`FrameInput`, or an integer local frame.
    """
    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []

    queue: FrameQueue | None = None
    rollback_buffer: RollbackBuffer | None = None
    cooldowns: Iterable[Any] = ()
    watermarks: Mapping[Hashable, int] | None = None
    local_frame = 0
    basis_frame: int | None = None
    snapshots: Iterable[Snapshot] = ()

    if isinstance(target, FrameQueue):
        queue = target
    elif isinstance(target, RollbackBuffer):
        rollback_buffer = target
    elif isinstance(target, Snapshot):
        snapshots = (target,)
    elif isinstance(target, Mapping):
        local_frame = int(target.get("local_frame", 0))
        basis_frame = target.get("confirmed_frame", confirmed_frame)
        cooldowns = target.get("cooldowns", ())
        watermarks = target.get("watermarks")
        snapshots = target.get("snapshots", ())
        if "queue" in target and isinstance(target["queue"], FrameQueue):
            queue = target["queue"]
    elif isinstance(target, int) and not isinstance(target, bool):
        local_frame = target
    elif target is None:
        pass
    elif isinstance(target, Sequence):
        queue = FrameQueue(0, players=required_players, capacity=capacity)
        queue.extend(target)
    else:
        local_frame = int(getattr(target, "local_frame", 0))
        basis_frame = getattr(target, "confirmed_frame", confirmed_frame)
        cooldowns = getattr(target, "cooldowns", ())
        watermarks = getattr(target, "watermarks", None)
        snapshots = getattr(target, "snapshots", getattr(target, "_snapshots", ()))

    if queue is not None:
        checks["frame_floor_non_negative"] = queue.floor_frame >= 0
        checks["backlog_within_capacity"] = queue.backlog <= queue.capacity
        if queue.floor_frame < 0:
            errors.append("frame floor is negative")
        if queue.backlog > queue.capacity:
            errors.append("input backlog exceeds capacity")
        if queue.duplicate_policy == "latest" and queue.stats.duplicates:
            warnings.append("duplicate inputs were replaced by latest policy")

    if rollback_buffer is not None:
        snapshots = tuple(rollback_buffer._snapshots)
        checks["rollback_cap_configured"] = rollback_buffer.max_compensation > 0
        if rollback_buffer.failed_rollbacks:
            warnings.append(f"{rollback_buffer.failed_rollbacks} rollback(s) required resync")

    snapshot_tuple = tuple(snapshots)
    if snapshot_tuple:
        checks["snapshots_carry_rng"] = all(isinstance(s.rng, RNGState) for s in snapshot_tuple)
        checks["snapshots_verified"] = all(s.verify() for s in snapshot_tuple)
        if not checks["snapshots_carry_rng"]:
            errors.append("snapshot without RNG state cannot be rolled back deterministically")
        if not checks["snapshots_verified"]:
            errors.append("snapshot digest mismatch")

    basis_ok, basis_errors = _cooldown_basis_ok(cooldowns)
    checks["cooldowns_use_logical_frames"] = basis_ok
    errors.extend(basis_errors)

    logical = logical_frame(local_frame, input_delay=input_delay, confirmed_frame=basis_frame)
    checks["logical_frame_not_ahead_of_confirmed"] = (
        basis_frame is None or logical <= basis_frame
    )
    if basis_frame is not None and logical > basis_frame:
        errors.append("logical frame is ahead of confirmed server frame")

    if watermarks is not None:
        monotonic = all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in watermarks.values()
        )
        checks["watermarks_monotonic"] = monotonic
        if not monotonic:
            errors.append("watermark is not a non-negative sequence number")

    checks.setdefault("no_errors", not errors)
    checks["no_errors"] = not errors
    digest = state_digest(
        {
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "logical_frame": logical,
        }
    )
    return ValidationReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=checks,
        digest=digest,
    )


@dataclass(frozen=True, slots=True)
class AuditEvent:
    kind: str
    player: Hashable | None = None
    server_frame: int | None = None
    seq: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise LockstepError("audit event kind must be a non-empty string")
        if self.server_frame is not None:
            _require_int(self.server_frame, "server_frame", minimum=0)
        if self.seq is not None:
            _require_int(self.seq, "seq", minimum=0)


@dataclass(frozen=True, slots=True)
class AuditLog:
    events: tuple[AuditEvent, ...]
    counts: dict[str, int]
    watermarks: dict[Hashable, int]
    duplicates: int
    stale: int
    regressions: int
    ok: bool
    digest: str

    def __len__(self) -> int:
        return len(self.events)

    def findings(self) -> tuple[AuditEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.kind in {"duplicate", "stale", "regressed_seq", "cooldown_basis_render", "rng_drift"}
        )


def audit(
    events: Iterable[Any] | int = (),
    *,
    floor_frame: int = 0,
    watermarks: Mapping[Hashable, int] | None = None,
) -> AuditLog:
    """Audit an input stream for duplicate/stale/out-of-order delivery.

    ``events`` may contain :class:`FrameInput`, :class:`AuditEvent` or mappings
    with ``player``/``server_frame``/``seq`` keys.  An integer argument is
    treated as the frame floor, producing an empty log.
    """
    if isinstance(events, int) and not isinstance(events, bool):
        floor_frame = events
        events = ()

    dedupe = InputDeduplicator(floor_frame=floor_frame, watermarks=watermarks)

    recorded: list[AuditEvent] = []
    counts: dict[str, int] = {}
    for raw in events:
        if isinstance(raw, FrameInput):
            result = dedupe.accept(raw)
            event = AuditEvent(
                kind=result.reason,
                player=result.player,
                server_frame=result.server_frame,
                seq=result.seq,
                detail="accepted" if result.accepted else result.reason,
            )
        elif isinstance(raw, AuditEvent):
            # A pre-classified event contributes its own reason; sequences at
            # or below an accepted one are treated as duplicates so ``ok`` is
            # still meaningful for hand-built logs.
            event = raw
            if event.player is not None and event.seq is not None:
                current = dedupe.watermark(event.player)
                if event.kind == "accepted" and event.seq > current:
                    dedupe._watermarks[event.player] = event.seq
                elif event.kind == "accepted":
                    event = AuditEvent(
                        "duplicate", event.player, event.server_frame, event.seq, event.detail
                    )
        elif isinstance(raw, Mapping):
            player = raw.get("player")
            frame = raw.get("server_frame", 0)
            seq = raw.get("seq", 1)
            item = FrameInput(player, frame, seq, tuple(raw.get("actions", ())))
            result = dedupe.accept(item)
            event = AuditEvent(
                kind=result.reason,
                player=result.player,
                server_frame=result.server_frame,
                seq=result.seq,
            )
        else:
            raise LockstepError(f"cannot audit event of type {type(raw).__name__}")

        counts[event.kind] = counts.get(event.kind, 0) + 1
        recorded.append(event)

    duplicates = counts.get("duplicate", 0) + counts.get("regressed_seq", 0)
    stale = counts.get("stale", 0)
    digest = state_digest([_canonical(e) for e in recorded])
    return AuditLog(
        events=tuple(recorded),
        counts=counts,
        watermarks=dedupe.watermarks,
        duplicates=duplicates,
        stale=stale,
        regressions=counts.get("regressed_seq", 0),
        ok=duplicates == 0 and stale == 0,
        digest=digest,
    )


# ---------------------------------------------------------------------------
# Fallback, latency compensation and metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FallbackPlan:
    """Concrete steps for recovering when catch-up or rollback exceeds its cap."""

    resync_required: bool
    reason: str
    drop_inputs_before: int
    request_snapshot: bool
    clear_presentation: bool
    reset_rng: bool
    replay_from: int | None
    steps: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.resync_required


def fallback(
    reason: Any = "catchup_cap",
    *,
    frame_floor: int = 0,
    snapshot_available: bool = True,
    authoritative_frame: int | None = None,
) -> FallbackPlan:
    """Build a recovery plan that never replays more than the cap allows.

    ``reason`` may be a :class:`CatchupPlan`/:class:`CompensationPlan`, a
    string, or an integer frame floor.  The plan always resyncs rather than
    extending an unbounded replay.
    """
    drop_before = _require_int(frame_floor, "frame_floor", minimum=0)
    label = "fallback"

    if isinstance(reason, CatchupPlan):
        resync = reason.resync_required
        label = reason.reason
    elif isinstance(reason, CompensationPlan):
        resync = reason.resync_required
        label = reason.reason
    elif isinstance(reason, RollbackResult):
        resync = reason.resync_required
        label = reason.reason
    elif isinstance(reason, int) and not isinstance(reason, bool):
        drop_before = _require_int(reason, "frame_floor", minimum=0)
        resync = True
        label = "manual_resync"
    elif isinstance(reason, str):
        resync = True
        label = reason
    else:
        resync = True
        label = "unknown"

    steps: list[str] = []
    if resync:
        steps.append("stop_local_simulation")
        steps.append(f"drop_inputs_before:{drop_before}")
        if snapshot_available:
            steps.append("request_authoritative_snapshot")
        else:
            steps.append("await_server_snapshot")
        steps.append("clear_presentation_samples")
        steps.append("restore_rng_with_snapshot")
        steps.append("resume_from_snapshot")
    else:
        steps.append("continue_within_cap")

    replay_from = authoritative_frame if resync else None
    return FallbackPlan(
        resync_required=resync,
        reason=label,
        drop_inputs_before=drop_before,
        request_snapshot=resync and snapshot_available,
        clear_presentation=resync,
        reset_rng=resync,
        replay_from=replay_from,
        steps=tuple(steps),
    )


@dataclass(frozen=True, slots=True)
class LatencyPlan:
    """Input delay and buffer depth derived from measured latency."""

    latency_frames: int
    tick_ms: float
    input_delay: int
    buffer_frames: int
    max_catchup: int
    reason: str

    @property
    def latency_ms(self) -> float:
        return self.latency_frames * self.tick_ms


def latency_comp(
    latency: int | float = 0,
    *,
    tick_ms: float = 33.0,
    safety_frames: int = 1,
    max_delay: int = 8,
    unit: str | None = None,
) -> LatencyPlan:
    """Choose a fixed input delay from round-trip latency.

    ``latency`` is in frames by default; pass ``unit="ms"`` (or a float, which
    is assumed to be milliseconds) to convert.  The delay is capped so a bad
    connection degrades into capped catch-up rather than unbounded buffering.
    """
    tick = _require_number(tick_ms, "tick_ms", minimum=0.001)
    safety = _require_int(safety_frames, "safety_frames", minimum=0)
    cap = _require_int(max_delay, "max_delay", minimum=1)

    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise LockstepError("latency must be a number of frames or milliseconds")
    value = float(latency)
    if not math.isfinite(value) or value < 0:
        raise LockstepError("latency must be finite and non-negative")

    if unit is None:
        unit = "ms" if isinstance(latency, float) else "frames"
    if unit not in {"frames", "ms"}:
        raise LockstepError("unit must be 'frames' or 'ms'")

    if unit == "ms":
        frames = math.ceil(value / tick)
    else:
        frames = int(round(value))

    delay = min(cap, frames + safety)
    return LatencyPlan(
        latency_frames=max(0, frames),
        tick_ms=tick,
        input_delay=delay,
        buffer_frames=max(1, delay + 1),
        max_catchup=delay,
        reason=f"derived_from_latency_{unit}",
    )


@dataclass(slots=True)
class Metrics:
    """Counters for one client session, mergeable for telemetry."""

    frames_simulated: int = 0
    frames_rolled_back: int = 0
    inputs_accepted: int = 0
    inputs_duplicate: int = 0
    inputs_stale: int = 0
    catchup_steps: int = 0
    resyncs: int = 0
    presentation_rebuilds: int = 0
    max_backlog: int = 0
    latency_frames: int = 0

    def record_queue(self, stats: QueueStats) -> "Metrics":
        self.inputs_accepted += stats.accepted
        self.inputs_duplicate += stats.duplicates
        self.inputs_stale += stats.stale
        self.frames_simulated += stats.frames_consumed
        return self

    def record_catchup(self, plan: CatchupPlan) -> "Metrics":
        self.catchup_steps += plan.steps
        self.max_backlog = max(self.max_backlog, plan.requested)
        if plan.resync_required:
            self.resyncs += 1
        return self

    def record_rollback(self, result: RollbackResult) -> "Metrics":
        if result.success:
            # Re-simulated frames are work, not progress; count them and do not
            # subtract the rewind again from frames_simulated (it was never
            # added for the discarded prediction).
            self.frames_rolled_back += result.frames_rewound
        elif result.resync_required:
            self.resyncs += 1
        return self

    def record_rebuild(self, rebuild: PresentRebuild) -> "Metrics":
        self.presentation_rebuilds += 1
        return self

    def merge(self, other: "Metrics") -> "Metrics":
        for f in fields(self):
            if f.name == "max_backlog":
                setattr(self, f.name, max(getattr(self, f.name), getattr(other, f.name)))
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self

    def as_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __len__(self) -> int:
        return len(fields(self))


def metrics(source: Any = None, **counters: int) -> Metrics:
    """Build/merge a :class:`Metrics` snapshot.

    ``source`` may be another :class:`Metrics`, a mapping of counter names, a
    :class:`QueueStats`, a :class:`CatchupPlan`, a :class:`RollbackResult`, or
    an integer (ignored count, useful for contract tests).
    """
    result = Metrics()
    if source is None or (isinstance(source, int) and not isinstance(source, bool)):
        pass
    elif isinstance(source, Metrics):
        result.merge(source)
    elif isinstance(source, QueueStats):
        result.record_queue(source)
    elif isinstance(source, CatchupPlan):
        result.record_catchup(source)
    elif isinstance(source, RollbackResult):
        result.record_rollback(source)
    elif isinstance(source, Mapping):
        for name, value in source.items():
            if hasattr(result, str(name)):
                setattr(result, str(name), _require_int(value, str(name), minimum=0))
    elif isinstance(source, Sequence):
        for item in source:
            result.merge(metrics(item))
    else:
        raise LockstepError(f"cannot build metrics from {type(source).__name__}")

    for name, value in counters.items():
        if not hasattr(result, name):
            raise LockstepError(f"unknown metric {name!r}")
        setattr(result, name, _require_int(value, name, minimum=0))
    return result


# ---------------------------------------------------------------------------
# Deterministic input ordering and determinism checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InputOrder:
    ordered: tuple[FrameInput, ...]
    per_frame: tuple[tuple[int, tuple[FrameInput, ...]], ...]
    players: tuple[Hashable, ...]
    duplicates: tuple[FrameInput, ...]
    stale: tuple[FrameInput, ...]
    digest: str

    def __len__(self) -> int:
        return len(self.ordered)

    def frame_inputs(self, server_frame: int) -> tuple[FrameInput, ...]:
        for frame, items in self.per_frame:
            if frame == server_frame:
                return items
        return ()


def input_order(
    inputs: Iterable[FrameInput] | int = (),
    *,
    floor_frame: int = 0,
) -> InputOrder:
    """Order inputs deterministically by ``(server_frame, player, seq)``.

    Only *duplicates* (same player+frame, or a replayed player+seq) are
    removed; network reordering that delivers a later frame first does not
    make the earlier frame stale.  Player identity is ordered by a stable
    repr, not by hash, so every client produces the same order even for
    non-string player ids.
    """
    if isinstance(inputs, int) and not isinstance(inputs, bool):
        floor_frame = inputs
        inputs = ()

    seen: dict[tuple[Hashable, int], FrameInput] = {}
    seen_seq: dict[Hashable, set[int]] = {}
    duplicates: list[FrameInput] = []
    stale: list[FrameInput] = []

    for item in inputs:
        if not isinstance(item, FrameInput):
            raise LockstepError("input_order expects FrameInput items")
        if item.server_frame < floor_frame:
            stale.append(item)
            continue
        key = (item.player, item.server_frame)
        if key in seen or item.seq in seen_seq.get(item.player, ()):
            duplicates.append(item)
            continue
        seen[key] = item
        seen_seq.setdefault(item.player, set()).add(item.seq)

    ordered = tuple(sorted(seen.values(), key=lambda i: (i.server_frame, _stable_key(i.player), i.seq)))
    grouped: dict[int, list[FrameInput]] = {}
    for item in ordered:
        grouped.setdefault(item.server_frame, []).append(item)

    per_frame = tuple((frame, tuple(grouped[frame])) for frame in sorted(grouped))
    players = tuple(sorted({item.player for item in ordered}, key=_stable_key))
    digest = state_digest(
        [[item.server_frame, repr(item.player), item.seq, _canonical(item.actions)] for item in ordered]
    )
    return InputOrder(
        ordered=ordered,
        per_frame=per_frame,
        players=players,
        duplicates=tuple(duplicates),
        stale=tuple(stale),
        digest=digest,
    )


@dataclass(frozen=True, slots=True)
class DeterminismReport:
    ok: bool
    digest: str
    reference: str
    divergences: tuple[str, ...] = ()
    checked: int = 0

    def __bool__(self) -> bool:
        return self.ok


def determinism(
    state: Any = None,
    *,
    reference: Any = _MISSING,
    reference_rng: Any = _MISSING,
    rng: Any = None,
    expected_digest: str | None = None,
) -> DeterminismReport:
    """Compare a state against a reference and report divergence.

    ``state`` may be a mapping of ``name -> value``; every entry is compared
    against the same key in ``reference``.  When ``rng`` is supplied its state
    is folded into the digest; pass ``reference_rng`` to require that the RNG
    state matches too (this is how a rollback test catches an RNG that was not
    rewound).  With neither a reference nor an expected digest the result is a
    baseline.
    """
    payload: Any = state
    if rng is not None:
        payload = {"state": state, "rng": rng_snapshot(rng).digest}

    digest = state_digest(payload)
    divergences: list[str] = []
    checked = 0

    if reference is not _MISSING:
        rng_checked = reference_rng is not _MISSING
        if rng_checked and rng is None:
            raise LockstepError("reference_rng requires rng")

        ref_payload: Any = reference
        if rng_checked:
            ref_payload = {"state": reference, "rng": rng_snapshot(reference_rng).digest}
        ref_digest = state_digest(ref_payload)

        if isinstance(state, Mapping) and isinstance(reference, Mapping):
            for key in sorted(set(state) | set(reference), key=_stable_key):
                checked += 1
                if key not in state:
                    divergences.append(f"missing:{key!r}")
                elif key not in reference:
                    divergences.append(f"extra:{key!r}")
                elif state_digest(state[key]) != state_digest(reference[key]):
                    divergences.append(f"differs:{key!r}")
        else:
            checked = 1
            if state_digest(state) != state_digest(reference):
                divergences.append("state_differs")

        if rng_checked:
            checked += 1
            if rng_snapshot(rng).digest != rng_snapshot(reference_rng).digest:
                divergences.append("rng_differs")

        return DeterminismReport(
            ok=not divergences,
            digest=digest,
            reference=ref_digest,
            divergences=tuple(divergences),
            checked=checked,
        )

    if expected_digest is not None:
        checked = 1
        if digest != expected_digest:
            divergences.append("digest_mismatch")
        return DeterminismReport(
            ok=not divergences,
            digest=digest,
            reference=expected_digest,
            divergences=tuple(divergences),
            checked=checked,
        )

    return DeterminismReport(ok=True, digest=digest, reference=digest, checked=1)
