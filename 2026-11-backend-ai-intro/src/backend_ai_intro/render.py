"""Render a recorded scheduler experiment without rerunning or predicting it.

The graphics show manager-side resource accounting. They never infer container
startup from a successful placement. Pillow's bundled Aileron font makes the
output independent of the fonts installed on the host operating system.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1200
HEIGHT = 760
SCALE = 2
BACKGROUND = "#ffffff"
INK = "#202833"
MUTED = "#596574"
LINE = "#d9dfe6"
FREE = ("#e9f6ed", "#43865b")
PALETTE = (
    ("#eef0f3", "#75808d"),
    ("#e8e3fb", "#7452bb"),
    ("#dceefc", "#267ab4"),
    ("#fff0d9", "#ad7527"),
    ("#f8e3ed", "#a65076"),
    ("#e0f4f2", "#357f78"),
)
EVENT_LABELS = {
    "initial_state": "Supplied initial state",
    "scheduling_pass": "Observed scheduling pass",
    "fixture_release": "Supplied release input",
}
STATUSES = {"allocated", "pending", "released"}
OUTCOMES = {"scheduled", "failed", "skipped"}


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _mode(recording: Mapping[str, Any]) -> str | None:
    data = _mapping(recording, "recording")
    experiment = _mapping(data.get("experiment", {}), "experiment")
    return experiment.get("mode")


def validate_recording(recording: Mapping[str, Any]) -> None:
    """Reject malformed data before drawing any files.

    This checks the visualization's input contract and accounting consistency;
    the experiment runner separately verifies the expected scheduler behavior.
    """
    data = _mapping(recording, "recording")
    if _mode(data) == "live_cpu":
        _live_checkpoints(data)
        return
    if _integer(data.get("schema_version"), "schema_version") != 1:
        raise ValueError("Only recording schema_version 1 is supported")
    events = _list(data.get("events"), "events")
    if not 1 <= len(events) <= 30:
        raise ValueError("events must contain between 1 and 30 entries")
    previous_sequence = -1
    previous_ids: set[str] | None = None
    previous_capacity: int | None = None
    for index, raw_event in enumerate(events):
        prefix = f"events[{index}]"
        event = _mapping(raw_event, prefix)
        sequence = _integer(event.get("sequence"), f"{prefix}.sequence")
        if sequence <= previous_sequence:
            raise ValueError("Event sequence numbers must be strictly increasing")
        previous_sequence = sequence
        if _text(event.get("kind"), f"{prefix}.kind") not in EVENT_LABELS:
            raise ValueError(f"{prefix}.kind is unsupported")
        action = _mapping(event.get("action"), f"{prefix}.action")
        _text(action.get("actor"), f"{prefix}.action.actor")
        _text(action.get("description"), f"{prefix}.action.description")
        snapshot = _mapping(event.get("snapshot"), f"{prefix}.snapshot")
        node = _mapping(snapshot.get("node"), f"{prefix}.snapshot.node")
        _text(node.get("id"), f"{prefix}.snapshot.node.id")
        capacity = _integer(node.get("capacity"), "node.capacity", minimum=1)
        if capacity > 8:
            raise ValueError("The diagram supports at most eight CPU slots")
        allocated = _integer(node.get("allocated"), "node.allocated")
        free = _integer(node.get("free"), "node.free")
        if free + allocated != capacity:
            raise ValueError("node.free + node.allocated must equal node.capacity")
        if previous_capacity is not None and capacity != previous_capacity:
            raise ValueError("Worker capacity must stay fixed across the recording")
        previous_capacity = capacity
        sessions = _list(snapshot.get("sessions"), f"{prefix}.snapshot.sessions")
        if not 1 <= len(sessions) <= 4:
            raise ValueError("The diagram supports between one and four sessions")
        known: dict[str, Mapping[str, Any]] = {}
        for raw_session in sessions:
            session = _mapping(raw_session, "session")
            session_id = _text(session.get("id"), "session.id")
            if session_id in known:
                raise ValueError(f"Duplicate session id {session_id!r}")
            known[session_id] = session
            _text(session.get("label"), "session.label")
            _integer(session.get("requested"), "session.requested", minimum=1)
            if _text(session.get("status"), "session.status") not in STATUSES:
                raise ValueError("Session status must be allocated, pending, or released")
            decision = session.get("last_decision")
            if decision is not None:
                decision = _mapping(decision, "session.last_decision")
                if _text(decision.get("outcome"), "session.last_decision.outcome") not in OUTCOMES:
                    raise ValueError("Unsupported session.last_decision.outcome")
                _text(decision.get("reason"), "session.last_decision.reason")
        if previous_ids is not None and set(known) != previous_ids:
            raise ValueError("Session identities must stay fixed across the recording")
        previous_ids = set(known)
        allocations = _list(node.get("allocations"), "node.allocations")
        assigned: dict[str, int] = {}
        for raw_allocation in allocations:
            allocation = _mapping(raw_allocation, "allocation")
            session_id = _text(allocation.get("session_id"), "allocation.session_id")
            if session_id not in known or session_id in assigned:
                raise ValueError("Allocations must reference unique, known sessions")
            slots = _integer(allocation.get("slots"), "allocation.slots", minimum=1)
            if known[session_id]["status"] != "allocated":
                raise ValueError("Every allocation must belong to an allocated session")
            if slots != known[session_id]["requested"]:
                raise ValueError("Allocated slots must equal the session's requested slots")
            assigned[session_id] = slots
        if sum(assigned.values()) != allocated:
            raise ValueError("Allocation amounts must sum to node.allocated")
        if set(assigned) != {sid for sid, s in known.items() if s["status"] == "allocated"}:
            raise ValueError("Each allocated session must have an allocation")
        pending = _list(snapshot.get("pending_order"), "snapshot.pending_order")
        if any(not isinstance(sid, str) for sid in pending):
            raise ValueError("Pending session ids must be strings")
        if len(pending) != len(set(pending)):
            raise ValueError("Pending order must not contain duplicate sessions")
        if set(pending) != {sid for sid, s in known.items() if s["status"] == "pending"}:
            raise ValueError("Pending order must contain exactly the pending sessions")
        decided: set[str] = set()
        for raw_decision in _list(event.get("decisions"), "event.decisions"):
            decision = _mapping(raw_decision, "decision")
            session_id = _text(decision.get("session_id"), "decision.session_id")
            if session_id not in known or session_id in decided:
                raise ValueError("Decisions must reference known sessions")
            decided.add(session_id)
            if _text(decision.get("outcome"), "decision.outcome") not in OUTCOMES:
                raise ValueError("Unsupported decision.outcome")
            _text(decision.get("reason"), "decision.reason")


@lru_cache(maxsize=24)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size * SCALE)


class _Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width * SCALE, height * SCALE), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)

    def text(
        self, x: float, y: float, text: str, size: int = 22, color: str = INK, anchor: str = "la"
    ) -> None:
        self.draw.text((x * SCALE, y * SCALE), text, font=_font(size), fill=color, anchor=anchor)

    def rect(
        self,
        xy: tuple[float, float, float, float],
        fill: str,
        outline: str | None = None,
        radius: int = 10,
    ) -> None:
        self.draw.rounded_rectangle(
            tuple(v * SCALE for v in xy),
            radius=radius * SCALE,
            fill=fill,
            outline=outline,
            width=SCALE,
        )

    def line(self, x: float, y: float, end: float) -> None:
        self.draw.line((x * SCALE, y * SCALE, end * SCALE, y * SCALE), fill=LINE, width=SCALE)

    def wrapped(
        self,
        x: float,
        y: float,
        text: str,
        width: float,
        *,
        size: int = 22,
        color: str = INK,
        lines: int = 2,
        leading: int = 29,
    ) -> None:
        words = text.split()
        output: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if self.draw.textlength(candidate, font=_font(size)) > width * SCALE and current:
                output.append(current)
                current = word
            else:
                current = candidate
        if current:
            output.append(current)
        if len(output) > lines:
            output = output[:lines]
            output[-1] += "..."
        for index, line in enumerate(output):
            # Also handle a single long identifier or an ellipsis at the edge.
            while line and self.draw.textlength(line, font=_font(size)) > width * SCALE:
                line = line[:-4] + "..." if len(line) > 4 else ""
            self.text(x, y + index * leading, line, size, color)

    def finish(self) -> Image.Image:
        return self.image.resize((self.width, self.height), Image.Resampling.LANCZOS)


def _session_colors(events: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    return {
        session["id"]: PALETTE[index % len(PALETTE)]
        for index, session in enumerate(events[0]["snapshot"]["sessions"])
    }


def _owners(event: Mapping[str, Any]) -> list[str | None]:
    node = event["snapshot"]["node"]
    return [a["session_id"] for a in node["allocations"] for _ in range(a["slots"])] + [
        None
    ] * node["free"]


def _status(session: Mapping[str, Any], event_kind: str) -> str:
    if session["status"] == "pending" and event_kind == "fixture_release":
        return "Pending / awaiting next pass"
    if session["status"] == "pending" and session["last_decision"] is not None:
        outcome = session["last_decision"]["outcome"]
        if outcome == "skipped":
            return "Pending / last pass skipped"
        if outcome == "failed":
            return "Pending / last check failed"
    return session["status"].capitalize()


def _draw_slots(
    canvas: _Canvas,
    event: Mapping[str, Any],
    x: float,
    y: float,
    width: float,
    colors: Mapping[str, tuple[str, str]],
    *,
    small: bool = False,
) -> None:
    labels = {s["id"]: s["label"] for s in event["snapshot"]["sessions"]}
    owners = _owners(event)
    gap = 10 if small else 14
    columns = min(len(owners), 4)
    cell = (width - gap * (columns - 1)) / columns
    for index, owner in enumerate(owners):
        xx = x + (index % columns) * (cell + gap)
        yy = y + (index // columns) * (69 if small else 102)
        fill, edge = FREE if owner is None else colors[owner]
        canvas.rect((xx, yy, xx + cell, yy + (54 if small else 82)), fill, edge)
        text = "Free" if owner is None else labels[owner]
        size = 20 if small else 24
        canvas.wrapped(xx + 9, yy + (17 if small else 26), text, cell - 18, size=size, lines=1)


def _draw_frame(
    event: Mapping[str, Any], index: int, total: int, colors: Mapping[str, tuple[str, str]]
) -> Image.Image:
    canvas = _Canvas(WIDTH, HEIGHT)
    canvas.text(40, 27, "Backend.AI scheduling experiment", 35)
    canvas.text(
        40, 79, "Real scheduler decisions  /  Synthetic CPU inventory  /  No containers", 21, MUTED
    )
    canvas.text(1160, 29, f"{index + 1} / {total}", 25, MUTED, anchor="ra")
    canvas.line(40, 120, 1160)
    canvas.text(40, 139, EVENT_LABELS[event["kind"]].upper(), 18, MUTED)
    canvas.wrapped(40, 172, event["action"]["description"], 1120, size=26, lines=2, leading=32)

    snapshot = event["snapshot"]
    node = snapshot["node"]
    canvas.text(40, 259, "SESSION", 18, MUTED)
    canvas.text(233, 259, "CPU", 18, MUTED)
    canvas.text(300, 259, "STATE / LAST DECISION", 18, MUTED)
    canvas.text(585, 259, "WORKER ALLOCATIONS", 18, MUTED)
    canvas.text(1160, 259, f"{node['free']} / {node['capacity']} free", 21, INK, anchor="ra")
    sessions = snapshot["sessions"]
    row_height = 54 if len(sessions) > 3 else 66
    for row, session in enumerate(sessions):
        y = 301 + row * row_height
        canvas.rect((40, y, 47, y + 39), colors[session["id"]][1], radius=3)
        canvas.wrapped(61, y + 8, session["label"], 157, size=23, lines=1)
        canvas.text(240, y + 8, str(session["requested"]), 23)
        canvas.wrapped(
            300, y + 9, _status(session, event["kind"]), 260, size=20, lines=2, leading=23
        )
        canvas.line(61, y + row_height - 7, 543)
    _draw_slots(canvas, event, 585, 309, 575, colors)
    slot_rows = (node["capacity"] + 3) // 4
    if slot_rows == 1:
        canvas.text(585, 418, "Each square is one synthetic CPU slot.", 20, MUTED)
        canvas.text(585, 448, "Allocated does not mean running.", 20, MUTED)

    canvas.line(40, 529, 1160)
    canvas.text(
        40, 548, "RECORDED DECISIONS" if event["decisions"] else "EXPERIMENT INPUT", 18, MUTED
    )
    labels = {s["id"]: s["label"] for s in sessions}
    if event["decisions"]:
        compact = len(event["decisions"]) > 2
        for row, decision in enumerate(event["decisions"]):
            y = 580 + row * (30 if compact else 57)
            canvas.wrapped(
                40,
                y,
                f"{labels[decision['session_id']]} / {decision['outcome']}",
                245,
                size=18 if compact else 22,
                lines=1,
            )
            canvas.wrapped(
                300,
                y,
                decision["reason"],
                860,
                size=18 if compact else 20,
                lines=1 if compact else 2,
                leading=24,
            )
    else:
        explanation = (
            "Starting allocation supplied by the experiment."
            if event["kind"] == "initial_state"
            else (
                "Resource release supplied by the experiment; "
                "no container termination was observed."
            )
        )
        canvas.wrapped(40, 583, explanation, 1120, size=23, lines=2, leading=30)
    canvas.text(
        40,
        722,
        "Frames follow the saved event order. Playback duration is not measured scheduler latency.",
        18,
        MUTED,
    )
    return canvas.finish()


def _draw_storyboard(
    events: list[dict[str, Any]], colors: Mapping[str, tuple[str, str]]
) -> Image.Image:
    columns = 2
    panel_height = 414 if events[0]["snapshot"]["node"]["capacity"] > 4 else 354
    rows = (len(events) + columns - 1) // columns
    canvas = _Canvas(1500, 145 + rows * panel_height)
    canvas.text(40, 25, "Backend.AI scheduler · recorded sequence", 34)
    canvas.text(
        40,
        78,
        "Real scheduler decisions · Synthetic CPU inventory · "
        "Releases are supplied inputs · No containers",
        22,
        MUTED,
    )
    for index, event in enumerate(events):
        x = 40 + (index % columns) * 740
        y = 139 + (index // columns) * panel_height
        canvas.text(x, y, f"{index + 1:02d}  {EVENT_LABELS[event['kind']]}", 24)
        canvas.wrapped(x, y + 40, event["action"]["description"], 660, size=20, lines=2, leading=26)
        _draw_slots(canvas, event, x, y + 108, 660, colors, small=True)
        sessions_y = y + (183 if event["snapshot"]["node"]["capacity"] <= 4 else 239)
        for row, session in enumerate(event["snapshot"]["sessions"]):
            # Compact statuses preserve failed versus not-attempted decisions.
            canvas.wrapped(
                x,
                sessions_y + row * 32,
                f"{session['label']} ({session['requested']} CPU)  ·  "
                f"{_status(session, event['kind'])}",
                660,
                size=20,
                lines=1,
            )
        canvas.line(x, y + panel_height - 22, x + 660)
    return canvas.finish()


def render_recording(recording: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write a GIF, static storyboard, and representative still from saved events.

    No experiment code is imported or executed here. Existing named outputs are
    replaced only after their replacements have been encoded successfully.
    """
    if _mode(recording) == "live_cpu":
        return render_live_recording(recording, output_dir)
    validate_recording(recording)
    events = recording["events"]
    colors = _session_colors(events)
    frames = [_draw_frame(event, i, len(events), colors) for i, event in enumerate(events)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gif": output_dir / "scheduler.gif",
        "storyboard": output_dir / "storyboard.png",
        "keyframe": output_dir / "keyframe.png",
    }
    durations = [5000 if e["kind"] == "scheduling_pass" else 3500 for e in events]
    temporary = {name: path.with_name(f".{path.name}.tmp") for name, path in paths.items()}
    try:
        frames[0].save(
            temporary["gif"],
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )
        _draw_storyboard(events, colors).save(temporary["storyboard"], format="PNG")
        selected = next((i for i, event in enumerate(events) if event["decisions"]), 0)
        frames[selected].save(temporary["keyframe"], format="PNG")
        for name, path in paths.items():
            temporary[name].replace(path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return paths


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative number")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{label} must be a nonnegative number") from error
    if not number.is_finite() or number < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _number(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _elapsed(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a nonnegative number")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _live_capacity(sample: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    agent = _mapping(sample.get("agent"), "sample.agent")
    available = _mapping(agent.get("available_slots"), "agent.available_slots")
    occupied = _mapping(agent.get("occupied_slots"), "agent.occupied_slots")
    capacity = _decimal(available.get("cpu"), "agent.available_slots.cpu")
    used = _decimal(occupied.get("cpu", 0), "agent.occupied_slots.cpu")
    if capacity <= 0 or capacity > 8:
        raise ValueError("The live diagram supports positive CPU capacity up to eight")
    return capacity, used


def _live_checkpoints(
    recording: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Validate live observations without treating separate API reads as atomic.

    In particular, session occupancy is not required to sum to the agent's
    occupancy. Each value is displayed as reported by its own API response.
    """
    if _integer(recording.get("schema_version"), "schema_version") != 1:
        raise ValueError("Only recording schema_version 1 is supported")
    for raw_action in _list(recording.get("actions", []), "actions"):
        action = _mapping(raw_action, "action")
        _text(action.get("operation"), "action.operation")
        _text(action.get("session"), "action.session")
        _elapsed(action.get("elapsed_seconds"), "action.elapsed_seconds")
    samples = _list(recording.get("samples"), "samples")
    if not samples:
        raise ValueError("A live recording must contain API samples")
    by_sequence: dict[int, Mapping[str, Any]] = {}
    previous = -1
    for raw_sample in samples:
        sample = _mapping(raw_sample, "sample")
        sequence = _integer(sample.get("sequence"), "sample.sequence")
        if sequence <= previous:
            raise ValueError("Live sample sequence numbers must be strictly increasing")
        previous = sequence
        by_sequence[sequence] = sample
    checkpoints = _list(recording.get("checkpoints"), "checkpoints")
    if not 1 <= len(checkpoints) <= 30:
        raise ValueError("A live diagram requires between one and thirty checkpoints")
    selected = []
    previous = -1
    for raw_checkpoint in checkpoints:
        checkpoint = _mapping(raw_checkpoint, "checkpoint")
        _text(checkpoint.get("label"), "checkpoint.label")
        sequence = _integer(checkpoint.get("sample_sequence"), "checkpoint.sample_sequence")
        if sequence <= previous or sequence not in by_sequence:
            raise ValueError("Checkpoints must reference ordered, distinct observed samples")
        previous = sequence
        sample = by_sequence[sequence]
        _elapsed(sample.get("elapsed_seconds"), "sample.elapsed_seconds")
        _live_capacity(sample)
        sessions = _list(sample.get("sessions"), "sample.sessions")
        if not 1 <= len(sessions) <= 4:
            raise ValueError("A live checkpoint must contain between one and four sessions")
        seen = set()
        for raw_session in sessions:
            session = _mapping(raw_session, "session")
            session_id = _text(session.get("id"), "session.id")
            if session_id in seen:
                raise ValueError("Live session ids must be unique within a sample")
            seen.add(session_id)
            _text(session.get("label"), "session.label")
            _text(session.get("status"), "session.status")
            _decimal(session.get("requested_cpu"), "session.requested_cpu")
        summary = checkpoint.get("decision_summary")
        if summary is not None:
            _text(summary, "checkpoint.decision_summary")
        selected.append((checkpoint, sample))
    return selected


def _draw_live_slots(
    canvas: _Canvas,
    sample: Mapping[str, Any],
    x: float,
    y: float,
    width: float,
    *,
    small: bool = False,
) -> None:
    capacity, used = _live_capacity(sample)
    count = int(capacity.to_integral_value(rounding="ROUND_CEILING"))
    columns = min(count, 4)
    gap = 10 if small else 14
    cell = (width - gap * (columns - 1)) / columns
    height = 54 if small else 82
    for index in range(count):
        xx = x + (index % columns) * (cell + gap)
        yy = y + (index // columns) * (height + 14)
        amount = min(Decimal(1), max(Decimal(0), used - index))
        canvas.rect((xx, yy, xx + cell, yy + height), FREE[0], FREE[1])
        if amount:
            canvas.rect(
                (xx, yy, xx + cell * float(amount), yy + height), PALETTE[2][0], PALETTE[2][1]
            )
        label = "Used" if amount == 1 else "Not reported used" if amount == 0 else "Partial"
        top = 7 if small and amount == 0 else 17 if small else 18 if amount == 0 else 28
        canvas.wrapped(
            xx + 9,
            yy + top,
            label,
            cell - 18,
            size=17 if small else 18 if amount == 0 else 21,
            lines=2 if amount == 0 else 1,
            leading=20 if small else 23,
        )


def _draw_live_frame(
    checkpoint: Mapping[str, Any],
    sample: Mapping[str, Any],
    index: int,
    total: int,
    action_label: str = "Observed checkpoint",
) -> Image.Image:
    canvas = _Canvas(WIDTH, HEIGHT)
    canvas.text(40, 27, "Backend.AI session lifecycle", 35)
    canvas.text(40, 79, "Live CPU-only installation  /  Recorded API observations", 21, MUTED)
    canvas.text(1160, 29, f"{index + 1} / {total}", 25, MUTED, anchor="ra")
    canvas.line(40, 120, 1160)
    canvas.wrapped(40, 139, action_label.upper(), 900, size=18, lines=1)
    canvas.text(1160, 139, f"t + {sample['elapsed_seconds']:.1f} s", 20, MUTED, anchor="ra")
    canvas.wrapped(40, 172, checkpoint["label"], 1120, size=28, lines=2, leading=32)
    canvas.text(40, 259, "SESSION", 18, MUTED)
    canvas.text(222, 259, "CPU REQUEST", 18, MUTED)
    canvas.text(380, 259, "API STATUS", 18, MUTED)
    capacity, used = _live_capacity(sample)
    canvas.text(600, 259, "AGENT-REPORTED USED CPU", 18, MUTED)
    canvas.text(1160, 259, f"{_number(used)} / {_number(capacity)} used", 20, INK, anchor="ra")
    row_height = 54 if len(sample["sessions"]) > 3 else 66
    for row, session in enumerate(sample["sessions"]):
        y = 301 + row * row_height
        canvas.wrapped(40, y + 9, session["label"], 164, size=23, lines=1)
        requested = _number(_decimal(session["requested_cpu"], "session.requested_cpu"))
        canvas.text(250, y + 9, requested, 23)
        canvas.wrapped(380, y + 9, session["status"], 198, size=20, lines=2, leading=23)
        canvas.line(40, y + row_height - 7, 570)
    _draw_live_slots(canvas, sample, 600, 309, 560)
    if capacity <= 4:
        unused = capacity - used
        canvas.text(600, 418, f"Capacity minus used = {_number(unused)} CPU slots", 19, MUTED)
        canvas.text(600, 448, "Reservations are not shown in this counter.", 19, MUTED)
    canvas.line(40, 529, 1160)
    canvas.text(40, 548, "OBSERVATION", 18, MUTED)
    summary = (
        checkpoint.get("decision_summary")
        or "Session states and agent counters are reported by the APIs."
    )
    canvas.wrapped(40, 582, summary, 1120, size=23, lines=3, leading=31)
    canvas.text(
        40, 696, "Agent used excludes reservations. Hardware activity is not measured.", 18, MUTED
    )
    canvas.text(
        40,
        728,
        "API reads are not atomic. Elapsed time is recorded; GIF durations are for readability.",
        18,
        MUTED,
    )
    return canvas.finish()


def _draw_live_storyboard(
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> Image.Image:
    extended = any(_live_capacity(sample)[0] > 4 for _, sample in selected)
    panel_height = 464 if extended else 424
    canvas = _Canvas(1500, 145 + ((len(selected) + 1) // 2) * panel_height)
    canvas.text(40, 25, "Backend.AI · observed session lifecycle", 34)
    canvas.text(
        40,
        78,
        "Live CPU-only installation · Agent used excludes reservations · "
        "CPU activity is not measured",
        22,
        MUTED,
    )
    for index, (checkpoint, sample) in enumerate(selected):
        x, y = 40 + (index % 2) * 740, 139 + (index // 2) * panel_height
        canvas.wrapped(
            x, y, f"{index + 1:02d}  {checkpoint['label']}", 520, size=23, lines=2, leading=29
        )
        canvas.text(x + 660, y + 2, f"+{sample['elapsed_seconds']:.1f} s", 20, MUTED, anchor="ra")
        summary = checkpoint.get("decision_summary", "")
        canvas.wrapped(x, y + 59, summary, 660, size=18, lines=2, leading=23)
        _draw_live_slots(canvas, sample, x, y + 124, 660, small=True)
        capacity, used = _live_capacity(sample)
        start = y + (264 if extended else 204)
        canvas.text(
            x, start, f"Agent-reported used CPU  {_number(used)} / {_number(capacity)}", 20, MUTED
        )
        for row, session in enumerate(sample["sessions"]):
            cpu = _number(_decimal(session["requested_cpu"], "session.requested_cpu"))
            canvas.wrapped(
                x,
                start + 39 + row * 31,
                f"{session['label']} ({cpu} CPU)  ·  {session['status']}",
                660,
                size=21,
                lines=1,
            )
        canvas.line(x, y + panel_height - 22, x + 660)
    return canvas.finish()


def render_live_recording(recording: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Render selected real API samples without synthesizing intermediate states."""
    selected = _live_checkpoints(recording)
    actions = _list(recording.get("actions", []), "actions")
    frames = []
    for index, (checkpoint, sample) in enumerate(selected):
        preceding = [
            action
            for action in actions
            if action.get("operation") in {"create_requested", "termination_requested"}
            and action.get("elapsed_seconds", float("inf")) <= sample["elapsed_seconds"]
        ]
        action_label = "Observed checkpoint"
        if preceding:
            action = max(preceding, key=lambda item: item["elapsed_seconds"])
            role = str(action.get("session", "session"))
            labels = {session["id"]: session["label"] for session in sample["sessions"]}
            label = labels.get(role, role)
            action_label = (
                f"Operator requested creation of {label}"
                if action["operation"] == "create_requested"
                else f"Operator requested termination of {label}"
            )
        frames.append(_draw_live_frame(checkpoint, sample, index, len(selected), action_label))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gif": output_dir / "scheduler.gif",
        "storyboard": output_dir / "storyboard.png",
        "keyframe": output_dir / "keyframe.png",
    }
    temporary = {name: path.with_name(f".{path.name}.tmp") for name, path in paths.items()}
    try:
        frames[0].save(
            temporary["gif"],
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=4500,
            loop=0,
            disposal=2,
            optimize=False,
        )
        _draw_live_storyboard(selected).save(temporary["storyboard"], format="PNG")
        frames[min(1, len(frames) - 1)].save(temporary["keyframe"], format="PNG")
        for name, path in paths.items():
            temporary[name].replace(path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return paths
