"""Behavioral checks against real scheduler output, without mocking scheduling."""

import asyncio
import json
import re

import pytest

from backend_ai_intro.experiment import run_experiment
from backend_ai_intro.upstream_adapter import ExperimentError, FixtureState, schedule_pass


def outcomes(observation):
    return {item["session_id"]: item["outcome"] for item in observation["decisions"]}


def controls_by_name(recording):
    return {control["name"]: control for control in recording["controls"]}


def test_capacity_failure_skips_bob_then_release_allows_progress(recording):
    passes = [event for event in recording["events"] if event["kind"] == "scheduling_pass"]

    assert [outcomes(event) for event in passes] == [
        {"alice": "failed", "bob": "skipped"},
        {"alice": "scheduled", "bob": "failed"},
        {"bob": "scheduled"},
    ]
    assert [event["snapshot"]["node"]["free"] for event in passes] == [2, 0, 3]
    assert [event["snapshot"]["pending_order"] for event in passes] == [
        ["alice", "bob"],
        ["bob"],
        [],
    ]


def test_every_snapshot_conserves_resources_and_session_states(recording):
    snapshots = [event["snapshot"] for event in recording["events"]]
    for control in recording["controls"]:
        observations = control.get("observations", [control.get("observation")])
        snapshots.extend(item["snapshot"] for item in observations)

    for snapshot in snapshots:
        node = snapshot["node"]
        sessions = {session["id"]: session for session in snapshot["sessions"]}
        allocated = {item["session_id"]: item["slots"] for item in node["allocations"]}
        pending = set(snapshot["pending_order"])
        assert len(pending) == len(snapshot["pending_order"])
        assert sum(allocated.values()) == node["allocated"]
        assert node["allocated"] + node["free"] == node["capacity"]
        assert 0 <= node["allocated"] <= node["capacity"]
        assert not set(allocated) & pending
        for name, session in sessions.items():
            assert session["status"] in {"allocated", "pending", "released"}
            assert (session["status"] == "allocated") == (name in allocated)
            assert (session["status"] == "pending") == (name in pending)
            if name in allocated:
                assert allocated[name] == session["requested"]


def test_release_inputs_remove_one_allocation_without_scheduling(recording):
    events = recording["events"]
    for index, event in enumerate(events):
        if event["kind"] != "fixture_release":
            continue
        previous = events[index - 1]["snapshot"]
        current = event["snapshot"]
        before = {item["session_id"]: item["slots"] for item in previous["node"]["allocations"]}
        after = {item["session_id"]: item["slots"] for item in current["node"]["allocations"]}
        removed = before.keys() - after.keys()
        assert len(removed) == 1
        assert after.items() <= before.items()
        assert current["node"]["free"] - previous["node"]["free"] == before[removed.pop()]
        assert current["pending_order"] == previous["pending_order"]
        assert event["decisions"] == []
        assert event["action"]["actor"] == "fixture"


def test_bob_first_control_changes_the_allocation(recording):
    control = controls_by_name(recording)["bob_first"]["observation"]
    assert outcomes(control) == {"bob": "scheduled", "alice": "failed"}
    assert control["snapshot"]["node"]["free"] == 1
    assert control["snapshot"]["pending_order"] == ["alice"]


def test_quota_rejection_occurs_before_placement_and_does_not_skip_bob(recording):
    control = controls_by_name(recording)["quota_does_not_block_bob"]["observation"]
    assert outcomes(control) == {"alice": "failed", "bob": "scheduled"}
    phases = control["upstream_records"]["alice"]["phases"]
    assert any(phase["name"] == "validation" and phase["status"] == "failed" for phase in phases)
    assert not any(phase["name"] == "agent_selection" for phase in phases)


def test_unchanged_retry_does_not_consume_capacity(recording):
    first, second = controls_by_name(recording)["unchanged_retry"]["observations"]
    assert outcomes(first) == outcomes(second) == {"alice": "failed", "bob": "skipped"}
    assert first["snapshot"] == second["snapshot"]


def test_recording_is_serializable_and_identifies_the_exercised_code(recording):
    assert json.loads(json.dumps(recording)) == recording
    assert recording["experiment"]["mode"] == "scheduler_component"
    assert recording["experiment"]["resource"] == "cpu"
    assert recording["provenance"]["distribution"] == "backend.ai-manager"
    assert recording["provenance"]["version"] == "26.9.7"
    hashes = recording["provenance"]["component_sha256"]
    assert any("SessionProvisioner" in name for name in hashes)
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())
    assert recording["verification"]["passed"] is True


def test_repeated_runs_preserve_decisions_and_snapshots(recording):
    repeated = run_experiment()
    semantic_keys = ("sequence", "kind", "decisions", "snapshot")
    assert [{key: event[key] for key in semantic_keys} for event in recording["events"]] == [
        {key: event[key] for key in semantic_keys} for event in repeated["events"]
    ]


def test_releasing_an_unallocated_session_does_not_change_the_fixture():
    state = FixtureState()
    before = state.snapshot()
    with pytest.raises(ExperimentError):
        state.release("bob")
    assert state.snapshot() == before


def test_duplicate_pending_identity_is_rejected_before_scheduling():
    state = FixtureState(pending=["alice", "bob", "bob"])
    with pytest.raises(ExperimentError):
        asyncio.run(schedule_pass(state))
