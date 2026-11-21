"""Read the frozen output head after every block on three recorded held-out items.

Run beside model_engine.py, carry.py, memory.py, and cue.py from the study.
Each condition is one forward trajectory. Intermediate projections are probes
of the running state; only the final depth is the checkpoint's output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

import carry
import cue
import memory
from model_engine import CHECKPOINTS, Engine


# Independently recorded margins from the original evidence/kaggle-results.zip.
# For memory, subtract the higher of the two alternate scene-box logits.
ORIGINAL_ARCHIVE_SHA256 = "9a26ba409eab9ccd02fedef298cbdb478875c7ca05e92beae3e57555c52c33ad"
ARCHIVE_MARGINS = {
    "ouro": {"carry": (-2.0390623807907104, -1.0),
             "memory": (0.2109375, -0.125, -1.033203125, -1.2421875),
             "locker": (0.59375, 0.5625)},
    "qwen": {"carry": (-1.1875, -0.5625),
             "memory": (2.9609375, 0.25, 5.1689453125, -0.34375),
             "locker": (-0.2890625, -0.203125)},
}


def single_token(engine, text):
    ids = engine.ids(" " + text)
    if len(ids) != 1:
        raise ValueError(f"Expected a single answer token for {text!r}, got {ids}")
    return ids[0]


def measure(engine, ids, position, tokens, key, label, *, edit=None, mask=None, events=()):
    # Retain only the answer position. A full-vocabulary head read on one row at
    # a time uses the same matrix shape as the original checkpoint scores.
    states = []

    def collect(stage, offset, hidden):
        changed = edit(stage, offset, hidden) if edit is not None else None
        active = hidden if changed is None else changed
        states.append(active[:, position:position + 1, :].detach().clone())
        return changed

    run = engine.run(ids, edit_fn=collect, mask_fn=mask, logits_positions=[position])
    points = []
    score_rows, top_ids = [], []
    index = torch.tensor(tokens, dtype=torch.long, device=engine.device)
    for state in states:
        logits = engine.readout(state)[0, 0]
        score_rows.append(logits.index_select(0, index).float())
        top_ids.append(logits.argmax())
    if len(states) != 4 * engine.width:
        raise ValueError("The adapter did not visit every expected decoder block")
    torch.testing.assert_close(logits, run.logits[0, 0], atol=0.0, rtol=0.0)
    rows = torch.stack(score_rows)
    if not torch.isfinite(rows).all():
        raise ValueError(f"Nonfinite projected answer score in {key}")
    for i, (row, top_id) in enumerate(zip(rows.tolist(), torch.stack(top_ids).tolist())):
        stage, offset = divmod(i, engine.width)
        negative = max(row[1:])
        points.append({"depth": i + 1, "block": offset + 1 if engine.name == "ouro" else i + 1,
                       "stage": stage + 1, "positive_logit": row[0], "negative_logit": negative,
                       "margin": row[0] - negative, "top_token_id": top_id,
                       "top_token": engine.tokenizer.decode([top_id], clean_up_tokenization_spaces=False)})
    return {"key": key, "label": label, "events": list(events), "points": points}


def unchanged_prefix(a, b, length):
    for left, right in zip(a["points"][:length], b["points"][:length]):
        if left != right:
            raise AssertionError(f"Unexpected difference before the intervention at depth {left['depth']}")


def carry_trace(engine):
    family = carry.families("heldout")[0]
    recipient_pair, donor_pair = family["sums"]
    recipient = carry.make_prompt(engine, recipient_pair, "English")
    donor = carry.make_prompt(engine, donor_pair, "English")
    position = recipient.positions["answer position"]
    site = (3, engine.width)
    depth = 3 * engine.width
    donor_run = engine.run(donor.ids, capture=[site], logits_positions=[donor.positions["answer position"]])
    value = donor_run.states[site][0, donor.positions["answer position"]].clone()
    del donor_run
    original = family["total"] // 10
    words = [carry.WORDS["English"][digit] for digit in (original + 1, original)]
    tokens = [single_token(engine, word) for word in words]
    for word, token in zip(words, tokens):
        if engine.ids(recipient.text + " " + word) != recipient.ids + [token]:
            raise ValueError("Unexpected answer boundary in the carry prompt")

    def swap(stage, offset, state):
        if (stage, offset) == site:
            patched = state.clone()
            patched[0, position] = value
            return patched
        return None

    clean = measure(engine, recipient.ids, position, tokens, "clean", "Clean prompt")
    patched = measure(engine, recipient.ids, position, tokens, "swap", "Swap the answer-position state",
                      edit=swap, events=[{"depth": depth, "label": "Replace state with donor state"}])
    unchanged_prefix(clean, patched, depth - 1)
    return {"subject": "Carry · held-out family 000", "prompt": recipient.text, "donor_prompt": donor.text,
            "metric": "Carry-predicted word minus correct word (logits)",
            "positive": words[0], "negative": words[1], "conditions": [clean, patched]}


def memory_trace(engine):
    family, boxes, witness = memory.families("heldout")[0]
    if family != 24 or boxes != ["L", "A", "B"] or witness != 0:
        raise ValueError("Unexpected memory scene")
    story = memory.facts("people", boxes, witness, False)
    question = memory.question("people", "world")
    source = engine.ids(story)
    slots = [single_token(engine, ".")] * memory.SLOTS
    ids, position = memory.question_branch(source, slots, engine.ids(question))
    frozen_slots = memory.read_memory(engine, source, slots)
    target = boxes[-1]
    alternatives = sorted(boxes[:-1])
    tokens = [single_token(engine, box) for box in [target, *alternatives]]
    direct = measure(engine, ids, position, tokens, "direct", "Full story")
    stored = measure(engine, ids, position, tokens, "memory", "Eight stored positions",
                     mask=memory.mask_for("memory", len(source)),
                     edit=memory.replace_memory(engine, len(source), frozen_slots),
                     events=[{"depth": 1, "label": "Question can read only its own tokens"},
                             {"depth": 2 * engine.width, "label": "Copy eight stored states"},
                             {"depth": 2 * engine.width + 1, "label": "Question can now read memory; story closed"}])
    return {"subject": "Moving coin · held-out scene 024",
            "prompt": engine.tokenizer.decode(ids[:position + 1], clean_up_tokenization_spaces=False),
            "metric": "Actual box minus strongest other box (logits)", "positive": target,
            "negative": "best of A and L", "conditions": [direct, stored]}


def locker_trace(engine):
    scene = next(item for item in cue.scenes("heldout") if item.number == 35)
    p = cue.Prompt(engine, scene)
    offset = (engine.width + 1) // 2
    early, late = engine.width + offset, 2 * engine.width + offset
    mask = cue.cut_mask(p.descriptions, p.first["target"], {(2, offset), (3, offset)})
    position = len(p.ids) - 1
    intact = measure(engine, p.ids, position, p.answers, "intact", "No cut")
    both = measure(engine, p.ids, position, p.answers, "target_both", "Cut both visits", mask=mask,
                   events=[{"depth": n, "label": "Block the clue-to-description attention connection"}
                           for n in (early, late)])
    # The first cut changes description tokens; the answer position can read
    # those changed states only at a later block.
    unchanged_prefix(intact, both, early)
    return {"subject": "Repeated clue · held-out scene 035", "prompt": p.text,
            "metric": "Correct fruit minus other fruit (logits)",
            "positive": p.answer_words[0], "negative": p.answer_words[1], "conditions": [intact, both]}


def compare_with_archive(engine, tasks):
    checks = []
    for name, task in tasks.items():
        depths = [3 * engine.width, 4 * engine.width] if name == "memory" else [4 * engine.width]
        observed = [(condition["key"], depth, condition["points"][depth - 1]["margin"])
                    for condition in task["conditions"] for depth in depths]
        expected = ARCHIVE_MARGINS[engine.name][name]
        for (condition, depth, value), target in zip(observed, expected, strict=True):
            # Old carry scores subtract the two log-softmax values in FP32;
            # single-token logit differences agree up to FP32 cancellation.
            error = abs(value - target)
            if not math.isclose(value, target, rel_tol=0, abs_tol=1e-5):
                raise AssertionError(f"Original archive differs: {name} {condition} depth {depth}: "
                                     f"{value} versus {target}")
            checks.append({"task": name, "condition": condition, "depth": depth,
                           "margin": value, "archived_margin": target, "abs_error": error})
    return checks


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=CHECKPOINTS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    engine = Engine.load(args.model)
    validation = engine.validate()
    tasks = {"carry": carry_trace(engine), "memory": memory_trace(engine), "locker": locker_trace(engine)}
    result = {"model": engine.name, "checkpoint": CHECKPOINTS[engine.name],
              "blocks_per_stage": engine.width, "executed_depth": 4 * engine.width,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(engine.device),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "original_archive_sha256": ORIGINAL_ARCHIVE_SHA256, "native_validation": validation,
              "tasks": tasks, "archive_checks": compare_with_archive(engine, tasks)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(f"{args.model}: {sum(len(c['points']) for t in tasks.values() for c in t['conditions'])} "
          f"depth readouts, {len(result['archive_checks'])} archive checks", flush=True)


if __name__ == "__main__":
    main()
