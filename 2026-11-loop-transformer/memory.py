"""Source-closed memory, witness counterfactuals and matched register controls."""
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch


BOXES = tuple("ABCDEFGHIJKL")
SLOTS = 8
QUESTION_WIDTH = 16
ROLES = ("first", "second", "world")
DOMAINS = ("people", "registers")
PHASE_SIZES = {"pilot": 24, "heldout": 120}
SEED = 170271
BOOTSTRAPS = 2000


def families(phase):
    rng = random.Random(SEED)
    layouts = list(itertools.combinations(BOXES, 3))
    rng.shuffle(layouts)
    items = []
    for index, layout in enumerate(layouts[:sum(PHASE_SIZES.values())]):
        boxes = list(layout)
        rng.shuffle(boxes)
        items.append((index, boxes, index % 2))
    n = PHASE_SIZES["pilot"]
    return items[:n] if phase == "pilot" else items[n:]


def facts(domain, boxes, witness, toggle):
    a, b, c = boxes
    watching = [witness == index or toggle for index in range(2)]
    if domain == "people":
        actors = ["Alice", "Bob"]
        rule = "People remember the last box they saw the coin in."
        start = f"The coin is in box {a}. Alice and Bob both see it."
        mid = " ".join(f"{name} sees this move: {'yes' if sees else 'no'}."
                       for name, sees in zip(actors, watching))
        end = "Neither person sees this move."
    else:
        actors = ["register A", "register B"]
        rule = "Registers store the last box they recorded the coin in."
        start = f"The coin is in box {a}. Both registers record its location."
        mid = " ".join(f"{name.capitalize()} records this move: {'yes' if sees else 'no'}."
                       for name, sees in zip(actors, watching))
        end = "Neither register records this move."
    return f"{rule}\n{start}\nThe coin moves to box {b}. {mid}\nThe coin moves to box {c}. {end}\n"


def question(domain, role):
    if role == "world":
        ask = "Where is the coin actually located now?"
    elif domain == "people":
        person = ("Alice", "Bob")[ROLES.index(role)]
        ask = f"Where does {person} believe the coin is?"
    else:
        register = ("A", "B")[ROLES.index(role)]
        ask = f"Which location is stored in register {register}?"
    return f"\nQuestion: {ask}\nAnswer: box"


def mask_for(mode, n_source):
    n_question = n_source + SLOTS

    def mask_fn(stage, offset, causal):
        if mode == "sham":
            return causal
        mask = causal
        blocked = torch.finfo(mask.dtype).min
        if mode == "closed" or stage > 2:
            mask[:, :, n_source:, :n_source] = blocked
        if stage <= 2:
            mask[:, :, n_question:, :n_question] = blocked
        return mask

    return mask_fn


def replace_memory(engine, n_source, memory):
    def edit(stage, offset, hidden):
        if (stage, offset) == (2, engine.width):
            hidden = hidden.clone()
            hidden[:, n_source:n_source + SLOTS] = memory
        return hidden
    return edit


def question_branch(source, slots, q):
    """Place all right padding strictly after the position read by the output head."""
    if len(q) > QUESTION_WIDTH:
        raise ValueError(f"Question length {len(q)} exceeds the frozen width {QUESTION_WIDTH}")
    end = len(source) + len(slots) + len(q) - 1
    ids = source + slots + q + [slots[0]] * (QUESTION_WIDTH - len(q))
    return ids, end


def read_memory(engine, source, slots):
    ids, end = question_branch(source, slots, [])
    key = (2, engine.width)
    result = engine.run(ids, capture=(key,), mask_fn=mask_for("memory", len(source)), logits_positions=[end])
    return result.states[key][:, len(source):len(source) + SLOTS].detach().clone()


def close_enough(left, right, label):
    left, right = left.float(), right.float()
    delta = (left - right).abs()
    if not torch.isfinite(delta).all().item():
        raise AssertionError(f"{label}: non-finite values")
    # Keep the predeclared FP16 tolerance for all same-shape comparisons.
    if not torch.allclose(left, right, rtol=0.002, atol=0.015625):
        raise AssertionError(f"{label}: max difference {float(delta.max())}")
    return float(delta.max())


def controls(engine, source, slots, q, alt_q, other, memory):
    ids, end = question_branch(source, slots, q)
    alt_ids, alt_end = question_branch(source, slots, alt_q)
    other_ids, _ = question_branch(other, slots, q)
    n = len(source)
    if len(other) != n:
        raise ValueError("The leak test requires facts with identical token lengths")
    if len(q) != len(alt_q) or q == alt_q:
        raise ValueError("Memory invariance requires distinct, equally long questions")
    sham = engine.run(ids, mask_fn=mask_for("sham", n), logits_positions=[end]).logits
    native = engine.run(ids, logits_positions=[end]).logits
    sham_diff = close_enough(sham, native, "sham versus native")
    del sham, native
    blocked = mask_for("closed", n)
    old = engine.run(ids, mask_fn=blocked, logits_positions=[end]).logits
    new = engine.run(other_ids, mask_fn=blocked, logits_positions=[end]).logits
    closed_diff = close_enough(old, new, "changing facts with all routes closed")
    del old, new
    key = (2, engine.width)
    masked = mask_for("memory", n)
    run = engine.run(ids, mask_fn=masked, capture=(key,), logits_positions=[end])
    alternate = engine.run(alt_ids, mask_fn=masked, capture=(key,), logits_positions=[alt_end])
    first = run.states[key][:, n:n + SLOTS]
    second = alternate.states[key][:, n:n + SLOTS]
    memory_diff = close_enough(memory, first, "neutral versus real question at the same shape")
    alt_diff = close_enough(first, second, "changing an equal-length question")
    del run, alternate, first, second
    # Reuse exactly the same boundary tensor while changing the facts.
    edit = replace_memory(engine, n, memory)
    before = engine.run(ids, mask_fn=masked, edit_fn=edit, logits_positions=[end]).logits
    after = engine.run(other_ids, mask_fn=masked, edit_fn=edit, logits_positions=[end]).logits
    fixed_diff = close_enough(before, after, "fixed memory, changed source")
    return dict(sham_max_abs=sham_diff, source_closed_max_abs=closed_diff,
                memory_independence_max_abs=memory_diff, changed_question_max_abs=alt_diff,
                fixed_memory_max_abs=fixed_diff)


def score(engine, logits, answer_ids, boxes, target):
    logits = logits[0, -1].detach().float()
    scores = logits[answer_ids]
    probs = (scores - torch.logsumexp(logits, dim=0)).exp().tolist()
    values = scores.tolist()
    if not np.isfinite(values + probs).all():
        raise AssertionError("Non-finite answer scores")
    options = [BOXES.index(box) for box in boxes]
    best = max(options, key=values.__getitem__)
    greedy = int(logits.argmax())
    return dict(letter_logits=values, winner=BOXES[best], greedy_id=greedy,
                greedy_text=engine.tokenizer.decode([greedy]), malformed=greedy not in answer_ids,
                outside_scene=greedy not in [answer_ids[i] for i in options],
                free_correct=greedy == answer_ids[BOXES.index(target)],
                answer_mass=sum(probs[i] for i in options), correct_mass=probs[BOXES.index(target)])


def ask(engine, source, slots, q_ids, memory, answer_ids, boxes, target):
    kwargs = {}
    if memory is not None:
        kwargs.update(mask_fn=mask_for("memory", len(source)),
                      edit_fn=replace_memory(engine, len(source), memory))
    ids, end = question_branch(source, slots, q_ids)
    result = engine.run(ids, capture=((3, engine.width),), logits_positions=[end], **kwargs)
    return {stage: score(engine, logits, answer_ids, boxes, target)
            for stage, logits in [(3, result.stage_logits(3, positions=[end])), (4, result.logits)]}


def collect(engine, phase, directory, slots, answer_ids):
    layouts = families(phase)
    for family, boxes, witness in layouts:
        path = directory / f"scene-{family:03d}.json"
        if path.exists() and json.loads(path.read_text()).get("question_width") == QUESTION_WIDTH:
            continue
        rows = []
        checks = {}
        # A different scene cyclically reassigns identical box names to events.
        # This derangement keeps the answer vocabulary and event permissions fixed.
        foil = boxes[1:] + boxes[:1]
        for domain in DOMAINS:
            for toggle in (False, True):
                source = engine.ids(facts(domain, boxes, witness, toggle))
                other = engine.ids(facts(domain, foil, witness, toggle))
                if len(source) != len(other):
                    raise AssertionError("Shuffled scene has a different position offset")
                mem = read_memory(engine, source, slots)
                memories = {"direct": None, "memory": mem}
                if not toggle:
                    memories["shuffled"] = read_memory(engine, other, slots)
                if family == layouts[0][0] and not toggle:
                    checks[domain] = controls(engine, source, slots, engine.ids(question(domain, "world")),
                                              engine.ids(question(domain, "first")), other, mem)
                for role_index, role in enumerate(ROLES):
                    q_ids = engine.ids(question(domain, role))
                    target_index = 2 if role == "world" else int(role_index == witness or toggle)
                    target = boxes[target_index]
                    for condition, selected in memories.items():
                        scores = ask(engine, source, slots, q_ids, selected, answer_ids, boxes, target)
                        for stage, scores_at_stage in scores.items():
                            rows.append(dict(family=family, boxes=boxes, witness=witness, domain=domain,
                                             toggle=toggle, role=role, condition=condition, stage=stage,
                                             target=target, donor_target=foil[target_index], **scores_at_stage))
        atomic(path, {"rows": rows, "checks": checks, "question_width": QUESTION_WIDTH})
        print(f"memory {engine.name} {phase}, scene {family} complete", flush=True)


def atomic(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_rows(directory):
    rows = []
    for path in sorted(directory.glob("scene-*.json")):
        scene = json.loads(path.read_text())
        if scene.get("question_width") != QUESTION_WIDTH:
            raise ValueError(f"Rerun {path.parent} with fixed question width before calibration")
        rows.extend(scene["rows"])
    return rows


def calibrate(train, evaluate):
    """Identical ridge readouts on all twelve box logits; train only on calibration scenes."""
    for stage in (3, 4):
        tr = [r for r in train if r["stage"] == stage and r["condition"] in ("memory", "direct")]
        te = [r for r in evaluate if r["stage"] == stage]
        x = np.asarray([r["letter_logits"] for r in tr])
        y = np.eye(len(BOXES))[[BOXES.index(r["target"]) for r in tr]]
        mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
        x = (x - mean) / std
        intercept = y.mean(0)
        beta = np.linalg.solve(x.T @ x + len(x) * 0.05 * np.eye(len(BOXES)), x.T @ (y - intercept))
        for r in te:
            pred = (np.asarray(r["letter_logits"]) - mean) / std @ beta + intercept
            r["calibrated_winner"] = max(r["boxes"], key=lambda b: pred[BOXES.index(b)])


def interval(family_values):
    vals = np.array(family_values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(vals, (BOOTSTRAPS, len(vals)), replace=True).mean(1)
    return dict(mean=float(vals.mean()), ci95=np.quantile(samples, [0.025, 0.975]).tolist(), families=len(vals))


def summarize(rows):
    result, contrasts = {}, {}
    families_in_run = sorted({r["family"] for r in rows})
    by = {(r["family"], r["domain"], r["toggle"], r["condition"], r["role"], r["stage"]): r for r in rows}
    for domain, toggle, cond, stage in itertools.product(DOMAINS, (False, True),
                                                        ("memory", "direct", "shuffled"), (3, 4)):
        if toggle and cond == "shuffled":
            continue
        scene_rows = [[by[f, domain, toggle, cond, role, stage] for role in ROLES] for f in families_in_run]
        key = f"{domain}/{'both' if toggle else 'one'}/{cond}/stage{stage}"
        metrics = {}
        for field, label in (("winner", "native"), ("calibrated_winner", "calibrated")):
            scores = [[r[field] == r["target"] for r in scene] for scene in scene_rows]
            metrics[label + "_mean_accuracy"] = interval(np.mean(scores, axis=1))
            metrics[label + "_all_three"] = interval(np.all(scores, axis=1))
            matrix = [[sum(r[field] == r["boxes"][col] for scene in scene_rows for r in [scene[i]])
                       for col in range(3)] for i in range(3)]
            metrics[label + "_event_confusion_counts"] = matrix
        for attr in ("answer_mass", "correct_mass", "malformed", "outside_scene", "free_correct"):
            metrics[attr] = interval([np.mean([r[attr] for r in scene]) for scene in scene_rows])
        result[key] = metrics
    for readout in ("winner", "calibrated_winner"):
        drifts = {}
        for domain in DOMAINS:
            shifts, selective, recall, donor, rightful = [], [], [], [], []
            for f in families_in_run:
                get = lambda cond, role, stage, both=False: by[f, domain, both, cond, role, stage]
                early = [get("memory", r, 3) for r in ROLES[:2]]
                late = [get("memory", r, 4) for r in ROLES[:2]]
                shifts.append(np.mean([int(v[readout] == v["boxes"][2]) - int(u[readout] == u["boxes"][2])
                                       for u, v in zip(early, late)]))
                orig = [get("memory", r, 4) for r in ROLES]
                switched = [get("memory", r, 4, True) for r in ROLES]
                affected = 1 - orig[0]["witness"]
                selective.append(all((a[readout] == a["target"] and b[readout] == b["target"])
                                     if idx == affected else (a[readout] == b[readout])
                                     for idx, (a, b) in enumerate(zip(orig, switched))))
                world, shuffle = orig[2], get("shuffled", "world", 4)
                recall.append(world[readout] == world["target"])
                donor.append(shuffle[readout] == shuffle["donor_target"])
                rightful.append(int(recall[-1]) - int(shuffle[readout] == shuffle["target"]))
            drifts[domain] = np.asarray(shifts)
            contrasts[f"{domain}/{readout}"] = dict(late_world_intrusion_change=interval(shifts),
                witness_only_counterfactual_correct=interval(selective), own_world_accuracy=interval(recall),
                shuffled_donor_world_accuracy=interval(donor), own_minus_shuffled_world_accuracy=interval(rightful))
        contrasts[f"people_minus_registers/{readout}"] = interval(drifts["people"] - drifts["registers"])
    return {"metrics": result, "paired_contrasts": contrasts}


def plots(summary, output):
    import matplotlib.pyplot as plt
    for method in ("native", "calibrated"):
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), gridspec_kw={"width_ratios": [1.4, 1, 1]})
        for i, domain in enumerate(DOMAINS):
            axes[i, 0].axis("off")
            text = ("People\nA coin goes from box 1 to 2 to 3.\nBoth people see its placement.\nOnly one sees the first move.\nNeither sees the second."
                    if i == 0 else "Registers\nBoth record the initial box.\nOnly one records the first move.\nNeither records the second.")
            axes[i, 0].text(0, .7, text, va="top", fontsize=9)
            for j, stage in enumerate((3, 4), 1):
                ax = axes[i, j]
                matrix = np.array(summary["metrics"][f"{domain}/one/memory/stage{stage}"][method + "_event_confusion_counts"])
                freq = matrix / matrix.sum(axis=1, keepdims=True)
                ax.imshow(freq, vmin=0, vmax=1, cmap="Blues")
                for y, x in itertools.product(range(3), repeat=2):
                    ax.text(x, y, f"{freq[y,x]:.0%}", ha="center", va="center",
                            color="white" if freq[y,x] > .55 else "black")
                ax.set_xticks(range(3), ["initial", "middle", "final"], fontsize=8)
                ax.set_yticks(range(3), ["Alice", "Bob", "world"] if i == 0 else ["reg. A", "reg. B", "world"])
                ax.set_title(f"Stage {stage}")
        fig.tight_layout()
        fig.savefig(output / f"{method}-confusion.png", dpi=140)
        plt.close(fig)


def rooms(engine, phase, slots, answer_ids):
    rows = []
    for family, boxes, witness in families(phase):
        rooms_and_boxes = dict(zip(("red", "blue"), boxes[:2]))
        sources, memories = {}, {}
        for color, box in rooms_and_boxes.items():
            source = engine.ids(f"In the {color} room, the key is in box {box}.\n")
            sources[color] = source
            memories[color] = read_memory(engine, source, slots)
        if len(sources["red"]) != len(sources["blue"]):
            raise AssertionError("Room memory offsets differ")
        blend = ((memories["red"].float() + memories["blue"].float()) * 0.5).to(memories["red"].dtype)
        for color, target in rooms_and_boxes.items():
            q = engine.ids(f"\nQuestion: Where is the key in the {color} room?\nAnswer: box")
            for condition, memory in ((color + "_alone", memories[color]), ("blend", blend)):
                measured = ask(engine, sources[color], slots, q, memory, answer_ids, boxes[:2], target)[4]
                rows.append(dict(family=family, color=color, target=target, condition=condition, **measured))
    return rows


@torch.no_grad()
def run(engine, phase, outdir):
    if phase not in PHASE_SIZES:
        raise ValueError("phase must be 'pilot' or 'heldout'")
    directory = Path(outdir) / "memory" / engine.name / phase
    directory.mkdir(parents=True, exist_ok=True)
    caps = [engine.ids(" " + box) for box in BOXES]
    neutral = engine.ids(" .")
    if any(len(seq) != 1 for seq in [*caps, neutral]) or len(set(seq[0] for seq in caps)) != len(BOXES):
        raise ValueError(f"Tokenizer must supply distinct, single-token box answers and neutral slots: {caps}, {neutral}")
    slots, answer_ids = neutral * SLOTS, [seq[0] for seq in caps]
    collect(engine, phase, directory, slots, answer_ids)
    rows = read_rows(directory)
    if phase == "pilot":
        for fold in range(4):
            calibrate([r for r in rows if r["family"] % 4 != fold],
                      [r for r in rows if r["family"] % 4 == fold])
    else:
        training = read_rows(directory.parent / "pilot")
        if len({r["family"] for r in training}) != PHASE_SIZES["pilot"]:
            raise ValueError("Need complete pilot calibration rows from the same engine before heldout")
        calibrate(training, rows)
    summary = summarize(rows)
    summary.update(checkpoint=engine.name, phase=phase, sizes=PHASE_SIZES,
                   answer_ids=dict(zip(BOXES, answer_ids)), slot_ids=slots, question_width=QUESTION_WIDTH, seed=SEED,
                   readout="Twelve box logits to twelve box scores, ridge penalty 0.05, per stage/model; pilot fourfold grouped crossfit.",
                   boundary_blocks=[2 * engine.width, 3 * engine.width, 4 * engine.width],
                   example={"facts": facts("people", ["A", "B", "C"], 1, False),
                            "question": question("people", "first")})
    gate_rule = "Pilot world-only own and shuffled-to-donor accuracy lower 95% bounds > 1/3, and own minus shuffled-to-recipient lower bound > 0."
    if phase == "pilot":
        main = summary["paired_contrasts"]["people/winner"]
        gate = (main["own_world_accuracy"]["ci95"][0] > 1/3
                and main["shuffled_donor_world_accuracy"]["ci95"][0] > 1/3
                and main["own_minus_shuffled_world_accuracy"]["ci95"][0] > 0)
    else:
        gate = json.loads((directory.parent / "pilot" / "summary.json").read_text())["room_gate"]["passed"]
    summary["room_gate"] = {"rule": gate_rule, "passed": bool(gate)}
    if gate:
        room_rows = rooms(engine, phase, slots, answer_ids)
        atomic(directory / "room-rows.json", room_rows)
        summary["room_blend"] = {f"{color}/{condition}": interval(
            [r["winner"] == r["target"] for r in room_rows if r["color"] == color and r["condition"] == condition])
            for color in ("red", "blue") for condition in (color + "_alone", "blend")}
    atomic(directory / "summary.json", summary)
    plots(summary, directory)
    return summary
