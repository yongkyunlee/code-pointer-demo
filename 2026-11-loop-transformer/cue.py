"""Paired attention cuts for the repeated-clue experiment (one prompt at a time)."""
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


COLORS = "red blue green black white gold pink gray brown".split()
NAMES = "Amy Ben Dan Emma John Kate Lisa Mark Mary Paul Sam Tom".split()
FRUITS = "pear plum apple orange peach cherry grape lemon lime banana mango fig coconut strawberry".split()
FILLER = " Each locker had a door. One locker had a dent. Each locker had a shelf. One locker had a mark."
PILOT_N, TEST_N = 24, 100
DESIGN = "cue-v1"


@dataclass(frozen=True)
class Scene:
    number: int
    colors: tuple
    fruits: tuple
    people: tuple
    choice: int
    decoy: int
    target_first: bool


def scenes(phase):
    rng = random.Random(20261121)
    items, seen = [], set()
    while len(items) < PILOT_N + TEST_N:
        colors, fruits, people = (tuple(rng.sample(v, 2)) for v in (COLORS, FRUITS, NAMES))
        key = (tuple(sorted(colors)), tuple(sorted(fruits)), tuple(sorted(people)))
        if key in seen:
            continue
        seen.add(key)
        idx = len(items)
        items.append(Scene(idx, colors, fruits, people, rng.randrange(2),
                           rng.randrange(2), idx % 2 == 0))
    if phase == "pilot":
        return items[:PILOT_N]
    if phase == "heldout":
        return items[PILOT_N:]
    raise ValueError(phase)


class Prompt:
    """Sentence-aligned token ranges, checked against whole-string tokenization."""

    def __init__(self, engine, scene, *, reverse=False, clue_first=False):
        choice = scene.choice ^ int(reverse)
        descriptions = ""
        for color, fruit in zip(scene.colors, scene.fruits):
            article = "an" if fruit[0] in "aeiou" else "a"
            descriptions += f" {color.capitalize()} locker holds {article} {fruit}."
        target = f" {scene.people[0]} uses the {scene.colors[choice]} locker."
        sham = f" {scene.people[1]} uses the {scene.colors[scene.decoy]} locker."
        clues = [("target", target), ("sham", sham)]
        if not scene.target_first:
            clues.reverse()
        parts = [("descriptions", descriptions)]
        parts = clues + parts if clue_first else parts + clues
        self.record = "".join(text for _, text in parts)
        question = f" Which fruit is in {scene.people[0]}'s locker? Answer:"
        self.text = self.record * 2 + question
        self.ids = []
        self.first = {}
        for name, part in parts:
            start = len(self.ids)
            self.ids.extend(engine.ids(part))
            self.first[name] = (start, len(self.ids))
        self.split = len(self.ids)
        self.ids = self.ids * 2 + engine.ids(question)
        assert self.ids == engine.ids(self.text)
        a, b = self.first["descriptions"]
        self.descriptions = (a + self.split, b + self.split)
        a, b = self.first["target"]
        c, d = self.first["sham"]
        assert b - a == d - c
        correct, wrong = scene.fruits[choice], scene.fruits[1 - choice]
        self.answer_words = (correct, wrong)
        ids = []
        for fruit in self.answer_words:
            continuation = engine.ids(" " + fruit)
            assert len(continuation) == 1 and engine.ids(self.text + " " + fruit) == self.ids + continuation
            ids.append(continuation[0])
        self.answers = tuple(ids)


def cut_mask(queries, keys, sites=None):
    """Mask exactly the stated query rows and key columns in all heads."""
    q0, q1 = queries
    k0, k1 = keys
    assert 0 <= k0 < k1 <= q0 < q1

    def apply(stage, offset, mask):
        if sites is None or (stage, offset) in sites:
            mask[..., q0:q1, k0:k1] = float("-inf")
        return mask

    return apply


def score(engine, prompt, ids=None, mask_fn=None):
    import torch

    logits = engine.run(prompt.ids if ids is None else ids, mask_fn=mask_fn,
                        logits_positions=[-1]).logits[0, -1].float()
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite final logits")
    a, b = prompt.answers
    probs = logits.softmax(0)
    top = int(logits.argmax())
    top_text = engine.tokenizer.decode([top], skip_special_tokens=False)
    top_word = top_text.strip().casefold()
    return dict(margin=float(logits[a] - logits[b]), correct_probability=float(probs[a]),
                other_probability=float(probs[b]), top_token=top, top_text=top_text,
                top_correct=top_word == prompt.answer_words[0],
                off_candidate=top_word not in prompt.answer_words)


def isolation_check(engine, prompt):
    """Offset-matched isolated suffix and same-shape, changed-prefix leakage check."""
    import torch

    split = prompt.split
    close = cut_mask((split, len(prompt.ids)), (0, split))
    def final(ids, *, mask=None, offset=0):
        return engine.run(ids, mask_fn=mask, position_offset=offset,
                          logits_positions=[-1]).logits[0, -1].float()
    closed = final(prompt.ids, mask=close)
    isolated = final(prompt.ids[split:], offset=split)
    replacement = list(reversed(prompt.ids[:split])) + prompt.ids[split:]
    assert replacement != prompt.ids
    changed = final(replacement, mask=close)
    isolated_err = (closed - isolated).abs()
    leak = float((changed - closed).abs().max())
    metrics = dict(position_offset=split, altered_prefix_max_abs=leak,
                   isolated_max_abs=float(isolated_err.max()),
                   isolated_rms=float(isolated_err.square().mean().sqrt()),
                   isolated_rms_limit=0.025, isolated_max_limit=0.25,
                   altered_prefix_limit=0.002, vocabulary_size=len(closed))
    if not (torch.isfinite(isolated_err).all() and metrics["isolated_max_abs"] <= 0.25
            and metrics["isolated_rms"] <= 0.025 and leak <= 0.002):
        raise AssertionError(f"Failed cross-copy isolation: {metrics}")
    return metrics


def run_scene(engine, scene, primary, neighbors):
    p = Prompt(engine, scene)
    def cut(key, visits=None):
        return score(engine, p, mask_fn=cut_mask(p.descriptions, p.first[key], visits))
    early = (2, primary)
    late = (3, primary)
    s = {"intact": score(engine, p)}
    for key in ("target", "sham"):
        for label, visits in (("early", {early}), ("late", {late}), ("both", {early, late})):
            s[key + "_" + label] = cut(key, visits)
    s["target_all_layers"] = cut("target")
    for offset in neighbors:
        site = (3, offset)
        s[f"neighbor_{offset}_late"] = cut("target", {site})
        s[f"neighbor_{offset}_both"] = cut("target", {early, site})
    s["reversed_assignment"] = score(engine, Prompt(engine, scene, reverse=True))
    s["one_copy"] = score(engine, p, ids=p.ids[p.split:])
    filler_ids = engine.ids(FILLER)
    assert len(filler_ids) == p.split and filler_ids + p.ids[p.split:] == engine.ids(FILLER + p.text[len(p.record):])
    s["filler"] = score(engine, p, ids=filler_ids + p.ids[p.split:])
    s["clue_first"] = score(engine, Prompt(engine, scene, clue_first=True))
    return dict(scene=asdict(scene), prompt=p.text, answers=list(zip(p.answer_words, p.answers)),
                split=p.split, copy1=p.first, copy2_queries=p.descriptions, scores=s)


def interaction(m, early, late, both):
    return (m[early] - m[both]) - (m["intact"] - m[late])


def scene_metrics(row, neighbor):
    scores = row["scores"]
    m = {k: v["margin"] for k, v in scores.items()}
    i = interaction(m, "target_early", "target_late", "target_both")
    sham = interaction(m, "sham_early", "sham_late", "sham_both")
    nearby = interaction(m, "target_early", f"neighbor_{neighbor}_late", f"neighbor_{neighbor}_both")
    metrics = dict(target_interaction=i, sham_interaction=sham, nearby_interaction=nearby,
                   target_minus_sham=i - sham, target_minus_nearby=i - nearby,
                   early_cost=m["intact"] - m["target_early"], late_cost=m["intact"] - m["target_late"],
                   late_cost_after_early=m["target_early"] - m["target_both"],
                   all_layers_cost=m["intact"] - m["target_all_layers"],
                   duplication_gain=m["intact"] - m["one_copy"], evidence_vs_filler=m["intact"] - m["filler"],
                   clue_order_gain=m["clue_first"] - m["intact"],
                   correct_both_assignments=float(m["intact"] > 0 and m["reversed_assignment"] > 0),
                   nearby_single_cut_gap=m[f"neighbor_{neighbor}_late"] - m["target_late"])
    for condition, v in scores.items():
        for k in ("margin", "correct_probability", "other_probability", "top_correct", "off_candidate"):
            metrics[f"{condition}.{k}"] = float(v[k])
        metrics[f"{condition}.candidate_accuracy"] = float(v["margin"] > 0)
        metrics[f"{condition}.tie"] = float(v["margin"] == 0)
    return metrics


def intervals(rows):
    rng = np.random.default_rng(20261122)
    idx = rng.integers(len(rows), size=(4000, len(rows)))
    result = {}
    for key in rows[0]:
        values = np.array([r[key] for r in rows], dtype=float)
        low, high = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
        result[key] = dict(mean=float(values.mean()), lo=float(low), hi=float(high))
    return result


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def pick_neighbor(rows, neighbors):
    """Match the single late-cut cost on pilot means before evaluating held-out scenes."""
    gaps = {}
    for offset in neighbors:
        diffs = [r["scores"][f"neighbor_{offset}_late"]["margin"] - r["scores"]["target_late"]["margin"]
                 for r in rows]
        gaps[offset] = float(np.mean(diffs))
    return min(neighbors, key=lambda o: (abs(gaps[o]), o)), gaps


def plot(estimates, path, title):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    for ax, keys, labels, heading in (
            (axes[0], [x + ".margin" for x in ("intact", "target_early", "target_late", "target_both")],
             ["Intact", "Stage 2", "Stage 3", "Both"], "Final correct-minus-other logit"),
            (axes[1], ["target_interaction", "sham_interaction", "nearby_interaction"],
             ["Clue", "Sham", "Nearby offset"], "Paired-cut interaction (logits)")):
        stats = [estimates[key] for key in keys]
        ax.axvline(0, color="0.7", linewidth=1)
        for j, stat in enumerate(stats):
            ax.plot([stat["lo"], stat["hi"]], [j, j], color="#326897")
            ax.plot(stat["mean"], j, "o", color="#19466e")
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(heading, fontsize=10)
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(engine, phase, outdir):
    out = Path(outdir) / "cue"
    out.mkdir(parents=True, exist_ok=True)
    model = engine.name
    path = out / f"{model}_{phase}_rows.json"
    cfg_path = out / f"{model}_frozen_sites.json"
    items = scenes(phase)
    primary = (engine.width + 1) // 2
    delta = max(1, round(engine.width / 8))
    candidates = [primary - delta, primary + delta]
    if phase == "heldout":
        cfg = json.loads(cfg_path.read_text())
        assert (cfg["design"], cfg["primary"], cfg["width"]) == (DESIGN, primary, engine.width)
        neighbors = [cfg["late_neighbor"]]
    else:
        neighbors = candidates
    records = json.loads(path.read_text()) if path.exists() else {"design": DESIGN, "rows": []}
    rows = records["rows"]
    assert records["design"] == DESIGN
    assert [r["scene"]["number"] for r in rows] == [x.number for x in items[:len(rows)]]
    if "isolation" not in records:
        records["isolation"] = isolation_check(engine, Prompt(engine, items[0]))
        write_json(path, records)
    for item in items[len(rows):]:
        rows.append(run_scene(engine, item, primary, neighbors))
        write_json(path, records)
        if len(rows) % 10 == 0 or len(rows) == len(items):
            print(f"Repeated clue {model} {phase}: {len(rows)}/{len(items)}", flush=True)
    # Verify exact balance of clue length, key position, and key-to-query distance.
    def geometry(row, key):
        a, b = row["copy1"][key]
        c, d = row["copy2_queries"]
        return (a, b, c, d)
    assert sorted(geometry(r, "target") for r in rows) == sorted(geometry(r, "sham") for r in rows)
    if phase == "pilot":
        neighbor, gaps = pick_neighbor(rows, neighbors)
        cfg = dict(design=DESIGN, model=model, width=engine.width, primary=primary,
                   late_neighbor=neighbor, pilot_single_cut_margin_differences=gaps,
                   selection="Smallest absolute difference in mean single-cut margin on pilot scenes")
        write_json(cfg_path, cfg)
    neighbor = cfg["late_neighbor"]
    estimates = intervals([scene_metrics(r, neighbor) for r in rows])
    summary = dict(design=DESIGN, model=model, phase=phase, n_families=len(rows), sites=cfg,
                   executed_blocks_1_based=[engine.width + primary, 2 * engine.width + primary],
                   alternate_late_block_1_based=2 * engine.width + neighbor,
                   isolation=records["isolation"], estimates=estimates,
                   bootstrap="4000 resamples of all scene families; percentile 95% intervals",
                   off_candidate="Decoded greedy next token falls outside the two named fruits; "
                                 "case and surrounding whitespace ignored")
    write_json(out / f"{model}_{phase}_summary.json", summary)
    plot(estimates, out / f"{model}_{phase}.png", f"{model} · repeated clue · {phase} (n={len(rows)})")
    if phase == "heldout" and model in ("qwen", "ouro"):
        paths = [out / f"{tag}_heldout_rows.json" for tag in ("ouro", "qwen")]
        if all(p.exists() for p in paths):
            datasets = [json.loads(p.read_text())["rows"] for p in paths]
            if all(len(ds) == TEST_N for ds in datasets):
                sites = [json.loads((out / f"{tag}_frozen_sites.json").read_text())["late_neighbor"]
                         for tag in ("ouro", "qwen")]
                differences = []
                common_success = []
                for ouro, qwen in zip(*datasets):
                    assert ouro["scene"] == qwen["scene"] and ouro["prompt"] == qwen["prompt"]
                    a, b = scene_metrics(ouro, sites[0]), scene_metrics(qwen, sites[1])
                    diff = {k: a[k] - b[k] for k in ("target_interaction", "target_minus_sham",
                            "target_minus_nearby", "duplication_gain", "intact.candidate_accuracy")}
                    differences.append(diff)
                    if a["intact.candidate_accuracy"] and b["intact.candidate_accuracy"]:
                        common_success.append(diff)
                write_json(out / "heldout_ouro_minus_qwen.json", dict(
                    n=TEST_N, estimates=intervals(differences), secondary_common_clean_success=dict(
                        n=len(common_success), estimates=intervals(common_success) if common_success else {})))
    return summary
