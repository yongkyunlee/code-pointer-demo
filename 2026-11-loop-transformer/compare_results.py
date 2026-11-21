"""Pair frozen-checkpoint results by family. Run locally after downloading Kaggle outputs."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_results import ENGINES, PHASES, TASKS, audit, check, file_hash, read_json


@dataclass
class Family:
    design: object
    cluster: object
    clean_success: bool
    scores: dict


GATES = {
    "carry": "Both clean sums return the correct first word in the requested language.",
    "memory": "All six direct-source, stage-four answers (three people queries and three register queries) have the correct unrestricted top token on the original scene.",
    "cue": "Both the intact and assignment-reversed scene return the correct fruit as the unrestricted first token.",
}


def carry_families(root, model, phase):
    directory = root / f"carry-{model}-{phase}"
    width = read_json(directory / "manifest.json")["width"]
    families = []
    for path in sorted(directory.glob("family-*.json")):
        raw = read_json(path)
        design = raw["family"]
        tens = design["total"] // 10
        correct_words = {r["carry"]: r["clean_word"] for r in raw["patches"]}
        scores, successes = {}, []
        for r in raw["clean"]:
            lang = r["language"]
            correct = r["first_word"] == correct_words[r["carry"]]
            successes.append(correct)
            scores.update({f"{lang}/clean/correct_word": correct,
                           f"{lang}/clean/correct_digit": r["digit"] == tens,
                           f"{lang}/clean/correct_word_probability": r["correct_word_probability"],
                           f"{lang}/clean/malformed": r["malformed"],
                           f"{lang}/clean/candidate_correct": r["best_word"] == correct_words[r["carry"]]})
        for r in raw["readers"]:
            base = f"{r['language']}/s{r['stage']}/reader"
            for name in ("fixed_knn", "fixed_ridge", "fresh_ridge"):
                scores[f"{base}/{name}"] = r[name] == r["carry"]
            scores[f"{base}/fresh_minus_fixed_ridge"] = int(r["fresh_ridge"] == r["carry"]) - int(r["fixed_ridge"] == r["carry"])
            scores[f"{base}/fresh_tens"] = r["fresh_tens"] == tens
        for r in raw["patches"]:
            if r["offset"] != width or r["role"] != "answer position":
                continue
            base = f"{r['language']}/s{r['stage']}/patch/{r['control']}"
            for key in ("predicted", "observed", "target_probability", "malformed"):
                scores[f"{base}/{key}"] = r[key]
            scores[f"{base}/gradient_error"] = r["predicted"] - r["observed"]
            scores[f"{base}/target_word"] = r["first_word"] == r["target_word"]
            scores[f"{base}/unambiguous_recipient_language"] = r["languages"] == [r["language"]]
        for lang in ("English", "French"):
            for stage in range(1, 5):
                base = f"{lang}/s{stage}/patch/"
                same = scores[base + "opposite/same language/observed"]
                cross = scores[base + "opposite/cross language/observed"]
                sham = scores[base + "same carry sham/observed"]
                scores[base + "opposite_minus_sham"] = same - sham
                scores[base + "cross_minus_same_language"] = cross - same
        families.append(Family(design, design["total"], all(successes), scores))
    return families


def memory_families(root, model, phase):
    import memory

    directory = root / "memory" / model
    rows = memory.read_rows(directory / phase)
    if phase == "pilot":
        for fold in range(4):
            memory.calibrate([r for r in rows if r["family"] % 4 != fold],
                             [r for r in rows if r["family"] % 4 == fold])
    else:
        memory.calibrate(memory.read_rows(directory / "pilot"), rows)
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    families = []
    for i, group in sorted(by_family.items()):
        indexed = {(r["domain"], r["condition"], r["toggle"], r["stage"], r["role"]): r for r in group}
        scores, successes = {}, []
        for domain in memory.DOMAINS:
            for condition in ("direct", "memory", "shuffled"):
                for stage in (3, 4):
                    scene = [indexed[domain, condition, False, stage, role] for role in memory.ROLES]
                    base = f"{domain}/{condition}/s{stage}"
                    for method, column in (("native", "winner"), ("calibrated", "calibrated_winner")):
                        correct = [r[column] == r["target"] for r in scene]
                        scores[f"{base}/{method}_accuracy"] = float(np.mean(correct))
                        scores[f"{base}/{method}_all_three"] = all(correct)
                        scores[f"{base}/{method}_observers_at_world"] = float(np.mean([
                            r[column] == r["boxes"][2] for r in scene[:2]]))
                    for field in ("answer_mass", "correct_mass", "free_correct", "malformed", "outside_scene"):
                        scores[f"{base}/{field}"] = float(np.mean([r[field] for r in scene]))
                    if condition == "direct" and stage == 4:
                        successes.extend(r["free_correct"] for r in scene)
            for method, col in (("native", "winner"), ("calibrated", "calibrated_winner")):
                delta = (scores[f"{domain}/memory/s4/{method}_observers_at_world"] -
                         scores[f"{domain}/memory/s3/{method}_observers_at_world"])
                scores[f"{domain}/{method}_late_world_drift"] = delta
                new_witness = 1 - group[0]["witness"]
                original = [indexed[domain, "memory", False, 4, role] for role in memory.ROLES]
                counter = [indexed[domain, "memory", True, 4, role] for role in memory.ROLES]
                scores[f"{domain}/{method}_witness_selectivity"] = all(
                    (a[col] == a["target"] and b[col] == b["target"]) if j == new_witness else a[col] == b[col]
                    for j, (a, b) in enumerate(zip(original, counter)))
        for method in ("native", "calibrated"):
            scores[f"people_minus_registers/{method}_late_world_drift"] = (
                scores[f"people/{method}_late_world_drift"] - scores[f"registers/{method}_late_world_drift"])
        design = [i, group[0]["boxes"], group[0]["witness"]]
        families.append(Family(design, i, all(successes), scores))
    saved = read_json(directory / phase / "summary.json")
    for domain in memory.DOMAINS:
        for stage in (3, 4):
            observed = np.mean([f.scores[f"{domain}/memory/s{stage}/calibrated_accuracy"] for f in families])
            expected = saved["metrics"][f"{domain}/one/memory/stage{stage}"]["calibrated_mean_accuracy"]["mean"]
            check(np.isclose(observed, expected, rtol=0, atol=1e-9), "Reconstructed memory calibration mismatch")
    return families


def cue_families(root, model, phase):
    from cue import scene_metrics

    directory = root / "cue"
    raw = read_json(directory / f"{model}_{phase}_rows.json")
    site = read_json(directory / f"{model}_frozen_sites.json")["late_neighbor"]
    families = []
    for row in raw["rows"]:
        scenes = row["scores"]
        scores = {k: v for k, v in scene_metrics(row, site).items() if not k.startswith("neighbor_")}
        clean = all(scenes[name]["top_correct"] for name in ("intact", "reversed_assignment"))
        families.append(Family(row["scene"], row["scene"]["number"], clean, scores))
    return families


def intervals(differences, cluster_ids):
    """Resample whole clusters, keeping their original number of eligible families."""
    labels, membership = np.unique(cluster_ids, return_inverse=True)
    if len(labels) < 2:
        return [None] * differences.shape[1]
    counts = np.bincount(membership)
    totals = np.zeros((len(labels), differences.shape[1]))
    np.add.at(totals, membership, differences)
    rng = np.random.default_rng(20261121)
    draws = rng.multinomial(len(labels), np.full(len(labels), 1/len(labels)), size=4000)
    averages = (draws @ totals) / (draws @ counts)[:, None]
    return np.quantile(averages, [0.025, 0.975], axis=0).T.tolist()


def compare(left, right, task, clean_only):
    check(len(left) == len(right) and all(a.design == b.design for a, b in zip(left, right)),
          "Compared checkpoints have different families or ordering")
    pairs = [(a, b) for a, b in zip(left, right) if not clean_only or (a.clean_success and b.clean_success)]
    summary = {"eligible_families": len(pairs), "total_families": len(left), "estimates": {}}
    if not pairs:
        return summary
    keys = sorted(pairs[0][0].scores)
    for pair in pairs:
        check(all(s.scores.keys() == set(keys) for s in pair), "Family measurements differ between checkpoints")
    a, b = (np.array([[pair[j].scores[k] for k in keys] for pair in pairs], dtype=float) for j in (0, 1))
    check(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite checkpoint score")
    delta = a - b
    bounds = intervals(delta, np.arange(len(pairs)))
    clusters = [p[0].cluster for p in pairs]
    sum_bounds = intervals(delta, clusters) if task == "carry" else None
    if sum_bounds is not None:
        summary["distinct_sum_totals"] = len(set(clusters))
    for j, (k, m1, m2, diff, ci) in enumerate(zip(keys, a.mean(0), b.mean(0), delta.mean(0), bounds)):
        result = dict(ouro=m1, qwen=m2, ouro_minus_qwen=diff, paired_ci95=ci)
        if sum_bounds is not None:
            result["sum_clustered_ci95"] = sum_bounds[j]
        summary["estimates"][k] = result
    return summary


def analyze(results, source, phases, tasks):
    checked = audit(results, source)
    frames = {"completed": {}, "unavailable": {},
              "comparison": "Ouro minus Qwen on matching families. Each family contributes once to each metric.",
              "intervals": "Percentile 95% intervals from 4,000 paired resamples over families. Carry also reports whole-total cluster resampling to account for shared shams. Intervals condition on these checkpoints, fitted readers, pilot site selection and observed clean-success eligibility. One-family or one-cluster intervals are unavailable.",
              "clean_success_rules": GATES,
              "carry_sites": "The cross-checkpoint analysis reports the answer position at each stage end. Individual checkpoint outputs retain the other measured sites.",
              "calibration": "Pilot memory readers use fourfold grouped cross-fitting. Held-out memory scores apply readers fitted on the complete pilot. Carry uses separate reader-fit totals. Nearby cue offsets are selected separately for each model on its pilot."}
    readers = {"memory": memory_families, "carry": carry_families, "cue": cue_families}
    for phase in phases:
        for task in tasks:
            name = f"{task}-{phase}"
            rec = {m: checked[f"{m}-{task}-{phase}"] for m in ENGINES}
            if not all(r["status"] == "verified" for r in rec.values()):
                frames["unavailable"][name] = {
                    m: dict(status=r["status"], reason=r.get("reason")) for m, r in rec.items()}
                continue
            # Local formula implementations come from the same experiment source as the receipts.
            if task in ("memory", "cue"):
                module = f"{task}.py"
                check(file_hash(Path(__file__).with_name(module)) == file_hash(source / module),
                      f"Analysis source differs from downloaded run: {module}")
            data = [readers[task](results, m, phase) for m in ENGINES]
            entry = {
                "receipt_sha256": {m: r["receipt_sha256"] for m, r in rec.items()},
                "family_data_sha256": rec["ouro"]["family_data_sha256"],
                "all_families": compare(*data, task, False),
                "common_clean_success": compare(*data, task, True),
                "clean_families_per_model": {m: sum(f.clean_success for f in rows)
                                             for m, rows in zip(ENGINES, data)},
            }
            if task == "memory":
                entry["room_followup_gate"] = {m: read_json(results / "memory" / m / phase / "summary.json")["room_gate"]
                                               for m in ENGINES}
            frames["completed"][name] = entry
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--phase", choices=PHASES, action="append")
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--json", type=Path, help="Save all metrics and paired intervals")
    args = parser.parse_args()
    if not args.results.is_dir():
        parser.error(f"Result directory does not exist: {args.results}")
    report = analyze(args.results, args.source, args.phase or PHASES, args.task or TASKS)
    report["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for which, state in report["unavailable"].items():
        print(f"{which}: {state}")
    for which, result in report["completed"].items():
        all_rows, shared = result["all_families"], result["common_clean_success"]
        print(f"{which}: {all_rows['eligible_families']} matched families; "
              f"{shared['eligible_families']} shared clean successes")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return int(bool(report["unavailable"]))


if __name__ == "__main__":
    raise SystemExit(main())
