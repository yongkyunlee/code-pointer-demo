"""Verify downloaded Kaggle outputs against the experiment code and raw observations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys


ENGINES = ("ouro", "qwen")
TASKS = ("carry", "memory", "cue")
PHASES = ("pilot", "heldout")


class AuditError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise AuditError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def family_records(families):
    entries = [json.dumps(f, sort_keys=True, separators=(",", ":")) for f in families]
    check(len(set(entries)) == len(entries), "Duplicate scene families")
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return {"n_families": len(entries), "family_data_sha256": digest, "_families": set(entries)}


def check_receipt(receipt, source):
    check(receipt["status"] == "complete", "Receipt did not report completion")
    check(receipt["platform"] == "kaggle", "Receipt does not identify a Kaggle run")
    gpu = receipt["gpu"]
    check(isinstance(gpu, str) and gpu.strip() and gpu.lower() != "cpu",
          "Receipt does not identify GPU hardware")
    for key in ("elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
        check(finite(receipt[key]) and receipt[key] > 0, f"Invalid hardware measurement: {key}")
    check(receipt["peak_reserved_bytes"] >= receipt["peak_allocated_bytes"],
          "GPU reservation is smaller than measured allocation")
    hashes = receipt["code_sha256"]
    check(isinstance(hashes, dict) and bool(hashes), "No source hashes in receipt")
    for name, expected in hashes.items():
        check(Path(name).name == name and name not in {".", ".."}, "Invalid source filename")
        check(file_hash(source / name) == expected, f"Source differs from Kaggle run: {name}")
    return {"gpu": gpu, "elapsed_seconds": receipt["elapsed_seconds"],
            "peak_allocated_bytes": receipt["peak_allocated_bytes"],
            "peak_reserved_bytes": receipt["peak_reserved_bytes"]}


def check_native(root, model, receipt):
    diag = read_json(root / f"{model}-native-validation.json")
    revision, width = {
        "ouro": ("574fa66cb8bf5abdc979642d01cf2b79b16bfab1", 24),
        "qwen": ("8faed761d45a263340a0528343f099c05c9a4323", 7),
    }[model]
    check(diag["checkpoint"][1] == revision and diag["blocks_per_stage"] == width
          and diag["executed_depth"] == 4*width, "Wrong checkpoint or executed depth")
    attention = "eager" if model == "ouro" else "sdpa_math"
    check(diag["attention"] == attention and diag["dtype"] == "torch.float16" and diag["use_cache"] is False,
          "Native equivalence test used a different model configuration")
    check(diag["gpu"] == receipt["gpu"] and diag["code_sha256"]["model_engine.py"] ==
          receipt["code_sha256"]["model_engine.py"], "Native equivalence and measurement provenance disagree")
    errors = diag["stage_native_max_abs_errors"]
    check(len(errors) == 4 and all(finite(x) and x >= 0 for x in errors), "Missing stage parity evidence")
    for key in ("final_native_max_abs_error", "stage_four_head_max_abs_error", "sham_final_max_abs_error"):
        check(finite(diag[key]) and diag[key] >= 0, f"Invalid native parity evidence: {key}")


def verify_carry(root, model, phase):
    directory = root / f"carry-{model}-{phase}"
    manifest = read_json(directory / "manifest.json")
    summary = {tuple(r["group"]): r for r in read_json(directory / "summary.json")}
    n = 24 if phase == "pilot" else 100
    check((manifest["model"], manifest["phase"], manifest["allocated_families"]) == (model, phase, n),
          "Carry manifest does not match requested split")
    groups = {"opposite/same language", "opposite/cross language", "same carry sham"}
    check(len(list(directory.glob("family-*.json"))) == n, "Wrong number of carry family files")
    languages = ("English", "French")
    correct = Counter()
    families = []
    for index in range(n):
        row = read_json(directory / f"family-{index:03d}.json")
        family = row["family"]
        families.append({k: family[k] for k in ("sham", "sums", "total")})
        check(family["index"] == index and family["total"] % 10 in manifest["split_units"][phase],
              "Carry family index or split mismatch")
        for group in ("sums", "sham"):
            for carry, operands in enumerate(family[group]):
                a, b = operands
                check(a + b == family["total"] and carry == int(a % 10 + b % 10 >= 10),
                      "Invalid arithmetic carry or matched total")
        check(len(row["clean"]) == 2 and {r["language"] for r in row["clean"]} == set(languages),
              "Missing bilingual clean scores")
        clean = {r["carry"]: r for r in row["clean"]}
        for r in clean.values():
            check(0 <= r["correct_word_probability"] <= 1 and finite(r["margin"]), "Invalid clean carry scores")
            correct[r["language"]] += r["digit"] == family["total"] // 10
        reader_counts = Counter((r["carry"], r["stage"]) for r in row["readers"])
        check(reader_counts == Counter(itertools.product(range(2), range(1,5))), "Missing carry readout")
        by_site = defaultdict(set)
        for patch in row["patches"]:
            c, stage, offset, role = (patch[k] for k in ("carry", "stage", "offset", "role"))
            observed = patch["margin"] - clean[c]["margin"]
            check(finite(patch["predicted"]) and math.isclose(observed, patch["observed"], abs_tol=1e-6),
                  "Carry patch margin is nonfinite or disagrees with its clean baseline")
            check(patch["target_digit"] == family["total"] // 10 + 1 - 2*c, "Wrong carry direction")
            if stage == 4 and offset == manifest["width"] and role != "answer position":
                check(abs(patch["predicted"]) < 1e-5 and abs(patch["observed"]) < 1e-5,
                      "A final-block non-answer position influenced the output")
            by_site[c, stage, offset, role].add(patch["control"])
        check(len(row["patches"]) == 54 and len(by_site) == 18 and all(v == groups for v in by_site.values()),
              "Missing or duplicate carry control, site, or direction")
        for c, stage in itertools.product(range(2), range(1,5)):
            check((c, stage, manifest["width"], "answer position") in by_site, "Missing end-of-stage carry patch")
    measures = {}
    for lang in languages:
        estimate = summary["clean", lang, "digit accuracy"]
        check(estimate["families"] == n and math.isclose(estimate["mean"], correct[lang]/n, abs_tol=1e-8),
              "Carry aggregate disagrees with raw responses")
        for metric in ("digit accuracy", "greedy correct word", "malformed", "correct word probability"):
            measures[f"{lang} {metric}"] = summary["clean", lang, metric]["mean"]
    return {**family_records(families), "measurements": measures}


def verify_memory(root, model, phase):
    directory = root / "memory" / model / phase
    summary = read_json(directory / "summary.json")
    first, stop = (0, 24) if phase == "pilot" else (24, 144)
    n = stop-first
    check((summary["checkpoint"], summary["phase"], summary["sizes"][phase]) == (model, phase, n),
          "Memory summary or split mismatch")
    check(len(list(directory.glob("scene-*.json"))) == n, "Wrong number of memory scenes")
    domains = ("people", "registers")
    roles = ("first", "second", "world")
    required = {(d, toggle, condition, role, stage)
                for d, toggle, role, stage in itertools.product(domains, (False, True), roles, (3,4))
                for condition in ("direct", "memory", "shuffled") if not (toggle and condition == "shuffled")}
    direct, through_memory = Counter(), Counter()
    families = []
    for i in range(first, stop):
        scene = read_json(directory / f"scene-{i:03d}.json")
        if i == first:
            for d in domains:
                for k in ("sham_max_abs", "source_closed_max_abs", "memory_independence_max_abs",
                          "fixed_memory_max_abs"):
                    diff = scene["checks"][d][k]
                    check(finite(diff) and diff >= 0, f"Missing source-closure control: {k}")
        rows = scene["rows"]
        families.append([rows[0]["boxes"], rows[0]["witness"]])
        observed = {tuple(r[k] for k in ("domain", "toggle", "condition", "role", "stage")) for r in rows}
        check(len(rows) == len(required) and observed == required, "Memory scene missing a condition or question")
        for r in rows:
            check(r["family"] == i and r["target"] in r["boxes"] and r["winner"] in r["boxes"],
                  "Invalid family or location in memory result")
            if not r["toggle"] and r["stage"] == 4:
                if r["condition"] == "direct":
                    direct[r["domain"]] += r["target"] == r["winner"]
                if r["condition"] == "memory":
                    through_memory[r["domain"]] += r["target"] == r["winner"]
            check(0 <= r["correct_mass"] <= r["answer_mass"] <= 1.00001, "Invalid box answer probabilities")
    output = {}
    for d in domains:
        for label, counts in (("direct", direct), ("memory", through_memory)):
            key = f"{d}/one/{label}/stage4"
            estimate = summary["metrics"][key]["native_mean_accuracy"]
            check(estimate["families"] == n and math.isclose(counts[d]/(3*n), estimate["mean"], abs_tol=1e-8),
                  "Memory accuracy disagrees with the raw answers")
            output[key] = estimate["mean"]
        output[f"{d}/late_world_intrusion"] = summary["paired_contrasts"][f"{d}/winner"]["late_world_intrusion_change"]
    gate = summary["room_gate"]["passed"]
    if gate:
        check(len(read_json(directory / "room-rows.json")) == 4*n, "Missing gated two-room measurements")
    output["two_room_gate_passed"] = gate
    return {**family_records(families), "measurements": output}


def verify_cue(root, model, phase):
    folder = root / "cue"
    raw = read_json(folder / f"{model}_{phase}_rows.json")
    summary = read_json(folder / f"{model}_{phase}_summary.json")
    sites = read_json(folder / f"{model}_frozen_sites.json")
    rows = raw["rows"]
    first, stop = (0, 24) if phase == "pilot" else (24, 124)
    check([r["scene"]["number"] for r in rows] == list(range(first, stop)),
          "Repeated-clue rows have missing or duplicate scene families")
    check((summary["model"], summary["phase"], summary["n_families"]) ==
          (model, phase, len(rows)) and summary["sites"] == sites, "Clue summary or site selection mismatch")
    check(summary["isolation"] == raw["isolation"], "Clue isolation results disagree")
    for measure, limit in (("isolated_max_abs", "isolated_max_limit"), ("isolated_rms", "isolated_rms_limit"),
                           ("altered_prefix_max_abs", "altered_prefix_limit")):
        iso = raw["isolation"]
        check(finite(iso[measure]) and 0 <= iso[measure] <= iso[limit], f"Cross-copy isolation failed: {measure}")
    neighbor = sites["late_neighbor"]
    required = {"intact", "target_all_layers", "reversed_assignment", "one_copy", "filler", "clue_first",
                f"neighbor_{neighbor}_late", f"neighbor_{neighbor}_both"}
    required.update(f"{region}_{cut}" for region in ("target", "sham") for cut in ("early", "late", "both"))
    measured = defaultdict(list)
    for row in rows:
        scores = row["scores"]
        check(required <= scores.keys(), "Clue row missing a control or cut")
        m = {}
        for name, s in scores.items():
            check(finite(s["margin"]), f"Nonfinite clue margin: {name}")
            probs = [s[k] for k in ("correct_probability", "other_probability")]
            check(all(finite(p) and 0 <= p <= 1 for p in probs) and sum(probs) <= 1.00001,
                  f"Invalid candidate probability mass: {name}")
            m[name] = s["margin"]
        measured["target_interaction"].append(m["target_early"] - m["target_both"] - m["intact"] + m["target_late"])
        measured["sham_interaction"].append(m["sham_early"] - m["sham_both"] - m["intact"] + m["sham_late"])
        measured["nearby_interaction"].append(m["target_early"] - m[f"neighbor_{neighbor}_both"]
                                             - m["intact"] + m[f"neighbor_{neighbor}_late"])
        measured["duplication_gain"].append(m["intact"] - m["one_copy"])
        measured["all_layers_cost"].append(m["intact"] - m["target_all_layers"])
        measured["intact.candidate_accuracy"].append(m["intact"] > 0)
    def geometry(row, key):
        return tuple(row["copy1"][key]) + tuple(row["copy2_queries"])
    check(sorted(geometry(r, "target") for r in rows) == sorted(geometry(r, "sham") for r in rows),
          "Clue and sham mask geometry differs across families")
    estimates = {}
    for key, sample in measured.items():
        estimate = summary["estimates"][key]
        check(all(finite(estimate[k]) for k in ("mean", "lo", "hi")) and estimate["lo"] <= estimate["hi"],
              f"Malformed clue interval: {key}")
        check(math.isclose(estimate["mean"], sum(sample) / len(sample), abs_tol=1e-8, rel_tol=1e-8),
              f"Clue aggregate differs from raw scores: {key}")
        estimates[key] = estimate
    families = [{k: v for k, v in r["scene"].items() if k != "number"} for r in rows]
    return {**family_records(families), "measurements": estimates}


def verify_paired(root, source):
    directory = root / "paired"
    summary = read_json(directory / "summary.json")
    cfg = summary["config"]
    seeds, names = (41, 73, 107), ("tied", "untied", "untied_budget")
    check(summary["all_runs_complete"] is True and summary["device"].startswith("cuda") and summary["gpu"],
          "Missing completed GPU training runs")
    check(tuple(cfg["seeds"]) == seeds and cfg["steps"] == 900 and cfg["n_pilot"] == 24
          and cfg["n_heldout"] == 128 and (cfg["stages"], cfg["blocks_per_stage"]) == (4, 2)
          and cfg["source_sha256"] == file_hash(source / "paired.py"), "Wrong matched-training design or source")
    metrics = {"carry": "s4_cross_language_observed", "memory": "observer_late_world_drift",
               "cue": "matched_interaction"}
    frames, parameters, samples = {}, {}, defaultdict(list)
    for seed in seeds:
        streams = set()
        for name in names:
            result = read_json(directory / f"{seed}_{name}.json")
            check(result["config"] == cfg and result["complete"] is True and
                  (result["architecture"], result["seed"]) == (name, seed), "Paired fit metadata mismatch")
            tr = result["training"]
            streams.add((tr["stream_sha256"], tr["train_input_tokens"]))
            check(tr["train_input_tokens"] > 0 and finite(tr["last_minibatch_loss"]), "Missing training evidence")
            parameters[name] = tr["parameters"]
            for task in TASKS:
                for phase, n in (("pilot", 24), ("heldout", 128)):
                    rows = result[phase][task]
                    scene_ids = [json.dumps(r["family"], separators=(",", ":")) for r in rows]
                    check(len(rows) == n and len(set(scene_ids)) == n, "Missing or duplicate paired families")
                    check(scene_ids == frames.setdefault((phase, task), scene_ids),
                          "Evaluation families differ between compared fits")
                    value = [r["metrics"][metrics[task]] for r in rows]
                    check(all(finite(v) for v in value), "Invalid paired fit score")
                    if phase == "heldout":
                        samples[task, name].append(sum(value) / n)
        check(len(streams) == 1, "Training streams differ across architectures within a seed")
    check(parameters["untied"] > parameters["tied"] and cfg["narrow_width"] < cfg["width"],
          "Unique parameter counts or control width do not match the paired design")
    contrasts = {}
    for task, metric in metrics.items():
        check(set(frames["pilot", task]).isdisjoint(frames["heldout", task]), "Pilot family reappeared in test")
        res = summary["contrasts"][task]["tied_minus_untied"][metric]
        differences = [a-b for a,b in zip(samples[task, "tied"], samples[task, "untied"])]
        check(len(res["by_seed"]) == len(seeds) and
              all(math.isclose(a,b,abs_tol=1e-6) for a,b in zip(differences,res["by_seed"])),
              "Paired seed differences disagree with held-out measurements")
        contrasts[task] = res
    return {"n_seeds": len(seeds), "parameters": parameters, "measurements": contrasts}


def audit(results, source):
    """Keep an independent status for each model/task/phase, including failures."""
    checks = [(engine, task, phase) for phase in PHASES for engine in ENGINES for task in TASKS]
    checks.append(("paired", "training", "control"))
    records = {}
    for engine, task, phase in checks:
        name = f"{engine}-{task}-{phase}"
        path = results / f"{name}.receipt.json"
        record = records[name] = {"status": "missing"}
        if not path.exists():
            failures = list(results.glob(f"{name}.*error*.json")) + list(results.glob(f"{name}.*fail*.json"))
            if failures:
                record.update(status="failed", files=sorted(p.name for p in failures))
            continue
        try:
            receipt = read_json(path)
            hardware = check_receipt(receipt, source)
            check((receipt["engine"], receipt["experiment"], receipt["phase"]) == (engine, task, phase),
                  "Receipt identifiers and filename disagree")
            if task == "training":
                measures = verify_paired(results, source)
            else:
                check_native(results, engine, receipt)
                checker = {"carry": verify_carry, "memory": verify_memory, "cue": verify_cue}[task]
                measures = checker(results, engine, phase)
            record.update(status="verified", receipt_sha256=file_hash(path),
                          hardware=hardware, **measures)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            record.update(status="invalid", reason=str(exc))
    for task in TASKS:
        for model in ENGINES:
            calibration, test = (records[f"{model}-{task}-{phase}"] for phase in PHASES)
            if "_families" in calibration and "_families" in test:
                if not calibration["_families"].isdisjoint(test["_families"]):
                    test.update(status="invalid", reason="Held-out family found in pilot")
        for phase in PHASES:
            pair = [records[f"{model}-{task}-{phase}"] for model in ENGINES]
            if all("family_data_sha256" in record for record in pair):
                if pair[0]["family_data_sha256"] != pair[1]["family_data_sha256"]:
                    for record in pair:
                        record.update(status="invalid", reason="Compared checkpoints scored different scene families")
    for record in records.values():
        record.pop("_families", None)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory containing downloaded receipts and raw results")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent,
                        help="Source snapshot to match against Kaggle receipts")
    parser.add_argument("--partial", action="store_true",
                        help="Allow absent jobs, but require at least one verified job and no invalid receipts")
    parser.add_argument("--json", type=Path, help="Write the full audit as JSON")
    args = parser.parse_args()
    if not args.results.is_dir():
        parser.error(f"Result directory does not exist: {args.results}")
    records = audit(args.results, args.source)
    counts = Counter(item["status"] for item in records.values())
    for name, record in records.items():
        detail = record.get("reason") or ", ".join(record.get("files", []))
        if record["status"] == "verified":
            detail = json.dumps(record.get("measurements"), ensure_ascii=False, sort_keys=True)
        print(f"{name:<24} {record['status']:<9} {detail}".rstrip())
    print(json.dumps(counts, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps({"jobs": records, "counts": counts}, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
    return int(bool(counts["invalid"] or counts["failed"] or not counts["verified"]
                    or (not args.partial and counts["missing"])))


if __name__ == "__main__":
    sys.exit(main())
