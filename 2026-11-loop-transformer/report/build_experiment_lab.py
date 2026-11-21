#!/usr/bin/env python3
"""Build the standalone experiment lab from the saved Kaggle measurements.

The lab is a companion to index.html. It reads the same archive, reuses the
report's scene extractors, embeds every value it displays and writes only
experiment-lab.html.
"""

import argparse
import json
import math
import re
from pathlib import Path
from zipfile import ZipFile

from build_swap_explorer import CONDITIONS, records as swap_records, summarize_answer
from locker_scene import SCENE_NUMBER, records as locker_records
from memory_scene import FAMILY as MEMORY_SCENE, ROLES, _read as memory_winners

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("lab_template.html")
MODELS = {"ouro": ("Ouro-1.4B", "Looped"), "qwen": ("Qwen2.5-1.5B", "Regular")}
CARRY_FAMILY = 0
DONORS = {  # Key used on the page -> archived control name.
    "same": "opposite/same language",
    "cross": "opposite/cross language",
    "sham": "same carry sham",
}
PAIRED_KEYS = {"same": "swap", "cross": "cross_language", "sham": "sham"}
BOXES = "ABCDEFGHIJKL"
CUE_KEYS = ("intact", "target_early", "target_late", "target_both",
            "sham_early", "sham_late", "sham_both")


def load(archive, name):
    return json.loads(archive.read("results/" + name))


def interval(mean, lo, hi):
    return {"mean": mean, "lo": lo, "hi": hi}


def close(value, expected, tol, label):
    if not math.isclose(value, expected, abs_tol=tol):
        raise ValueError(f"{label} is {value}, expected {expected} from the verified findings")


def architecture(archive):
    out = {}
    for model, (name, kind) in MODELS.items():
        native = load(archive, f"{model}-native-validation.json")
        width, depth = native["blocks_per_stage"], native["executed_depth"]
        if depth != 4 * width or max(native["stage_native_max_abs_errors"]) != 0:
            raise ValueError(f"{model} does not match the four-stage native layout")
        out[model] = {"name": name, "kind": kind, "width": width, "depth": depth,
                      "unique": width if model == "ouro" else depth,
                      "revision": native["checkpoint"][1]}
    return out


# Experiment 1 ---------------------------------------------------------------

def carry(archive, arch):
    data = {"models": {}, "aggregate": {}, "readers": {}, "clean": {}}
    stage3 = swap_records(archive.filename)
    for model in MODELS:
        width = arch[model]["width"]
        manifest = load(archive, f"carry-{model}-heldout/manifest.json")
        if manifest["primary_offset"] != width:
            raise ValueError("Carry swaps must sit at the last block of each stage")
        prefix = manifest["prompt"]
        family = load(archive, f"carry-{model}-heldout/family-{CARRY_FAMILY:03}.json")
        if family["family"]["sums"] != [[60, 15], [28, 47]] or family["family"]["total"] != 75:
            raise ValueError("Family 000 no longer holds 60 + 15 and 28 + 47")
        clean, = [r for r in family["clean"] if r["language"] == "English"]
        if clean["operands"] != [60, 15] or clean["carry"] != 0 or clean["target_digit"] != 8:
            raise ValueError("Unexpected English recipient in family 000")
        if clean["prompt"] != prefix + "60 + 15 | English:":
            raise ValueError("Recipient prompt differs from the manifest prefix")
        stages = {}
        for stage in range(1, 5):
            rows = {}
            for key, control in DONORS.items():
                patch, = [p for p in family["patches"] if
                          (p["language"], p["role"], p["stage"], p["offset"], p["control"]) ==
                          ("English", "answer position", stage, width, control)]
                if (patch["target_word"], patch["clean_word"]) != ("eight", "seven"):
                    raise ValueError("Family 000 must compare eight with seven")
                margin = patch["scores"]["eight"] - patch["scores"]["seven"]
                if not math.isclose(margin - clean["margin"], patch["observed"], abs_tol=1e-6):
                    raise ValueError("Recorded change disagrees with the two margins")
                summary = summarize_answer(patch)
                rows[key] = {"donor": patch["donor_operands"], "donor_lang": patch["donor_language"],
                             "word": summary["word"], "info": summary["info"],
                             "text": patch["answer_text"], "malformed": patch["malformed"],
                             "margin": margin, "delta": patch["observed"]}
            stages[stage] = rows
        # The report's explorer already extracted the stage-three swaps. Both must agree.
        explorer = stage3[model]["English"][CARRY_FAMILY]["rows"]
        for i, key in enumerate(("same", "cross", "sham"), 1):
            mine, theirs = stages[3][key], explorer[i]
            if (mine["word"], mine["donor"]) != (theirs["word"], theirs["donor"]) or \
                    not math.isclose(mine["delta"], theirs["delta"], abs_tol=1e-9):
                raise ValueError(f"Stage-three swap differs from build_swap_explorer for {model} {key}")
        clean_summary = summarize_answer(clean)
        data["models"][model] = {
            "clean": {"word": clean_summary["word"], "info": clean_summary["info"],
                      "text": clean["answer_text"], "margin": clean["margin"]},
            "stages": stages,
        }
        summary = {tuple(r["group"]): r for r in load(archive, f"carry-{model}-heldout/summary.json")}
        agg, readers, accuracy = {}, {}, {}
        for lang in ("English", "French"):
            accuracy[lang] = summary["clean", lang, "greedy correct word"]["mean"]
            agg[lang], readers[lang] = {}, {}
            for stage in range(1, 5):
                row = {}
                for key, control in DONORS.items():
                    r = summary["patch", lang, stage, width, "answer position", control, "observed"]
                    row[key] = interval(r["mean"], r["lo"], r["hi"])
                r = summary["paired", lang, stage, width, "answer position", "opposite minus sham"]
                row["vs_sham"] = interval(r["mean"], r["lo"], r["hi"])
                agg[lang][stage] = row
                readers[lang][stage] = {
                    kind: interval(*(summary["reader", lang, stage, name][k] for k in ("mean", "lo", "hi")))
                    for kind, name in (("fresh", "fresh_ridge"), ("fixed", "fixed_ridge"))}
        data["aggregate"][model], data["readers"][model], data["clean"][model] = agg, readers, accuracy
        data["prompt_prefix"] = prefix
    return data


# Experiment 2 ---------------------------------------------------------------

def story(boxes, witness):
    """Mirror memory.facts for the people domain with one witness."""
    a, b, c = boxes
    mid = " ".join(f"{name} sees this move: {'yes' if witness == i else 'no'}."
                   for i, name in enumerate(("Alice", "Bob")))
    return (f"People remember the last box they saw the coin in.\nThe coin is in box {a}. "
            f"Alice and Bob both see it.\nThe coin moves to box {b}. {mid}\n"
            f"The coin moves to box {c}. Neither person sees this move.\n")


def memory(archive, arch):
    boxes, witness, winners = memory_winners(archive)
    data = {"scene": MEMORY_SCENE, "boxes": boxes, "witness": witness,
            "gold": [boxes[int(i == witness)] for i in range(2)] + [boxes[2]],
            "story": story(boxes, witness), "models": {}, "aggregate": {}}
    for model in MODELS:
        summary = load(archive, f"memory/{model}/heldout/summary.json")
        width = arch[model]["width"]
        if summary["boundary_blocks"] != [2 * width, 3 * width, 4 * width] or len(summary["slot_ids"]) != 8:
            raise ValueError("Memory layout differs from the eight-slot, stage-two design")
        if summary["example"]["facts"] != story(["A", "B", "C"], 1):
            raise ValueError("Story text no longer mirrors memory.facts")
        if list(summary["answer_ids"]) != list(BOXES):
            raise ValueError("Letter logits are no longer ordered A to L")
        scene = load(archive, f"memory/{model}/heldout/scene-{MEMORY_SCENE:03}.json")
        ranks = {}
        for row in scene["rows"]:
            if row["domain"] != "people" or row["toggle"] or row["condition"] not in ("direct", "memory"):
                continue
            order = sorted(row["boxes"], key=lambda b: -row["letter_logits"][BOXES.index(b)])
            if order[0] != row["winner"]:
                raise ValueError("Box ranking disagrees with the recorded winner")
            if row["stage"] == 4 and winners[model, row["condition"], row["role"]] != row["winner"]:
                raise ValueError("Scene ranking differs from memory_scene")
            ranks.setdefault(row["condition"], {}).setdefault(str(row["stage"]), {})[row["role"]] = {
                "order": order, "text": row["greedy_text"],
                "logits": [row["letter_logits"][BOXES.index(b)] for b in order]}
        data["models"][model] = ranks
        # Recount where every answer went across the held-out scenes.
        events = {}
        names = [n for n in archive.namelist()
                 if re.fullmatch(rf"results/memory/{model}/heldout/scene-\d{{3}}\.json", n)]
        if len(names) != summary["sizes"]["heldout"]:
            raise ValueError("Missing held-out memory scenes")
        for name in names:
            for row in json.loads(archive.read(name))["rows"]:
                if row["domain"] != "people" or row["toggle"] or row["condition"] not in ("direct", "memory"):
                    continue
                kind = "world" if row["role"] == "world" else \
                    "saw" if ROLES.index(row["role"]) == row["witness"] else "missed"
                key = f"{row['condition']}/{row['stage']}"
                cell = events.setdefault(key, {k: [0, 0, 0] for k in ("saw", "missed", "world")})
                cell[kind][row["boxes"].index(row["winner"])] += 1
        agg = {}
        for key, counts in events.items():
            condition, stage = key.split("/")
            metric = summary["metrics"][f"people/one/{condition}/stage{stage}"]
            correct = counts["saw"][1] + counts["missed"][0] + counts["world"][2]
            close(correct / 360, metric["native_mean_accuracy"]["mean"], 1e-9, f"{model} {key} recount")
            agg[key] = {"counts": counts, "correct": correct,
                        "accuracy": interval(metric["native_mean_accuracy"]["mean"],
                                             *metric["native_mean_accuracy"]["ci95"]),
                        "all_three": round(metric["native_all_three"]["mean"] * metric["native_all_three"]["families"]),
                        "mass": metric["answer_mass"]["mean"]}
        data["aggregate"][model] = agg
    return data


# Experiment 3 ---------------------------------------------------------------

def locker(archive):
    rows, summaries = locker_records(archive)
    scene = rows["ouro"]["scene"]
    data = {"scene": scene, "prompt": rows["ouro"]["prompt"], "answers": [a[0] for a in rows["ouro"]["answers"]],
            "models": {}, "aggregate": {}}
    if scene["number"] != SCENE_NUMBER or data["answers"] != ["apple", "orange"]:
        raise ValueError("Scene 035 no longer asks for Tom's apple")
    for model in MODELS:
        scores = rows[model]["scores"]
        data["models"][model] = {k: {"margin": scores[k]["margin"], "top": scores[k]["top_text"],
                                     "top_correct": scores[k]["top_correct"],
                                     "off": scores[k]["off_candidate"]} for k in CUE_KEYS}
        est = summaries[model]["estimates"]
        pick = lambda k: interval(est[k]["mean"], est[k]["lo"], est[k]["hi"])
        data["aggregate"][model] = {
            "n": summaries[model]["n_families"],
            "blocks": summaries[model]["executed_blocks_1_based"],
            "margins": {k: pick(k + ".margin") for k in CUE_KEYS},
            "target_interaction": pick("target_interaction"),
            "sham_interaction": pick("sham_interaction"),
            "target_minus_sham": pick("target_minus_sham"),
            "all_layers_cost": pick("all_layers_cost"),
            "candidate_accuracy": est["intact.candidate_accuracy"]["mean"],
            "top_correct": est["intact.top_correct"]["mean"],
            "off_candidate": est["intact.off_candidate"]["mean"],
        }
    diff = load(archive, "cue/heldout_ouro_minus_qwen.json")["estimates"]["target_interaction"]
    data["ouro_minus_qwen"] = interval(diff["mean"], diff["lo"], diff["hi"])
    return data


# Matched small models -------------------------------------------------------

def paired(archive):
    summary = load(archive, "paired/summary.json")
    config, means, contrasts = summary["config"], summary["means"], summary["contrasts"]
    if (config["stages"], config["blocks_per_stage"]) != (4, 2) or not summary["all_runs_complete"]:
        raise ValueError("Unexpected small-model design")
    parameters = {}
    for arch in ("tied", "untied", "untied_budget"):
        counts = {load(archive, f"paired/{seed}_{arch}.json")["training"]["parameters"] for seed in config["seeds"]}
        streams = {load(archive, f"paired/{seed}_{arch}.json")["training"]["stream_sha256"] for seed in config["seeds"]}
        if len(counts) != 1:
            raise ValueError("Parameter count varies across seeds")
        parameters[arch] = counts.pop()
    for seed in config["seeds"]:
        if len({load(archive, f"paired/{seed}_{a}.json")["training"]["stream_sha256"]
                for a in ("tied", "untied", "untied_budget")}) != 1:
            raise ValueError("Training streams differ within a seed")

    def entry(task, key):
        c = contrasts[task]["tied_minus_untied"][key]
        return {"tied": means[task]["tied"][key]["mean"], "untied": means[task]["untied"][key]["mean"],
                "budget": means[task]["untied_budget"][key]["mean"],
                "diff": interval(c["mean"], c["low"], c["high"]), "seeds": c["by_seed"]}

    carry_rows = {page: {stage: entry("carry", f"s{stage}_{name}_observed") for stage in range(1, 5)}
                  for page, name in PAIRED_KEYS.items()}
    lang = {stage: {k: means["carry"][k][f"s{stage}_cross_language_recipient_language"]["mean"]
                    for k in ("tied", "untied", "untied_budget")} for stage in range(1, 5)}
    readers = {stage: {"fixed": entry("carry", f"s{stage}_reader_fixed"),
                       "local": entry("carry", f"s{stage}_reader_local")} for stage in range(1, 5)}
    return {
        "config": {k: config[k] for k in ("seeds", "width", "narrow_width", "n_heldout", "steps")},
        "parameters": parameters,
        "tokens": load(archive, f"paired/{config['seeds'][0]}_tied.json")["training"]["train_input_tokens"],
        "carry": carry_rows, "readers": readers, "lang": lang,
        "clean": {k: means["carry"][k]["clean_accuracy"]["mean"] for k in ("tied", "untied", "untied_budget")},
        "memory": {k: entry("memory", k) for k in ("observer_s3_accuracy", "observer_s4_accuracy",
                                                    "observer_all_three", "observer_shuffled_accuracy")},
        "cue": {k: entry("cue", k) for k in ("two_person_accuracy", "target_interaction",
                                              "irrelevant_interaction", "target_all_layer_cost")},
    }


def check_findings(d):
    """Hold the embedded values to the audited findings in the results notes."""
    c, m, k, p = d["carry"], d["memory"], d["locker"], d["paired"]
    close(c["clean"]["ouro"]["English"], .98, 1e-9, "Ouro English clean accuracy")
    close(c["clean"]["ouro"]["French"], .10, 1e-9, "Ouro French clean accuracy")
    close(c["aggregate"]["ouro"]["English"][3]["cross"]["mean"], 1.154, 5e-4, "Ouro cross-language change")
    close(c["aggregate"]["ouro"]["English"][3]["vs_sham"]["mean"], -.034, 5e-4, "Ouro English vs sham")
    close(c["aggregate"]["ouro"]["French"][3]["vs_sham"]["mean"], .208, 5e-4, "Ouro French vs sham")
    close(c["aggregate"]["qwen"]["English"][3]["vs_sham"]["mean"], .042, 5e-4, "Qwen English vs sham")
    close(c["readers"]["ouro"]["English"][3]["fresh"]["mean"], .93, 1e-9, "Ouro stage-3 reader")
    if c["models"]["ouro"]["stages"][3]["cross"]["word"] != "sixt":
        raise ValueError("Ouro's French-donor output is no longer 'sixt'")
    if m["aggregate"]["ouro"]["direct/4"]["correct"] != 143 or m["aggregate"]["qwen"]["direct/4"]["correct"] != 168:
        raise ValueError("Full-story memory counts differ from 143 and 168 of 360")
    close(m["aggregate"]["ouro"]["memory/4"]["accuracy"]["mean"], .322, 5e-4, "Ouro memory accuracy")
    close(m["aggregate"]["qwen"]["memory/4"]["accuracy"]["mean"], .325, 5e-4, "Qwen memory accuracy")
    close(k["aggregate"]["ouro"]["margins"]["intact"]["mean"], 2.0480, 5e-5, "Ouro intact margin")
    close(k["aggregate"]["qwen"]["margins"]["intact"]["mean"], 0.2413, 5e-5, "Qwen intact margin")
    close(k["aggregate"]["ouro"]["target_minus_sham"]["mean"], .00008, 1e-5, "Ouro target minus sham")
    close(k["aggregate"]["qwen"]["target_interaction"]["mean"], -.00063, 1e-5, "Qwen interaction")
    close(p["carry"]["same"][1]["diff"]["mean"], .320, 5e-4, "Small-model stage-1 difference")
    if p["parameters"] != {"tied": 97113, "untied": 347481, "untied_budget": 91865}:
        raise ValueError("Small-model parameter counts changed")


def facts(d):
    """Strings substituted into the template prose."""
    a, p = d["arch"], d["paired"]
    o, q = a["ouro"], a["qwen"]
    return {
        "OURO_WIDTH": str(o["width"]), "OURO_DEPTH": str(o["depth"]),
        "QWEN_WIDTH": str(q["width"]), "QWEN_DEPTH": str(q["depth"]),
        "OURO_REV": o["revision"][:7], "QWEN_REV": q["revision"][:7],
        "SMALL_WIDTH": str(p["config"]["width"]), "NARROW_WIDTH": str(p["config"]["narrow_width"]),
        "SEEDS": str(len(p["config"]["seeds"])), "SMALL_N": str(p["config"]["n_heldout"]),
        "P_TIED": f"{p['parameters']['tied']:,}", "P_UNTIED": f"{p['parameters']['untied']:,}",
        "P_BUDGET": f"{p['parameters']['untied_budget']:,}", "TOKENS": f"{p['tokens']:,}",
        "MEMORY_SCENE": f"{d['memory']['scene']:03}", "LOCKER_SCENE": f"{d['locker']['scene']['number']:03}",
        "OURO_CUE_BLOCK": str(d["locker"]["aggregate"]["ouro"]["blocks"][0] - o["width"]),
        "OURO_CUE_DEPTHS": " and ".join(map(str, d["locker"]["aggregate"]["ouro"]["blocks"])),
        "QWEN_CUE_BLOCKS": " and ".join(map(str, d["locker"]["aggregate"]["qwen"]["blocks"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence/kaggle-results.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "experiment-lab.html")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "index.html").resolve():
        raise SystemExit("The lab must not overwrite the full report")
    with ZipFile(args.archive) as archive:
        arch = architecture(archive)
        data = {"arch": arch, "carry": carry(archive, arch), "memory": memory(archive, arch),
                "locker": locker(archive), "paired": paired(archive)}
    check_findings(data)
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in facts(data).items():
        page = page.replace("{{" + key + "}}", value)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for char in "<>&  ":
        payload = payload.replace(char, f"\\u{ord(char):04x}")
    page = page.replace("{{DATA}}", payload)
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", page)
    if leftover:
        raise ValueError(f"Unfilled template fields: {sorted(set(leftover))}")
    args.output.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
