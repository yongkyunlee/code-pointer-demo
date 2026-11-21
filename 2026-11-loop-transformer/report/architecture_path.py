"""Map each frozen-checkpoint intervention to the layers actually executed."""
import json


def load(archive, path):
    return json.loads(archive.read("results/" + path))


def block_path(model, width, stage):
    start = (stage - 1) * width + 1
    end = stage * width
    first_weight = 1 if model == "ouro" else start
    last_weight = width if model == "ouro" else end
    label = "Pass" if model == "ouro" else "Group"
    return (f'<span class="ap-stage">{label} {stage} · depth {start}–{end}</span>'
            f'<strong class="ap-weights">Blocks {first_weight}–{last_weight}</strong>')


def memory_step(summary, stage, end):
    memory_text = {
        1: "Slots read the story", 2: f"Save 8 states after {end}",
        3: "Read slots · story closed", 4: "Read slots · story closed",
    }[stage]
    items = []
    for mode, operation in (("memory", memory_text), ("direct", "Full story available")):
        score = ""
        if stage >= 3:
            measure = summary["metrics"][f"people/one/{mode}/stage{stage}"]["native_mean_accuracy"]
            score = f'<b class="ap-score">{measure["mean"]:.1%} correct</b>'
        items.append(f'<span class="ap-{mode}">{operation}{score}</span>')
    return "".join(items)


def render(archive, task):
    headings = {
        "carry": "Where the hidden state moves",
        "memory": "Where the questions lose access to the story",
        "locker": "Which blocks can read the first clue",
    }
    notes = {
        "carry": "The explorer uses the end of stage 3 in both checkpoints. Ouro has used block 24 three times by this point. Qwen reaches block 21 for the first time. Both then finish stage 4 before producing an answer.",
        "memory": "Correct means the native readout ranked the right box highest among three. Each percentage covers 360 questions in 120 scenes. Chance is 33.3%. Stage 3 uses an intermediate readout, and stage 4 uses the final readout.",
        "locker": "Ouro returns to the same attention block on two passes. Qwen reaches two different attention blocks at corresponding points in its stack. A cut blocks the specified clue-to-description connection at that visit.",
    }
    selector = ""
    if task == "memory":
        selector = """
<fieldset class="ap-controls"><legend>Access condition</legend>
<label><input type="radio" name="ap-access" id="ap-memory" checked><span>Eight stored positions</span></label>
<label><input type="radio" name="ap-access" id="ap-direct"><span>Full story</span></label>
</fieldset>"""
    lanes = []
    for model, title, architecture in (("ouro", "Ouro-1.4B", "Looped"),
                                       ("qwen", "Qwen2.5-1.5B", "Regular")):
        native = load(archive, f"{model}-native-validation.json")
        width, depth = native["blocks_per_stage"], native["executed_depth"]
        if depth != width * 4:
            raise ValueError(f"Expected four stages for {model}")
        if task == "memory":
            summary = load(archive, f"memory/{model}/heldout/summary.json")
            if summary["boundary_blocks"] != [width * n for n in (2, 3, 4)] or len(summary["slot_ids"]) != 8:
                raise ValueError("Archived memory layout differs from the report")
        if task == "locker":
            cue = load(archive, f"cue/{model}_heldout_summary.json")
            early, late = cue["executed_blocks_1_based"]
        cells = []
        for stage in range(1, 5):
            end = stage * width
            mark = ""
            detail = ""
            if task == "memory":
                mark = " ap-memory-stage" if stage >= 2 else ""
                detail = memory_step(summary, stage, end)
            elif task == "carry" and stage == 3:
                mark = " ap-carry-site"
                detail = f'<span class="ap-swap">Swap after depth {end}</span><span class="ap-clean">Clean · no swap</span>'
            elif task == "locker" and stage in (2, 3):
                execution = early if stage == 2 else late
                weight = execution - (stage - 1) * width if model == "ouro" else execution
                mark = " ap-cut-site"
                detail = (f'<span>Block {weight} · depth {execution}</span>'
                          '<span class="ap-open">Open</span><span class="ap-cut">Cut here</span>')
            cells.append(f'<li class="ap-cell{mark}" data-stage="{stage}">{block_path(model, width, stage)}'
                         f'<div class="ap-action">{detail}</div></li>')
        lanes.append(f'<div class="ap-track ap-{model}"><div class="ap-name"><b>{architecture}</b>'
                     f'<strong>{title}</strong></div><ol aria-label="{title} block visits">{"".join(cells)}</ol></div>')
    return (f'<figure class="arch-path arch-{task}" id="path-{task}" aria-labelledby="path-{task}-title">'
            f'<h4 id="path-{task}-title">{headings[task]}</h4>{selector}'
            f'<div class="ap-lanes">{"".join(lanes)}</div><figcaption>{notes[task]}</figcaption></figure>')
