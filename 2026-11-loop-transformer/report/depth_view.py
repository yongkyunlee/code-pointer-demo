"""Render the recorded block-by-block answer readouts for the two checkpoints."""

import math
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib import rc_context
from matplotlib.ticker import MaxNLocator

from plot_reader_figures import write_svg


MODELS = {
    "ouro": ("Ouro-1.4B", 24, "Looped", "Pass"),
    "qwen": ("Qwen2.5-1.5B", 7, "Regular", "Group"),
}
TASKS = {"memory": "Moving coin", "carry": "Carry swap", "locker": "Locker clue"}
EXAMPLES = {
    "carry": "28 + 47 and 60 + 15 both equal 75. We copied a hidden state from the first sum, which needs a carry, into the second sum. If only the carry transferred, its tens digit would become eight.",
    "memory": "The coin moved from L to A to B. Where is it now? In one run the question could read the whole story. In the other, eight positions read the story for the first half, then the question could read those stored states.",
    "locker": "Tom uses the gray locker, which holds an apple. The model saw the record twice. At two depths we blocked the second copy’s locker descriptions from reading Tom’s assignment in the first copy.",
}
TEMPLATE = Path(__file__).with_name("depth_view.html")
LINE_COLORS = ("#425963", "#a94325")


def number(value):
    if abs(value) < .0005:
        return "0.000"
    return f"{value:+.3f}".replace("-", "−")


def visible_token(point):
    token = point["top_token"]
    if token == "":
        return f"empty token {point['top_token_id']}"
    return (token.replace(" ", "␠").replace("\n", "↵").replace("\r", "␍")
            .replace("\t", "⇥").replace("\u00a0", "⍽"))


def tile_color(value, limit):
    magnitude = min(1, abs(value) / limit)
    endpoint = (183, 78, 45) if value < 0 else (15, 100, 92)
    neutral = (238, 239, 229)
    rgb = [round(low + (high - low) * magnitude) for low, high in zip(neutral, endpoint)]
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def validate(model, entry):
    width = MODELS[model][1]
    conditions = entry["conditions"]
    if len(conditions) != 2:
        raise ValueError("Each depth example must include the two measured conditions")
    for condition in conditions:
        points = condition["points"]
        if len(points) != 4 * width:
            raise ValueError(f"{model} has an incomplete measured depth trace")
        for depth, point in enumerate(points, 1):
            stage, column = divmod(depth - 1, width)
            block = column + 1 if model == "ouro" else depth
            if (point["depth"], point["block"], point["stage"]) != (depth, block, stage + 1):
                raise ValueError(f"Incorrect block identity for {model} depth {depth}")
            if (not math.isfinite(point["margin"]) or not isinstance(point["top_token"], str)
                    or not isinstance(point["top_token_id"], int) or point["top_token_id"] < 0):
                raise ValueError(f"Incomplete readout at {model} depth {depth}")
        for event in condition.get("events", []):
            if event["depth"] not in range(1, 4 * width + 1):
                raise ValueError(f"Event outside {model}'s block stack")


def focus_depth(task, model, entry):
    if task == "memory":
        return 4 * MODELS[model][1]
    events = entry["conditions"][1].get("events", [])
    if events:
        return events[0]["depth"]
    return 3 * MODELS[model][1]


def chart(model, task, conditions, events, limit, start, temp):
    width = MODELS[model][1]
    end = 4 * width
    fig, ax = plt.subplots(figsize=(5.7, 2.34), dpi=72)
    fig.subplots_adjust(left=.12, right=.96, top=.94, bottom=.20)
    fig.patch.set_facecolor("#fffefa")
    ax.set_facecolor("#fffefa")
    ax.set(xlim=(1, end), ylim=(-limit, limit))
    ax.set_xticks([1, width, 2 * width, 3 * width, end])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True, prune="both"))
    ax.tick_params(axis="both", labelsize=8, colors="#53655e", length=0, pad=4)
    ax.set_xlabel("Executed depth", size=8, labelpad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#dbe0d6", lw=.6)
    ax.axhline(0, color="#85978e", lw=.8)
    for k in (1, 2, 3):
        ax.axvline(k * width + .5, color="#c5cec4", lw=.7, linestyle=":")
    for event in events:
        ax.plot(event, .975, marker="v", markersize=4.2, color="#753a27",
                transform=ax.get_xaxis_transform(), clip_on=False)
    xs = range(1, end + 1)
    for condition, color, linestyle in zip(conditions, LINE_COLORS, ("-", (0, (4, 2)))):
        ax.plot(xs, [p["margin"] for p in condition["points"]], lw=1.65,
                color=color, linestyle=linestyle, solid_capstyle="round")
    fig.canvas.draw()
    bottom, top = (fig.bbox.height - ax.bbox.ymin, fig.bbox.height - ax.bbox.ymax)
    left, right = ax.bbox.xmin, ax.bbox.xmax
    path = temp / f"depth-{model}-{task}.svg"
    description = " and ".join(c["label"] for c in conditions)
    write_svg(fig, path, f"{MODELS[model][0]} answer margin at every executed depth",
              f"{description} across {end} blocks. The point values are in the all-depths table.")
    x = left + (start - 1) * (right - left) / (end - 1)
    cursor = (f'<g data-chart-cursor="" data-left="{left:.4f}" data-right="{right:.4f}" '
              f'data-top="{top:.4f}" data-bottom="{bottom:.4f}" data-limit="{limit}" '
              f'aria-hidden="true"><line x1="{x:.4f}" x2="{x:.4f}" y1="{top:.4f}" '
              f'y2="{bottom:.4f}" stroke="#213e37" stroke-width=".9" stroke-dasharray="2 2"/>')
    for condition, color in zip(conditions, LINE_COLORS):
        value = condition["points"][start - 1]["margin"]
        y = bottom - (value + limit) / (2 * limit) * (bottom - top)
        cursor += (f'<circle cx="{x:.4f}" cy="{y:.4f}" r="2.8" fill="{color}" '
                   'stroke="#fffefa" stroke-width="1"/>')
    svg = path.read_text(encoding="utf-8").replace("</svg>", cursor + "</g></svg>")
    return svg.replace("<svg ", '<svg class="dv-chart" ', 1)


def detail_row(condition, depth):
    points = condition["points"]
    point = points[depth - 1]
    change = number(point["margin"] - points[depth - 2]["margin"]) if depth > 1 else "—"
    return (f'<tr><th scope="row">{escape(condition["label"])}</th><td data-margin="">{number(point["margin"])}</td>'
            f'<td data-prior-change="">{change}</td><td><code data-token="">{escape(visible_token(point))}</code></td></tr>')


def card(model, task, entry, plot_limit, color_limit, temp):
    name, width, architecture, stage_label = MODELS[model]
    reference, treatment = entry["conditions"]
    color_run = 0 if task == "memory" else 1
    a, b = reference["points"], treatment["points"]
    start = focus_depth(task, model, entry)
    chosen = b[start - 1]
    end = len(b)
    by_depth = {}
    for event in treatment.get("events", []):
        by_depth.setdefault(event["depth"], []).append(event["label"])
    event_text = {d: treatment["label"] + " · " + " · ".join(values) for d, values in by_depth.items()}
    stage_rows = []
    table_rows = []
    for stage in range(1, 5):
        cells = []
        for depth in range((stage - 1) * width + 1, stage * width + 1):
            before, after = a[depth - 1], b[depth - 1]
            block = after["block"]
            value = after["margin"]
            event = event_text.get(depth, "")
            attrs = {
                "depth": str(depth), "block": str(block), "stage": str(stage),
                "a": str(before["margin"]), "b": str(value), "ta": visible_token(before),
                "tb": visible_token(after), "event": event,
            }
            encoded = " ".join(f'data-{k}="{escape(v, quote=True)}"' for k, v in attrs.items())
            active_margin = before["margin"] if color_run == 0 else value
            active_label = entry["conditions"][color_run]["label"]
            text = f"Depth {depth}, block {block}, {active_label} margin {number(active_margin)}"
            if event:
                text += ". " + event
            same_weight = " dv-reused" if model == "ouro" and block == chosen["block"] else ""
            current = " dv-current" if depth == start else ""
            has_event = " dv-event" if event else ""
            fill_a, fill_b = tile_color(before["margin"], color_limit), tile_color(value, color_limit)
            cells.append(f'<button type="button" class="dv-tile{current}{same_weight}{has_event}" '
                         f'style="--dv-fill-a:{fill_a};--dv-fill-b:{fill_b}" {encoded} tabindex="-1" disabled '
                         f'aria-label="{escape(text, quote=True)}" title="{escape(text, quote=True)}" '
                         f'aria-pressed="{str(depth == start).lower()}"><span>{block}</span></button>')
            table_rows.append(f'<tr><td>{depth}</td><td>{stage}</td><td>{block}</td>'
                              f'<td>{number(before["margin"])}</td><td>{number(value)}</td></tr>')
        stage_rows.append(f'<div class="dv-maprow"><button class="dv-jump" type="button" '
                          f'data-jump="{stage}" disabled>{stage_label} {stage}</button>'
                          f'<div class="dv-tiles">{"".join(cells)}</div></div>')
    ticks = []
    for n in range(1, width + 1):
        text = str(n) if n in (1, width) or (width == 24 and n % 4 == 0) else ""
        ticks.append(f'<span>{text}</span>')
    axis = ('<div class="dv-maprow dv-axis" aria-hidden="true"><span>Block</span>'
            f'<div class="dv-tiles">{"".join(ticks)}</div></div>') if model == "ouro" else ""
    description = ("The same 24 blocks run on each pass. A column follows one set of weights."
                   if model == "ouro" else "Each numbered block has its own weights and runs once.")
    offset = (start - 1) % width + 1
    repeat_rows = []
    for stage in range(1, 5):
        depth = (stage - 1) * width + offset
        before, after = a[depth - 1], b[depth - 1]
        selected = str(depth == start).lower()
        cells = "".join(
            f'<td><code data-visit-token="">{escape(visible_token(point))}</code>'
            f'<small data-visit-margin="">{number(point["margin"])}</small></td>'
            for point in (before, after)
        )
        repeat_rows.append(
            f'<tr data-same-stage="{stage}"><th scope="row"><button type="button" data-visit="{stage}" '
            f'aria-pressed="{selected}" disabled><b>{stage_label} {stage}</b>'
            f'<span data-visit-site="">Depth {depth} · block {after["block"]}</span></button></th>{cells}</tr>'
        )
    heading = ("Shared block" if model == "ouro" else "Position")
    explanation = ("across four passes" if model == "ouro" else "in four distinct block groups")
    repeat = (f'<div class="dv-repeat"><h5>{heading} <span data-repeat-block="">{offset}</span> {explanation}</h5>'
              f'<table><caption>Projected top token and answer margin at each visit</caption><thead>'
              f'<tr><th scope="col">Block visit</th><th scope="col">{escape(reference["label"])}</th>'
              f'<th scope="col">{escape(treatment["label"])}</th></tr></thead><tbody>'
              f'{"".join(repeat_rows)}</tbody></table></div>')
    detail = detail_row(reference, start) + detail_row(treatment, start)
    legend = "".join(f'<span>{escape(condition["label"])}</span>' for condition in (reference, treatment))
    unit = "pass" if model == "ouro" else "group"
    return f"""
<article class="dv-card dv-{model}" data-model="{model}" data-width="{width}" data-unit="{unit}">
<header class="dv-cardhead"><p>{architecture} Transformer</p><h4>{name}</h4><div>{description}</div></header>
<div class="dv-map" style="--dv-count:{width}">{axis}{"".join(stage_rows)}</div>
<div class="dv-slider" hidden><button type="button" data-step="-1" aria-label="Previous depth in {name}">−</button>
<label><span class="dv-sr">Executed depth for {name}, {TASKS[task]}</span>
<input type="range" min="1" max="{end}" value="{start}" step="1" aria-label="Executed depth for {name}, {TASKS[task]}"></label>
<button type="button" data-step="1" aria-label="Next depth in {name}">+</button></div>
<div class="dv-inspect" aria-live="polite" aria-atomic="true"><h5 data-location="">Depth {start} of {end} · {unit} {chosen['stage']} · block {chosen['block']}</h5>
<p class="dv-eventnote" data-event-note=""{'' if start in event_text else ' hidden'}>{escape(event_text.get(start, ''))}</p>
<table><thead><tr><th scope="col">Run</th><th scope="col">Margin</th><th scope="col">Δ from prior depth</th><th scope="col">Top token</th></tr></thead><tbody>{detail}</tbody></table></div>
{repeat}
<figure class="dv-curve"><figcaption>Answer margin across all {end} depths</figcaption>
<div class="dv-lines">{legend}</div>{chart(model, task, (reference, treatment), by_depth, plot_limit, start, temp)}</figure>
<details class="dv-all"><summary>Read all {end} block scores</summary><div tabindex="0" aria-label="All block scores for {name}">
<table><thead><tr><th>Depth</th><th>{stage_label}</th><th>Block</th><th>{escape(reference['label'])}</th><th>{escape(treatment['label'])}</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div></details></article>"""


def render(data: dict) -> str:
    """Return a standalone, progressively enhanced explorer from the measured traces."""
    panels = []
    with TemporaryDirectory() as work, rc_context({"svg.fonttype": "none", "svg.hashsalt": "loop-depth-view",
                                                   "font.family": "sans-serif", "font.size": 9}):
        for task in TASKS:
            for model in MODELS:
                validate(model, data[model]["tasks"][task])
            entry = data["ouro"]["tasks"][task]
            other = data["qwen"]["tasks"][task]
            if any(entry[k] != other[k] for k in ("positive", "negative")):
                raise ValueError("Token margin definitions must agree across the two models")
            values = [abs(p["margin"]) for m in MODELS for c in data[m]["tasks"][task]["conditions"]
                      for p in c["points"]]
            limit = max(1, math.ceil(max(values)))
            shade = "a" if task == "memory" else "b"
            labels = [c["label"] for c in entry["conditions"]]
            selected_label = labels[0 if shade == "a" else 1]
            switches = []
            for key, label in zip(("a", "b"), labels):
                checked = " checked" if key == shade else ""
                switches.append(f'<label><input type="radio" name="dv-shade-{task}" value="{key}"{checked}>'
                                f'<span>{escape(label)}</span></label>')
            both = "".join(card(model, task, data[model]["tasks"][task], limit, limit, Path(work))
                           for model in MODELS)
            positive, negative = escape(entry["positive"]), escape(entry["negative"])
            prompt = f'<pre>{escape(entry["prompt"])}</pre>'
            if entry["prompt"] != other["prompt"]:
                prompt = "".join(f'<p>{MODELS[m][0]}</p><pre>{escape(data[m]["tasks"][task]["prompt"])}</pre>'
                                 for m in MODELS)
            if task == "memory":
                width = MODELS["ouro"][1]
                visits = entry["conditions"][0]["points"][width - 1::width]
                chain = " → ".join(escape(point["top_token"].strip()) for point in visits)
                orientation = (f'<p class="dv-example"><b>Follow block {width}.</b> With the full story, '
                               f'Ouro’s head read <strong>{chain}</strong> from the same block at the ends of '
                               'its four passes. Qwen reaches four different blocks at the ends of its groups.</p>')
            elif task == "locker":
                first_changes = []
                for checkpoint in (entry, other):
                    intact, cut = checkpoint["conditions"]
                    first_changes.append(next(p["depth"] for p, q in zip(intact["points"], cut["points"])
                                              if p["margin"] != q["margin"]))
                orientation = (f'<p class="dv-example"><b>Step forward from the first cut.</b> The cut changed '
                               f'which words the locker descriptions could read. The fruit margin at the answer '
                               f'position first changed one block later, at depth {first_changes[0]} in Ouro '
                               f'and {first_changes[1]} in Qwen.</p>')
            else:
                ouro_swap = entry["conditions"][1]["events"][0]["depth"]
                qwen_swap = other["conditions"][1]["events"][0]["depth"]
                orientation = (f'<p class="dv-example"><b>Inspect the swap.</b> At depth {ouro_swap} in Ouro '
                               f'and {qwen_swap} in Qwen, compare the two runs at the selected square. Step back '
                               'one block and their scores agree.</p>')
            panels.append(f"""
<div class="dv-panel dv-panel-{task}" data-task="{task}" data-shade="{shade}">
<div class="dv-taskhead"><h3>{escape(entry['subject'])}</h3>
<p>{EXAMPLES[task]}</p>{orientation}<p>The margin is the logit for <b>{positive}</b> minus the logit for <b>{negative}</b>. Both charts share the same vertical scale.</p>
<details><summary>Read the input</summary>{prompt}</details></div>
<div class="dv-key"><b class="dv-color-static">Tile color · {escape(selected_label)}</b>
<fieldset class="dv-shades" hidden><legend>Color blocks by</legend>{"".join(switches)}</fieldset>
<span>{negative} <i aria-hidden="true"></i> {positive}</span>
<small>−{limit} to +{limit} logits</small></div>
<div class="dv-cards">{both}</div></div>""")
    return TEMPLATE.read_text(encoding="utf-8").replace("{{PANELS}}", "".join(panels))
