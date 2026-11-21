"""Create the reader figures from the downloaded Kaggle ZIP or its results directory."""
import argparse
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from contextlib import ExitStack
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INK = "#222d2c"
TEAL = "#006b62"
RUST = "#a0462c"
SLATE = "#485c59"
MUTED = "#54635f"
RULE = "#dfe5e1"
SVG = "http://www.w3.org/2000/svg"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
        "font.size": 10, "axes.unicode_minus": True, "svg.fonttype": "none",
        "svg.hashsalt": "code-pointer-loop-experiments", "figure.facecolor": "white",
    })


def write_svg(fig, filename, title, description):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor="white", metadata={"Date": None, "Creator": "Matplotlib"})
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg "):]
    # Matplotlib reuses element IDs across files. Prefix them for inline HTML.
    prefix = filename.stem + "-"
    ids = set(re.findall(r'\bid="([^"]+)"', svg))
    svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m[1]}"', svg)
    svg = re.sub(r'#([\w.-]+)', lambda m: "#" + prefix + m[1] if m[1] in ids else m[0], svg)
    title_id, desc_id = prefix + "title", prefix + "description"
    start = svg.index(">")
    from xml.sax.saxutils import escape
    svg = (svg[:start] + f' role="img" aria-labelledby="{title_id} {desc_id}">' +
           f'<title id="{title_id}">{escape(title)}</title>\n' +
           f'<desc id="{desc_id}">{escape(description)}</desc>\n' + svg[start + 1:])
    ET.fromstring(svg)  # Validate the edited document before writing.
    filename.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")


def label(fig, x, y, text, **kwargs):
    fig.text(x, y, text, va="center", **kwargs)


def signed(v, places):
    return f"{v:+.{places}f}".replace("-", "−")


def carry_figure(read, out):
    summary = read("paired/summary.json")
    seeds = summary["config"]["seeds"]
    mean = summary["contrasts"]["carry"]["tied_minus_untied"]
    fits = {(seed, arch): read(f"paired/{seed}_{arch}.json")["heldout"]["carry"]
            for seed in seeds for arch in ("tied", "untied")}
    fig = plt.figure(figsize=(5, 3.8))
    label(fig, .045, .926, "Carry transfer across four stages", size=15.2, weight="bold")
    label(fig, .045, .845, "Tied minus untied change in target margin (logits)", size=9.6, color=MUTED)
    ax = fig.add_axes([.18, .236, .65, .528])
    ax.set(xlim=(-.32, .46), ylim=(3.55, -.53))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([-.2, 0, .2, .4])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else signed(x, 1)))
    ax.tick_params(axis="both", length=0, labelsize=9, pad=5)
    ax.set_yticks(range(4), [f"Stage {i}" for i in range(1, 5)])
    ax.get_yticklabels()[0].set(color=TEAL, fontweight="bold")
    ax.axvspan(0, .46, facecolor="#f4f9f7", zorder=0)
    for x in (-.2, .2, .4):
        ax.axvline(x, color=RULE, linewidth=.6, zorder=1)
    ax.axvline(0, color="#778780", linewidth=.9, zorder=1)
    values = []
    for row in range(4):
        key = f"s{row+1}_swap_observed"
        entry = mean[key]
        color = TEAL if row == 0 else SLATE
        by_seed = []
        for seed in seeds:
            tied, untied = [fits[seed, arch] for arch in ("tied", "untied")]
            assert [r["family"] for r in tied] == [r["family"] for r in untied]
            diffs = [a["metrics"][key] - b["metrics"][key] for a, b in zip(tied, untied)]
            by_seed.append(sum(diffs) / len(diffs))
        assert all(abs(a-b) < 1e-7 for a,b in zip(by_seed, entry["by_seed"]))
        lo, point, hi = (entry[k] for k in ("low", "mean", "high"))
        ax.plot([lo, hi], [row, row], lw=2.6, color=color, solid_capstyle="round", zorder=3)
        ax.scatter(point, row, s=40, color=color, edgecolors="white", linewidths=.8, zorder=4)
        ax.scatter(by_seed, [row+.22]*len(seeds), marker="|", s=27, linewidths=1.2, color=RUST, zorder=3)
        ax.text(1.04, row, signed(point, 3), va="center", ha="left",
                transform=ax.get_yaxis_transform(), size=9.2, color=color, weight="bold" if row == 0 else "normal")
        values.append(f"stage {row+1}, {signed(point,3)} [{signed(lo,3)}, {signed(hi,3)}]")
    # Use figure-relative positions so legends are independent of the data limits.
    legend = fig.add_axes([.045, .112, .93, .05])
    legend.set(xlim=(0,1), ylim=(0,1)); legend.axis("off")
    legend.plot([.008, .078], [.5, .5], lw=2.5, color=TEAL, solid_capstyle="round")
    legend.scatter([.044], [.5], s=25, color=TEAL, edgecolors="white", linewidths=.65, zorder=3)
    legend.text(.098, .5, "Mean and 95% interval", va="center", size=9.1)
    legend.scatter([.568, .584, .600], [.5]*3, marker="|", s=30, color=RUST)
    legend.text(.627, .5, "Individual seeds", va="center", size=9.1)
    label(fig, .045, .058, "Held-out families grouped by sum. Intervals condition on three seeds.", size=8, color=MUTED)
    desc = ("Same-language opposite-carry swaps. Difference in mean logit change, tied minus untied. " +
            "; ".join(values) + ". Individual seeds are shown below each interval.")
    write_svg(fig, out / "carry-stage-differences.svg", "Carry transfer across four stages", desc)


def cue_figure(read, out):
    fig = plt.figure(figsize=(5, 3.12))
    label(fig, .045, .918, "Which fruit did the model choose?", size=15.2, weight="bold")
    label(fig, .045, .82, "Correct answers on the same 100 intact locker scenes", size=9.5, color=MUTED)
    ax = fig.add_axes([.33, .203, .565, .512])
    ax.set(xlim=(0, 100), ylim=(3.55, -.65))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_yticks([.38, 2.38], ["Ouro-1.4B", "Qwen2.5-1.5B"])
    ax.tick_params(axis="both", length=0, labelsize=9.5, pad=7)
    for item in ax.get_yticklabels():
        item.set_fontweight("bold")
    for x in (0,25,50,75,100):
        ax.axvline(x, color=RULE, linewidth=.7, zorder=0)
    counts = []
    for i, model in enumerate(("ouro", "qwen")):
        rows = read(f"cue/{model}_heldout_rows.json")["rows"]
        summary = read(f"cue/{model}_heldout_summary.json")
        n = len(rows)
        assert n == 100
        scores = [r["scores"]["intact"] for r in rows]
        counts.append((sum(s["margin"] > 0 for s in scores), sum(s["top_correct"] for s in scores)))
        for j, (key, value, color) in enumerate(zip(("candidate_accuracy", "top_correct"), counts[-1], (TEAL, RUST))):
            assert abs(value/n - summary["estimates"]["intact."+key]["mean"]) < 1e-6
            y = 2*i + j*.72
            if value:
                ax.barh(y, value, height=.39, color=color, zorder=2)
            else:
                ax.plot(0, y, marker="|", color=color, markersize=9, zorder=2, clip_on=False)
            ax.annotate(str(value), (value, y), xytext=(5, 0), textcoords="offset points",
                        va="center", size=9.5, color=color, weight="bold")
    leg = fig.add_axes([.046, .026, .93, .065])
    leg.set(xlim=(0,1), ylim=(0,1));leg.axis("off")
    for pos, color, name in ((.013,TEAL,"Ranked first of two fruits"),(.547,RUST,"Top token overall")):
        leg.scatter(pos,.5,marker="s",s=35,color=color)
        leg.text(pos+.032,.5,name,va="center",size=9.1)
    desc = f"Ouro ranked the correct fruit first in {counts[0][0]} of 100 scenes and selected it as its top token in {counts[0][1]}. Qwen did so in {counts[1][0]} and {counts[1][1]} scenes, respectively."
    write_svg(fig, out / "locker-answer-scoring.svg", "Which fruit did the model choose?", desc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path, help="Kaggle results ZIP, extracted results directory, or its parent")
    p.add_argument("--outdir", type=Path, default=Path.cwd())
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    style()
    with ExitStack() as stack:
        if args.evidence.is_file():
            archive = stack.enter_context(zipfile.ZipFile(args.evidence))
            def read(name):
                return json.loads(archive.read("results/" + name))
        else:
            root = args.evidence / "results" if (args.evidence / "results").is_dir() else args.evidence
            def read(name):
                return json.loads((root / name).read_text(encoding="utf-8"))
        carry_figure(read, args.outdir)
        cue_figure(read, args.outdir)


if __name__ == "__main__":
    main()
