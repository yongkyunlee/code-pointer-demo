"""Rebuild the standalone reader report from the saved Kaggle measurements."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from build_swap_explorer import records, render
from locker_scene import render as locker_scene
from memory_scene import render as memory_scene
from stage_map import render as stage_map
from plot_reader_figures import carry_figure, cue_figure, style


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("template.html")


def signed(value):
    return f"{value:+.3f}".replace("-", "−")


def stage_table(summary):
    contrasts = summary["contrasts"]["carry"]["tied_minus_untied"]
    rows = []
    for stage in range(1, 5):
        effect = contrasts[f"s{stage}_swap_observed"]
        rows.append(
            f'<tr><th scope="row">{stage}</th><td>{signed(effect["mean"])}</td>'
            f'<td>{signed(effect["low"])} to {signed(effect["high"])}</td></tr>'
        )
    return (
        '<table class="matrix"><caption>Tied minus untied, in logits</caption>'
        '<thead><tr><th scope="col">Stage</th><th scope="col">Mean</th>'
        '<th scope="col">95% interval</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence/kaggle-results.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "index.html")
    args = parser.parse_args()
    html = TEMPLATE.read_text(encoding="utf-8")
    with ZipFile(args.archive) as archive, TemporaryDirectory() as temp:
        def read_result(name):
            return json.loads(archive.read("results/" + name))

        style()
        fig_dir = Path(temp)
        carry_figure(read_result, fig_dir)
        cue_figure(read_result, fig_dir)
        for marker, filename in (
            ("EFFECT_PLOT", "carry-stage-differences.svg"),
            ("CLUE_PLOT", "locker-answer-scoring.svg"),
        ):
            svg = (fig_dir / filename).read_text(encoding="utf-8")
            html = html.replace("{{" + marker + "}}", svg.replace("<svg ", '<svg class="plot" ', 1))
        summary = read_result("paired/summary.json")
        html = html.replace("{{EFFECT_TABLE}}", stage_table(summary))
        html = html.replace("{{STAGE_MAP}}", stage_map(summary))
        html = html.replace("{{MEMORY_SCENE}}", memory_scene(archive))
        html = html.replace("{{LOCKER_SCENE}}", locker_scene(archive))
    html = html.replace("{{SWAP_EXPLORER}}", render(records(args.archive)))
    args.output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
