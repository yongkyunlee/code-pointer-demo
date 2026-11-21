#!/usr/bin/env python3
"""Make a self-contained HTML fragment from a completed Kaggle result archive."""

import argparse
import html
import json
import math
from pathlib import Path
from zipfile import ZipFile

MODELS = {"ouro": ("Ouro-1.4B", 24), "qwen": ("Qwen2.5-1.5B", 7)}
LANGUAGES = ("English", "French")
CONDITIONS = (
    ("opposite/same language", "Opposite carry", "Same language"),
    ("opposite/cross language", "Opposite carry", "Other language"),
    ("same carry sham", "Same carry", "Sham swap"),
)


def summarize_answer(row):
    languages = row["languages"]
    if row["malformed"]:
        info = "Outside digit-word vocabulary"
    elif len(languages) == 2:
        info = "English and French"
    else:
        info = languages[0]
    return {"word": row["first_word"] or "No complete word", "info": info}


def records(archive):
    data = {}
    with ZipFile(archive) as z:
        for model, (_, width) in MODELS.items():
            data[model] = {lang: [] for lang in LANGUAGES}
            root = f"results/carry-{model}-heldout/"
            paths = [n for n in z.namelist() if n.startswith(root + "family-") and n.endswith(".json")]
            if len(paths) != 100:
                raise ValueError(f"{root} contains {len(paths)} family records")
            for i in range(100):
                obj = json.loads(z.read(f"{root}family-{i:03}.json"))
                family = obj["family"]
                if family["index"] != i or len(obj["clean"]) != 2:
                    raise ValueError(f"Invalid family {i} in {root}")
                for lang in LANGUAGES:
                    clean, = [r for r in obj["clean"] if r["language"] == lang]
                    rows = [{"donor": None, "donor_lang": None, **summarize_answer(clean),
                             "margin": clean["margin"], "delta": 0.0}]
                    words = set()
                    for control, _, _ in CONDITIONS:
                        patch, = [r for r in obj["patches"] if
                                  (r["language"], r["role"], r["stage"], r["offset"], r["control"]) ==
                                  (lang, "answer position", 3, width, control)]
                        words.add((patch["clean_word"], patch["target_word"]))
                        observed = patch["observed"]
                        score_margin = patch["scores"][patch["target_word"]] - patch["scores"][patch["clean_word"]]
                        assert math.isclose(score_margin, patch["margin"], abs_tol=1e-7)
                        assert math.isclose(observed, score_margin - clean["margin"], abs_tol=1e-7)
                        assert sum(patch["donor_operands"]) == sum(clean["operands"]) == family["total"]
                        rows.append({"donor": patch["donor_operands"], "donor_lang": patch["donor_language"],
                                     **summarize_answer(patch), "margin": score_margin, "delta": observed})
                    original, target = words.pop()
                    if words:
                        raise ValueError("Digit words differ across conditions")
                    data[model][lang].append(dict(pair=clean["operands"], total=family["total"],
                                                  correct=original, target=target,
                                                  digit=family["total"]//10, target_digit=clean["target_digit"],
                                                  carry=clean["carry"], rows=rows))
    return data


CSS = """
<style>
.cp-swap { --sw-ink:#172c2b; --sw-teal:#164f4b; --sw-muted:#475d58; --sw-line:#c8d4cc; container-type:inline-size; font:inherit; color:var(--sw-ink); background:#fffdf9; border:1px solid var(--sw-line); border-radius:.7rem; padding:clamp(.8rem,2.5vw,1.45rem); margin:1.2rem 0; }
.cp-swap * { box-sizing:border-box; }
.cp-swap [hidden] { display:none!important; }
.cp-swap .sw-controls { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; padding:0; margin:0 0 1.15rem; border:0; min-width:0; }
.cp-swap .sw-controls label,.cp-swap .sw-col { display:block; font-size:.86em; color:var(--sw-muted); font-weight:650; }
.cp-swap select { min-height:2.7rem; width:100%; max-width:100%; font:inherit; font-size:1rem; color:var(--sw-ink); background:#fffdf9; border:1px solid #718d84; border-radius:.35rem; padding:.34rem .4rem; margin-top:.28rem; }
.cp-swap select:focus-visible { outline:3px solid var(--sw-teal); outline-offset:2px; }
.cp-swap .sw-scene { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem .9rem; border-left:3px solid var(--sw-teal); background:#eef4ee; padding:.7rem .8rem; margin:.55rem 0 1.05rem; }
.cp-swap .sw-scene strong { color:var(--sw-ink); font-weight:650; font-size:1.15em; }
.cp-swap .sw-meta,.cp-swap .sw-guide { color:var(--sw-muted); margin:.45rem 0; line-height:1.45; font-size:.88em; }
.cp-swap table.sw-results { width:100%; max-width:100%; border-collapse:collapse; font:inherit; table-layout:auto; margin:0; color:var(--sw-ink); }
.cp-swap .sw-results th,.cp-swap .sw-results td { border:0; border-bottom:1px solid var(--sw-line); padding:.58rem .42rem; background:transparent; vertical-align:top; text-align:left; color:inherit; font-size:.94em; line-height:1.3; }
.cp-swap .sw-results thead th { color:var(--sw-muted); font-size:.84em; }
.cp-swap .sw-results tbody th { font-weight:650; }
.cp-swap .sw-results .sw-num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.cp-swap .sw-results td b { font-weight:650; font-size:1.06em; color:var(--sw-teal); }
.cp-swap .sw-small { display:block; color:var(--sw-muted); font-weight:400; font-size:.85em; line-height:1.35; margin-top:.15rem; }
.cp-swap .sw-mobile { display:none; }
.cp-swap .sw-guide { margin:.75rem 0 0; }
.cp-swap .sw-sr { position:absolute!important; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); white-space:nowrap; }
@container (max-width:34rem) {
  .cp-swap .sw-controls { grid-template-columns:1fr 1fr; }
  .cp-swap .sw-controls label:last-child { grid-column:1/-1; }
  .cp-swap .sw-scene { grid-template-columns:1fr 1fr; }
  .cp-swap .sw-scene > div:first-child { grid-column:1/-1; }
  .cp-swap .sw-results,.cp-swap .sw-results tbody { display:block; }
  .cp-swap .sw-results thead { position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); }
  .cp-swap .sw-results tbody tr { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:.1rem .8rem; padding:.6rem 0; border-top:1px solid var(--sw-line); }
  .cp-swap .sw-results tbody th { grid-column:1/-1; }
  .cp-swap .sw-results tbody th .sw-small { display:inline; margin-left:.5rem; }
  .cp-swap .sw-results td,.cp-swap .sw-results tbody th { border:0; padding:.25rem 0; text-align:left; }
  .cp-swap .sw-mobile { display:block; font-size:.81em; color:var(--sw-muted); font-weight:600; }
}
</style>
"""


def esc(v):
    return html.escape(str(v), quote=True)


def sumtext(pair):
    return " + ".join(map(str, pair))


def signed(value):
    if abs(value) < .0005:
        return "0.000"
    return ("+" if value > 0 else "−") + f"{abs(value):.3f}"


def options(values):
    return "".join(f'<option value="{esc(key)}">{esc(label)}</option>' for key, label in values)


def render(data):
    first = data["ouro"]["English"][0]
    names = [("Clean", "Original prompt"), *((title, sub) for _, title, sub in CONDITIONS)]
    table = []
    for i, row in enumerate(first["rows"]):
        title, sub = names[i]
        table.append(
            f'<tr><th scope="row">{esc(title)}<span class="sw-small">{esc(sub)}</span></th>'
            '<td><span class="sw-mobile" aria-hidden="true">Donor</span>'
            f'<span data-field="donor">{esc(sumtext(row["donor"]) if row["donor"] else "None")}</span>'
            f'<span class="sw-small" data-field="donor_lang">{esc(row["donor_lang"] or "")}</span></td>'
            '<td><span class="sw-mobile" aria-hidden="true">First word</span>'
            f'<b data-field="word">{esc(row["word"])}</b>'
            f'<span class="sw-small" data-field="info">{esc(row["info"])}</span></td>'
            '<td class="sw-num"><span class="sw-mobile" aria-hidden="true">Margin</span>'
            f'<span data-field="margin">{signed(row["margin"])}</span></td>'
            '<td class="sw-num"><span class="sw-mobile" aria-hidden="true">Change</span>'
            f'<span data-field="delta">{signed(row["delta"])}</span></td></tr>'
        )
    family_options = options((i, f"{i:03} · total {item['total']}")
                             for i, item in enumerate(data["ouro"]["English"]))
    selectors = (
        '<fieldset class="sw-controls" hidden><legend class="sw-sr">Choose a recorded swap</legend>'
        f'<label>Checkpoint<select data-select="model">{options((k,v[0]) for k,v in MODELS.items())}</select></label>'
        f'<label>Recipient language<select data-select="lang">{options((k,k) for k in LANGUAGES)}</select></label>'
        f'<label>Held-out family<select data-select="family">{family_options}</select></label></fieldset>'
    )
    data_json = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    for char in "<>&\u2028\u2029":
        data_json = data_json.replace(char, f"\\u{ord(char):04x}")
    return CSS + f"""
<div class="cp-swap" role="region" aria-label="Recorded stage-three carry swaps">
{selectors}
<p class="sw-meta" data-field="intro">Ouro-1.4B · stage 3 · shared block 24 · family 000 · English</p>
<div class="sw-scene">
  <div><span class="sw-col">Recipient sum</span><strong data-field="sum">{esc(sumtext(first['pair']))} = {first['total']}</strong></div>
  <div><span class="sw-col">Correct tens digit</span><strong data-field="correct">{esc(first['correct'])} ({first['digit']})</strong></div>
  <div><span class="sw-col">With donor’s opposite carry</span><strong data-field="target">{esc(first['target'])} ({first['target_digit']})</strong></div>
</div>
<table class="sw-results" role="table"><caption class="sw-sr">Greedy first words and full-word scores after changing one answer-position state</caption>
<thead><tr><th scope="col">Condition</th><th scope="col">Donor</th><th scope="col">First word</th><th scope="col" class="sw-num">Margin</th><th scope="col" class="sw-num">Change</th></tr></thead>
<tbody>{''.join(table)}</tbody></table>
<p class="sw-guide">The margin subtracts the log probability of the correct word from the log probability of the word predicted by transferring the opposite carry. Change is relative to the clean prompt. Scores use natural logarithms and include every token of each word.</p>
<p class="sw-sr" aria-live="polite" aria-atomic="true" data-field="status"></p>
<template class="sw-records">{data_json}</template>
{SCRIPT}
</div>
"""


SCRIPT = """
<script>
(() => {
  const root = document.querySelector(".cp-swap");
  const records = JSON.parse(root.querySelector(".sw-records").content.textContent);
  const field = (name, parent = root) => parent.querySelector('[data-field="' + name + '"]');
  const choice = name => root.querySelector('[data-select="' + name + '"]');
  const num = value => Math.abs(value) < .0005 ? "0.000" : (value > 0 ? "+" : "−") + Math.abs(value).toFixed(3);
  const sum = operands => operands.join(" + ");
  function update() {
    const model = choice("model").value;
    const lang = choice("lang").value;
    const family = Number(choice("family").value);
    const rec = records[model][lang][family];
    const label = model === "ouro" ? "Ouro-1.4B · stage 3 · shared block 24" : "Qwen2.5-1.5B · stage 3 · block 21";
    const status = label + " · family " + String(family).padStart(3, "0") + " · " + lang;
    field("intro").textContent = status;
    field("sum").textContent = sum(rec.pair) + " = " + rec.total;
    field("correct").textContent = rec.correct + " (" + rec.digit + ")";
    field("target").textContent = rec.target + " (" + rec.target_digit + ")";
    root.querySelectorAll(".sw-results tbody tr").forEach((tr, i) => {
      const item = rec.rows[i];
      field("donor",tr).textContent = item.donor ? sum(item.donor) : "None";
      field("donor_lang",tr).textContent = item.donor_lang || "";
      field("word",tr).textContent = item.word;
      field("info",tr).textContent = item.info;
      field("margin",tr).textContent = num(item.margin);
      field("delta",tr).textContent = num(item.delta);
    });
    field("status").textContent = "Results updated. " + status + ". Recipient sum " + sum(rec.pair) + ".";
  }
  root.querySelector(".sw-controls").addEventListener("change", update);
  root.querySelector(".sw-controls").hidden = false;
})();
</script>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.write_text(render(records(args.archive)), encoding="utf-8")


if __name__ == "__main__":
    main()
