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
    # Both columns must describe the same donor and recipient for every setting.
    for lang in LANGUAGES:
        for i, (ouro, qwen) in enumerate(zip(data["ouro"][lang], data["qwen"][lang], strict=True)):
            if any(ouro[k] != qwen[k] for k in ("pair", "total", "correct", "target", "carry")):
                raise ValueError(f"Recipient mismatch across checkpoints for {lang} family {i}")
            for left, right in zip(ouro["rows"], qwen["rows"], strict=True):
                if (left["donor"], left["donor_lang"]) != (right["donor"], right["donor_lang"]):
                    raise ValueError(f"Donor mismatch across checkpoints for {lang} family {i}")
    return data


CSS = """
<style>
.cp-swap { --sw-ink:#172c2b; --sw-teal:#164f4b; --sw-muted:#475d58; --sw-line:#c8d4cc; container-type:inline-size; font:inherit; color:var(--sw-ink); background:#fffdf9; border:1px solid var(--sw-line); border-radius:.7rem; padding:clamp(.8rem,2.5vw,1.45rem); margin:1.2rem 0; }
.cp-swap * { box-sizing:border-box; }
.cp-swap [hidden] { display:none!important; }
.cp-swap .sw-controls { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.8rem; padding:0; margin:0 0 1.15rem; border:0; min-width:0; }
.cp-swap .sw-controls label,.cp-swap .sw-col { display:block; font-size:.86em; color:var(--sw-muted); font-weight:650; }
.cp-swap select { min-height:2.7rem; width:100%; max-width:100%; font:inherit; font-size:1rem; color:var(--sw-ink); background:#fffdf9; border:1px solid #718d84; border-radius:.35rem; padding:.34rem .4rem; margin-top:.28rem; }
.cp-swap select:focus-visible { outline:3px solid var(--sw-teal); outline-offset:2px; }
.cp-swap .sw-picker { margin:.2rem 0 .9rem; display:flex; align-items:baseline; flex-wrap:wrap; gap:.4rem .8rem; }
.cp-swap .sw-picker label { font-size:.9em; font-weight:650; }
.cp-swap .sw-picker select { width:auto; max-width:100%; margin-top:0; min-height:2.5rem; }
.cp-swap .sw-pair { display:grid; grid-template-columns:minmax(0,1fr) clamp(34px,9cqw,96px) minmax(0,1fr); gap:.35rem; margin:.7rem 0; }
.cp-swap .sw-board { border:1px solid var(--sw-line); background:#f4f5ee; border-radius:.5rem; padding:clamp(.55rem,2cqw,1rem); min-width:0; }
.cp-swap [data-board="recipient"] { background:#edf4ee; }
.cp-swap .sw-arrow { align-self:center; text-align:center; color:var(--sw-teal); font-weight:650; line-height:1.2; }
.cp-swap .sw-arrow b { display:block; font-size:clamp(21px,3cqw,31px); }
.cp-swap .sw-arrow small { font-size:.75rem; }
.cp-swap .sw-pair:has([data-board="donor"][hidden]) { grid-template-columns:repeat(2,minmax(0,1fr)); }
.cp-swap .sw-pair:has([data-board="donor"][hidden]) .sw-arrow { display:none; }
@container (max-width:34rem) { .cp-swap .sw-arrow small { display:none; } }
.cp-swap .sw-board-title { display:flex; gap:.2rem .6rem; flex-wrap:wrap; justify-content:space-between; align-items:baseline; font-size:.88em; }
.cp-swap .sw-board-title strong { color:var(--sw-teal); }
.cp-swap .sw-board-title span { color:var(--sw-muted); }
.cp-swap .sw-columns { width:max-content; margin:.6rem auto; color:var(--sw-ink); font:500 clamp(1.55rem,4.3cqw,2.2rem)/1.15 ui-monospace,SFMono-Regular,Consolas,monospace; }
.cp-swap .sw-line { display:grid; grid-template-columns:.75em 1em 1em; text-align:center; }
.cp-swap .sw-line span:last-child { background:#d9e7d9; }
.cp-swap .sw-line:last-child { border-top:2px solid #546f64; font-weight:700; padding-top:.07em; }
.cp-swap .sw-calc { line-height:1.4; margin:.35rem 0 0; text-align:center; font:600 clamp(.75rem,2.1cqw,.96rem)/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }
.cp-swap .sw-empty { min-height:100%; display:grid; place-content:center; text-align:center; border:1px dashed var(--sw-line); border-radius:.5rem; padding:.65rem; color:var(--sw-muted); }
.cp-swap .sw-prediction { margin:.9rem 0 .65rem; color:var(--sw-ink); }
.cp-swap .sw-prediction strong { color:var(--sw-teal); }
.cp-swap table.sw-both { width:100%; table-layout:fixed; border-collapse:collapse; line-height:1.35; }
.cp-swap .sw-both th,.cp-swap .sw-both td { padding:.45rem .55rem; border-bottom:1px solid var(--sw-line); text-align:left; vertical-align:top; overflow-wrap:anywhere; }
.cp-swap .sw-both th:first-child { width:30%; }
.cp-swap .sw-both tbody th { font-size:.84em; font-weight:600; }
.cp-swap .sw-both thead th { font-size:.92em; color:var(--sw-teal); background:#edf4ee; }
.cp-swap .sw-both .sw-measured { font-weight:650; color:var(--sw-teal); font-variant-numeric:tabular-nums; }
.cp-swap .sw-both caption { text-align:left; font-weight:650; padding:.35rem 0 .55rem; }
.cp-swap .sw-trace-picker { display:flex; align-items:baseline; flex-wrap:wrap; gap:.6rem; margin:.7rem 0; }
.cp-swap .sw-trace-picker select { width:auto; margin:0; }
.cp-swap .sw-comparison { margin:.8rem 0 0; }
.cp-swap .sw-comparison summary { color:var(--sw-teal); cursor:pointer; font-weight:650; }
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
  .cp-swap .sw-both th,.cp-swap .sw-both td { padding:.4rem .22rem; font-size:.84em; }
  .cp-swap .sw-board-title { flex-direction:column; }
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


def addition_board(role, pair, lang):
    """Keep the visual digits quiet to readers; provide the equation in one phrase."""
    a, b = pair
    units = a % 10 + b % 10
    def line(operator, number, label):
        return (f'<div class="sw-line"><span>{operator}</span>'
                f'<span data-digit="{label}-tens">{number // 10}</span>'
                f'<span data-digit="{label}-ones">{number % 10}</span></div>')
    return (
        f'<div class="sw-board" data-board="{role.lower()}">'
        f'<div class="sw-board-title"><strong>{role}</strong><span data-part="language">{esc(lang)}</span></div>'
        f'<span class="sw-sr" data-part="equation">{a} + {b} = {a+b}.</span>'
        f'<div class="sw-columns" aria-hidden="true">{line("",a,"a")}{line("+",b,"b")}{line("",a+b,"result")}</div>'
        f'<p class="sw-calc" data-part="units">{a%10} + {b%10} = {units} → carry {units//10}</p></div>'
    )


def model_comparison(data):
    rows = []
    sites = {
        "ouro": "Shared block 24 on pass 3 (visit 72 of 96)",
        "qwen": "Distinct block 21 of 28",
    }
    for measure, label in (("site", "End of stage 3"), ("clean", "Clean first word"),
                           ("recorded", "First word with this setting"), ("delta", "Change in word margin")):
        cells = []
        for key in MODELS:
            runs = data[key]["English"][0]["rows"]
            values = {"site": sites[key], "clean": runs[0]["word"],
                      "recorded": runs[2]["word"], "delta": signed(runs[2]["delta"])}
            tracked = ' data-field="recorded"' if measure == "recorded" and key == "ouro" else ""
            css = ' class="sw-measured"' if measure in ("recorded", "delta") else ""
            cells.append(f'<td data-out="{key}-{measure}"{tracked}{css}>{esc(values[measure])}</td>')
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    return ('<table class="sw-both"><caption>Both models on the selected prompt</caption><thead><tr>'
            '<th scope="col">Measurement</th><th scope="col">Looped<span class="sw-small">Ouro-1.4B</span></th>'
            '<th scope="col">Regular<span class="sw-small">Qwen2.5-1.5B</span></th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def render(data, architecture_path=""):
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
    selected = first["rows"][2]
    a, b = first["pair"]
    donor_carry = sum(number % 10 for number in selected["donor"]) // 10
    prediction = f"{a // 10} + {b // 10} + {donor_carry} = {first['target_digit']}"
    selectors = (
        '<fieldset class="sw-controls" hidden><legend class="sw-sr">Choose a recorded swap</legend>'
        f'<label>Recipient language<select data-select="lang">{options((k,k) for k in LANGUAGES)}</select></label>'
        f'<label>Held-out family<select data-select="family">{family_options}</select></label></fieldset>'
    )
    data_json = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    for char in "<>&\u2028\u2029":
        data_json = data_json.replace(char, f"\\u{ord(char):04x}")
    return CSS + f"""
<div class="cp-swap" role="region" aria-label="Recorded stage-three carry swaps">
{selectors}
<p class="sw-meta" data-field="selection">Family 000 · English · Opposite carry · other language</p>
<div class="sw-picker" hidden><label for="sw-condition">Intervention</label>
<select id="sw-condition" data-select="condition">
<option value="0">Clean prompt</option>
<option value="1">Opposite carry · same language</option>
<option value="2" selected>Opposite carry · other language</option>
<option value="3">Same carry · sham swap</option>
</select></div>
{architecture_path}
<div class="sw-pair">
{addition_board("Donor",selected['donor'],selected['donor_lang'])}
<div class="sw-empty" data-part="empty-donor" hidden>Clean prompt<br>Original state</div>
<div class="sw-arrow" aria-hidden="true"><b>→</b><small>Copy state</small></div>
{addition_board("Recipient",first['pair'],"English")}
</div>
<p class="sw-meta" data-field="operation">The run replaced the recipient’s whole hidden state at its last prompt token with the donor’s state. If only the carry transferred, the recipient would keep its own tens operands.</p>
<p class="sw-prediction"><span data-field="prediction-label">Carry-only prediction</span> · <strong data-field="prediction">{esc(first['target'])}</strong>
  (<span data-field="tens-calculation">{esc(prediction)}</span>). Correct tens word · <strong data-field="clean-correct">{esc(first['correct'])}</strong>.</p>
{model_comparison(data)}
<p class="sw-guide">A positive change means the full word <b data-field="margin-target">{esc(first['target'])}</b> gained log probability relative to <b data-field="margin-original">{esc(first['correct'])}</b>, compared with that model’s clean prompt.</p>
<details class="sw-comparison"><summary>Inspect all four settings for one checkpoint</summary>
<label class="sw-trace-picker" hidden>Detailed scores for <select data-select="model">{options((key,v[0]) for key,v in MODELS.items())}</select></label>
<p class="sw-meta" data-field="intro">Ouro-1.4B · family 000 · English</p>
<table class="sw-results" role="table"><caption class="sw-sr">Greedy first words and full-word scores after changing one answer-position state</caption>
<thead><tr><th scope="col">Condition</th><th scope="col">Donor</th><th scope="col">First word</th><th scope="col" class="sw-num">Margin</th><th scope="col" class="sw-num">Change</th></tr></thead>
<tbody>{''.join(table)}</tbody></table>
<p class="sw-guide">Each margin uses the opposite-carry word minus the correct word, including in the same-carry sham. Scores include every token of each word and use natural logarithms.</p></details>
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
  const part = (name, parent = root) => parent.querySelector('[data-part="' + name + '"]');
  function setBoard(role, operands, lang) {
    const board = root.querySelector('[data-board="' + role + '"]');
    board.hidden = !operands;
    if (!operands) return;
    const [a,b] = operands;
    const total = a+b;
    for (const [name, value] of [["a",a],["b",b],["result",total]]) {
      board.querySelector('[data-digit="' + name + '-tens"]').textContent = Math.floor(value/10);
      board.querySelector('[data-digit="' + name + '-ones"]').textContent = value % 10;
    }
    const units = a%10 + b%10;
    part("units",board).textContent = (a%10) + " + " + (b%10) + " = " + units + " → carry " + Math.floor(units/10);
    part("language",board).textContent = lang;
    part("equation",board).textContent = sum(operands) + " = " + total + ".";
  }
  function update() {
    const model = choice("model").value;
    const lang = choice("lang").value;
    const family = Number(choice("family").value);
    const rec = records[model][lang][family];
    const label = choice("model").selectedOptions[0].text;
    const status = label + " · family " + String(family).padStart(3, "0") + " · " + lang;
    field("intro").textContent = status;
    const condition = Number(choice("condition").value);
    const row = rec.rows[condition];
    setBoard("recipient", rec.pair, lang);
    setBoard("donor", row.donor, row.donor_lang);
    part("empty-donor").hidden = Boolean(row.donor);
    field("operation").textContent = row.donor
      ? "The run replaced the recipient’s whole hidden state at its last prompt token with the donor’s state. If only the carry transferred, the recipient would keep its own tens operands."
      : "The clean run kept the recipient’s original hidden state. Its correct tens digit comes from adding its two tens operands and its own carry.";
    const pair = row.donor || rec.pair;
    const carry = Math.floor((pair[0]%10 + pair[1]%10) / 10);
    const [left, right] = rec.pair.map(value => Math.floor(value/10));
    const predictedDigit = left + right + carry;
    const predictedWord = predictedDigit === rec.digit ? rec.correct : rec.target;
    field("clean-correct").textContent = rec.correct;
    field("prediction-label").textContent = row.donor ? "Carry-only prediction" : "Arithmetic prediction";
    field("prediction").textContent = predictedWord;
    field("tens-calculation").textContent = left + " + " + right + " + " + carry + " = " + predictedDigit;
    const words = [];
    for (const [key, byLanguage] of Object.entries(records)) {
      const runs = byLanguage[lang][family].rows;
      const output = measure => root.querySelector('[data-out="' + key + '-' + measure + '"]');
      output("clean").textContent = runs[0].word;
      const result = output("recorded");
      result.textContent = runs[condition].word;
      if (key === model) result.dataset.field = "recorded";
      else result.removeAttribute("data-field");
      output("delta").textContent = num(runs[condition].delta);
      words.push((key === "ouro" ? "Ouro" : "Qwen") + " said " + runs[condition].word);
    }
    field("selection").textContent = "Family " + String(family).padStart(3, "0") + " · " + lang + " · " + choice("condition").selectedOptions[0].text;
    field("margin-target").textContent = rec.target;
    field("margin-original").textContent = rec.correct;
    root.querySelectorAll(".sw-results tbody tr").forEach((tr, i) => {
      const item = rec.rows[i];
      field("donor",tr).textContent = item.donor ? sum(item.donor) : "None";
      field("donor_lang",tr).textContent = item.donor_lang || "";
      field("word",tr).textContent = item.word;
      field("info",tr).textContent = item.info;
      field("margin",tr).textContent = num(item.margin);
      field("delta",tr).textContent = num(item.delta);
    });
    field("status").textContent = status + ". " + choice("condition").selectedOptions[0].text + ". Prediction " + predictedWord + ". " + words.join(". ") + ".";
  }
  root.querySelector(".sw-controls").addEventListener("change", update);
  choice("condition").addEventListener("change", update);
  choice("model").addEventListener("change", update);
  root.querySelector(".sw-trace-picker").hidden = false;
  root.querySelector(".sw-controls").hidden = false;
  root.querySelector(".sw-picker").hidden = false;
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
