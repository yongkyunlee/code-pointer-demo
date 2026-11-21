"""Render one recorded repeated-clue intervention, including its four outcomes."""
import json
from html import escape
from zipfile import ZipFile


CONDITIONS = (
    ("intact", "No cut", ()),
    ("target_early", "Stage 2 only", (2,)),
    ("target_late", "Stage 3 only", (3,)),
    ("target_both", "Both stages", (2, 3)),
)
MODELS = (("ouro", "Ouro-1.4B"), ("qwen", "Qwen2.5-1.5B"))
SCENE_NUMBER = 35


def signed(value):
    return f"{value:+.3f}".replace("-", "−")


def records(archive):
    rows, summaries = {}, {}
    for key, _ in MODELS:
        items = json.loads(archive.read(f"results/cue/{key}_heldout_rows.json"))["rows"]
        rows[key] = next(row for row in items if row["scene"]["number"] == SCENE_NUMBER)
        summaries[key] = json.loads(archive.read(f"results/cue/{key}_heldout_summary.json"))
    assert rows["ouro"]["scene"] == rows["qwen"]["scene"]
    assert rows["ouro"]["prompt"] == rows["qwen"]["prompt"]
    return rows, summaries


def record_card(scene, number):
    colors, fruits = scene["colors"], scene["fruits"]
    target, decoy = scene["people"]
    art = []
    for color, fruit in zip(colors, fruits):
        art.append(f'<div class="lk-door lk-{escape(color)}"><div class="lk-vent" aria-hidden="true"></div>'
                   f'<strong>{escape(color.title())} locker</strong><span>{escape(fruit)}</span>'
                   '<i class="lk-handle" aria-hidden="true"></i></div>')
    target_line = f"{target} uses the {colors[scene['choice']]} locker."
    decoy_line = f"{decoy} uses the {colors[scene['decoy']]} locker."
    lines = [(target_line, True), (decoy_line, False)]
    if not scene["target_first"]:
        lines.reverse()
    notes = "".join(
        f'<p class="{"lk-key" if selected and number == 1 else "lk-line"}">{escape(line)}</p>'
        for line, selected in lines)
    mark = " lk-queries" if number == 2 else ""
    return (f'<section class="lk-record" aria-label="Record copy {number}">'
            f'<h4>Record copy {number}</h4><div class="lk-doors{mark}">{"".join(art)}</div>'
            f'<div class="lk-lines">{notes}</div></section>')


def token_label(text):
    marks = {" ": "␠", "\n": "↵", "\r": "␍", "\t": "⇥"}
    return escape("".join(marks.get(character, character) for character in text) or "(empty token)")


def results_for(key, label, rows, summaries):
    cards = []
    for model, name in MODELS:
        row = rows[model]
        value = row["scores"][key]
        right, wrong = (item[0] for item in row["answers"])
        preferred = right if value["margin"] > 0 else wrong if value["margin"] < 0 else "Tie"
        summary = summaries[model]
        mean = summary["estimates"][key + ".margin"]["mean"]
        cards.append(
            f'<section class="lk-result"><h5>{name}</h5><dl>'
            f'<div><dt>Scene margin</dt><dd class="lk-score">{signed(value["margin"])}</dd></div>'
            f'<div><dt>Higher-ranked fruit</dt><dd>{escape(preferred)}</dd></div>'
            f'<div><dt>Greedy next token</dt><dd>“{token_label(value["top_text"])}”</dd></div>'
            f'<div class="lk-mean"><dt>Mean margin, {summary["n_families"]} scenes</dt>'
            f'<dd>{signed(mean)}</dd></div></dl></section>')
    return (f'<div class="lk-outcome lk-for-{key}" role="group" aria-label="{label} scores">'
            f'{"".join(cards)}</div>')


CSS = """
<style>
#locker-scene{--lk-ink:#213833;--lk-muted:#4b6059;--lk-teal:#175950;--lk-rust:#92452e;--lk-line:#c2ccc0;
 color:var(--lk-ink);background:#faf8f0;border:1px solid var(--lk-line);border-radius:10px;
 padding:clamp(14px,3vw,30px);margin-block:22px;min-width:0;container-type:inline-size;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#locker-scene *{box-sizing:border-box}
#locker-scene h3,#locker-scene h4,#locker-scene h5,#locker-scene p,#locker-scene dl,#locker-scene dd{margin:0}
#locker-scene h3{font:normal clamp(24px,3.2vw,36px)/1.13 Georgia,serif;color:var(--lk-ink);letter-spacing:-.035em}
#locker-scene .lk-intro,#locker-scene .lk-foot{margin-top:8px;font-size:15px;line-height:1.55;color:var(--lk-muted);max-width:76ch}
#locker-scene fieldset{margin:20px 0 16px;min-width:0;border:0;padding:0}
#locker-scene legend,#locker-scene .lk-result-title{font-family:inherit;font-size:15px;font-weight:650;line-height:1.4}
#locker-scene .lk-options{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}
#locker-scene .lk-options label{position:relative;cursor:pointer}
#locker-scene .lk-options input{position:absolute;width:1px;height:1px;opacity:0}
#locker-scene .lk-options span{display:block;padding:7px 13px;border:1px solid var(--lk-line);border-radius:20px;font-size:14px;line-height:1.4;font-weight:600;background:#fffdf7}
#locker-scene .lk-options input:checked+span{background:var(--lk-teal);border-color:var(--lk-teal);color:#fff}
#locker-scene .lk-options input:focus-visible+span{outline:3px solid #99552d;outline-offset:2px}
#locker-scene .lk-map{display:grid;grid-template-columns:minmax(0,1fr) minmax(130px,.58fr) minmax(0,1fr);
 gap:clamp(10px,1.8vw,18px);align-items:center}
#locker-scene .lk-record{align-self:stretch;background:#fffdf8;border:1px solid var(--lk-line);border-radius:7px;padding:12px}
#locker-scene .lk-record h4{color:var(--lk-ink);font-size:15px;line-height:1.5;font-weight:600;margin:0 0 8px}
#locker-scene .lk-doors{display:grid;grid-template-columns:1fr 1fr;gap:5px;padding:5px;border:1px solid transparent;border-radius:5px}
#locker-scene .lk-queries{border:1px dashed var(--lk-teal);background:#edf4ec}
#locker-scene .lk-door{position:relative;background:var(--lk-door);border:1px solid #758680;
 border-radius:4px;padding:9px 8px 13px;min-width:0;color:#20342f;box-shadow:inset 3px 0 0 #ffffff69}
#locker-scene .lk-blue{--lk-door:#c7dbdc}#locker-scene .lk-gray{--lk-door:#d5d9cf}
#locker-scene .lk-vent{width:22px;height:6px;border-block:2px solid #687b72;margin:0 0 9px}
#locker-scene .lk-door strong{display:block;line-height:1.15;font-size:12px;font-weight:650}
#locker-scene .lk-door span{display:inline-block;margin-top:5px;padding:1px 4px;border:1px solid #7b897b;border-radius:2px;
 background:#fff9e8;font-family:ui-monospace,monospace;font-size:12px;line-height:1.3}
#locker-scene .lk-handle{display:block;position:absolute;width:4px;height:10px;border-radius:2px;background:#566961;bottom:5px;right:4px}
#locker-scene .lk-lines{margin-top:8px;display:grid;gap:4px}
#locker-scene .lk-lines p{font-size:13px;line-height:1.4;padding:3px 6px}
#locker-scene .lk-key{background:#dfeee3;border-left:3px solid var(--lk-teal);font-weight:650}
#locker-scene .lk-route{display:grid;gap:10px;color:var(--lk-teal)}
#locker-scene .lk-route h4{margin:0;color:var(--lk-muted);font-size:12px;line-height:1.4;font-weight:600}
#locker-scene .lk-lane{display:grid;gap:2px}
#locker-scene .lk-lane small{font:600 11px/1.4 ui-monospace,monospace}
#locker-scene .lk-gate{display:flex;align-items:center;gap:4px;color:var(--lk-teal);font-size:12px;font-weight:700;line-height:1.3}
#locker-scene .lk-gate:before,#locker-scene .lk-gate:after{content:"";border-top:2px solid currentColor;flex:1;min-width:4px}
#locker-scene .lk-gate:after{flex:0 1 13px}#locker-scene .lk-end{order:1;width:0;height:0;border-block:4px solid transparent;
 border-left:6px solid currentColor;margin-left:-6px}
#locker-scene .lk-gate .lk-stop{display:none}
#locker-scene:has(#lk-target_early:checked) .lk-s2 .lk-gate,#locker-scene:has(#lk-target_late:checked) .lk-s3 .lk-gate,
#locker-scene:has(#lk-target_both:checked) .lk-gate{color:var(--lk-rust)}
#locker-scene:has(#lk-target_early:checked) .lk-s2 .lk-go,#locker-scene:has(#lk-target_late:checked) .lk-s3 .lk-go,
#locker-scene:has(#lk-target_both:checked) .lk-go{display:none}
#locker-scene:has(#lk-target_early:checked) .lk-s2 .lk-stop,#locker-scene:has(#lk-target_late:checked) .lk-s3 .lk-stop,
#locker-scene:has(#lk-target_both:checked) .lk-stop{display:inline}
#locker-scene .lk-explain{margin:11px 0 17px;font-size:14px;line-height:1.5;color:var(--lk-muted)}
#locker-scene .lk-results{border-top:1px solid var(--lk-line);padding-top:15px}
#locker-scene .lk-outcome{display:none;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:8px}
#locker-scene:has(#lk-intact:checked) .lk-for-intact,#locker-scene:has(#lk-target_early:checked) .lk-for-target_early,
#locker-scene:has(#lk-target_late:checked) .lk-for-target_late,#locker-scene:has(#lk-target_both:checked) .lk-for-target_both{display:grid}
#locker-scene .lk-result{background:#fffdf8;border:1px solid var(--lk-line);border-radius:6px;padding:10px 12px}
#locker-scene .lk-result h5{font-size:14px;color:var(--lk-teal);margin:0 0 5px}
#locker-scene .lk-result dl>div{display:flex;align-items:baseline;justify-content:space-between;gap:8px;line-height:1.45;font-size:13px;padding-block:2px}
#locker-scene .lk-result dt{color:var(--lk-muted)}#locker-scene .lk-result dd{font-weight:650}
#locker-scene .lk-score,#locker-scene .lk-mean dd{font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums}
#locker-scene .lk-result .lk-mean{margin-top:5px;padding-top:6px;border-top:1px dotted var(--lk-line)}
#locker-scene details{margin-top:10px}#locker-scene details summary{font-size:13px;color:var(--lk-teal);cursor:pointer}
#locker-scene .lk-prompt{margin-top:6px;color:var(--lk-muted);font-size:13px;line-height:1.6}
@container(max-width:630px){
 #locker-scene .lk-map{grid-template-columns:minmax(0,1fr);gap:8px}
 #locker-scene .lk-record{padding:10px}#locker-scene .lk-doors{max-width:340px}
 #locker-scene .lk-route{grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 14px;padding:0 8px}
 #locker-scene .lk-route h4{grid-column:1/-1}#locker-scene .lk-end{transform:rotate(90deg)}
}
@container(max-width:440px){
 #locker-scene .lk-outcome{grid-template-columns:minmax(0,1fr);gap:6px}
 #locker-scene .lk-options span{padding:6px 9px}
}
@media print{#locker-scene fieldset{display:none}#locker-scene .lk-result,#locker-scene .lk-record{break-inside:avoid}}
</style>
"""


def render(archive: ZipFile) -> str:
    rows, summaries = records(archive)
    scene = rows["ouro"]["scene"]
    target = escape(scene["people"][0])
    color = escape(scene["colors"][scene["choice"]])
    correct, other = (escape(item[0]) for item in rows["ouro"]["answers"])
    options = []
    for key, label, _ in CONDITIONS:
        checked = " checked" if key == "intact" else ""
        options.append(f'<label><input type="radio" name="lk-mask" id="lk-{key}" value="{key}"'
                       f' aria-controls="lk-route lk-results"{checked}><span>{label}</span></label>')
    lanes = "".join(
        f'<div class="lk-lane lk-s{stage}" data-stage="{stage}"><small>Stage {stage}</small>'
        '<span class="lk-gate"><span class="lk-go">Open</span><span class="lk-stop">× Blocked</span>'
        '<i class="lk-end" aria-hidden="true"></i></span></div>' for stage in (2, 3))
    results = "".join(results_for(key, label, rows, summaries) for key, label, _ in CONDITIONS)
    return CSS + f"""
<section id="locker-scene" aria-labelledby="lk-title">
 <h3 id="lk-title">Which fruit is in {target}’s locker?</h3>
 <p class="lk-intro">In held-out scene {SCENE_NUMBER}, {target} uses the {color} locker. It holds an {correct}. The model sees the record twice before answering.</p>
 <fieldset><legend>Block access to {target}’s first clue</legend><div class="lk-options">{''.join(options)}</div></fieldset>
 <div class="lk-map">
  {record_card(scene, 1)}
  <div class="lk-route" id="lk-route" aria-label="Information flow from the first clue to the second copy’s descriptions">
   <h4>{target}’s clue → repeated locker descriptions</h4>{lanes}</div>
  {record_card(scene, 2)}
 </div>
 <p class="lk-explain">At a blocked visit, the two outlined locker descriptions in copy 2 cannot read the highlighted clue in copy 1. Every word stays in the prompt. The final question can still attend to both copies.</p>
 <div class="lk-results" id="lk-results" aria-live="polite" aria-atomic="true">
  <h4 class="lk-result-title">Final answer with this mask</h4>{results}
 </div>
 <p class="lk-foot">The margin compares “{correct}” with “{other}” in logits. A positive value favors {correct}. The greedy next token is the model’s first choice across its full vocabulary. The symbol ␠ marks a space.</p>
 <details><summary>Read the original prompt and block numbers</summary>
 <p class="lk-prompt">{escape(rows['ouro']['prompt'].strip())}</p>
 <p class="lk-prompt">Ouro visits shared block 12 at depths {summaries['ouro']['executed_blocks_1_based'][0]} and {summaries['ouro']['executed_blocks_1_based'][1]}. Qwen visits distinct blocks {summaries['qwen']['executed_blocks_1_based'][0]} and {summaries['qwen']['executed_blocks_1_based'][1]}.</p></details>
</section>
"""
