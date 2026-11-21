"""One recorded witness scene with native prediction and answer controls."""
import json
from string import ascii_uppercase
from zipfile import ZipFile


FAMILY = 24  # First held-out scene, selected independently of model outcomes.
ROLES = ("first", "second", "world")
LABELS = ("Alice", "Bob", "Actual location")
MODELS = {"ouro": "Ouro-1.4B", "qwen": "Qwen2.5-1.5B"}


CSS = """<style>
.cp-memory-scene{container-type:inline-size;--mc-ink:#1c302d;--mc-body:#354c46;--mc-teal:#19564f;--mc-line:#cbd4c9;--mc-rust:#8e442e;background:#fffdf8;border:1px solid var(--mc-line);border-radius:9px;color:var(--mc-ink);padding:clamp(14px,2.4vw,29px);margin:20px 0;font-family:inherit}
.cp-memory-scene *{box-sizing:border-box}
.cp-memory-scene .mc-id{margin:0 0 .35rem;color:var(--mc-teal);font:650 .72rem/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase}
.cp-memory-scene h3{margin:0;font:normal clamp(1.65rem,3vw,2.15rem)/1.12 Georgia,serif;letter-spacing:-.035em}
.cp-memory-scene p.mc-intro,.cp-memory-scene .mc-note{font-size:.94rem;line-height:1.5;color:var(--mc-body);margin:.55rem 0 0;max-width:77ch}
.cp-memory-scene ol.mc-events{list-style:none;padding:0;margin:1.15rem 0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;border:1px solid var(--mc-line);border-radius:5px;overflow:hidden;background:var(--mc-line)}
.cp-memory-scene .mc-event{margin:0;min-width:0;padding:.8rem;background:#f0f3eb;display:flex;flex-direction:column}
.cp-memory-scene .mc-event b{font-family:inherit;font-weight:650;font-size:.95rem;line-height:1.4;color:var(--mc-teal)}
.cp-memory-scene .mc-boxes{align-self:center;display:flex;align-items:end;justify-content:center;gap:clamp(4px,1vw,12px);width:100%;height:70px}
.cp-memory-scene .mc-box{position:relative;display:grid;place-items:center;width:clamp(28px,5cqw,50px);height:29px;background:#fdfcf7;border:1.5px solid #61796e;border-top:3px double #61796e;border-radius:1px 1px 5px 5px;font:600 13px/1 ui-monospace,monospace;color:#36594c}
.cp-memory-scene .mc-box[data-occupied]{background:#fff0d9;border-color:#a45836;color:#713a24}
.cp-memory-scene .mc-coin{display:block;position:absolute;bottom:calc(100% + 3px);left:50%;transform:translateX(-50%);width:21px;height:21px;border:1.5px solid #945031;border-radius:50%;background:#e2ac60;box-shadow:inset 0 0 0 2px #f6d294,inset 0 0 0 3px #af7544}
.cp-memory-scene .mc-event p{font-size:.87rem;line-height:1.38;margin:.65rem 0 0;color:var(--mc-body)}
.cp-memory-scene .mc-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
.cp-memory-scene .mc-prompt{font-size:1rem;line-height:1.45;font-weight:650;margin:1rem 0 .55rem}
.cp-memory-scene .mc-choices{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.85rem;margin:0 0 1.1rem}
.cp-memory-scene fieldset{border:0;min-width:0;margin:0;padding:0}
.cp-memory-scene legend{padding:0;margin:0 0 .4rem;color:var(--mc-body);font-size:.88rem;line-height:1.45}
.cp-memory-scene .mc-options{display:inline-flex;max-width:100%;border:1px solid #7d9386;border-radius:4px}
.cp-memory-scene label{position:relative;display:inline-flex;align-items:center;justify-content:center;cursor:pointer;min-width:32px;min-height:36px;color:var(--mc-body)}
.cp-memory-scene label+label{border-left:1px solid #a9b8ac}
.cp-memory-scene label span{display:grid;place-items:center;align-self:stretch;min-width:100%;padding:0 .43rem;font:650 .9rem/1.2 ui-monospace,monospace}
.cp-memory-scene label:first-child span{border-radius:2px 0 0 2px}
.cp-memory-scene label:last-child span{border-radius:0 2px 2px 0}
.cp-memory-scene label:hover span{background:#e9efe7}
.cp-memory-scene input:checked+span{background:var(--mc-teal);color:white;text-decoration:underline;text-underline-offset:3px}
.cp-memory-scene input:focus-visible+span{outline:3px solid #a9552d;outline-offset:2px;z-index:1}
.cp-memory-scene details{border-top:1px solid var(--mc-line);padding-top:.7rem}
.cp-memory-scene summary{padding:.25rem .1rem;cursor:pointer;font:650 .96rem/1.5 system-ui,sans-serif;color:var(--mc-teal)}
.cp-memory-scene summary:focus-visible{outline:3px solid var(--mc-rust);outline-offset:3px}
.cp-memory-scene h4.mc-feedback-title{font-size:1rem;font-weight:650;margin:.85rem 0 .35rem}
.cp-memory-scene .mc-feedback{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.5rem;margin:0 0 .9rem}
.cp-memory-scene .mc-feedback div{border-left:2px solid var(--mc-line);padding-left:.55rem}
.cp-memory-scene .mc-feedback dt{font-size:.85rem;font-weight:650;color:var(--mc-teal)}
.cp-memory-scene .mc-feedback dd{margin:.15rem 0 0;font-size:.86rem;line-height:1.45;color:var(--mc-body)}
.cp-memory-scene .mc-feedback dd span[data-choice]{display:none}
.cp-memory-scene table{border-collapse:collapse;width:100%;margin:.8rem 0 .45rem;line-height:1.35;color:var(--mc-ink)}
.cp-memory-scene table th,.cp-memory-scene table td{text-align:center;padding:.47rem .24rem;border:0;border-bottom:1px solid var(--mc-line);font-size:.9rem;background:transparent}
.cp-memory-scene table th:first-child{text-align:left}
.cp-memory-scene thead th{color:var(--mc-body);font-size:.82rem}
.cp-memory-scene table .mc-access th{padding-top:.8rem;padding-bottom:.3rem;border-bottom:0;font-size:.8rem;color:var(--mc-teal);font-weight:650}
.cp-memory-scene table .mc-model{font-weight:500}
.cp-memory-scene td{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.cp-memory-scene .mc-truth{background:#e9efe6}
.cp-memory-scene .mc-truth td{font-weight:700}
.cp-memory-scene .mc-match{font-weight:700;color:var(--mc-teal)}
.cp-memory-scene .mc-note{font-size:.84rem}
@container(max-width:36rem){
  .cp-memory-scene ol.mc-events,.cp-memory-scene .mc-feedback{grid-template-columns:1fr}
  .cp-memory-scene .mc-event{display:grid;grid-template-columns:minmax(88px, .8fr) minmax(0,1fr);grid-template-rows:auto auto;column-gap:.8rem;align-items:start;padding:.55rem .75rem}
  .cp-memory-scene .mc-event b{grid-column:2;grid-row:1;align-self:end}
  .cp-memory-scene .mc-event p{grid-column:2;grid-row:2;margin:.14rem 0 0}
  .cp-memory-scene .mc-boxes{grid-column:1;grid-row:1 / 3;height:57px;align-self:center;align-items:end;padding-bottom:3px}
  .cp-memory-scene .mc-choices{grid-template-columns:1fr;gap:.45rem}
  .cp-memory-scene fieldset{display:grid;grid-template-columns:1fr auto;align-items:center}
  .cp-memory-scene legend{float:left;margin:0}
  .cp-memory-scene .mc-options{justify-self:end}
  .cp-memory-scene label{min-width:39px;min-height:36px}
}
@container(max-width:25rem){.cp-memory-scene table td,.cp-memory-scene table th{font-size:.77rem}.cp-memory-scene .mc-event{column-gap:.55rem}}
</style>"""


def _read(archive):
    winners = {}
    reference = None
    for model in MODELS:
        path = f"results/memory/{model}/heldout/scene-{FAMILY:03d}.json"
        scene = json.loads(archive.read(path))
        rows = [r for r in scene["rows"] if r["domain"] == "people" and not r["toggle"]
                and r["stage"] == 4 and r["condition"] in ("direct", "memory")]
        if len(rows) != 6:
            raise ValueError(f"Expected six final-stage person-scene rankings in {path}")
        for row in rows:
            boxes, witness = tuple(row["boxes"]), row["witness"]
            if not (row["family"] == FAMILY and len(set(boxes)) == len(boxes) == 3
                    and all(b in ascii_uppercase and len(b) == 1 for b in boxes)
                    and witness in (0, 1) and row["winner"] in boxes):
                raise ValueError(f"Invalid scene record in {path}")
            if reference is None:
                reference = boxes, witness
            elif reference != (boxes, witness):
                raise ValueError("Scene events differ across rankings")
            role = ROLES.index(row["role"])
            expected = boxes[2 if role == 2 else int(role == witness)]
            if row["target"] != expected:
                raise ValueError(f"Witness interpretation disagrees with {path}")
            winners[model, row["condition"], row["role"]] = row["winner"]
    boxes, witness = reference
    return boxes, witness, winners


def _event(title, caption, boxes, active):
    shapes = []
    for index, box in enumerate(boxes):
        occupied = ' data-occupied="true"' if index == active else ""
        coin = '<i class="mc-coin"></i>' if occupied else ""
        shapes.append(f'<span class="mc-box"{occupied}>{coin}{box}</span>')
    return (f'<li class="mc-event"><b>{title}</b><div class="mc-boxes" aria-hidden="true">'
            f'{"".join(shapes)}</div><p><span class="mc-sr">The coin is in box {boxes[active]}. </span>'
            f'{caption}</p></li>')


def _choices(boxes):
    fields = []
    for role, perspective in zip(ROLES, ("Alice’s belief", "Bob’s belief", "Actual location")):
        options = "".join(f'<label><input class="mc-sr" name="mc-024-{role}" type="radio" value="{box}" '
                          f'aria-label="Box {box}"><span aria-hidden="true">{box}</span></label>' for box in boxes)
        fields.append(f'<fieldset><legend>{perspective}</legend><div class="mc-options">{options}</div></fieldset>')
    return "".join(fields)


def _feedback(boxes, gold):
    answers, selectors = [], []
    for role, name, target in zip(ROLES, LABELS, gold):
        reason = f"{name} last saw box {target}." if role != "world" else f"The coin ended in box {target}."
        spans = [f'<span class="mc-unanswered">No guess selected. {reason}</span>']
        chosen = f'.cp-memory-scene:has(input[name="mc-024-{role}"]'
        output = f'.mc-feedback [data-role="{role}"]'
        selectors.append(f'{chosen}:checked) {output} .mc-unanswered{{display:none}}')
        for box in boxes:
            feedback = f"You chose {box}. Correct." if box == target else f"You chose {box}. {reason}"
            spans.append(f'<span data-choice="{box}">{feedback}</span>')
            selectors.append(f'{chosen}[value="{box}"]:checked) {output} span[data-choice="{box}"]{{display:inline}}')
        answers.append(f'<div data-role="{role}"><dt>{name}</dt><dd>{"".join(spans)}</dd></div>')
    return "<style>" + "\n".join(selectors) + "</style>", (
        '<h4 class="mc-feedback-title">Your predictions</h4><dl class="mc-feedback" aria-live="polite">'
        + "".join(answers) + '</dl>')



def render(archive: ZipFile) -> str:
    boxes, witness, winners = _read(archive)
    gold = [boxes[int(n == witness)] for n in range(2)] + [boxes[2]]
    feedback_style, feedback = _feedback(boxes, gold)
    events = [_event("Placement", "Alice and Bob see the coin.", boxes, 0),
              _event("First move", f"Only {LABELS[witness]} sees the move.", boxes, 1),
              _event("Final move", "Alice and Bob both miss the move.", boxes, 2)]
    rows = []
    for condition, title in (("direct", "Full story"), ("memory", "Eight stored positions")):
        rows.append(f'<tbody><tr class="mc-access"><th scope="rowgroup" colspan="4">{title}</th></tr>')
        for model, model_label in MODELS.items():
            cells = []
            for role, correct in zip(ROLES, gold):
                winner = winners[model, condition, role]
                if winner == correct:
                    cell = f'<td class="mc-match">{winner}<span aria-hidden="true"> ✓</span><span class="mc-sr"> correct</span></td>'
                else:
                    cell = f'<td>{winner}<span class="mc-sr"> incorrect</span></td>'
                cells.append(cell)
            rows.append(f'<tr><th class="mc-model" scope="row">{model_label}</th>{"".join(cells)}</tr>')
        rows.append('</tbody>')
    return f"""{CSS}{feedback_style}
<section class="cp-memory-scene" id="cp-memory-scene" aria-labelledby="mc-scene-title">
<p class="mc-id">Held-out scene {FAMILY:03d}</p><h3 id="mc-scene-title">Follow the coin</h3>
<p class="mc-intro">Alice and Bob remember the last box where they saw the coin.</p>
<ol class="mc-events">{"".join(events)}</ol>
<p class="mc-prompt">Choose a box for each answer.</p>
<div class="mc-choices">{_choices(boxes)}</div>
<details class="mc-reveal"><summary>Reveal the answers and model rankings</summary>
{feedback}
<p class="mc-note">In the full-story run, each question could read every move. In the memory run, eight positions encoded the story during stages one and two. All three questions received the same eight states at the start of stage three. From there, the questions could read the memory positions and their own text. Access to the story was closed.</p>
<table><caption class="mc-sr">Final-stage rankings for scene {FAMILY:03d}. A check marks a correct box.</caption>
<thead><tr><th scope="col">Answer source</th>{"".join(f'<th scope="col">{label}</th>' for label in LABELS)}</tr></thead>
<tbody><tr class="mc-truth"><th scope="row">Correct box</th>{"".join(f'<td>{box}</td>' for box in gold)}</tr></tbody>
{"".join(rows)}</table>
<p class="mc-note">Each letter is the box the model ranked highest among {boxes[0]}, {boxes[1]} and {boxes[2]}. A check marks a correct ranking.</p>
</details></section>"""
