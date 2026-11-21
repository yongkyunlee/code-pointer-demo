"""Carry-directed residual swaps on matched two-digit addition problems."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch


WORDS = {
    "English": "zero one two three four five six seven eight nine".split(),
    "French": "zéro un deux trois quatre cinq six sept huit neuf".split(),
}
LANGUAGES = tuple(WORDS)
ALL_WORDS = tuple(dict.fromkeys(word for words in WORDS.values() for word in words))
INSTRUCTION = (
    "Write the tens digit of the sum as one word in the requested language.\n"
    "14 + 18 | English: three\n23 + 21 | French: quatre\n"
)
# Different totals, even across reader fitting, pilot and held-out splits.
UNITS = {"train": [0, 2, 8], "pilot": [1, 7], "heldout": [3, 4, 5, 6]}
SIZES = {"train": 36, "pilot": 24, "heldout": 100}
SEEDS = {"train": 9187, "pilot": 9188, "heldout": 9189}


def families(split):
    rng = random.Random(SEEDS[split])
    pools = {}
    for tens in range(3, 9):
        for unit in UNITS[split]:
            total = 10 * tens + unit
            if total in (32, 44):  # The two worked examples in the fixed prompt.
                continue
            groups = [[], []]
            for a in range(10, total // 2 + 1):
                b = total - a
                if a == b or b > 99:
                    continue
                groups[int(a % 10 + b % 10 >= 10)].append((a, b))
            for group in groups:
                rng.shuffle(group)
            if min(map(len, groups)) >= 2:
                # Reserve same-carry sham operands separate from all clean operands.
                # The secondary total-clustered interval accounts for reused shams.
                shams = [[group.pop() for _ in range(max(1, len(group)//5))]
                         for group in groups]
                pools[total] = groups, shams
    items = []
    # Use totals uniformly round-robin, then reuse a total with new operands.
    while len(items) < SIZES[split]:
        totals = list(pools)
        rng.shuffle(totals)
        previous = len(items)
        for total in totals:
            groups, reserved = pools[total]
            if min(map(len, groups)) < 1:
                continue
            def order(pair):
                return pair if rng.randrange(2) else pair[::-1]
            clean, sham = [], []
            for group, options in zip(groups, reserved):
                clean.append(order(group.pop()))
                sham.append(order(rng.choice(options)))
            items.append(dict(index=len(items), total=total, sums=clean, sham=sham))
            if len(items) == SIZES[split]:
                break
        if previous == len(items):
            raise ValueError(f"Insufficient distinct operands in {split}")
    return items


@dataclass
class Prompt:
    text: str
    ids: list
    positions: dict


def make_prompt(engine, operands, language):
    a, b = operands
    text = f"{INSTRUCTION}{a} + {b} | {language}:"
    encoded = engine.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    def token_at(char):
        return next(i for i, (start, end) in enumerate(offsets) if start <= char < end)
    positions = {
        "left operand": token_at(len(INSTRUCTION) + len(str(a)) - 1),
        "right operand": token_at(len(INSTRUCTION) + len(f"{a} + {b}") - 1),
        "answer position": len(ids) - 1,
    }
    return Prompt(text, ids, positions)


class Scorer:
    """Score every token of each word. Share forwards with the same token prefix."""

    def __init__(self, engine, prompt):
        self.engine = engine
        self.prompt = prompt
        self.paths = {}
        for word in ALL_WORDS:
            whole = engine.ids(prompt.text + " " + word)
            if whole[:len(prompt.ids)] != prompt.ids:
                raise ValueError(f"Unstable answer boundary for {word!r}")
            suffix = tuple(whole[len(prompt.ids):])
            if not suffix:
                raise ValueError(f"Empty continuation for {word!r}")
            self.paths[word] = suffix
        if len(set(self.paths.values())) != len(self.paths):
            raise ValueError("Distinct answer strings collapse under the tokenizer")

    def scores(self, words, first_logits, edit=None):
        logits = {(): first_logits}
        def distribution(prefix):
            if prefix not in logits:
                logits[prefix] = self.engine.run(
                    self.prompt.ids + list(prefix), edit_fn=edit, logits_positions=[-1]
                ).logits[0, -1].detach().float()
            return logits[prefix]
        distributions = {}
        scores = {}
        for word in dict.fromkeys(words):
            sequence = self.paths[word]
            score = 0.0
            for i, tok in enumerate(sequence):
                prefix = sequence[:i]
                if prefix not in distributions:
                    distributions[prefix] = distribution(prefix).log_softmax(-1)
                score += distributions[prefix][tok].item()
            scores[word] = score

        generated = []
        ended = False
        word = ""
        text = ""
        eos = self.engine.tokenizer.eos_token_id
        for _ in range(7):
            tok = int(distribution(tuple(generated)).argmax().item())
            if tok == eos:
                ended = True
                break
            generated.append(tok)
            text = self.engine.tokenizer.decode(generated, skip_special_tokens=False)
            # A boundary following the first word makes its identity unambiguous.
            match = re.match(r"^\s*([^\W_]+|[^\w\s])", text, re.UNICODE)
            if match:
                word = match.group(1).casefold()
                if match.end() < len(text) or not word[0].isalnum():
                    ended = True
                    break
        if not ended:
            word = ""  # A truncated word cannot establish the response.
        labels = [language for language, vocab in WORDS.items() if word in vocab]
        digit = next((WORDS[language].index(word) for language in labels), None)
        if digit is None and len(word) == 1 and word in "0123456789":
            digit = int(word)
        return dict(scores=scores, answer_text=text, first_word=word,
                    digit=digit, languages=labels, malformed=not labels,
                    terminated=ended, best_word=max(scores, key=scores.get))

    def gradients(self, target, original, sites, positions):
        # Linearize the very same full-continuation log-probability margin used
        # in the intervened runs, including later conditional token probabilities.
        coefficients = defaultdict(lambda: defaultdict(int))
        for word, sign in [(target, 1), (original, -1)]:
            path = self.paths[word]
            for i, tok in enumerate(path):
                coefficients[path[:i]][tok] += sign
        gradients = {}
        for prefix, counts in coefficients.items():
            counts = {tok: weight for tok, weight in counts.items() if weight}
            if not counts:
                continue
            with torch.enable_grad():
                result = self.engine.run(self.prompt.ids + list(prefix), capture=sites, grad=True, logits_positions=[-1])
                logp = result.logits[0, -1].float().log_softmax(-1)
                objective = sum(weight * logp[tok] for tok, weight in counts.items())
                derivatives = torch.autograd.grad(objective, [result.states[s] for s in sites])
            for site, deriv in zip(sites, derivatives):
                reduced = deriv.detach()[0, positions].float().clone()
                gradients[site] = gradients.get(site, 0) + reduced
            del objective, derivatives, result, logp
        return gradients


def snapshot(engine, prompt, sites):
    run = engine.run(prompt.ids, capture=sites, logits_positions=[-1])
    pos = list(prompt.positions.values())
    return ({s: run.states[s][0, pos].detach().clone() for s in sites},
            run.logits[0, -1].detach().float().clone())


def unit_rows(rows):
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True).clip(1e-8)


class Readers:
    """A frozen cosine k-NN reader and ridge probes see one answer-position vector."""

    def __init__(self, engine):
        sites = [(i, engine.width) for i in range(1, 5)]
        data = defaultdict(list)
        labels, digits = [], []
        for family in families("train"):
            for carry, pair in enumerate(family["sums"]):
                for language in LANGUAGES:
                    prompt = make_prompt(engine, pair, language)
                    run = engine.run(prompt.ids, capture=sites, logits_positions=[-1])
                    for stage, offset in sites:
                        v = run.states[stage, offset][0, -1].detach().float().cpu().numpy()
                        data[stage].append(v.copy())
                    labels.append(carry)
                    digits.append(family["total"] // 10)
                    del run
        self.labels = np.asarray(labels)
        self.vectors, self.weights = {}, {}
        outputs = np.concatenate(((2 * self.labels[:, None] - 1),
                                  np.eye(10)[digits]), axis=1)
        self.bias = outputs.mean(axis=0)
        for stage, vectors in data.items():
            x = unit_rows(np.asarray(vectors, dtype=np.float64))
            self.vectors[stage] = x
            centered = x - x.mean(axis=0)
            dual = np.linalg.solve(centered @ centered.T + 0.05 * np.eye(len(x)),
                                   outputs - self.bias)
            self.weights[stage] = centered.T @ dual

    def read(self, raw, stage):
        x = unit_rows(raw.detach().float().cpu().numpy().astype(np.float64))
        def ridge(train_stage):
            center = self.vectors[train_stage].mean(axis=0)
            return (x - center) @ self.weights[train_stage] + self.bias
        similarity = self.vectors[1] @ x
        nearest = np.argsort(similarity, kind="stable")[-5:]
        fresh = ridge(stage)
        return dict(fixed_knn=int(self.labels[nearest].sum() >= 3),
                    fixed_ridge=int(ridge(1)[0] >= 0), fresh_ridge=int(fresh[0] >= 0),
                    fresh_tens=int(fresh[1:].argmax()))


def measure_family(engine, family, readers):
    idx = family["index"]
    offset = engine.width
    primary_sites = [(stage, offset) for stage in range(1, 5)]
    near_stage = 1 + (idx // 2) % 4
    sites = primary_sites + [(near_stage, offset - 1)] if offset > 1 else primary_sites
    cache = {}
    def acquire(pair, lang):
        key = (tuple(pair), lang)
        if key not in cache:
            p = make_prompt(engine, pair, lang)
            cache[key] = (p, *snapshot(engine, p, sites))
        return cache[key]

    rows, reader_rows, clean_rows = [], [], []
    true_digit = family["total"] // 10
    for carry, pair in enumerate(family["sums"]):
        language = LANGUAGES[(idx + carry) % 2]
        other = LANGUAGES[1 - LANGUAGES.index(language)]
        prompt, clean_states, logits = acquire(pair, language)
        scorer = Scorer(engine, prompt)
        clean = scorer.scores(ALL_WORDS, logits)
        original = WORDS[language][true_digit]
        target_digit = true_digit + 1 - 2 * carry
        target = WORDS[language][target_digit]
        clean_margin = clean["scores"][target] - clean["scores"][original]
        clean_rows.append(dict(carry=carry, language=language, operands=pair,
                               target_digit=target_digit, margin=clean_margin,
                               correct_word_probability=math.exp(clean["scores"][original]),
                               prompt=prompt.text, ids=prompt.ids, positions=prompt.positions,
                               word_token_ids=scorer.paths, **clean))
        for s in primary_sites:
            reader_rows.append(dict(stage=s[0], language=language, carry=carry,
                                    **readers.read(clean_states[s][-1], s[0])))
        roles = list(prompt.positions)
        measured = [(s, "answer position") for s in sites]
        # Rotate operand probes across families; allocation is independent of gradients.
        measured += [(s, roles[(idx // 2 + s[0]) % 2]) for s in primary_sites]
        pos = list(prompt.positions.values())
        gradients = scorer.gradients(target, original, sites, pos)
        words = list(dict.fromkeys([target, original, WORDS[other][target_digit],
                                   WORDS[other][true_digit]]))
        controls = [
            ("opposite/same language", family["sums"][1-carry], language),
            ("opposite/cross language", family["sums"][1-carry], other),
            ("same carry sham", family["sham"][carry], language),
        ]
        for control, donor_pair, donor_lang in controls:
            donor, donor_states, _ = acquire(donor_pair, donor_lang)
            for s, role in measured:
                i = roles.index(role)
                dp, rp = donor.positions[role], prompt.positions[role]
                value = donor_states[s][i]
                delta = value.float() - clean_states[s][i].float()
                predicted = float(torch.dot(gradients[s][i], delta).item())
                def edit(stage, block, state):
                    if (stage, block) != s:
                        return state
                    patched = state.clone()
                    patched[0, rp] = value
                    return patched
                patched_logits = engine.run(prompt.ids, edit_fn=edit, logits_positions=[-1]).logits[0, -1].detach().float()
                result = scorer.scores(words, patched_logits, edit)
                margin = result["scores"][target] - result["scores"][original]
                rows.append(dict(stage=s[0], offset=s[1], role=role, carry=carry,
                                 control=control, language=language, donor_language=donor_lang,
                                 donor_operands=donor_pair, recipient_pos=rp, donor_pos=dp,
                                 recipient_token_id=prompt.ids[rp], donor_token_id=donor.ids[dp],
                                 target_digit=target_digit, target_word=target, clean_word=original,
                                 margin=margin, predicted=predicted, observed=margin-clean_margin,
                                 target_probability=math.exp(result["scores"][target]), **result))
    return dict(family=family, clean=clean_rows, readers=reader_rows, patches=rows)


def bootstrap(records):
    """Resample family means to preserve all within-family dependencies."""
    groups = defaultdict(list)
    for family, value in records:
        groups[family].append(value)
    x = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(143)
    samples = x[rng.integers(len(x), size=(2000, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), lo=float(np.quantile(samples, .025)),
                hi=float(np.quantile(samples, .975)), families=len(x))


def summarize(results, width):
    groups = defaultdict(list)
    def add(key, value):
        groups[key].append((idx, value))
    shuffled = defaultdict(list)
    for result in results:
        f = result["family"]
        idx = f["index"]
        tens = f["total"] // 10
        for row in result["clean"]:
            lang = row["language"]
            correct_word = WORDS[lang][tens]
            add(("clean", lang, "digit accuracy"), row["digit"] == tens)
            add(("clean", lang, "greedy correct word"), row["first_word"] == correct_word)
            add(("clean", lang, "candidate correct word"), row["best_word"] == correct_word)
            add(("clean", lang, "malformed"), row["malformed"])
            add(("clean", lang, "language unambiguous correct"), row["languages"] == [lang])
            add(("clean", lang, "language ambiguous"), len(row["languages"]) > 1)
            add(("clean", lang, "correct word probability"), row["correct_word_probability"])
        for row in result["readers"]:
            key = (row["language"], row["stage"])
            for name in ("fixed_knn", "fixed_ridge", "fresh_ridge"):
                add(("reader", *key, name), row[name] == row["carry"])
            add(("reader", *key, "fresh_tens"), row["fresh_tens"] == tens)
            shuffled[key].append((idx, row["fixed_knn"], row["carry"]))
        paired = defaultdict(dict)
        for row in result["patches"]:
            key = (row["language"], row["stage"], row["offset"], row["role"])
            base = ("patch", *key, row["control"])
            for metric in ("observed", "predicted", "target_probability"):
                add((*base, metric), row[metric])
            add((*base, "linearization error"), row["predicted"] - row["observed"])
            add((*base, "greedy target digit"), row["digit"] == row["target_digit"])
            add((*base, "greedy target word"), row["first_word"] == row["target_word"])
            add((*base, "language unambiguous correct"), row["languages"] == [row["language"]])
            add((*base, "malformed"), row["malformed"])
            paired[key][row["control"]] = row["observed"]
        for key, scores in paired.items():
            add(("paired", *key, "opposite minus sham"),
                scores["opposite/same language"] - scores["same carry sham"])
            add(("paired", *key, "cross minus same language"),
                scores["opposite/cross language"] - scores["opposite/same language"])
    rng = random.Random(3209)
    for key, rows in shuffled.items():
        preds = [p for _, p, _ in rows]
        rng.shuffle(preds)
        for (idx, _, carry), pred in zip(rows, preds):
            add(("reader", *key, "shuffled vector"), int(pred == carry))
    totals = {r["family"]["index"]: r["family"]["total"] for r in results}
    output = []
    for key, values in groups.items():
        total_interval = bootstrap([(totals[idx], value) for idx, value in values])
        output.append(dict(group=list(key), **bootstrap(values),
                           total_clustered=total_interval))
    return output


def figure(summary, width, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {tuple(row["group"]): row["mean"] for row in summary}
    roles = ["left operand", "right operand", "answer position"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), gridspec_kw={"height_ratios":[3,1]},
                            constrained_layout=True)
    panels = []
    for lang in LANGUAGES:
        for metric in ("predicted", "observed"):
            a = [[lookup.get(("patch", lang, s, width, role, "opposite/cross language",
                              metric), np.nan) for s in range(1, 5)] for role in roles]
            panels.append(np.array(a))
    lim = max(1e-6, float(np.nanmax(np.abs(panels))))
    for col, values in enumerate(panels):
        lang = LANGUAGES[col//2]
        ax = axes[0,col]
        heat = ax.imshow(values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"{lang} · {['gradient prediction','actual swap'][col%2]}")
        ax.set_yticks(range(len(roles)), roles if col == 0 else [])
        ax.set_xticks(range(4), [1,2,3,4])
        ax.set_xlabel("Stage")
        ax = axes[1,col]
        for metric, label in (("fixed_ridge", "Stage 1 ridge"), ("fresh_ridge", "Stage-specific")):
            ax.plot(range(1,5), [lookup["reader", lang, s, metric] for s in range(1,5)],
                    marker="o", label=label)
        ax.axhline(.5, color="gray", linestyle=":")
        ax.set_ylim(0,1.04)
        ax.set_xticks(range(1,5))
        if col == 0:
            ax.set_ylabel("Carry accuracy")
            ax.legend(fontsize=7)
    fig.colorbar(heat, ax=axes[0,:], label="Change in full-word log-probability margin")
    fig.suptitle("Carry transfer from an opposite-language donor")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False))
    tmp.replace(path)


def run(engine, phase, outdir):
    if phase not in ("smoke", "pilot", "heldout"):
        raise ValueError(phase)
    out = Path(outdir) / f"carry-{engine.name}-{phase}"
    out.mkdir(parents=True, exist_ok=True)
    items = families("heldout" if phase == "heldout" else "pilot")
    if phase == "smoke":
        items = items[:1]
    # Reject stale resumptions when experiment code or the engine has changed.
    import inspect
    fingerprint = hashlib.sha256((inspect.getsource(inspect.getmodule(run)) +
                                  inspect.getsource(type(engine))).encode()).hexdigest()
    meta = dict(phase=phase, model=engine.name, width=engine.width, fingerprint=fingerprint,
                allocated_families=len(items), reader_training_families=SIZES["train"],
                stages=[1,2,3,4], primary_offset=engine.width, nearby_offset=engine.width-1,
                split_units=UNITS, demo_totals_excluded=[32, 44], seeds=SEEDS, prompt=INSTRUCTION,
                primary_physical_blocks_one_based=[engine.width if engine.name == "ouro"
                                                   else stage * engine.width for stage in (1,2,3,4)],
                scoring="full-word log probability, canonical leading space; greedy first word",
                reader="cosine 5-NN and ridge (0.05), labels yes=1, no=0")
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != meta:
        raise ValueError(f"Different carry run already exists at {out}")
    save_json(manifest, meta)
    readers = None
    results = []
    for family in items:
        path = out / f"family-{family['index']:03d}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            if readers is None:
                readers = Readers(engine)
            result = measure_family(engine, family, readers)
            save_json(path, result)
            print(f"carry {engine.name} {phase}: {family['index']+1}/{len(items)}", flush=True)
        results.append(result)
    summary = summarize(results, engine.width)
    save_json(out / "summary.json", summary)
    figure(summary, engine.width, out / "carry.png")
    return dict(families=len(results), patches=sum(len(r["patches"]) for r in results),
                directory=out.name)
