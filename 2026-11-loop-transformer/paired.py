"""Controlled comparison on three symbolic tasks. Run with paired.run(output_path).

Every architecture sees the same fixed-length symbolic prompts in the same order
within seed. Stage-end answer supervision is identical. Untied stages start as
exact copies of the first stage and then train independently. The narrow untied
control changes width to approximate the tied parameter count. Findings apply to
these supervised tasks and training budget, independently of the frozen LMs.
"""
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STAGES = 4
BLOCKS = 2
WIDTH = 64
HEADS = 4
BATCH = 48
STEPS = 900
SEEDS = (41, 73, 107)
HELDOUT = 128
TASKS = ("carry", "memory", "cue")
NUMBERS = ("zero one two three four five six seven eight nine".split(),
           "zéro un deux trois quatre cinq six sept huit neuf".split())
PLACES = [f"box{i}" for i in range(16)]
FRUITS = "apple pear plum grape mango lemon lime peach".split()
COLORS = "red blue green pink teal gold white black".split()
PEOPLE = "Rae Jo Li Pat Kim Max Sam Lou".split()
PREFIX = "carry ten plus EN FR answer scene put move nobody slot ask Alice Bob R1 R2 world holds uses has".split()
VOCAB = list(dict.fromkeys(PREFIX + list("0123456789") + sum([list(v) for v in NUMBERS], []) +
                           PLACES + FRUITS + COLORS + PEOPLE))
IDS = {token: i for i, token in enumerate(VOCAB)}
SOURCE, QUESTION = 9, 17


def partition(key):
    # Split before drawing question, language, witness, or operand order.
    bucket = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:2], "big") % 10
    return "train" if bucket < 6 else "reader" if bucket == 6 else "pilot" if bucket == 7 else "heldout"


def arithmetic_pools():
    pooled_pairs = defaultdict(list)
    for a in range(10, 80):
        for b in range(a, 80):
            total = a + b
            if 30 <= total < 90:
                carry = int(a % 10 + b % 10 >= 10)
                pooled_pairs[total, carry].append((a, b))
    # Stratify assignments within each total and carry class. The raw operand
    # pairs stay disjoint, with enough same-carry shams in evaluation strata.
    schedule = ("train", "heldout", "pilot", "reader", "train", "heldout", "pilot", "reader",
                "train", "train", "heldout", "pilot")
    pools = defaultdict(lambda: defaultdict(list))
    for key, pairs in pooled_pairs.items():
        random.Random(repr(key)).shuffle(pairs)
        for i, pair in enumerate(pairs):
            pools[schedule[i % len(schedule)]][key].append(pair)
    return pools


POOLS = arithmetic_pools()
TOTALS = {split: [t for t in range(30, 90) if min(len(pool[t,c]) for c in (0,1)) >= 2]
          for split, pool in POOLS.items()}


def family(task, split, rng):
    if task == "carry":
        pools = POOLS[split]
        total = rng.choice(TOTALS[split])
        return tuple(rng.sample(pools[total, 0], 2) + rng.sample(pools[total, 1], 2))
    while True:
        if task == "memory":
            scene = tuple(rng.sample(PLACES, 3))
        else:
            scene = (*rng.sample(COLORS, 2), *rng.sample(FRUITS, 2), rng.choice(PEOPLE))
        key = (task, *scene) if task == "memory" else (task, tuple(sorted(zip(scene[:2], scene[2:4]))), scene[4])
        if partition(key) == split:
            return scene


def carry_item(pair, language):
    a, b = pair
    seq = ["carry", *str(a), "plus", *str(b), "ten", ("EN", "FR")[language], "answer"]
    return seq, NUMBERS[language][(a + b) // 10]


def memory_item(scene, register, witness, query):
    actors = ("R1", "R2") if register else ("Alice", "Bob")
    source = ["scene", "put", scene[0], "move", scene[1], actors[witness], "move", scene[2], "nobody"]
    question = ["ask", (*actors, "world")[query], "answer"]
    answer = scene[2] if query == 2 else scene[int(query == witness)]
    return source + ["slot"] * 8 + question, answer


def cue_item(scene, selected, variant="two"):
    c0, c1, f0, f1, person = scene
    desc = [c0, "holds", f0, c1, "holds", f1]
    clue = [person, "uses", (c0, c1)[selected]]
    record = (clue + desc) if variant == "first" else (desc + clue)
    prefix = [] if variant == "one" else ["slot"] * 9 if variant == "filler" else record
    return prefix + record + [person, "has", "answer"], (f0, f1)[selected]


def cue_decoy_item(scene, selected, decoy_choice, relevant_first):
    c0, c1, f0, f1, person = scene
    decoy = PEOPLE[(PEOPLE.index(person) + 1) % len(PEOPLE)]
    target = [person, "uses", (c0, c1)[selected]]
    irrelevant = [decoy, "uses", (c0, c1)[decoy_choice]]
    descriptions = [c0, "holds", f0, c1, "holds", f1]
    record = descriptions + (target + irrelevant if relevant_first else irrelevant + target)
    return record * 2 + [person, "has", "answer"], (f0, f1)[selected]


def encode(items, device):
    tokens, labels = zip(*items)
    return (torch.tensor([[IDS[word] for word in seq] for seq in tokens], device=device),
            torch.tensor([IDS[word] for word in labels], device=device))


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 3 * width), nn.GELU(), nn.Linear(3 * width, width))

    def forward(self, x, allowed):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, HEADS, width // HEADS).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        x = x + self.project(attention.transpose(1, 2).reshape_as(x))
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, width, tied):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), width)
        self.position = nn.Embedding(32, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, len(VOCAB))
        first = nn.ModuleList(Block(width) for _ in range(BLOCKS))
        self.stacks = nn.ModuleList([first] + ([] if tied else [copy.deepcopy(first) for _ in range(3)]))
        self.register_buffer("allowed", torch.ones(32, 32, dtype=torch.bool).tril(), persistent=False)

    def forward(self, tokens, mode="causal", cuts=(), edit=None, return_states=False,
                offset=0, cut_rows=slice(9, 15), cut_cols=slice(6, 9)):
        length = tokens.shape[1]
        x = self.embedding(tokens) + self.position(torch.arange(offset, offset + length, device=tokens.device))
        causal = self.allowed[:length, :length]
        logits, states = [], []
        for stage in range(STAGES):
            for block, layer in enumerate(self.stacks[stage % len(self.stacks)]):
                site = stage * BLOCKS + block  # zero-based executed block
                mask = causal
                if mode in ("memory", "sealed"):
                    mask = causal.clone()
                    if stage < 2:
                        mask[QUESTION:, :QUESTION] = False
                    if stage >= 2 or mode == "sealed":
                        mask[SOURCE:, :SOURCE] = False
                elif mode == "isolated":
                    mask = causal.clone()
                    mask[9:, :9] = False
                if site in cuts:
                    mask = mask.clone()
                    mask[cut_rows, cut_cols] = False
                x = layer(x, mask)
                if edit is not None:
                    x = edit(site, x)
                if return_states:
                    states.append(x)
            logits.append(self.head(self.norm(x[:, -1])))
        return torch.stack(logits), states


def clean_metrics(logits, answers, allowed):
    probability = logits.softmax(-1)
    guess = logits.argmax(-1)
    return {"accuracy": (guess == answers).float(), "answer_mass": probability.gather(1, answers[:, None])[:, 0],
            "candidate_mass": probability[:, allowed].sum(1), "malformed": (~torch.isin(guess, allowed)).float()}


def training_item(task, scene, rng):
    if task == "carry":
        return carry_item(rng.choice(scene), rng.randrange(2))
    if task == "memory":
        return memory_item(scene, rng.randrange(2), rng.randrange(2), rng.randrange(3))
    return cue_item(scene, rng.randrange(2))


def train(model, device, seed, steps, batch):
    rng = random.Random(200000 + seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.01)
    token_count, digest, loss = 0, hashlib.sha256(), 0.
    model.train()
    for i in range(steps):
        task = TASKS[i % len(TASKS)]
        # Keep token lengths uniform inside each minibatch. Expose both clue formats.
        if task == "cue" and (i // 3) % 2:
            items = [cue_decoy_item(family(task, "train", rng), rng.randrange(2), rng.randrange(2),
                                    rng.randrange(2)) for _ in range(batch)]
        else:
            items = [training_item(task, family(task, "train", rng), rng) for _ in range(batch)]
        x, y = encode(items, "cpu")
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
        token_count += x.numel()
        x, y = x.to(device), y.to(device)
        # Both architectures learn the same readout at every answerable stage.
        logits, _ = model(x, mode="memory" if task == "memory" else "causal")
        stages = logits[2:] if task == "memory" else logits
        loss = F.cross_entropy(stages.flatten(0, 1), y.repeat(stages.shape[0]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        for group in optimizer.param_groups:
            group["lr"] = .0002 + .0013 * (1 + math.cos(math.pi * i / steps)) / 2
        optimizer.step()
        if (i + 1) % 300 == 0:
            print(f"Controlled training seed {seed}, step {i+1}/{steps}, loss {loss.item():.3f}", flush=True)
    model.eval()
    return {"train_input_tokens": token_count, "stream_sha256": digest.hexdigest(),
            "last_minibatch_loss": loss.detach().item(), "parameters": sum(p.numel() for p in model.parameters())}


def dataset(task, split, size):
    rng = random.Random(7812 if split == "pilot" else 39452)
    scenes = {}
    for attempt in range(size * 1000):
        if len(scenes) == size:
            break
        scene = family(task, split, rng)
        # A repeated carry pair in reverse sham order does not create a fresh family.
        key = tuple(sorted(scene)) if task == "carry" else scene
        scenes[key] = scene
    if len(scenes) != size:
        raise RuntimeError(f"Found {len(scenes)} of {size} unique {task} {split} families")
    return list(scenes.values())


def frozen_reader(states, labels):
    x = torch.cat([states, torch.ones_like(states[:, :1])], dim=1).double()
    reg = .1 * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    return torch.linalg.solve(x.T @ x + reg, x.T @ (labels.double() * 2 - 1))


def read(reader, states):
    return states.double() @ reader[:-1] + reader[-1]


@torch.no_grad()
def fit_readers(model, device):
    scenes = dataset("carry", "reader", 32)
    inputs = [(carry_item(scene[carry * 2], lang), carry) for scene in scenes for carry in range(2) for lang in range(2)]
    codes = [[] for _ in range(STAGES)]
    for start in range(0, len(inputs), 32):
        x, _ = encode([item for item, _ in inputs[start:start+32]], device)
        _, states = model(x, return_states=True)
        for stage in range(STAGES):
            codes[stage].append(states[stage * BLOCKS + BLOCKS - 1][:, -1])
    labels = torch.tensor([c for _, c in inputs], device=device)
    return [frozen_reader(torch.cat(xs), labels) for xs in codes]


def carry_eval(model, scenes, device, readers):
    metrics = defaultdict(list)
    candidates = torch.tensor([IDS[n] for lang in NUMBERS for n in lang], device=device).unique()
    reader_codes = defaultdict(list)
    with torch.enable_grad():
        for lang in (0, 1):
            for carry in (0, 1):
                x, answer = encode([carry_item(s[2*carry], lang) for s in scenes], device)
                counter = torch.tensor([IDS[NUMBERS[lang][sum(s[0]) // 10 + 1 - 2 * carry]] for s in scenes], device=device)
                logits, states = model(x, return_states=True)
                margin = (logits[-1].gather(1, counter[:, None]) - logits[-1].gather(1, answer[:, None]))[:, 0]
                sites = [1, 3, 5, 7]
                grad = torch.autograd.grad(margin.sum(), [states[i] for i in sites])
                with torch.no_grad():
                    for k, v in clean_metrics(logits[-1], answer, candidates).items():
                        metrics["clean_" + k].append(v)
                    for stage, site in enumerate(sites):
                        code = states[site][:, -1]
                        reader_codes[stage].append(code.detach())
                        metrics[f"s{stage+1}_reader_fixed"].append((read(readers[0], code).gt(0) == bool(carry)).float())
                        metrics[f"s{stage+1}_reader_local"].append((read(readers[stage], code).gt(0) == bool(carry)).float())
                    for control, index, donor_lang in (("swap", 2*(1-carry), lang),
                                                        ("cross_language", 2*(1-carry), 1-lang),
                                                        ("sham", 2*carry+1, lang)):
                        donor, _ = encode([carry_item(s[index], donor_lang) for s in scenes], device)
                        _, donor_states = model(donor, return_states=True)
                        for stage, site in enumerate(sites):
                            value = donor_states[site][:, -1]
                            predicted = (grad[stage][:, -1] * (value - states[site][:, -1])).sum(-1)
                            def swap(idx, hidden):
                                if idx != site:
                                    return hidden
                                z = hidden.clone()
                                z[:, -1] = value
                                return z
                            patched = model(x, edit=swap)[0][-1]
                            actual = (patched.gather(1, counter[:, None]) - patched.gather(1, answer[:, None]))[:, 0] - margin
                            stem = f"s{stage+1}_{control}"
                            metrics[stem + "_observed"].append(actual)
                            if control != "sham":
                                direction = "remove" if carry else "add"
                                lang_tag = "FR" if lang else "EN"
                                metrics[f"{stem}_{lang_tag}_{direction}"].append(actual)
                            metrics[stem + "_predicted"].append(predicted)
                            lang_ids = torch.tensor([IDS[n] for n in NUMBERS[lang]], device=device)
                            metrics[stem + "_recipient_language"].append(torch.isin(patched.argmax(-1), lang_ids).float())
    for stage, codes in reader_codes.items():
        population = torch.cat(codes)
        generator = torch.Generator().manual_seed(521 + stage)
        order = torch.randperm(len(population), generator=generator).to(device)
        labels = torch.tensor([0, 1, 0, 1], device=device).repeat_interleave(len(scenes))
        hits = read(readers[0], population[order]).gt(0).eq(labels).float()
        metrics[f"s{stage+1}_reader_shuffled"].append(hits.reshape(4, -1).mean(0))
    return {k: torch.stack(v).detach().mean(0) for k, v in metrics.items()}


@torch.no_grad()
def memory_eval(model, scenes, device):
    metrics = defaultdict(list)
    candidates = torch.tensor([IDS[n] for n in PLACES], device=device)
    for register, group in ((0, "observer"), (1, "register")):
        for witness in (0, 1):
            common, predictions = None, []
            for query in range(3):
                x, answer = encode([memory_item(s, register, witness, query) for s in scenes], device)
                original, states = model(x, mode="memory", return_states=True)
                memory = states[3][:, SOURCE:QUESTION]
                if common is None:
                    common = memory
                else:
                    torch.testing.assert_close(memory, common, atol=1e-5, rtol=1e-5)
                def replace_memory(value):
                    def edit(site, h):
                        if site != 3:
                            return h
                        z = h.clone()
                        z[:, SOURCE:QUESTION] = value
                        return z
                    return edit
                logits = model(x, mode="memory", edit=replace_memory(common))[0]
                torch.testing.assert_close(original, logits, atol=1e-5, rtol=1e-5)
                shuffled = model(x, mode="memory", edit=replace_memory(common.roll(1, 0)))[0][-1]
                direct = model(x)[0][-1]
                sealed = model(x, mode="sealed")[0][-1]
                changed = x.clone()
                changed[:, :SOURCE] = x.roll(1, 0)[:, :SOURCE]
                torch.testing.assert_close(sealed, model(changed, mode="sealed")[0][-1], atol=1e-5, rtol=1e-5)
                for name, head in (("direct", direct), ("shuffled", shuffled), ("s3", logits[2]), ("s4", logits[3])):
                    for k, v in clean_metrics(head, answer, candidates).items():
                        metrics[f"{group}_{name}_{k}"].append(v)
                if query < 2:
                    world = torch.tensor([IDS[s[2]] for s in scenes], device=device)
                    metrics[group+"_late_world_drift"].append((logits[3].argmax(-1) == world).float() -
                                                               (logits[2].argmax(-1) == world).float())
                predictions.append(logits[3].argmax(-1) == answer)
            metrics[group+"_all_three"].append(torch.stack(predictions).all(0).float())
    return {k: torch.stack(v).mean(0) for k, v in metrics.items()}


@torch.no_grad()
def cue_eval(model, scenes, device):
    metrics = defaultdict(list)
    candidates = torch.tensor([IDS[n] for n in FRUITS], device=device)
    for selected in (0, 1):
        x, y = encode([cue_item(s, selected) for s in scenes], device)
        other = torch.tensor([IDS[s[3 - selected]] for s in scenes], device=device)
        isolated = model(x, mode="isolated")[0]
        torch.testing.assert_close(isolated, model(x[:, 9:], offset=9)[0], atol=2e-5, rtol=2e-5)
        for variant in ("two", "one", "filler", "first"):
            inp, _ = encode([cue_item(s, selected, variant) for s in scenes], device)
            for k, v in clean_metrics(model(inp)[0][-1], y, candidates).items():
                metrics[f"{variant}_{k}"].append(v)
        for control, early, late, rows, cols in (("matched", 2, 4, slice(9, 15), slice(6, 9)),
                                                  ("offset", 2, 5, slice(9, 15), slice(6, 9))):
            margins = []
            for label, cuts in (("open", ()), ("early", (early,)), ("late", (late,)), ("both", (early, late))):
                logits = model(x, cuts=cuts, cut_rows=rows, cut_cols=cols)[0][-1]
                margin = (logits.gather(1, y[:, None]) - logits.gather(1, other[:, None]))[:, 0]
                metrics[f"{control}_{label}_margin"].append(margin)
                margins.append(margin)
            metrics[control+"_interaction"].append(margins[1] - margins[3] - margins[0] + margins[2])
        # In the two-person prompt, compare equal-area cuts and counterbalance
        # both clues' locations. Independent decoy choices cancel incidental agreement.
        for decoy_choice in (0, 1):
            for relevant_first in (False, True):
                inputs, _ = encode([cue_decoy_item(s, selected, decoy_choice, relevant_first) for s in scenes], device)
                intact = model(inputs)[0][-1]
                metrics["two_person_accuracy"].append(intact.argmax(1).eq(y).float())
                for relevant in (False, True):
                    start = 6 if relevant == relevant_first else 9
                    prefix = "target" if relevant else "irrelevant"
                    margins = []
                    for cuts in ((), (2,), (4,), (2, 4), tuple(range(8))):
                        logits = (model(inputs, cuts=cuts, cut_rows=slice(12, 18),
                                        cut_cols=slice(start, start+3))[0][-1] if cuts else intact)
                        margins.append((logits.gather(1, y[:, None]) - logits.gather(1, other[:, None]))[:, 0])
                    metrics[prefix+"_interaction"].append(margins[1] - margins[3] - margins[0] + margins[2])
                    metrics[prefix+"_all_layer_cost"].append(margins[0] - margins[4])
    return {k: torch.stack(v).mean(0) for k, v in metrics.items()}


def evaluate(model, device, split, n, readers):
    rows = {}
    for task, evaluate_task in zip(TASKS, (carry_eval, memory_eval, cue_eval)):
        scenes = dataset(task, split, n)
        task_rows = []
        for start in range(0, n, 24):
            chunk = scenes[start:start + 24]
            if len(chunk) < 2 and task == "memory":
                raise ValueError("A shuffled-memory evaluation batch needs at least two scenes")
            values = evaluate_task(model, chunk, device, readers) if task == "carry" else evaluate_task(model, chunk, device)
            values = {key: vector.cpu().tolist() for key, vector in values.items()}
            for i, scene in enumerate(chunk):
                if task == "carry":
                    cluster = str(sum(scene[0]))
                elif task == "cue":
                    cluster = repr((sorted(zip(scene[:2], scene[2:4])), scene[4]))
                else:
                    cluster = repr(scene)
                task_rows.append({"family": scene, "cluster": cluster, "metrics": {k: v[i] for k, v in values.items()}})
        rows[task] = task_rows
    return rows


def write_json(path, data):
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def count(tied, width):
    return sum(p.numel() for p in TinyLM(width, tied).parameters())


def matched_width(width):
    reference = count(True, width)
    return min(range(HEADS, width + 1, HEADS), key=lambda d: abs(count(False, d) - reference))


def bootstrap(values, clusters, rng):
    # Resample complete clusters. Carry families sharing a total stay together.
    keys, ix = np.unique(clusters, return_inverse=True)
    totals = np.bincount(ix, weights=values)
    sizes = np.bincount(ix)
    draws = rng.integers(len(keys), size=(2000, len(keys)))
    samples = totals[draws].sum(1) / sizes[draws].sum(1)
    return {"mean": float(np.mean(values)), "low": float(np.quantile(samples, .025)),
            "high": float(np.quantile(samples, .975)), "clusters": len(keys)}


def summarize(results, seeds):
    rng = np.random.default_rng(924051)
    comparisons, individual = {}, {}
    for task in TASKS:
        sample = results[seeds[0], "tied"]["heldout"][task]
        clusters = [r["cluster"] for r in sample]
        for key in sample[0]["metrics"]:
            by_arch = {}
            for name in ("tied", "untied", "untied_budget"):
                arr = np.stack([[r["metrics"][key] for r in results[seed, name]["heldout"][task]] for seed in seeds])
                by_arch[name] = arr
                individual.setdefault(task, {}).setdefault(name, {})[key] = {
                    "mean": float(arr.mean()), "by_seed": arr.mean(1).tolist()}
            for name in ("untied", "untied_budget"):
                delta = by_arch["tied"] - by_arch[name]
                stats = bootstrap(delta.mean(0), clusters, rng)
                stats["by_seed"] = delta.mean(1).tolist()
                comparisons.setdefault(task, {}).setdefault("tied_minus_" + name, {})[key] = stats
    return individual, comparisons


def _study(outdir, device, steps=STEPS, batch=BATCH, width=WIDTH, seeds=SEEDS, n=HELDOUT):
    directory = Path(outdir) / "paired"
    directory.mkdir(parents=True, exist_ok=True)
    small = matched_width(width)
    archs = {"tied": (width, True), "untied": (width, False), "untied_budget": (small, False)}
    settings = {"seeds": list(seeds), "steps": steps, "batch": batch, "width": width, "n_heldout": n,
                "n_pilot": 24, "blocks_per_stage": BLOCKS, "stages": STAGES, "narrow_width": small,
                "heads": HEADS, "optimizer": "AdamW", "readout_stages": {"memory": [3, 4], "carry": [1, 2, 3, 4],
                                                                        "cue": [1, 2, 3, 4]},
                "vocabulary": VOCAB, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    results = {}
    for seed in seeds:
        # Functional-equivalence check using complete outputs at every stage.
        # Perform it in FP32 before gradient updates.
        with torch.no_grad():
            probe, _ = encode([cue_item(dataset("cue", "pilot", 1)[0], 0)], device)
            torch.manual_seed(seed)
            ref = TinyLM(width, True).to(device)(probe)[0]
            torch.manual_seed(seed)
            other = TinyLM(width, False).to(device)(probe)[0]
            torch.testing.assert_close(ref, other, atol=0, rtol=0)
        for name, (d, tied) in archs.items():
            path = directory / f"{seed}_{name}.json"
            previous = json.loads(path.read_text()) if path.is_file() else {}
            if previous.get("config") == settings and previous.get("complete") is True:
                results[seed, name] = previous
                continue
            torch.manual_seed(seed)
            model = TinyLM(d, tied).to(device)
            started = time.perf_counter()
            info = train(model, device, seed, steps, batch)
            for param in model.parameters():
                param.requires_grad_(False)
            # Input embeddings participate in activation attribution, never optimization here.
            model.embedding.weight.requires_grad_(True)
            readers = fit_readers(model, device)
            output = {"config": settings, "architecture": name, "seed": seed, "training": info,
                      "pilot": evaluate(model, device, "pilot", 24, readers),
                      "heldout": evaluate(model, device, "heldout", n, readers),
                      "elapsed_seconds": time.perf_counter() - started, "complete": True}
            write_json(path, output)
            results[seed, name] = output
            print(f"Controlled {name}, seed {seed} saved ({output['elapsed_seconds']:.1f}s)", flush=True)
            del model
        streams = {results[seed, name]["training"]["stream_sha256"] for name in archs}
        if len(streams) != 1:
            raise RuntimeError("Training streams differed between architectures")
    individual, comparisons = summarize(results, seeds)
    summary = {"all_runs_complete": True, "config": settings, "means": individual, "contrasts": comparisons,
               "scope": "Symbolic task-trained models. Intervals resample held-out scene families (sums grouped by total), conditional on these training seeds. Per-seed means are included. Carry swaps target the final prompt token. Identical French/English word 'six' is compatible with either language. The cue alternative offset was fixed before testing, without matching individual effect magnitudes. The narrow parameter-count control changes width and forward compute.",
               "device": str(device), "torch_version": torch.__version__,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    write_json(directory / "summary.json", summary)
    return summary


def run(outdir):
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle GPU before running the controlled training experiment")
    return _study(outdir, torch.device("cuda"))
