"""Compare the two same-width architectures in the held-out training study."""


def signed(value, digits):
    return f"{value:+.{digits}f}".replace("-", "−")


def render(summary: dict) -> str:
    config = summary["config"]
    means = summary["means"]
    comparisons = summary["contrasts"]
    outcomes = (
        ("carry", "s1_swap_observed", "Carry · Final score shift after block 2", "logits"),
        ("memory", "observer_s4_accuracy", "Coin · Correct box after block 8", "%"),
        ("memory", "observer_all_three", "Coin · All three answers correct after block 8", "%"),
        ("cue", "two_person_accuracy", "Lockers · Correct fruit, both copies, no cut", "%"),
    )
    rows = []
    for task, key, label, unit in outcomes:
        tied = means[task]["tied"][key]["mean"]
        regular = means[task]["untied"][key]["mean"]
        difference = comparisons[task]["tied_minus_untied"][key]["mean"]
        if unit == "logits":
            values = (signed(tied, 3), signed(regular, 3), signed(difference, 3) + " logits")
        else:
            values = (f"{100 * tied:.1f}%", f"{100 * regular:.1f}%", signed(100 * difference, 1) + " points")
        cells = "".join(f"<td>{value}</td>" for value in values)
        rows.append(f'<tr><th scope="row">{label}</th>{cells}</tr>')

    carry = comparisons["carry"]["tied_minus_untied"]["s1_swap_observed"]
    cut = "target_interaction"
    tied_cut = means["cue"]["tied"][cut]["mean"]
    regular_cut = means["cue"]["untied"][cut]["mean"]
    return f"""
<section class="training-comparison" aria-labelledby="training-comparison-title">
<h3 class="sub" id="training-comparison-title">The same two architectures on all three tasks</h3>
<p>These small models learned the three tasks using a shared vocabulary of symbolic tokens. Ouro and Qwen answered natural-language prompts in the checkpoint experiments.</p>
<table class="matrix"><caption>Means across {config["n_heldout"]} held-out families per task and {len(config["seeds"])} training seeds. Differences use unrounded scores.</caption>
<thead><tr><th scope="col">Measurement</th><th scope="col">Looped</th><th scope="col">Regular</th><th scope="col">Looped minus regular</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>The early carry swap favored the looped model in all three seeds. Its paired difference had a 95% interval of {signed(carry["low"], 3)} to {signed(carry["high"], 3)} logits when held-out families were resampled by sum total. Regular weights scored higher on individual coin questions in two seeds. The looped model scored higher in the third.</p>
<p>We also tested the locker attention cuts. After cutting the target clue at block 3, the cost of cutting it at block 5 changed by {signed(tied_cut, 6)} logits in the looped model and {signed(regular_cut, 6)} in the regular model. These visits reused block A in the looped model and hit distinct blocks 3 and 5 in the regular model.</p>
<details class="numbers"><summary>Scoring and model sizes</summary>
<p>Each carry family contributed both transfer directions in English and French. The shift measures the final score of the carry-predicted digit relative to the original digit. Each coin family supplied three questions under each of two witness assignments. After block 4, all three questions received the same eight stored states. We scored the first-choice token after block 8. Each two-person locker family supplied eight versions, varying the two people’s locker assignments and clue order. Success required the correct fruit to be the model’s first-choice token.</p>
<p>Both models in the table use width {config["width"]} and execute eight blocks. Reusing a pair of blocks reduces the number of trainable parameters. A separate regular model with width {config["narrow_width"]} brought its parameter count closer to the looped model’s and used less computation per block.</p>
</details></section>"""
