"""Render the four patch sites in the controlled architecture experiment."""


def render(summary):
    config = summary["config"]
    if (config["stages"], config["blocks_per_stage"]) != (4, 2):
        raise ValueError("The block map depicts four stages of two blocks each")

    columns = []
    scores = []
    for stage in range(1, 5):
        end = 2 * stage
        checked = " checked" if stage == 1 else ""
        columns.append(f"""
          <div class="map-col"><input class="sr-only" type="radio" name="patch-stage"
            id="patch-stage-{stage}" aria-label="Stage {stage}, after block {end}"{checked}>
          <label for="patch-stage-{stage}">
            <span class="map-step">Stage <b>{stage}</b></span>
            <span class="block-pair"><i class="block-a">A</i><i class="block-b">B</i></span>
            <span class="block-pair independent"><i>{end-1}</i><i>{end}</i></span>
          </label></div>""")
        key = f"s{stage}_swap_observed"
        row = []
        for arch, title in (("tied", "Looped"), ("untied", "Regular")):
            effect = summary["means"]["carry"][arch][key]["mean"]
            value = f"{effect:+.3f}".replace("-", "−")
            row.append(f'<span><span>{title}</span><strong>{value}</strong></span>')
        scores.append(f'<div class="stage-scores stage-{stage}">'
                      f'<div>Swap after block {end}<small>Mean shift · logits</small></div>'
                      + "".join(row) + "</div>")

    return ("""
<figure class="map-card" aria-labelledby="map-title">
  <p class="overline">Small models · matched training</p>
  <h2 id="map-title">Eight block visits, two ways</h2>
  <p class="map-intro">The looped model revisits the same two blocks, A and B. The regular Transformer has eight distinct blocks. Choose where to copy a state between two sums.</p>
  <fieldset class="map-chooser"><legend class="sr-only">Choose a patch stage in both models</legend>
    <div class="map-grid">
      <div class="map-axis" aria-hidden="true"><span></span><b>Looped</b><b>Regular</b></div>
""" + "".join(columns) + """
    </div>
    <div class="map-output" aria-live="polite" aria-atomic="true">""" + "".join(scores) + """</div>
  </fieldset>
  <figcaption>For example, 15 + 39 and 13 + 41 both equal 54. Moving only the first sum’s carry into the second would change its tens digit from five to six. Each stage is a separate swap at the last input token. Scores average 128 sum pairs and three seeds. They measure how much the swap changed the final logit of the carry-predicted digit relative to the original digit, with donor and recipient in the same language.</figcaption>
</figure>""")
