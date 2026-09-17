

Step 1 — one activation vector per token, for all features (not just top-N)

For a sentence with seq_len tokens, running it through GPT-2 and encoding through the SAE gives you:


acts = sae.encode(cache[sae.cfg.metadata.hook_name])[0]  # shape: [seq_len, n_features]
n_features is ~24,576 for this SAE. So acts is a full matrix: every token gets a value for every feature (most of them are 0, since SAEs are trained to be sparse, but the full row exists — we haven't picked "top N per token" yet at this point).

Step 2 — collapse across tokens, per feature (this is the key step)


max_vals, argmax_pos = acts_no_bos.max(dim=0)   # dim=0 = collapse over the token axis
dim=0 means: for each of the ~24,576 columns (features), look down that column across all token positions and take the single highest value. The result, max_vals, is a vector of length n_features — one number per feature, representing "the strongest this feature ever fired anywhere in this sentence." argmax_pos records which token position gave that peak, per feature.

This is the step that turns "per-token" into "per-sentence": we're not asking "what's big at this token," we're asking "what's the peak value each feature ever reaches across the whole sentence."

(For contrast: Neuronpedia's search-topk-by-token endpoint does the opposite order — top-k per token position, kept separate per position — which is a different, finer-grained view than what we're computing here.)

---------- Still to clean the features from strong activations that are unrelated to the countries

---

## Mutual information feature selection (Step 2)

The goal: out of ~16k SAE features, find the ones that actually distinguish countries from each other, so the steering vector isn't built out of noise.

**The formula** (mutual information between a feature `A_j` and the country label `C`):

```
I(A_j; C) = Σ_{a_j, c} P(a_j, c) · log( P(a_j, c) / (P(a_j) · P(c)) )
```

In words: for every combination of "did feature `j` fire? (yes/no)" and "which country?", compare how often that combination *actually* happens (`P(a_j, c)`) against how often it *would* happen if firing and country were independent (`P(a_j) · P(c)`). If a feature fires much more (or less) often for one country than chance would predict, it contributes a lot of MI. A feature that fires at the same rate regardless of country contributes ~0.

**How it's implemented** (`compute_mi` in the notebook):

- Activations are discretized to binary: `active = (X > 0)` — did the feature fire at all on this assertion, ignoring *how much*.
- `P(c)` — how common each country is in the dataset (`p_g`, one value per country).
- `P(a_j = 1)` — how often each feature fires overall, across all assertions (`p_active`, one value per feature).
- For each country `c` in turn:
  - `P(a_j = 1 | c)` — how often the feature fires *within just that country's* assertions.
  - `P(a_j = 1, c) = P(a_j = 1 | c) · P(c)` — the joint probability, built from the conditional.
  - That joint is plugged into the log-ratio term above and added to a running total (`mi`) — once for "fired" (`a_j = 1`) and once for "silent" (`a_j = 0`), since MI sums over both outcomes.
- This repeats per country and accumulates into one MI value per feature — so the whole thing is vectorized over all ~16k features at once, with only a small Python loop over the 22 countries (not over features or assertions).

**Taking the values**: the output `mi` is a single vector of length `n_features`. Higher = more country-informative. `select_top_mi` sorts features by MI descending and keeps the smallest prefix whose *cumulative* MI reaches a fraction `rho` (default 0.1 = 10%) of the total MI mass — so instead of picking a fixed count like "top 50," it keeps however many features are needed to capture that share of the total signal (in practice, 223 out of 16384 features here).

### Is MI per-country or per-feature? (and how does it become a per-country steering vector?)

`mi[j]` is **one scalar per feature, already summed across all 22 countries** inside `compute_mi`'s loop — it answers "how generally useful is feature `j` for telling countries apart," not "which country is feature `j` about." `select_top_mi` then does **one single global selection pass** over that flat vector, producing one feature subset `S` shared by every country — not a separate top-MI selection per country. This matches the paper (Sec 2.3): rank features globally, pick one `S`, and have every country's embedding live inside that same shared coordinate system.

That raises the natural question: if `S` is identical for every country, where does the country-specific steering direction actually come from? Two more steps, after MI:

1. **`build_prototypes` — per-country averages within the shared basis `S`.**
   ```python
   CuE = X[:, S]                             # restrict every assertion to just the S feature columns
   prototypes[g] = CuE[y == g].mean(axis=0)  # average, but only over THIS country's assertions
   ```
   Same `S` columns for every country, but a separate average computed per country. A feature in `S` that fires almost only for Japan will average out near-zero in every other country's prototype — so country identity shows up as *which coordinates of the shared basis are non-trivial*, not as a different feature set per country. (`S` isn't "features every country cares about equally" — MI rewards any feature diagnostic of *something* country-related, including one that's essentially Japan-only.)

2. **`build_steering_vector` — contrast the target against the rest, then decode.**
   ```python
   delta = prototypes[target] - mean(prototypes[other 21 countries])   # still in |S|-dim space
   delta_full[S] = delta                                                # zero-pad back to full n_features
   v_cue = delta_full @ sae.W_dec                                       # feature space -> residual-stream space
   ```
   Subtracting the mean of the *other* countries' prototypes cancels out whatever's generically "cultural-assertion-shaped" across all of them, leaving only what's distinctively true of the target. `@ sae.W_dec` then decodes that sparse direction back out of SAE-feature space into the model's actual residual-stream space — this is the `v_cue` that `steering_hook` adds into every token's activation during generation.

So the pipeline is: **MI (global, country-blind) → `S`, a shared basis → prototypes (per-country averages within that basis) → delta (per-country contrast against the rest) → `W_dec` decode → steering vector.**

**A caveat worth flagging**: because `S` is chosen by one global ranking, it can end up imbalanced — if a handful of countries have unusually separable assertions, their features could dominate the top of the ranking before the cumulative-10% cutoff is reached, leaving other countries with thinner, weaker prototypes (fewer of `S`'s dimensions actually "belong" to them). This is consistent with — though not a confirmed explanation of — the paper's own RQ5 finding that some countries steer more easily than others. Not yet checked empirically here; a quick diagnostic would be counting, per country, how many of `S`'s features have a non-trivial value in that country's prototype.

---

## Evaluation: two stages, two different questions

Two-line version, before the detail:

- **Stage 1** asks: *"before touching steering at all — can this feature space even tell countries apart?"*
- **Stage 2** asks: *"does nudging generation toward a country actually make it look more like that country than doing nothing would?"*

Stage 1 is a sanity gate that has to pass before Stage 2's numbers mean anything. Both live in `cultural_neurons/evaluation.py`.

### Stage 1 — sanity gate (`evaluate_holdout_separability`)

No generation, no steering vector — this only checks the representation itself. Take real held-out text (CANDLE/augmented assertions the training step never saw, country name stripped out same as training data), encode it, and ask: *"whose cultural fingerprint (prototype) does this text land closest to?"* Check that against the true label and report accuracy over the whole held-out set — plus a per-country breakdown, since some countries may separate better than others.

Why this needs to be checked rather than assumed: stripping the country's own name out of every assertion (so `China` isn't just detected by the literal word "China") is necessary to avoid a trivial shortcut — but it's also a real risk in the other direction. Strip too aggressively and the remaining text might be so generic that *nothing* could tell it apart, regardless of how good the model's cultural representation actually is. Stage 1 answers that empirically: if accuracy is near the random baseline (`1/22 ≈ 4.5%` for our 22 countries), the feature set or the stripping is destroying real signal, and no steering result downstream can be trusted no matter how good it looks. If it's well above baseline, the harder, stripped setting still carries genuine signal, and Stage 2 is worth running.

### Stage 2 — does steering work? (`evaluate_generation`, looped per target country in `main.py`)

For a culture-agnostic prompt (e.g. *"Describe a typical breakfast."*), generate two continuations: **unsteered** (no intervention) and **steered** (the target country's steering vector added in). Encode both the same way training data was encoded, then check which country's *held-out* prototype each one lands closest to — held-out, not the training prototype, so the eval isn't just confirming "we recreated the exact data the steering vector was built from" (see the `prototypes_train` vs `prototypes_holdout` note above).

**Why the unsteered baseline gets a target label too, even though it wasn't aiming at anything**: this trips people up, reasonably, so it's worth spelling out. `alpha=0` means no steering vector is added at all — the unsteered text is the same regardless of which country we're about to evaluate it against, and `main.py` generates it exactly once per prompt, then reuses it for every target country. Scoring that *same* unsteered text against, say, China's prototype isn't asking "did this text succeed at being about China" — it never tried to be. It's establishing a **baseline**: *"before any intervention, how close does the model's default output already sit to China?"*

That baseline is what makes the steered number mean anything. A steered rank of #1 is a strong result if the unsteered baseline was #15 for the same prompt and target — steering did real work. It's a much weaker result if the unsteered baseline was already #2 — the model was basically already there, steering barely mattered. You can't tell those two situations apart from the steered number alone; you need the matched unsteered-vs-steered comparison, *for the same target*, to isolate what steering actually contributed. (As a bonus, averaging the unsteered `predicted` country across many prompts and targets — free, since it's the same cached generations — gives a rough readout of the model's implicit cultural default, the same kind of thing the paper's RQ3 measures: without any steering, does the model just default to the US/UK regardless of what's asked?)

**What gets reported**, per target country, averaged over the 10 prompts in `data/eval_prompts.json`:
- `avg_rank` — where the target country landed in the full country similarity ranking (1 = best), for unsteered and steered separately, averaged across all 10 prompts.
- `hit_rate` — the *fraction* of those 10 prompts where the target country was the #1 nearest prototype (so `0.6` = 6 of 10 prompts), unsteered vs steered.

They're not measuring the same thing, which is why both get reported rather than picking one: `hit_rate` is binary per prompt (either the target won #1 or it didn't) — coarse, but directly answers "how often did steering actually win." `avg_rank` is continuous — it still registers a country moving from rank 4 to rank 2 as progress, even though that's a `0` in `hit_rate` both before and after. A country stuck permanently at rank 2 would show `hit_rate = 0%` forever while genuinely getting closer over time, which only `avg_rank` would catch.

**Worked example**, from a real run (`results/gemma-3-1b_layer17_c4_train0.67_holdout0.33_20260915-181449.json`, 4 target countries, alpha=1.5) — one single prompt for target `Mexico`, "Describe a typical breakfast.":

- *Unsteered* ranking: `Canada 0.455, Mexico 0.431, Japan -0.336, India -0.558` → Mexico is 2nd → this prompt's `target_rank = 2`, doesn't count toward `hit_rate`.
- *Steered* (toward Mexico) ranking: `Mexico 0.298, Canada 0.272, Japan -0.219, India -0.362` → Mexico flipped to 1st → `target_rank = 1`, this prompt now counts as a hit.

That's one prompt. Averaged over all 10 prompts for the `Mexico` target in that same run: `avg_rank_unsteered = 1.4 → avg_rank_steered = 1.3` (small improvement — Mexico was already landing 1st or 2nd most of the time even unsteered) and `hit_rate_unsteered = 0.6 → hit_rate_steered = 0.7` (went from winning 6/10 prompts to 7/10 — the breakfast prompt above is that 7th win). Modest, but a real, consistent gap in the direction steering should push.

Contrast with `India` in that same run: `avg_rank_unsteered = 3.8 → avg_rank_steered = 3.8` and `hit_rate = 0.0 → 0.0`, completely unchanged (out of 4 target countries here, rank 3.8 is nearly worst-possible). Steering did essentially nothing for India in this run — consistent with the "some countries steer more easily than others" caveat noted earlier (thinner `S`-prototype coverage, or this run's single-layer/alpha limitations from the "Why was generated text so low quality?" section below).

A convincing steering effect looks like a clear gap in both directions at once, like the Mexico case: steered `avg_rank` meaningfully lower (closer to 1) *and* `hit_rate` meaningfully higher than the unsteered baseline, for the same target. India shows what "no effect" looks like in this same schema — both numbers identical before and after.

**What this evaluation is *not***: it's a proxy for "did the internal representation move toward the target," entirely inside the model's own SAE-feature space. It says nothing about surface text quality, fluency, or whether a human would actually read the output as culturally faithful — that's what the paper's LLM-as-judge evaluation is for, and we don't have that infrastructure here (see the data augmentation section above for the related tradeoff of not having a truly independent verifier model).

---

## Why was generated text so low quality? (and what we changed)

Investigated by re-reading the actual paper (Khanuja et al. 2026, *Steering LLMs for Culturally Localized Generation*, arXiv:2603.23301 — found and PDF-extracted via `pypdf` since the bundled PDF's own text layer isn't machine-searchable without `poppler`) plus general literature on activation steering and prompting base models. Three separate causes stacked on top of each other:

**1. Biggest factor: we were generating from the base (`-pt`) model, but prompting it like an instruction-tuned one.** Our eval prompts are direct commands ("Write me a recipe for a local dish."). A base/pretrained model was never trained to *comply* with instructions — it was only trained to continue text statistically. Given a command it doesn't recognize as a pattern from pretraining data, it just continues however is locally plausible, which reads as "trash": off-topic, repetitive, or a continuation of the *prompt's shape* rather than an answer. This isn't specific to our reproduction — the paper hit the exact same tension and explicitly solved it: *"For Gemma models, we use the instruction-tuned variants to support open-ended generation; SAEs trained on the base models have been shown to transfer effectively to these variants"* (Sec 3, "Models"). In other words: keep the SAE trained on the base checkpoint (that's the only thing GemmaScope publishes), but swap the actual model doing the generating (and the assertion-encoding) to the `-it` checkpoint. Same architecture, same hook points, so the base-trained SAE decomposes the `-it` model's residual stream just fine.

   **What changed**: `config.PRESETS` now points every Gemma preset's `"model"` field at the `-it` checkpoint (`google/gemma-3-1b-it`, `gemma-2-9b-it`, etc.) while `"release"` stays on the base-trained SAE release, exactly matching the paper's setup. `gemma-1-2b` (predates GemmaScope, unrelated community SAE) and `llama-3.1-8b` (paper's transfer claim is scoped to "Gemma models" only) were left as-is.

**2. `ALPHA` was set far outside anything the paper actually validated.** App. C: they sweep `alpha ∈ {0.25, 0.5, 1, 2}` **per country**, and explicitly *discard* any alpha that produces low-fluency generations before reporting results — fluency is used purely as a filter, never optimized past. We had `ALPHA = 3.0` hardcoded globally, well above their max tested value, with no fluency check of any kind. At that strength the steering hook can push the residual stream far enough off-distribution to break coherence outright, independent of the base/instruct issue above. Lowered the default to `1.0` (mid of their tested range) as a safer starting point — a real fix would be a per-country fluency-filtered sweep like the paper's, which we don't have yet (`evaluate_generation` has no fluency signal at all currently — see the "not" caveat just above).

**3. Structural ceiling, not easily fixable: single-layer steering vs. the paper's all-layer steering.** The paper applies its steering vector at *every layer* of the model simultaneously (Sec 3, App. C) for the SAE widths where that's feasible (Gemma-2-2B-16K, Gemma-2-9B-16K, Llama-3.1-8B-32K) — spreading the intervention thin across the whole network. Our reproduction, on `gemma-3-1b` via GemmaScope-2, can only steer at **one** of just 3 available layers (13, 17, or 22 — GemmaScope-2 doesn't ship per-layer SAEs for Gemma 3 the way the original GemmaScope does for Gemma 2). A single-layer intervention needs a proportionally larger `alpha` to produce a comparable behavioral shift to the paper's distributed one — which pushes straight into the fluency-destroying regime from point 2. This is a real limitation of reproducing on Gemma-3 specifically; the closest faithful reproduction of the paper's own setup would be switching `PRESET_NAME` to `gemma-2-9b` (their actual model, with full per-layer 16k SAE coverage) or `gemma-2-2b` as a lighter middle ground — not fixable by prompt or alpha tuning alone on Gemma-3.

**Side finding, not yet acted on**: the paper's actual evaluation prompts (App. D, Table 4 — 24 prompts, e.g. "Write me a short story about a boy and his kite", "Explain photosynthesis as if I'm five years old") are pure creative-writing prompts where culture only shows up incidentally, distinct in character from several of our 10 hand-written prompts (e.g. "Describe what people wear to a wedding") which ask about cultural practices more directly. Not a bug, just worth knowing our eval set isn't a literal reproduction of Table 4 if exact comparability to the paper's numbers ever matters.

---

## Multi-layer steering (implemented)

Point 3 above (single-layer steering) is what this addresses. Confirmed working end-to-end via a small smoke test (`gemma-3-1b`, 2 countries, no augmented data): layer discovery found `[7, 13, 17, 22]` (correcting the stale "13/17/22 only" comment - `discover_layers` catches this kind of drift automatically now), MI selected `S` unevenly across those layers (16/58/53/11 features respectively - confirming `S` really does concentrate unevenly, not spread flat), and steering installed hooks at all 4 layers and produced different steered vs. unsteered generations without error.

- **Architecture**: `feature_selection.py`, `steering.build_prototypes`, and all of `evaluation.py` are unchanged - literally untouched by this refactor. A feature's real coordinate is `(layer, local_feature_idx)`, but everything above the forward-pass layer only ever sees one flat concatenated axis (`layers.concat_vectors`), matching how the paper itself describes MI: "ranked by mutual information across all layers" (Sec 2.3) - one global `S`, not a per-layer top-k.
- **`cultural_neurons/layers.py`** (new module) owns everything layer-specific: `discover_layers()` reads SAELens' own `pretrained_saes.yaml` metadata directly (no network probing, no per-layer try/except), `load_saes()` loads one SAE per layer, `layer_boundaries()` fixes the column ranges each layer occupies within the joint axis, `concat_vectors()` builds a joint per-assertion vector, and `split_S_by_layer()` translates `S` back into per-layer local indices for decoding.
- **Decode/steering is the one genuinely per-layer piece** (`steering.build_steering_vectors`, plural): the joint `delta = p_target - mean(p_others)` is computed once, but each layer's slice is zero-padded to that layer's own full SAE width and decoded through *that layer's own* `W_dec` - no cross-layer mixing, since `W_dec_l` only knows how to decode layer `l`'s feature space. `steering.hooked_generate` installs one hook per layer present in `v_cues` (layers with zero selected features are simply absent - nothing to steer there, no hook installed).
- **`activation_cache.get_feature_vectors_multi_layer`**: one cache file per layer still (unchanged fingerprint scheme), but for any assertion missing at least one layer, ONE shared `model.run_with_cache` call (with `names_filter` restricted to only the still-missing hook points) feeds every layer's SAE - the expensive part (the transformer forward pass) is never repeated per layer, only the cheap SAE-encode step is.
- **`config.LAYERS = "all"`**: resolved at the start of `main.py` via `layers.discover_layers`, with a loud banner printing exactly which layers were found before any real work starts - never silently treated as the paper's dense per-layer setup when GemmaScope-2 (Gemma 3) only ships a handful of layers per model size (unlike GemmaScope's dense coverage for Gemma 2).
- **`ALPHA`**: pinned to `1.0`, layer-uniform (no per-layer magnitude normalization) - a manual judgment call on raising it, not an auto-tuned value.
- **Validated on `gemma-3-1b` first** (cheap: only 4 layers) before attempting `gemma-2-2b`/`9b` (26/42 layers - much heavier; back-of-envelope, loading every layer's SAE simultaneously for `gemma-2-9b` is ~4.9B params of SAEs alone, on top of the 9B base model - likely infeasible locally, same constraint the paper hit on a single A100 and worked around with a stride-4 layer subset).

**Deferred to future work, explicitly not done now**: signal-driven layer selection (using Stage 1 separability or per-layer MI mass to pick a handful of high-signal layers, rather than every available layer or a fixed stride) - the paper's own RQ1 probe shows exactly this kind of signal exists (cultural information concentrates more in later layers) but never uses it to *choose* a sparse layer subset, only to justify "steer everywhere." Worth revisiting once exhaustive-layer results are in hand as a baseline to compare against.

