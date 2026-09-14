# cultural_sae_exploration
Step 1 — one activation vector per token, for all features (not just top-N)

For a sentence with seq_len tokens, running it through GPT-2 and encoding through the SAE gives you:


acts = sae.encode(cache[sae.cfg.metadata.hook_name])[0]  # shape: [seq_len, n_features]
n_features is ~24,576 for this SAE. So acts is a full matrix: every token gets a value for every feature (most of them are 0, since SAEs are trained to be sparse, but the full row exists — we haven't picked "top N per token" yet at this point).

Step 2 — collapse across tokens, per feature (this is the key step)


max_vals, argmax_pos = acts_no_bos.max(dim=0)   # dim=0 = collapse over the token axis
dim=0 means: for each of the ~24,576 columns (features), look down that column across all token positions and take the single highest value. The result, max_vals, is a vector of length n_features — one number per feature, representing "the strongest this feature ever fired anywhere in this sentence." argmax_pos records which token position gave that peak, per feature.

This is the step that turns "per-token" into "per-sentence": we're not asking "what's big at this token," we're asking "what's the peak value each feature ever reaches across the whole sentence."

Step 3 — now take top-k, but over features, not tokens


top = torch.topk(max_vals, k)   # top k features by their per-sentence peak
Now we pick the top N — but we're picking from the already-collapsed per-feature vector (length n_features), not from a per-token list. So the ordering you get is "features ranked by their single best moment anywhere in this sentence," and argmax_pos tells you which token that moment happened at (used to recover the token in the returned tuple).

So to correct the assumption in your question: we don't compute a top-N per token and then merge those lists. We compute the full dense activation matrix for every token × every feature, reduce it down to one number per feature via max over the token axis, and only take "top N" once, at the very end, over features. A feature makes the sentence-level top-10 by having one standout moment somewhere in the text — it doesn't matter if it's silent everywhere else.

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

