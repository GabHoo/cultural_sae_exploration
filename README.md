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

