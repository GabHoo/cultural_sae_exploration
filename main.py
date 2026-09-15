"""
End-to-end pipeline: data prep -> forward pass -> pooling -> feature selection ->
prototypes + steering vector -> generation -> held-out evaluation.

All tunable options live in config.py - edit that file to change model/layer/data/
steering behavior, not this one. See NOTES.md for a detailed walkthrough of each
stage's mechanics.
"""

import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch

import config
from cultural_neurons import activation_cache, data_prep, evaluation, feature_selection, forward_pass, pooling, steering


def results_path(n_countries: int) -> str:
    """
    Build a unique output path under config.EVAL_RESULTS_DIR so results from different
    runs (model preset, country subset, split fractions) never overwrite each other.

    Input:
        n_countries — how many countries this run actually used (post config.COUNTRIES filter)

    Output:
        e.g. "results/gemma-3-1b_layer17_c5_train0.67_holdout0.33_20260915-101530.json"
    """
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = (
        f"{config.PRESET_NAME}_layer{config.LAYER}_c{n_countries}"
        f"_train{config.TRAIN_FRACTION}_holdout{config.HELDOUT_FRACTION}_{timestamp}.json"
    )
    return os.path.join(config.EVAL_RESULTS_DIR, filename)


def main():
    torch.set_grad_enabled(False)   # everything here is inference only

    device = forward_pass.get_device()
    preset = config.PRESETS[config.PRESET_NAME]
    print(f"loading {config.PRESET_NAME} (layer {config.LAYER}) on {device}...")
    model, sae = forward_pass.load_model_and_sae(preset, config.LAYER, device)
    pool_fn = pooling.POOLING_FUNCS[config.POOLING_STRATEGY]

    # --- Data prep ---
    strip_country = data_prep.make_stripper(config.COUNTRY_ALIASES)

    def restrict(groups: dict[str, list[str]]) -> dict[str, list[str]]:
        if config.COUNTRIES is None:
            return groups
        return {c: texts for c, texts in groups.items() if c in config.COUNTRIES}

    # --- Forward pass + pooling, via the activation cache ---
    # Embeds every assertion in each source ONCE, keyed by (country, raw text); a later
    # run that only changes TRAIN_FRACTION / HELDOUT_FRACTION / COUNTRIES is served
    # straight from disk instead of re-running the model. See activation_cache.py.
    cache_meta = activation_cache.cache_meta(
        preset["model"], preset["release"], config.LAYER, config.POOLING_STRATEGY, config.COUNTRY_ALIASES,
    )
    raw_groups = restrict(data_prep.load_candle_groups(config.COUNTRIES_PATH, "country"))
    vectors = activation_cache.get_feature_vectors(
        raw_groups, model, sae, pool_fn, strip_country, config.CACHE_DIR, config.CANDLE_CACHE_NAME, cache_meta,
    )

    if config.USE_AUGMENTED:
        raw_augmented_groups = restrict(data_prep.load_candle_groups(config.AUGMENTED_COUNTRIES_PATH, "country"))
        augmented_vectors = activation_cache.get_feature_vectors(
            raw_augmented_groups, model, sae, pool_fn, strip_country,
            config.CACHE_DIR, config.AUGMENTED_CACHE_NAME, cache_meta,
        )
        raw_groups = data_prep.merge_groups(raw_groups, raw_augmented_groups)
        vectors = {
            country: {**vectors.get(country, {}), **augmented_vectors.get(country, {})}
            for country in raw_groups
        }

    countries = sorted(raw_groups.keys())
    if config.COUNTRIES is not None:
        print(f"restricted to {len(countries)} countries: {countries}")
    print(f"{sum(len(texts) for texts in raw_groups.values())} total assertions available "
          f"({'with' if config.USE_AUGMENTED else 'without'} augmented data)")

    # --- Sampling ---
    train_groups, holdout_groups = data_prep.train_holdout_split(
        raw_groups, config.TRAIN_FRACTION, config.HELDOUT_FRACTION, config.RANDOM_SEED,
    )
    # Stripping already happened inside get_feature_vectors before caching - here we just
    # need the raw text back out, as the lookup key into `vectors`.
    texts_train, labels_train = data_prep.build_dataset(train_groups, countries, lambda text, label: text)
    texts_holdout, labels_holdout = data_prep.build_dataset(holdout_groups, countries, lambda text, label: text)
    print(f"{len(texts_train)} train assertions, {len(texts_holdout)} held-out assertions")

    X_train = np.stack([vectors[label][text] for text, label in zip(texts_train, labels_train)])
    y_train = np.array(labels_train)
    X_holdout = np.stack([vectors[label][text] for text, label in zip(texts_holdout, labels_holdout)])
    y_holdout = np.array(labels_holdout)

    # --- Feature selection ---
    mi = feature_selection.compute_mi(X_train, y_train, countries)
    S, order = feature_selection.select_top_mi(mi, config.MI_RHO)
    print(f"selected |S| = {len(S)} / {len(mi)} features ({len(S) / len(mi) * 100:.2f}%)")

    # --- Prototypes ---
    prototypes_train = steering.build_prototypes(X_train, y_train, countries, S)
    prototypes_holdout = steering.build_prototypes(X_holdout, y_holdout, countries, S)


    # --- Sanity gate: is held-out text still separable against the train prototypes? ---
    # Stage 1 eval (see evaluation.py docstring) - run before touching steering. If this
    # isn't well above the 1/len(countries) random baseline, S or the name-stripping is
    # destroying real signal, and no steering result below would mean anything.
    separability = evaluation.evaluate_holdout_separability(X_holdout, y_holdout, S, prototypes_train)
    random_baseline = 1 / len(countries)
    print(f"\nheld-out separability: {separability['accuracy'] * 100:.1f}% accuracy "
          f"(n={separability['n']}, random baseline={random_baseline * 100:.1f}%)")
    if separability["accuracy"] < 2 * random_baseline:
        print("  WARNING: barely above random baseline - steering results below may not be meaningful")

    out_path = results_path(len(countries))
    with open(out_path, "w") as f:
        json.dump({
            "preset": config.PRESET_NAME,
            "layer": config.LAYER,
            "countries": countries,
            "train_fraction": config.TRAIN_FRACTION,
            "holdout_fraction": config.HELDOUT_FRACTION,
            "separability": separability,
        }, f, indent=2)
    print(f"\nsanity-check results written to {out_path}")

    # --- Steering + evaluation, looped over every target country ---
    # Commented out for now - only running the Stage 1 sanity check above until that's
    # trusted. Everything below is otherwise unchanged; uncomment to resume Stage 2.
    
    # N_TARGET_SAMPLE lets you try a handful of countries quickly (e.g. 5) before
    # committing to the full len(countries) sweep, which is ~22x the generation cost.
    if config.N_TARGET_SAMPLE is None:
        target_countries = countries
    else:
        target_countries = random.Random(config.RANDOM_SEED).sample(countries, config.N_TARGET_SAMPLE)
    print(f"\nevaluating {len(target_countries)} target countries: {target_countries}")
    
    eval_prompts = data_prep.load_eval_prompts(config.EVAL_PROMPTS_PATH)
    gen_kwargs = dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        seed=config.GENERATION_SEED,
    )
    
    # Unsteered generation doesn't depend on the target country - alpha=0 skips the
    # steering hook entirely regardless of v_cue - so generate it once per prompt and
    # reuse it for every target's evaluation below, instead of redoing it per country.
    print("generating unsteered baselines (shared across all targets)...")
    unsteered_by_prompt = {
        prompt: steering.hooked_generate(model, sae, prompt, v_cue=torch.zeros_like(sae.W_dec[0]), alpha=0.0, **gen_kwargs)
        for prompt in eval_prompts
    }
    
    all_results = {}
    for target_country in target_countries:
        # Scored against prototypes_holdout, not prototypes_train, so evaluation isn't
        # circular against the exact data v_cue was built from (see NOTES.md).
        v_cue = steering.build_steering_vector(
            target_country, countries, X_train.shape[1], S, prototypes_train, sae, device,
        )
    
        prompt_results = []
        for prompt in eval_prompts:
            unsteered = unsteered_by_prompt[prompt]
            steered = steering.hooked_generate(model, sae, prompt, v_cue, alpha=config.ALPHA, **gen_kwargs)
            eval_unsteered = evaluation.evaluate_generation(
                unsteered, target_country, S, prototypes_holdout, model, sae, pool_fn,
            )
            eval_steered = evaluation.evaluate_generation(
                steered, target_country, S, prototypes_holdout, model, sae, pool_fn,
            )
            prompt_results.append({
                "prompt": prompt,
                "unsteered_text": unsteered,
                "steered_text": steered,
                "unsteered": eval_unsteered,
                "steered": eval_steered,
            })
    
        avg_rank_unsteered = float(np.mean([r["unsteered"]["target_rank"] for r in prompt_results]))
        avg_rank_steered = float(np.mean([r["steered"]["target_rank"] for r in prompt_results]))
        hit_rate_unsteered = float(np.mean([r["unsteered"]["predicted"] == target_country for r in prompt_results]))
        hit_rate_steered = float(np.mean([r["steered"]["predicted"] == target_country for r in prompt_results]))
    
        all_results[target_country] = {
            "avg_rank_unsteered": avg_rank_unsteered,
            "avg_rank_steered": avg_rank_steered,
            "hit_rate_unsteered": hit_rate_unsteered,
            "hit_rate_steered": hit_rate_steered,
            "prompts": prompt_results,
        }
        print(f"{target_country:>15}:  avg_rank unsteered={avg_rank_unsteered:5.2f} steered={avg_rank_steered:5.2f}"
              f"   hit_rate unsteered={hit_rate_unsteered * 100:4.0f}% steered={hit_rate_steered * 100:4.0f}%")
    
    mean_rank_unsteered = np.mean([r["avg_rank_unsteered"] for r in all_results.values()])
    mean_rank_steered = np.mean([r["avg_rank_steered"] for r in all_results.values()])
    mean_hit_unsteered = np.mean([r["hit_rate_unsteered"] for r in all_results.values()])
    mean_hit_steered = np.mean([r["hit_rate_steered"] for r in all_results.values()])
    print(f"\nOVERALL across {len(target_countries)} target countries x {len(eval_prompts)} prompts, alpha={config.ALPHA}:")
    print(f"  avg target_rank (lower=better, best=1):  unsteered={mean_rank_unsteered:.2f}  steered={mean_rank_steered:.2f}")
    print(f"  top-1 hit rate:                          unsteered={mean_hit_unsteered * 100:.1f}%  steered={mean_hit_steered * 100:.1f}%")
    
    with open(out_path, "w") as f:
        json.dump({
            "preset": config.PRESET_NAME,
            "layer": config.LAYER,
            "target_countries": target_countries,
            "alpha": config.ALPHA,
            "separability": separability,
            "results": all_results,
        }, f, indent=2)
    print(f"\nfull per-prompt results written to {out_path}")


if __name__ == "__main__":
    main()
