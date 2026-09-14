"""
End-to-end pipeline: data prep -> forward pass -> pooling -> feature selection ->
prototypes + steering vector -> generation -> held-out evaluation.

All tunable options live in config.py - edit that file to change model/layer/data/
steering behavior, not this one. See NOTES.md for a detailed walkthrough of each
stage's mechanics.
"""

import numpy as np
import torch

import config
from cultural_neurons import data_prep, evaluation, feature_selection, forward_pass, pooling, steering


def main():
    torch.set_grad_enabled(False)   # everything here is inference only

    device = forward_pass.get_device()
    preset = config.PRESETS[config.PRESET_NAME]
    print(f"loading {config.PRESET_NAME} (layer {config.LAYER}) on {device}...")
    model, sae = forward_pass.load_model_and_sae(preset, config.LAYER, device)
    pool_fn = pooling.POOLING_FUNCS[config.POOLING_STRATEGY]

    # --- Data prep ---
    country_groups = data_prep.load_candle_groups(config.COUNTRIES_PATH, "country")
    countries = sorted(country_groups.keys())
    train_groups, holdout_groups = data_prep.train_holdout_split(
        country_groups, config.N_PER_GROUP_TRAIN, config.N_PER_GROUP_HELDOUT, config.RANDOM_SEED,
    )
    strip_country = data_prep.make_stripper(config.COUNTRY_ALIASES)
    assertions_train, labels_train = data_prep.build_dataset(train_groups, countries, strip_country)
    assertions_holdout, labels_holdout = data_prep.build_dataset(holdout_groups, countries, strip_country)
    print(f"{len(assertions_train)} train assertions, {len(assertions_holdout)} held-out assertions")

    # --- Forward pass + pooling ---
    print("extracting train activations...")
    X_train = forward_pass.extract_feature_matrix(assertions_train, model, sae, pool_fn)
    y_train = np.array(labels_train)

    print("extracting held-out activations...")
    X_holdout = forward_pass.extract_feature_matrix(assertions_holdout, model, sae, pool_fn)
    y_holdout = np.array(labels_holdout)

    # --- Feature selection ---
    mi = feature_selection.compute_mi(X_train, y_train, countries)
    S, order = feature_selection.select_top_mi(mi, config.MI_RHO)
    print(f"selected |S| = {len(S)} / {len(mi)} features ({len(S) / len(mi) * 100:.2f}%)")

    # --- Prototypes ---
    prototypes_train = steering.build_prototypes(X_train, y_train, countries, S)
    prototypes_holdout = steering.build_prototypes(X_holdout, y_holdout, countries, S)

    # --- Steering vector ---
    v_cue = steering.build_steering_vector(
        config.TARGET_COUNTRY, countries, X_train.shape[1], S, prototypes_train, sae, device,
    )

    # --- Generation ---
    gen_kwargs = dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        seed=config.GENERATION_SEED,
    )
    unsteered = steering.hooked_generate(model, sae, config.PROMPT, v_cue, alpha=0.0, **gen_kwargs)
    steered = steering.hooked_generate(model, sae, config.PROMPT, v_cue, alpha=config.ALPHA, **gen_kwargs)
    print(f"\nUNSTEERED:\n{unsteered}")
    print(f"\nSTEERED toward {config.TARGET_COUNTRY} (alpha={config.ALPHA}):\n{steered}")

    # --- Evaluation (against held-out prototypes, never used to build v_cue) ---
    eval_unsteered = evaluation.evaluate_generation(
        unsteered, config.TARGET_COUNTRY, S, prototypes_holdout, model, sae, pool_fn,
    )
    eval_steered = evaluation.evaluate_generation(
        steered, config.TARGET_COUNTRY, S, prototypes_holdout, model, sae, pool_fn,
    )
    print(f"\nEVAL unsteered: predicted={eval_unsteered['predicted']!r}  "
          f"target_rank={eval_unsteered['target_rank']}/{len(countries)}  "
          f"target_similarity={eval_unsteered['target_similarity']:.3f}")
    print(f"EVAL steered:   predicted={eval_steered['predicted']!r}  "
          f"target_rank={eval_steered['target_rank']}/{len(countries)}  "
          f"target_similarity={eval_steered['target_similarity']:.3f}")


if __name__ == "__main__":
    main()
