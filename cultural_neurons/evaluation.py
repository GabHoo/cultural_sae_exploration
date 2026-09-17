"""
Two evaluation stages, meant to run in order.

Stage 1 (evaluate_holdout_separability) is a sanity gate, not a steering eval at all:
it checks whether held-out assertions - real CANDLE/augmented text, never touched by
generation or steering - still classify correctly against country prototypes built from
TRAINING data alone. Both train and held-out have the country's own name stripped (see
data_prep.strip_country), on purpose: the steered generations scored in Stage 2 never
contain the literal country name either, so held-out has to be stripped the same way for
the comparison to be fair. But stripping could in principle make the text too generic to
separate at all - this stage checks that empirically instead of assuming it either way.
If accuracy here isn't well above the 1/22 ~= 4.5% random baseline, the feature set S (or
the stripping) is destroying real signal, and no steering vector built from this pipeline
will evaluate meaningfully downstream.

Stage 2 (evaluate_generation) is the actual steering evaluation: encode a generated
continuation the same way training data was encoded, then check whether it lands closer
to the target country's held-out prototype than to any other country's. This is a proxy
for "did the internal representation move toward the target," not for surface-level text
quality - see NOTES.md for the tradeoff against an LLM-judge eval.

Both stages center prototypes/query vectors by subtracting the global mean prototype
(mu_global, paper Sec 2.4) before computing cosine similarity - see center_prototypes().
"""

from collections import defaultdict

import numpy as np
from sae_lens import SAE
from transformer_lens import HookedTransformer

from cultural_neurons import layers as layers_module
from cultural_neurons.forward_pass import get_activations


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Input:
        a, b — 1D np.ndarray, same shape

    Output:
        Cosine similarity in [-1, 1]; 0.0 if either vector is all-zero.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def center_prototypes(prototypes: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Paper Sec 2.4 (Step 3a): mu_global = mean of ALL countries' prototypes. A feature can
    clear the MI bar (its firing rate does vary by country) while still carrying a large
    activation component shared by almost every country - e.g. a "food ritual" feature
    that fires somewhat more for one country but is active across most assertions
    regardless. Left uncentered, that shared component dominates cosine similarity and
    dilutes the country-specific angle MI selected S for in the first place. Subtracting
    mu_global before comparing removes it - this is a SEPARATE operation from
    build_steering_vector's "subtract mean of the 21 OTHER countries" contrast (which
    stays target-exclusive on purpose); this one subtracts the mean of every country,
    including whichever one is being scored, purely to re-center the coordinate system.

    Input:
        prototypes — {label: prototype}, e.g. from steering.build_prototypes()

    Output:
        (centered_prototypes, mu_global) — centered_prototypes has the same keys/shapes
        as the input, each with mu_global subtracted; mu_global is [len(S)], the mean
        prototype across every label in the input, for centering query vectors the same way.
    """
    mu_global = np.mean(list(prototypes.values()), axis=0)
    centered = {label: proto - mu_global for label, proto in prototypes.items()}
    return centered, mu_global


def rank_by_similarity(
    vec: np.ndarray,
    prototypes: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    """
    Rank every label's prototype by cosine similarity to vec, highest first.

    Input:
        vec        — np.ndarray [len(S)], e.g. a generated continuation's pooled
                     activations restricted to S
        prototypes — {label: prototype}, e.g. held-out prototypes from steering.build_prototypes()

    Output:
        list of (label, similarity), sorted descending by similarity.
    """
    scored = [(label, cosine_similarity(vec, proto)) for label, proto in prototypes.items()]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


def evaluate_holdout_separability(
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    S: np.ndarray,
    prototypes: dict[str, np.ndarray],
) -> dict:
    """
    Stage 1 sanity gate - run this BEFORE building any steering vector. For every
    held-out assertion, find its nearest country prototype by cosine similarity and
    check whether it matches that assertion's true label. No generation, no steering:
    this only asks whether real held-out text is still separable in this feature space.

    Input:
        X_holdout  — np.ndarray [n_holdout, n_features], pooled SAE activations for
                     held-out assertions (forward_pass.extract_feature_matrix() on
                     data_prep's held-out split)
        y_holdout  — np.ndarray [n_holdout] of true country labels, parallel to X_holdout's rows
        S          — selected feature indices, from feature_selection.select_top_mi()
        prototypes — {country: prototype}, built from TRAINING data only
                     (steering.build_prototypes() on the train split - never the held-out one)

    Output:
        dict with:
            "accuracy"     — fraction of held-out assertions whose nearest prototype
                             matches their true country label
            "n"            — total held-out assertions evaluated
            "per_country"  — {country: accuracy restricted to that country's held-out rows}
    """
    centered_prototypes, mu_global = center_prototypes(prototypes)
    vecs = X_holdout[:, S] - mu_global
    correct = 0
    per_country_correct: dict[str, int] = defaultdict(int)
    per_country_total: dict[str, int] = defaultdict(int)

    for vec, true_label in zip(vecs, y_holdout):
        predicted = rank_by_similarity(vec, centered_prototypes)[0][0]
        per_country_total[true_label] += 1
        if predicted == true_label:
            correct += 1
            per_country_correct[true_label] += 1

    return {
        "accuracy": correct / len(y_holdout),
        "n": len(y_holdout),
        "per_country": {
            country: per_country_correct[country] / total
            for country, total in per_country_total.items()
        },
    }


def evaluate_generation(
    text: str,
    target_label: str,
    S: np.ndarray,
    boundaries: dict[int, tuple[int, int]],
    holdout_prototypes: dict[str, np.ndarray],
    model: HookedTransformer,
    saes: dict[int, SAE],
    pool_fn,
) -> dict:
    """
    Score one generated text against the held-out prototypes. Encodes the text through
    EVERY layer's SAE and concatenates in the same layer order used everywhere else
    (layers.concat_vectors), so the resulting vector lives in the same joint feature
    space X_train/X_holdout and the prototypes were built in - S then restricts it
    exactly as it restricts those.

    Input:
        text               — a generated continuation (steered or unsteered)
        target_label       — the country steering was aimed at
        S                  — selected GLOBAL feature indices (joint axis across layers),
                             from feature_selection.select_top_mi()
        boundaries          — {layer: (start, end)}, from layers.layer_boundaries()
        holdout_prototypes — {label: prototype}, built from held-out CANDLE assertions
                              (never used to build the steering vector)
        model, saes         — model from forward_pass.load_model(); saes is {layer: SAE},
                              e.g. from layers.load_saes()
        pool_fn            — pooling function, e.g. pooling.POOLING_FUNCS["max"]

    Output:
        dict with:
            "ranking"        — full list of (label, similarity), sorted descending
            "predicted"      — top-1 label
            "target_rank"    — target_label's 1-indexed position in the ranking
            "target_similarity" — target_label's raw cosine similarity
    """
    centered_prototypes, mu_global = center_prototypes(holdout_prototypes)
    pooled_by_layer = {
        layer: pool_fn(get_activations(text, model, sae)).cpu().numpy()
        for layer, sae in saes.items()
    }
    pooled = layers_module.concat_vectors(pooled_by_layer, boundaries)
    vec = pooled[S] - mu_global
    ranking = rank_by_similarity(vec, centered_prototypes)
    labels_in_order = [label for label, _ in ranking]
    return {
        "ranking": ranking,
        "predicted": labels_in_order[0],
        "target_rank": labels_in_order.index(target_label) + 1,
        "target_similarity": dict(ranking)[target_label],
    }
