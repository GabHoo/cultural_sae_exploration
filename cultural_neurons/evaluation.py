"""
Quick, LLM-free steering evaluation: encode a generated continuation the same way
training data was encoded, then check whether it lands closer to the target country's
HELD-OUT prototype (built from real CANDLE assertions the steering vector never saw)
than to any other country's. This is a proxy for "did the internal representation move
toward the target," not for surface-level text quality - see NOTES.md for the tradeoff
against an LLM-judge eval.
"""

import numpy as np
from sae_lens import SAE
from transformer_lens import HookedTransformer

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


def evaluate_generation(
    text: str,
    target_label: str,
    S: np.ndarray,
    holdout_prototypes: dict[str, np.ndarray],
    model: HookedTransformer,
    sae: SAE,
    pool_fn,
) -> dict:
    """
    Score one generated text against the held-out prototypes.

    Input:
        text               — a generated continuation (steered or unsteered)
        target_label       — the country steering was aimed at
        S                  — selected feature indices, from feature_selection.select_top_mi()
        holdout_prototypes — {label: prototype}, built from held-out CANDLE assertions
                              (never used to build the steering vector)
        model, sae         — as returned by forward_pass.load_model_and_sae()
        pool_fn            — pooling function, e.g. pooling.POOLING_FUNCS["max"]

    Output:
        dict with:
            "ranking"        — full list of (label, similarity), sorted descending
            "predicted"      — top-1 label
            "target_rank"    — target_label's 1-indexed position in the ranking
            "target_similarity" — target_label's raw cosine similarity
    """
    pooled = pool_fn(get_activations(text, model, sae)).cpu().numpy()
    vec = pooled[S]
    ranking = rank_by_similarity(vec, holdout_prototypes)
    labels_in_order = [label for label, _ in ranking]
    return {
        "ranking": ranking,
        "predicted": labels_in_order[0],
        "target_rank": labels_in_order.index(target_label) + 1,
        "target_similarity": dict(ranking)[target_label],
    }
