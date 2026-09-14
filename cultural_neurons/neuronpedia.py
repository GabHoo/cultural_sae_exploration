"""
Optional interpretability helper: looks up a human-readable description of an SAE
feature on Neuronpedia. Not on the main pipeline's critical path - only needed if you
want to inspect what a selected feature actually represents.
"""

from sae_lens import SAE
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_feature


def parse_neuronpedia_ids(sae: SAE) -> tuple[str, str, str]:
    """
    Read the model/layer/dataset identifiers Neuronpedia uses for this SAE's features,
    from the SAE's own metadata (not from config.py) - stays correct automatically no
    matter which preset/layer is loaded.

    Input:
        sae — a loaded SAE

    Output:
        (model_id, layer_str, dataset) — e.g. ("gpt2-small", "6", "res-jb").

    Note: assumes the release has a Neuronpedia mapping in this "model/layer-dataset"
    format (true for gpt2-small-res-jb; not guaranteed for every release/layer - thin
    Gemma 3 releases may not have one, in which case explain() below will just fail to
    find anything).
    """
    model_id, source_id = sae.cfg.metadata.neuronpedia_id.split("/")
    layer_str, dataset = source_id.split("-", 1)
    return model_id, layer_str, dataset


def make_explainer(sae: SAE):
    """
    Build a cached feature-explanation lookup function for one SAE.

    Input:
        sae — a loaded SAE

    Output:
        explain(feature_idx: int) -> list[str] — human-readable description(s) of that
        feature from Neuronpedia, cached so each feature is only fetched once.
    """
    model_id, layer_str, dataset = parse_neuronpedia_ids(sae)
    cache: dict[int, list[str]] = {}

    def explain(feature_idx: int) -> list[str]:
        if feature_idx not in cache:
            data = get_neuronpedia_feature(feature=feature_idx, layer=int(layer_str), model=model_id, dataset=dataset)
            cache[feature_idx] = [e["description"] for e in data["explanations"]]
        return cache[feature_idx]

    return explain
