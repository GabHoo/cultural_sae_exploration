"""
Multi-layer bookkeeping. Everything else in this codebase (feature_selection.py,
steering.build_prototypes, evaluation.py) operates on ONE joint feature axis - a
feature's real coordinate is (layer, local_feature_idx), but MI/prototypes/evaluation
never need to know that, because per-assertion vectors are pre-concatenated across
layers before reaching them (see concat_vectors). This module owns everything that DOES
need to know about individual layers: which layers actually have a published SAE,
loading them, where each layer's slice lives within the joint axis, and translating a
selected feature set S (global column indices) back into per-layer local indices for
decoding (see steering.py's multi-layer build_steering_vectors).
"""

import re

import numpy as np
from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory


def discover_layers(release: str, hook_template: str) -> list[int]:
    """
    Find every layer a given SAE release actually publishes, by reading SAELens' own
    bundled pretrained_saes.yaml metadata - no network calls, so this can't fail on a
    flaky connection and doesn't need try/except-per-layer probing.

    Input:
        release       — e.g. "gemma-scope-2-1b-pt-res" (preset["release"])
        hook_template — e.g. "layer_{layer}_width_16k_l0_medium" (preset["hook_template"])

    Output:
        Sorted list of ints - every layer with a published sae_id matching hook_template
        for this release. Raises ValueError if the release is unknown to SAELens, or if
        no sae_id matches the template (e.g. a typo'd hook_template).
    """
    directory = get_pretrained_saes_directory()
    if release not in directory:
        raise ValueError(f"Unknown SAE release '{release}' - not in SAELens' pretrained_saes.yaml")
    pattern = re.compile("^" + re.escape(hook_template).replace(re.escape("{layer}"), r"(\d+)") + "$")
    layers = sorted({
        int(m.group(1))
        for sae_id in directory[release].saes_map
        if (m := pattern.match(sae_id))
    })
    if not layers:
        raise ValueError(f"No SAEs found for release '{release}' matching hook template '{hook_template}'")
    return layers


def load_saes(preset: dict, layers: list[int], device: str) -> dict[int, SAE]:
    """
    Load one SAE per layer.

    Input:
        preset — one value from config.PRESETS
        layers — which layers to load, e.g. from discover_layers()
        device — "mps" / "cuda" / "cpu"

    Output:
        {layer: SAE}, one entry per requested layer.
    """
    return {
        layer: SAE.from_pretrained(
            release=preset["release"],
            sae_id=preset["hook_template"].format(layer=layer),
            device=device,
        )
        for layer in sorted(layers)
    }


def layer_boundaries(saes: dict[int, SAE]) -> dict[int, tuple[int, int]]:
    """
    Fixed column ranges for each layer within the joint concatenated feature axis, in
    ascending layer order. This ordering is the single source of truth for how per-layer
    vectors get concatenated (concat_vectors) and how a global feature index maps back
    to (layer, local_idx) (split_S_by_layer) - both must agree with this, and do, by
    construction (they're the only two places that read this dict).

    Input:
        saes — {layer: SAE}, e.g. from load_saes()

    Output:
        {layer: (start, end)} - layer's slice within the joint axis is [start, end).
        n_features_total = end of the last layer's range.
    """
    boundaries = {}
    offset = 0
    for layer in sorted(saes.keys()):
        width = saes[layer].W_dec.shape[0]   # d_sae - see steering.py's existing single-layer decode for this convention
        boundaries[layer] = (offset, offset + width)
        offset += width
    return boundaries


def concat_vectors(vectors_by_layer: dict[int, np.ndarray], boundaries: dict[int, tuple[int, int]]) -> np.ndarray:
    """
    Concatenate one assertion's per-layer pooled vectors into a single joint vector,
    in the same layer order layer_boundaries() used to define the axis.

    Input:
        vectors_by_layer — {layer: np.ndarray[n_features_layer]}, one vector per layer
                           (e.g. one text's pooled activations at each layer)
        boundaries        — from layer_boundaries()

    Output:
        np.ndarray[n_features_total] - the joint vector.
    """
    return np.concatenate([vectors_by_layer[layer] for layer in sorted(boundaries.keys())])


def split_S_by_layer(
    S: np.ndarray, boundaries: dict[int, tuple[int, int]]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Translate a selected feature set S (global column indices into the joint axis) back
    into per-layer pieces, for decoding a steering delta through each layer's own W_dec.

    Input:
        S          — np.ndarray of global column indices, from feature_selection.select_top_mi()
        boundaries — from layer_boundaries()

    Output:
        {layer: (positions_in_S, local_feature_indices)} - only for layers that actually
        have at least one S feature (a layer with none is omitted entirely, so callers
        can skip installing a steering hook there - see steering.py).
        positions_in_S indexes into the |S|-length delta vector; local_feature_indices
        is where those same features live within that layer's own d_sae-width space.
    """
    result = {}
    for layer, (start, end) in boundaries.items():
        mask = (S >= start) & (S < end)
        positions_in_S = np.where(mask)[0]
        if len(positions_in_S) > 0:
            result[layer] = (positions_in_S, S[mask] - start)
    return result
