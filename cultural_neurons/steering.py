"""
Turns per-country activation patterns (within the selected feature set S) into
per-layer steering vectors, and applies them during generation via residual-stream
hooks - one hook per layer that has any selected features (see layers.py and NOTES.md
for why the joint delta has to be split and decoded per layer, not once globally).
"""

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from cultural_neurons import layers as layers_module


def build_prototypes(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    S: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Sec 2.3: CuE(x) = a(x)[S] (restrict activations to the selected feature set), then
    average CuE(x) over all assertions of each group to get that group's prototype
    p_CuE^(c). Same S columns for every group - country-specificity comes from the
    per-group average, not from a different feature set (see NOTES.md).

    Input:
        X      — np.ndarray [n_assertions, n_features], pooled SAE activations
        y      — np.ndarray [n_assertions] of label strings, parallel to X's rows
        groups — list of labels to build a prototype for
        S      — np.ndarray of selected feature indices, from feature_selection.select_top_mi()

    Output:
        dict mapping each label to its prototype, an np.ndarray of shape [len(S)].
    """
    CuE = X[:, S]
    return {g: CuE[y == g].mean(axis=0) for g in groups}


def build_steering_vectors(
    target_label: str,
    groups: list[str],
    S: np.ndarray,
    boundaries: dict[int, tuple[int, int]],
    prototypes: dict[str, np.ndarray],
    saes: dict[int, SAE],
    device: str,
) -> dict[int, torch.Tensor]:
    """
    Sec 2.5, generalized across layers: delta = p_CuE^(target) - mean(p_CuE^(other
    groups)) is computed ONCE, in the joint |S|-dimensional space spanning every layer -
    subtracting the other groups' mean cancels out whatever's generic across all of
    them, leaving only what's distinctive of the target. Decoding is where layers
    separate: each layer's own slice of delta must be decoded through THAT layer's own
    SAE decoder (a layer's W_dec only knows how to decode its own feature space - see
    layers.split_S_by_layer and NOTES.md's "Planned: multi-layer steering" section for
    why this can't be one global decode). Layers with zero selected features are simply
    absent from the output - there's nothing to steer there, so no hook is needed.

    Input:
        target_label — which label to build a steering vector toward, e.g. "Japan"
        groups       — all labels (target_label must be one of them)
        S            — np.ndarray of selected GLOBAL feature indices (joint axis across
                       all layers), from feature_selection.select_top_mi()
        boundaries   — {layer: (start, end)}, from layers.layer_boundaries()
        prototypes   — {label: prototype}, from build_prototypes() - shape [len(S)] each
        saes         — {layer: SAE}, e.g. from layers.load_saes()
        device       — "mps" / "cuda" / "cpu", must match the model/saes' device

    Output:
        {layer: Tensor[d_model]} - one steering vector per layer that has at least one
        selected feature.
    """
    others = [g for g in groups if g != target_label]
    p_target = prototypes[target_label]
    p_others_mean = np.mean([prototypes[g] for g in others], axis=0)
    delta = p_target - p_others_mean   # shape [len(S)], joint space across all layers

    v_cues: dict[int, torch.Tensor] = {}
    for layer, (positions_in_S, local_indices) in layers_module.split_S_by_layer(S, boundaries).items():
        sae = saes[layer]
        n_features_layer = boundaries[layer][1] - boundaries[layer][0]
        delta_full = np.zeros(n_features_layer, dtype=np.float32)
        delta_full[local_indices] = delta[positions_in_S]
        delta_full_t = torch.tensor(delta_full, device=device, dtype=sae.W_dec.dtype)
        v_cues[layer] = delta_full_t @ sae.W_dec
    return v_cues


def hooked_generate(
    model: HookedTransformer,
    saes: dict[int, SAE],
    prompt: str,
    v_cues: dict[int, torch.Tensor],
    alpha: float = 1.0,
    max_new_tokens: int = 40,
    temperature: float = 0.9,
    seed: int = 0,
) -> str:
    """
    Generate text with steering vectors added into the residual stream at every token
    position, at EVERY layer in v_cues simultaneously - one hook per layer, each adding
    only that layer's own v_cue into that layer's own residual (never a global vector
    applied everywhere). alpha=0 (or an empty v_cues) reproduces the unsteered baseline -
    no hooks installed at all, rather than hooks that add zero.

    Input:
        model          — a loaded HookedTransformer
        saes           — {layer: SAE}, used only for each layer's hook_name
        prompt         — the text to continue, ideally culture-agnostic
        v_cues         — {layer: Tensor[d_model]}, e.g. from build_steering_vectors() -
                         a layer absent from this dict gets no hook
        alpha          — steering strength; 0 = unsteered baseline
        max_new_tokens — how many new tokens to sample
        temperature    — sampling temperature
        seed           — generation seed, fixed so unsteered/steered outputs are comparable

    Output:
        The generated string (prompt + continuation), including the BOS token as
        rendered by the tokenizer.
    """
    torch.manual_seed(seed)
    tokens = model.to_tokens(prompt)

    def make_hook(v_cue: torch.Tensor):
        def steering_hook(resid, hook):
            resid[:, :, :] += alpha * v_cue
            return resid
        return steering_hook

    fwd_hooks = (
        [(saes[layer].cfg.metadata.hook_name, make_hook(v_cue)) for layer, v_cue in v_cues.items()]
        if alpha != 0 else []
    )
    with model.hooks(fwd_hooks=fwd_hooks):
        out = model.generate(
            input=tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            stop_at_eos=False,   # avoids a known bug on MPS where generation can hang/crash on EOS
        )
    return model.to_string(out[0])
