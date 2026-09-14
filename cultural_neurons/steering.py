"""
Turns per-country activation patterns (within the selected feature set S) into a
single steering vector, and applies it during generation via a residual-stream hook.
"""

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


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


def build_steering_vector(
    target_label: str,
    groups: list[str],
    n_features: int,
    S: np.ndarray,
    prototypes: dict[str, np.ndarray],
    sae: SAE,
    device: str,
) -> torch.Tensor:
    """
    Sec 2.5: delta = p_CuE^(target) - mean(p_CuE^(other groups)), restricted to S, then
    decoded out of SAE-feature space into the model's residual-stream space via the
    SAE's decoder. Subtracting the other groups' mean cancels out whatever's generic
    across all of them, leaving only what's distinctive of the target.

    Input:
        target_label — which label to build a steering vector toward, e.g. "Japan"
        groups       — all labels (target_label must be one of them)
        n_features   — total SAE feature count (X.shape[1] from wherever prototypes came from)
        S            — np.ndarray of selected feature indices, matching prototypes' shape
        prototypes   — {label: prototype}, from build_prototypes()
        sae          — the SAE whose decoder (W_dec) projects feature space -> residual space
        device       — "mps" / "cuda" / "cpu", must match the model/sae's device

    Output:
        Tensor of shape [d_model] — the steering vector v_cue, ready to add into the
        residual stream (see steering_hook()).
    """
    others = [g for g in groups if g != target_label]
    p_target = prototypes[target_label]
    p_others_mean = np.mean([prototypes[g] for g in others], axis=0)
    delta = p_target - p_others_mean

    delta_full = np.zeros(n_features, dtype=np.float32)
    delta_full[S] = delta

    delta_full_t = torch.tensor(delta_full, device=device, dtype=sae.W_dec.dtype)
    return delta_full_t @ sae.W_dec


def hooked_generate(
    model: HookedTransformer,
    sae: SAE,
    prompt: str,
    v_cue: torch.Tensor,
    alpha: float = 1.0,
    max_new_tokens: int = 40,
    temperature: float = 0.9,
    seed: int = 0,
) -> str:
    """
    Generate text with an optional steering vector added into the residual stream at
    every token position, every generation step. alpha=0 reproduces the unsteered
    baseline (the hook is skipped entirely rather than added as a no-op).

    Input:
        model, sae     — as returned by forward_pass.load_model_and_sae()
        prompt         — the text to continue, ideally culture-agnostic
        v_cue          — steering vector, e.g. from build_steering_vector()
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

    def steering_hook(resid, hook):
        resid[:, :, :] += alpha * v_cue
        return resid

    fwd_hooks = [(sae.cfg.metadata.hook_name, steering_hook)] if alpha != 0 else []
    with model.hooks(fwd_hooks=fwd_hooks):
        out = model.generate(
            input=tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            stop_at_eos=False,   # avoids a known bug on MPS where generation can hang/crash on EOS
        )
    return model.to_string(out[0])
