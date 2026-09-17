"""
Model-agnostic forward pass: load a (model, SAE) pair from a config preset, and turn
text into per-token SAE feature activations. Nothing in here pools or reduces the
activations - that's pooling.py's job, kept separate so it can be swapped independently.
"""

import numpy as np
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE


def get_device() -> str:
    """
    Pick the best available device: MPS on Apple Silicon, else CUDA, else CPU.

    Output:
        "mps", "cuda", or "cpu"
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(preset: dict, device: str) -> HookedTransformer:
    """
    Load just the base language model from a PRESETS entry - no SAE. Split out from
    load_model_and_sae() so multi-layer setups can load the model once and then load N
    SAEs against it separately (see cultural_neurons/layers.py's load_saes()).

    Input:
        preset — one value from config.PRESETS, e.g. {"model": ..., "release": ..., "hook_template": ...}
        device — "mps" / "cuda" / "cpu", from get_device()

    Output:
        A loaded HookedTransformer.
    """
    return HookedTransformer.from_pretrained(preset["model"], device=device)


def load_model_and_sae(preset: dict, layer: int, device: str) -> tuple[HookedTransformer, SAE]:
    """
    Load the base language model and its matching SAE for one layer, from a PRESETS entry.
    Single-layer convenience wrapper around load_model() - for multi-layer setups, use
    load_model() + cultural_neurons.layers.load_saes() instead.

    Input:
        preset — one value from config.PRESETS, e.g. {"model": ..., "release": ..., "hook_template": ...}
        layer  — which layer to load the SAE for; must be valid for this preset (see preset["valid_layers"])
        device — "mps" / "cuda" / "cpu", from get_device()

    Output:
        (model, sae) — model is a loaded HookedTransformer, sae is a loaded SAE matched to one of
        model's hook points. The actual hook point name is sae.cfg.metadata.hook_name - it does not
        necessarily match preset["hook_template"] (see note in config.py).
    """
    model = load_model(preset, device)
    sae = SAE.from_pretrained(
        release=preset["release"],
        sae_id=preset["hook_template"].format(layer=layer),
        device=device,
    )
    return model, sae


def get_activations(text: str, model: HookedTransformer, sae: SAE) -> torch.Tensor:
    """
    Run one text through the model and decompose its residual stream into SAE features.

    Input:
        text  — a single string, untokenized
        model — a loaded HookedTransformer
        sae   — a loaded SAE, already matched to one of model's hook points

    Output:
        Tensor of shape [seq_len - 1, n_features] — per-token SAE activations, BOS token
        dropped (its activation is content-independent and would dominate every result),
        not yet pooled across tokens.
    """
    tokens = model.to_tokens(text)                              # tokenize (BOS prepended by default)
    _, cache = model.run_with_cache(tokens, prepend_bos=True)   # run the model, keep every internal activation
    acts = sae.encode(cache[sae.cfg.metadata.hook_name])[0]     # decompose into SAE features: [seq, n_features]
    return acts[1:]                                              # drop BOS position


def extract_feature_matrix(
    assertions: list[str],
    model: HookedTransformer,
    sae: SAE,
    pool_fn,
) -> np.ndarray:
    """
    Paper's Step 1: build one pooled feature vector per assertion, stacked into a matrix.

    Input:
        assertions — list of n strings (e.g. CANDLE assertions with country names stripped)
        model, sae — as returned by load_model_and_sae()
        pool_fn    — a function [seq_len, n_features] Tensor -> [n_features] Tensor,
                     e.g. pooling.POOLING_FUNCS["max"]

    Output:
        np.ndarray of shape [n_assertions, n_features], dtype float32 — row i is the
        pooled activation vector for assertions[i].
    """
    X = None   # allocated once n_features is known, from the first assertion
    for i, text in enumerate(tqdm(assertions)):
        vec = pool_fn(get_activations(text, model, sae)).cpu().numpy()
        if X is None:
            X = np.zeros((len(assertions), vec.shape[0]), dtype=np.float32)
        X[i] = vec
    return X
