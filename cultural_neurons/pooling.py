"""
Collapses per-token SAE activations into one vector per sentence. Isolated from
forward_pass.py on purpose: this is the piece most likely to change (max-pool today,
maybe mean or top-k-mean later) - add a new strategy here and register it in
POOLING_FUNCS, nothing else in the codebase needs to change.
"""

import torch


def max_pool(acts: torch.Tensor) -> torch.Tensor:
    """
    Per feature, take its single strongest activation anywhere in the sentence.

    This is the step that turns "per-token" into "per-sentence": not "what's big at
    this token" but "what's the peak value each feature ever reaches across the whole
    sentence." A feature can dominate the pooled vector by having one standout moment
    somewhere in the text - it doesn't matter if it's silent everywhere else.

    Input:
        acts — Tensor of shape [seq_len, n_features], e.g. from forward_pass.get_activations()

    Output:
        Tensor of shape [n_features] — this is a(x) from the paper's Sec 2.2.
    """
    return acts.max(dim=0).values


POOLING_FUNCS = {
    "max": max_pool,
}
