"""
Ranks SAE features by mutual information with the country label, and picks a shared
feature subset S. See NOTES.md for the full formula walkthrough and the "is this
per-country or per-feature" discussion. Isolated from everything else so this method
can be swapped for something simpler without touching data prep, forward pass, or
steering.
"""

import numpy as np


def compute_mi(X: np.ndarray, y: np.ndarray, groups: list[str]) -> np.ndarray:
    """
    I(A_j; C) = sum_{a_j,c} P(a_j,c) * log( P(a_j,c) / (P(a_j)*P(c)) ), one value per
    feature, already summed across all groups (countries) - see NOTES.md for why this
    is NOT per-country. Activations are discretized to binary: "fired" (>0) vs "silent".

    Input:
        X      — np.ndarray [n_assertions, n_features], pooled SAE activations (e.g. from
                 forward_pass.extract_feature_matrix())
        y      — np.ndarray [n_assertions] of label strings, parallel to X's rows
        groups — list of all possible labels (e.g. the 22 country names)

    Output:
        np.ndarray [n_features] — mutual information between each feature and the
        label. Higher = more informative about which group an assertion belongs to.
    """
    active = (X > 0)                                          # binary: did feature j fire on assertion i?
    label_idx = {g: i for i, g in enumerate(groups)}
    y_idx = np.array([label_idx[l] for l in y])
    n_groups = len(groups)
    p_g = np.array([(y_idx == i).mean() for i in range(n_groups)])   # P(c)

    mi = np.zeros(active.shape[1], dtype=np.float64)
    eps = 1e-12
    p_active = active.mean(axis=0)          # P(a_j = 1)
    p_inactive = 1 - p_active               # P(a_j = 0)

    for i in range(n_groups):                            # 22 iterations, each vectorized over all features
        mask = (y_idx == i)
        p_active_given_g = active[mask].mean(axis=0)       # P(a_j=1 | c)
        joint1 = p_active_given_g * p_g[i]                  # P(a_j=1, c)
        mi += joint1 * np.log((joint1 + eps) / (p_active * p_g[i] + eps))

        p_inactive_given_g = 1 - p_active_given_g           # P(a_j=0 | c)
        joint0 = p_inactive_given_g * p_g[i]                # P(a_j=0, c)
        mi += joint0 * np.log((joint0 + eps) / (p_inactive * p_g[i] + eps))
    return mi


def select_top_mi(mi: np.ndarray, rho: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank all features globally by MI, keep the smallest top-MI prefix whose cumulative
    MI reaches `rho` fraction of the total (App. C default). One single global selection
    pass - not run once per country (see NOTES.md).

    Input:
        mi  — np.ndarray [n_features], from compute_mi()
        rho — target fraction of total MI mass to capture, e.g. 0.1 = 10%

    Output:
        (S, order) —
            S:     np.ndarray of selected feature indices (the paper's S), sorted by
                   MI descending.
            order: np.ndarray of ALL feature indices sorted by MI descending
                   (S == order[:len(S)]; handy for inspecting the full ranking).
    """
    order = np.argsort(-mi)
    cum = np.cumsum(mi[order])
    cutoff = int(np.searchsorted(cum, rho * cum[-1])) + 1
    return order[:cutoff], order
