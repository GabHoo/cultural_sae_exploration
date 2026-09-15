"""
Caches per-assertion SAE feature vectors on disk, keyed by (country, raw assertion
text). The forward pass is by far the slowest step in main.py, but train/holdout
fractions and which countries get used only change the *sampling* - the underlying
assertion text and its embedding never change. So we embed each assertion once, cache
it, and every later run (different fractions, different config.COUNTRIES subset) just
resamples from what's already on disk instead of re-running the model.

The cache is invalidated automatically (silently recomputed from scratch) if the model
preset, layer, pooling strategy, or country-alias stripping rules change, since any of
those would make the stored vectors wrong.
"""

import hashlib
import json
import os

import numpy as np

from cultural_neurons import forward_pass


def cache_meta(model_name: str, sae_release: str, layer: int, pooling: str, alias_map: dict) -> dict:
    """
    Build the metadata fingerprint a cache file is validated against. Fingerprints the
    actual HF checkpoint/release strings (not config.PRESET_NAME) - a preset's "model"
    field can change (e.g. swapping -pt for -it) while keeping the same preset key, and
    that must invalidate old caches even though PRESET_NAME didn't change.

    Input:
        model_name  — preset["model"], the HookedTransformer checkpoint actually loaded
        sae_release — preset["release"], the SAE release actually loaded
        layer        — config.LAYER
        pooling      — config.POOLING_STRATEGY
        alias_map    — config.COUNTRY_ALIASES (affects what text actually gets embedded)

    Output:
        dict, stable/hashable, safe to compare with ==
    """
    alias_hash = hashlib.sha256(json.dumps(alias_map, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "model": model_name, "release": sae_release,
        "layer": layer, "pooling": pooling, "alias_hash": alias_hash,
    }


def _load(path: str, meta: dict) -> dict[tuple[str, str], np.ndarray]:
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=False)
    if json.loads(str(data["meta"])) != meta:
        return {}   # stale - model/layer/pooling/aliases changed since this was written
    keys = data["keys"]
    vectors = data["vectors"]
    return {tuple(k.split("\x1f", 1)): v for k, v in zip(keys, vectors)}


def _save(path: str, cache: dict[tuple[str, str], np.ndarray], meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = np.array([f"{country}\x1f{text}" for country, text in cache.keys()])
    vectors = np.stack(list(cache.values()))
    np.savez_compressed(path, keys=keys, vectors=vectors, meta=json.dumps(meta))


def get_feature_vectors(
    groups: dict[str, list[str]],
    model,
    sae,
    pool_fn,
    strip_fn,
    cache_path: str,
    meta: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Return one pooled SAE feature vector per assertion in `groups`, filling cache misses
    with a real forward pass (country name stripped first, same as always) and persisting
    the updated cache to disk before returning.

    Input:
        groups     — {country: [raw_assertion, ...]}, e.g. from load_candle_groups()
        model, sae — from forward_pass.load_model_and_sae()
        pool_fn    — from pooling.POOLING_FUNCS
        strip_fn   — from data_prep.make_stripper()
        cache_path — .npz file to read/write (created if missing)
        meta       — from cache_meta() - identifies what this cache is valid for

    Output:
        {country: {raw_assertion: feature_vector}} covering every assertion in `groups`,
        vectors drawn from the cache where possible and freshly computed otherwise.
    """
    cache = _load(cache_path, meta)

    missing: list[tuple[str, str]] = [
        (country, raw)
        for country, raws in groups.items()
        for raw in raws
        if (country, raw) not in cache
    ]
    if missing:
        print(f"activation cache [{cache_path}]: {len(missing)} new assertions to embed "
              f"({len(cache)} already cached)")
        stripped = [strip_fn(raw, country) for country, raw in missing]
        vectors = forward_pass.extract_feature_matrix(stripped, model, sae, pool_fn)
        for (country, raw), vec in zip(missing, vectors):
            cache[(country, raw)] = vec
        _save(cache_path, cache, meta)
    else:
        print(f"activation cache [{cache_path}]: full hit, no forward pass needed ({len(cache)} cached)")

    out: dict[str, dict[str, np.ndarray]] = {country: {} for country in groups}
    for (country, raw), vec in cache.items():
        if country in out:
            out[country][raw] = vec
    return out
