"""
Caches per-assertion SAE feature vectors on disk, keyed by (country, raw assertion
text). The forward pass is by far the slowest step in main.py, but train/holdout
fractions and which countries get used only change the *sampling* - the underlying
assertion text and its embedding never change. So we embed each assertion once, cache
it, and every later run (different fractions, different config.COUNTRIES subset) just
resamples from what's already on disk instead of re-running the model.

Each (model, SAE release, layer, pooling, alias-rules) combination gets its OWN cache
file, named after that config - so switching between presets (e.g. -pt vs -it, or
gemma-3-1b vs gemma-2-9b) never overwrites another config's cache; both stay on disk
and switching back is instant. The stored metadata is also checked on load as a safety
net against filename-sanitization collisions, but the filename is what actually keeps
different configs apart.
"""

import hashlib
import json
import os
import re

import numpy as np
from sae_lens import SAE
from tqdm.auto import tqdm

from cultural_neurons import forward_pass


def cache_meta(model_name: str, sae_release: str, layer: int, pooling: str, alias_map: dict) -> dict:
    """
    Build the metadata fingerprint a cache file is both named after and validated
    against. Fingerprints the actual HF checkpoint/release strings (not
    config.PRESET_NAME) - a preset's "model" field can change (e.g. swapping -pt for
    -it) while keeping the same preset key, and that must still get its own cache file.

    Input:
        model_name  — preset["model"], the HookedTransformer checkpoint actually loaded
        sae_release — preset["release"], the SAE release actually loaded
        layer        — one entry of the resolved config.LAYERS (a single layer's cache is
                       always per-layer, even in multi-layer mode - see main.py)
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


def cache_path(cache_dir: str, source_name: str, meta: dict) -> str:
    """
    Build the on-disk cache filename for one data source under one config - distinct
    configs never share a file, so caches for different models/SAEs/layers coexist.

    Input:
        cache_dir   — directory to hold cache files (created on write if missing)
        source_name — short tag for the data source, e.g. "candle_countries"
        meta        — from cache_meta()

    Output:
        e.g. "data/cache/candle_countries__google-gemma-3-1b-it__gemma-scope-2-1b-pt-res__L17__max.npz"
    """
    sanitize = lambda s: re.sub(r"[^A-Za-z0-9_.-]+", "-", s)
    filename = (
        f"{source_name}__{sanitize(meta['model'])}__{sanitize(meta['release'])}"
        f"__L{meta['layer']}__{meta['pooling']}__{meta['alias_hash']}.npz"
    )
    return os.path.join(cache_dir, filename)


def _load(path: str, meta: dict) -> dict[tuple[str, str], np.ndarray]:
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=False)
    if json.loads(str(data["meta"])) != meta:
        return {}   # stale - filename collision after sanitization (should be rare)
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
    cache_dir: str,
    source_name: str,
    meta: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Return one pooled SAE feature vector per assertion in `groups`, filling cache misses
    with a real forward pass (country name stripped first, same as always) and persisting
    the updated cache to disk before returning.

    Input:
        groups      — {country: [raw_assertion, ...]}, e.g. from load_candle_groups()
        model, sae  — from forward_pass.load_model_and_sae()
        pool_fn     — from pooling.POOLING_FUNCS
        strip_fn    — from data_prep.make_stripper()
        cache_dir   — directory holding cache files (created if missing)
        source_name — short tag for this data source, e.g. "candle_countries" - see cache_path()
        meta        — from cache_meta() - identifies what this cache is valid for, and
                      which file (this exact config gets its own, via cache_path())

    Output:
        {country: {raw_assertion: feature_vector}} covering every assertion in `groups`,
        vectors drawn from the cache where possible and freshly computed otherwise.
    """
    path = cache_path(cache_dir, source_name, meta)
    cache = _load(path, meta)

    missing: list[tuple[str, str]] = [
        (country, raw)
        for country, raws in groups.items()
        for raw in raws
        if (country, raw) not in cache
    ]
    if missing:
        print(f"activation cache [{path}]: {len(missing)} new assertions to embed "
              f"({len(cache)} already cached)")
        stripped = [strip_fn(raw, country) for country, raw in missing]
        vectors = forward_pass.extract_feature_matrix(stripped, model, sae, pool_fn)
        for (country, raw), vec in zip(missing, vectors):
            cache[(country, raw)] = vec
        _save(path, cache, meta)
    else:
        print(f"activation cache [{path}]: full hit, no forward pass needed ({len(cache)} cached)")

    out: dict[str, dict[str, np.ndarray]] = {country: {} for country in groups}
    for (country, raw), vec in cache.items():
        if country in out:
            out[country][raw] = vec
    return out


def get_feature_vectors_multi_layer(
    groups: dict[str, list[str]],
    model,
    saes: dict[int, SAE],
    pool_fn,
    strip_fn,
    cache_dir: str,
    source_name: str,
    meta_by_layer: dict[int, dict],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """
    Multi-layer version of get_feature_vectors(): one pooled feature vector per
    assertion, PER LAYER in `saes`, each layer cached in its own file (via cache_meta/
    cache_path per layer, exactly as the single-layer path does). The one thing this
    does differently from just calling get_feature_vectors() once per layer: the
    expensive part of a forward pass is the transformer itself, which is shared across
    every layer's SAE - so for any assertion missing at least one layer, this runs ONE
    model.run_with_cache (fetching only the still-missing layers' hook points) and
    encodes through each missing layer's SAE from that single cache, instead of
    re-running the full transformer forward pass once per layer.

    Input:
        groups        — {country: [raw_assertion, ...]}, e.g. from load_candle_groups()
        model         — from forward_pass.load_model_and_sae() (or any HookedTransformer)
        saes          — {layer: SAE}, e.g. from layers.load_saes()
        pool_fn       — from pooling.POOLING_FUNCS
        strip_fn      — from data_prep.make_stripper()
        cache_dir     — directory holding cache files (created if missing)
        source_name   — short tag for this data source, e.g. "candle_countries"
        meta_by_layer — {layer: cache_meta(...)}, one fingerprint per layer

    Output:
        {layer: {country: {raw_assertion: feature_vector}}} covering every assertion in
        `groups`, for every layer in `saes`.
    """
    paths = {layer: cache_path(cache_dir, source_name, meta_by_layer[layer]) for layer in saes}
    caches = {layer: _load(paths[layer], meta_by_layer[layer]) for layer in saes}

    missing_layers_by_text: dict[tuple[str, str], list[int]] = {}
    for country, raws in groups.items():
        for raw in raws:
            missing = [layer for layer in saes if (country, raw) not in caches[layer]]
            if missing:
                missing_layers_by_text[(country, raw)] = missing

    if missing_layers_by_text:
        print(f"activation cache (multi-layer) [{source_name}]: {len(missing_layers_by_text)} assertions "
              f"need at least one of {sorted(saes.keys())} layers embedded")
        for (country, raw), missing in tqdm(missing_layers_by_text.items()):
            stripped = strip_fn(raw, country)
            tokens = model.to_tokens(stripped)
            needed_hook_names = {saes[layer].cfg.metadata.hook_name for layer in missing}
            _, cache = model.run_with_cache(tokens, prepend_bos=True, names_filter=lambda n: n in needed_hook_names)
            for layer in missing:
                acts = saes[layer].encode(cache[saes[layer].cfg.metadata.hook_name])[0][1:]   # drop BOS
                caches[layer][(country, raw)] = pool_fn(acts).cpu().numpy()
        for layer in saes:
            _save(paths[layer], caches[layer], meta_by_layer[layer])
    else:
        print(f"activation cache (multi-layer) [{source_name}]: full hit across all {len(saes)} layers, "
              f"no forward pass needed")

    out: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for layer in saes:
        layer_out: dict[str, dict[str, np.ndarray]] = {country: {} for country in groups}
        for (country, raw), vec in caches[layer].items():
            if country in layer_out:
                layer_out[country][raw] = vec
        out[layer] = layer_out
    return out
