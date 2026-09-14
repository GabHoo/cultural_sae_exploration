"""
Loads CANDLE assertions, splits them into disjoint train/held-out samples per country,
and strips each country's own name out of its assertions so features can't just
pattern-match the literal name token.
"""

import json
import random
import re


def load_candle_groups(path: str, label_key: str) -> dict[str, list[str]]:
    """
    Read a CANDLE JSONL subset file and group assertion text by a chosen label column.
    Duplicate assertion strings within the same group are dropped (the source data has
    some - e.g. Japan has 513 raw rows but only 498 unique strings) so a duplicated
    sentence can never land in both the train and held-out split at once.

    Input:
        path      — path to a CANDLE JSONL file, one JSON object per line
        label_key — which column to group by, e.g. "country"

    Output:
        dict mapping each label (e.g. "Japan") to a list of its unique assertion
        strings, first-occurrence order preserved.
    """
    groups: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            label, assertion = row[label_key], row["assertion"]
            if assertion in seen.setdefault(label, set()):
                continue
            seen[label].add(assertion)
            groups.setdefault(label, []).append(assertion)
    return groups


def merge_groups(*group_dicts: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Concatenate several {label: [text, ...]} dicts into one, e.g. to combine the
    original CANDLE assertions with augment_data.py's LLM-generated ones before
    sampling/splitting.

    Input:
        *group_dicts — any number of dicts with the same label keys

    Output:
        dict mapping each label to the concatenation of its lists across all inputs,
        in the order the dicts were passed.
    """
    merged: dict[str, list[str]] = {}
    for groups in group_dicts:
        for label, texts in groups.items():
            merged.setdefault(label, []).extend(texts)
    return merged


def train_holdout_split(
    groups: dict[str, list[str]],
    n_train: int,
    n_holdout: int,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Per group, draw two DISJOINT random samples: one for training (feature selection +
    training prototypes), one held out for evaluation (see NOTES.md - held-out prototypes
    exist so steering can be scored against real text the steering vector was never built
    from, instead of comparing against its own training data).

    Input:
        groups    — {label: [assertion, ...]}, e.g. from load_candle_groups()
        n_train   — assertions to sample per group for training (capped at what's available)
        n_holdout — assertions to sample per group for held-out eval, from what's left
                    after n_train is removed (capped at what's available)
        seed      — random seed, for reproducible sampling across reruns

    Output:
        (train_groups, holdout_groups) — same shape as `groups`, but every group's list
        in train_groups and holdout_groups is guaranteed non-overlapping.
    """
    rng = random.Random(seed)
    train_groups: dict[str, list[str]] = {}
    holdout_groups: dict[str, list[str]] = {}
    for label, texts in groups.items():
        shuffled = texts[:]
        rng.shuffle(shuffled)
        train_groups[label] = shuffled[:n_train]
        holdout_groups[label] = shuffled[n_train:n_train + n_holdout]
    return train_groups, holdout_groups


def make_stripper(alias_map: dict[str, list[str]]):
    """
    Build a function that removes a label's own name/aliases from its assertion text.

    Input:
        alias_map — {label: [surface forms to strip, e.g. "Japan", "Japanese"]}, e.g. config.COUNTRY_ALIASES

    Output:
        strip(text, label) -> str — a function that removes every match of that label's
        aliases from text (whole-word, case-insensitive, plurals/suffixes included via \\w*)
        and collapses the resulting whitespace.
    """
    patterns = {
        label: re.compile("|".join(rf"\b{re.escape(a)}\w*\b" for a in aliases), re.IGNORECASE)
        for label, aliases in alias_map.items()
    }

    def strip(text: str, label: str) -> str:
        text = patterns[label].sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    return strip


def build_dataset(
    sampled: dict[str, list[str]],
    groups: list[str],
    strip_fn,
) -> tuple[list[str], list[str]]:
    """
    Flatten a {label: [texts]} dict into parallel (texts, labels) lists, with each
    text's own label name stripped out.

    Input:
        sampled  — {label: [assertion, ...]}, e.g. one side of train_holdout_split()'s output
        groups   — labels in a fixed iteration order (e.g. sorted country names)
        strip_fn — from make_stripper(), removes a label's own name from its text

    Output:
        (texts, labels) — parallel lists, same length, same order; labels[i] is the
        group that texts[i] came from.
    """
    texts: list[str] = []
    labels: list[str] = []
    for g in groups:
        for text in sampled[g]:
            texts.append(strip_fn(text, g))
            labels.append(g)
    return texts, labels
