# Cultural Neurons

Reproduction of Steps 1–2–3b of *Steering LLMs for Culturally Localized Generation*
(Khanuja et al. — the "CuE" paper), using SAE features to find and steer toward
country-specific representations inside a language model.

Grounded in the real **CANDLE** dataset (Nguyen et al., 2023), restricted to the
`rituals` and `traditions` facets.

See [NOTES.md](NOTES.md) for detailed write-ups of the pipeline's internals
(pooling logic, the mutual-information formula, how a shared feature basis becomes
a per-country steering vector, open caveats).

## Pipeline

1. **Data prep** (`cultural_neurons/data_prep.py`) — sample country-labeled CANDLE assertions
   into disjoint train/held-out sets, strip each country's own name out of its text.
2. **Forward pass** (`cultural_neurons/forward_pass.py`) — model-agnostic: load a (model, SAE)
   pair from a config preset, run each assertion through it, decompose the residual stream
   into SAE features.
3. **Pooling** (`cultural_neurons/pooling.py`) — collapse each assertion's per-token activations
   into one per-sentence feature vector. Max-pool today, swappable.
4. **Feature selection** (`cultural_neurons/feature_selection.py`) — rank features by mutual
   information with country, keep the top ones (a shared set `S`, not one per country).
5. **Prototypes + steering vector** (`cultural_neurons/steering.py`) — average per-country
   activations within `S`, build a contrastive direction, decode back into the model's
   residual-stream space, generate with it hooked in.
6. **Evaluation** (`cultural_neurons/evaluation.py`) — check whether a steered generation lands
   closer to the target country's held-out prototype (built from CANDLE assertions the
   steering vector never saw) than to any other country's. A quick, LLM-free proxy metric —
   not a substitute for the paper's LLM-as-judge eval, see NOTES.md for the tradeoff.

## Setup

Dependencies are pinned in `pyproject.toml`. With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync
```

This creates/updates `.venv` to match `pyproject.toml` exactly.

## Usage

```bash
uv run main.py
```

Runs the full pipeline end-to-end: loads the model/SAE, extracts features, selects
culturally-informative ones, builds a steering vector, generates an unsteered and a
steered continuation, and prints a held-out evaluation for both.

`sandbox.ipynb` remains available for ad hoc exploration outside the scripted pipeline.

## Config

All tunable options — model/SAE preset, layer, data paths, sample sizes, MI threshold,
target country, steering strength (alpha), generation params — live in `config.py`. Edit
that file to change pipeline behavior; nothing else should need editing for routine
experiments.

## Data

- `data/candle_countries_subset.jsonl` — CANDLE assertions labeled by country (used).
- `data/candle_religions_subset.jsonl` — CANDLE assertions labeled by religion (currently
  unused — see NOTES.md for why the religion pipeline is disabled).
- `data/WorldView-Bench Dataset.csv` — used in an earlier exploration (`sandbox.ipynb`),
  not part of `main.py`'s pipeline.
