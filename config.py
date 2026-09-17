"""
All tunable options for the pipeline, in one place. Edit this file to change
model/layer/data/steering behavior — nothing else in the codebase should need editing.
"""

# ============================================================
# Model / SAE preset
# ============================================================
# HOOK_TEMPLATE is NOT the model's internal hook point name in general - it's
# the template for the `sae_id` string that SAE.from_pretrained() uses to find
# the right file within that release's repo. For gpt2-small-res-jb it happens to
# match TransformerLens's real hook name, but other releases use unrelated naming
# (path-like, or terse codes) - the actual hook point is read back afterward from
# sae.cfg.metadata.hook_name, so nothing downstream needs to know which scheme is used.
#
# Gemma 3 SAEs are released as "gemma-scope-2-*" (the "2" = 2nd-gen GemmaScope suite,
# applied to Gemma 3 - a confusing name, NOT "for Gemma 2"). Unlike gpt2-small/Gemma 2
# "canonical" releases (one SAE per layer, every layer), Gemma 3 SAEs only exist at a
# handful of layers per model size. The "valid_layers" strings below are a human-readable
# hint ONLY, not authoritative (one was already found stale - see layers.py) - the real
# list is discovered live at runtime via layers.discover_layers(), which reads SAELens'
# own pretrained_saes.yaml metadata (no network calls, no guessing).
#
# "model" vs "release", and why Gemma entries point "model" at an -it checkpoint: the SAE
# (`release`) is always the one trained on the BASE/pretrained checkpoint - that's the only
# thing GemmaScope publishes. But `model`, the checkpoint actually loaded for generation
# (and for encoding CANDLE assertions), is the INSTRUCTION-TUNED variant. This matches the
# CuE paper itself (Khanuja et al. 2026, arXiv:2603.23301, Sec 3 "Models"): "For Gemma
# models, we use the instruction-tuned variants to support open-ended generation; SAEs
# trained on the base models have been shown to transfer effectively to these variants." A
# base/-pt model given an instruction-style prompt ("Write me a recipe...") just continues
# the text statistically rather than complying with it - that mismatch, not model size, is
# the biggest single cause of low-quality generations. Confirmed for Gemma-2-2B/9B in the
# paper; extended here to Gemma-3 by the same architectural logic (not paper-verified for
# Gemma-3, but the base->it transfer claim is stated as a property of Gemma models generally,
# not a 2-9B-specific fluke). gemma-1-2b is left on -pt: it predates GemmaScope entirely and
# uses an unrelated community SAE (Joseph Bloom's gemma-2b-res-jb), with no transfer evidence
# either way. llama-3.1-8b is also left as-is: the paper's "For Gemma models..." sentence is
# scoped to Gemma only, and Llama Scope's SAEs are themselves trained on the base checkpoint
# with no stated -Instruct transfer claim.
PRESETS = {
    "gpt2-small":   dict(model="gpt2-small",              release="gpt2-small-res-jb",                hook_template="blocks.{layer}.hook_resid_pre",     valid_layers="any of 0-11"),
    "gemma-1-2b":   dict(model="gemma-2b",                release="gemma-2b-res-jb",                  hook_template="blocks.{layer}.hook_resid_post",    valid_layers="any of 0-17"),
    "gemma-2-2b":   dict(model="gemma-2-2b-it",           release="gemma-scope-2b-pt-res-canonical",  hook_template="layer_{layer}/width_16k/canonical", valid_layers="any of 0-25"),
    "gemma-2-9b":   dict(model="gemma-2-9b-it",           release="gemma-scope-9b-pt-res-canonical",  hook_template="layer_{layer}/width_16k/canonical", valid_layers="any of 0-41 (model used in the CuE paper)"),
    "gemma-3-1b":   dict(model="google/gemma-3-1b-it",    release="gemma-scope-2-1b-pt-res",          hook_template="layer_{layer}_width_16k_l0_medium", valid_layers="7, 13, 17, or 22 (see layers.discover_layers - this string previously missed layer 7)"),
    "gemma-3-4b":   dict(model="google/gemma-3-4b-it",    release="gemma-scope-2-4b-pt-res",          hook_template="layer_{layer}_width_16k_l0_medium", valid_layers="9, 17, 22, or 29 only"),
    "gemma-3-12b":  dict(model="google/gemma-3-12b-it",   release="gemma-scope-2-12b-pt-res",         hook_template="layer_{layer}_width_16k_l0_medium", valid_layers="24, 31, or 41 only"),
    "gemma-3-27b":  dict(model="google/gemma-3-27b-it",   release="gemma-scope-2-27b-pt-res",         hook_template="layer_{layer}_width_16k_l0_medium", valid_layers="31, 40, or 53 only"),
    "llama-3.1-8b": dict(model="meta-llama/Llama-3.1-8B", release="llama_scope_lxr_32x",              hook_template="l{layer}r_32x",                     valid_layers="any of 0-31 (note: no width/sparsity choice in this release)"),
}

PRESET_NAME = "gemma-3-1b"   # <-- pick one of the keys in PRESETS above
LAYERS = "all"                # "all" = every layer layers.discover_layers() finds published for this
                               # preset's SAE release (see cultural_neurons/layers.py) - or an explicit
                               # list[int], e.g. [13, 17], to use only specific layers. Steering itself
                               # (paper Sec 3/App. C) is applied at every layer that ends up with any
                               # selected feature in S - not necessarily every layer loaded here.

# ============================================================
# Data
# ============================================================
COUNTRIES_PATH = "data/candle_countries_subset.jsonl"
RELIGIONS_PATH = "data/candle_religions_subset.jsonl"   # currently unused - religion pipeline is disabled (see NOTES.md)
WORLDVIEW_BENCH_PATH = "data/WorldView-Bench Dataset.csv"

COUNTRIES = None  # e.g. ["Japan", "Mexico", "Canada"] to restrict the whole pipeline (MI, prototypes,
                    # sanity eval) to just those countries - faster iteration. None = use all countries
                    # found in COUNTRIES_PATH.

TRAIN_FRACTION = 0.67     # fraction of each country's assertions used for feature selection / training prototypes
HELDOUT_FRACTION = 0.33   # fraction of each country's assertions used for held-out evaluation, drawn from
                           # what's left after TRAIN_FRACTION is removed (disjoint from train either way)
RANDOM_SEED = 0            # fixed seed so train/held-out sampling is identical across reruns

# CANDLE domain="countries", facet in {rituals, traditions} only, restricted (in the source file) to
# the 22 countries with >=500 total assertions per the paper's Sec 2 selection.

# Paper removes explicit label names via an LLM rewrite before feature discovery (Sec 2.1, App. A) so
# features don't just learn "detect the literal name token." We approximate with a regex strip of each
# label's name + common adjective/demonym forms - cruder than an LLM rewrite, but dependency-free.
COUNTRY_ALIASES = {                       # country label -> surface forms to strip out of its own assertions
    "Australia": ["Australia", "Australian"],
    "Canada": ["Canada", "Canadian"],
    "China": ["China", "Chinese"],
    "Egypt": ["Egypt", "Egyptian"],
    "England": ["England", "English"],
    "France": ["France", "French"],
    "Germany": ["Germany", "German"],
    "Greece": ["Greece", "Greek"],
    "India": ["India", "Indian"],
    "Ireland": ["Ireland", "Irish"],
    "Israel": ["Israel", "Israeli"],
    "Italy": ["Italy", "Italian"],
    "Japan": ["Japan", "Japanese"],
    "Mexico": ["Mexico", "Mexican"],
    "Russia": ["Russia", "Russian"],
    "Scotland": ["Scotland", "Scottish", "Scots"],
    "South Korea": ["South Korea", "Korean", "Korea"],
    "Spain": ["Spain", "Spanish", "Spaniard"],
    "Thailand": ["Thailand", "Thai"],
    "Turkey": ["Turkey", "Turkish", "Turk"],
    "United Kingdom": ["United Kingdom", "UK", "British", "Britain"],
    "United States": ["United States", "USA", "U.S.", "US", "American"],
}

# ============================================================
# Data augmentation (paper Appendix A - LLM-generated assertions via GPT-4.1;
# here generated via the Claude API instead - see augment_data.py)
# ============================================================
USE_AUGMENTED = True                 # merge AUGMENTED_COUNTRIES_PATH into the CANDLE data before sampling
AUGMENTED_COUNTRIES_PATH = "data/candle_countries_augmented.jsonl"
AUGMENTATION_MODEL = "claude-opus-5"
AUGMENTATION_NUM_ASSERTIONS = 100    # new assertions to generate per country
AUGMENTATION_IN_CONTEXT_N = 100      # existing CANDLE assertions shown in-context, to avoid duplication
AUGMENTATION_FACET = None            # optional: "food" / "drink" / "traditions" / "customs" / "daily life" / "values" / "social norms" / "celebrations"
AUGMENTATION_VERIFY = False          # if True, run a second-pass faithfulness check - see the caveat in augmentation.py (not a truly independent judge, single provider)
AUGMENTATION_VERIFY_MODEL = "claude-sonnet-5"

# ============================================================
# Activation cache (see cultural_neurons/activation_cache.py)
# ============================================================
# Per-assertion pooled SAE feature vectors, keyed by (country, raw assertion text),
# covering ALL assertions in a source (not just a sampled subset), so changing
# TRAIN_FRACTION / HELDOUT_FRACTION / COUNTRIES never needs a new forward pass - only
# new data does. Each (model, SAE release, layer, pooling, aliases) config gets its own
# file under CACHE_DIR (see activation_cache.cache_path()) - ONE file per layer, even in
# multi-layer mode - so switching PRESET_NAME/LAYERS back and forth, or adding/removing
# a layer, reuses whichever per-layer caches already exist instead of overwriting them.
CACHE_DIR = "data/cache"
CANDLE_CACHE_NAME = "candle_countries"
AUGMENTED_CACHE_NAME = "candle_countries_augmented"

# ============================================================
# Pooling
# ============================================================
POOLING_STRATEGY = "max"   # key into pooling.POOLING_FUNCS - swap to try a different per-sentence reduction

# ============================================================
# Feature selection (mutual information)
# ============================================================
MI_RHO = 0.1   # keep the smallest top-MI feature prefix whose cumulative MI reaches this fraction of the total (App. C default)

# ============================================================
# Steering
# ============================================================
ALPHA = 1.0                 # steering strength; alpha=0 reproduces the unsteered baseline. The paper
                             # (App. C) sweeps alpha in {0.25, 0.5, 1, 2} per country and DISCARDS any
                             # value that produces low-fluency generations - it never uses one fixed
                             # value for every country. We don't have that fluency-filtered sweep here
                             # yet; pinned to 1.0 deliberately for now (manual judgment call on whether
                             # to raise it, once multi-layer steering is in) rather than auto-tuned.
EVAL_PROMPTS_PATH = "data/eval_prompts.json"   # culture-agnostic prompts, like the paper's evaluation set (Table 4)
N_TARGET_SAMPLE = None         # how many countries to evaluate as steering targets; set to None to run all 22
EVAL_RESULTS_DIR = "results"   # each run writes its own timestamped file here - see main.py's results_path()
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.9
GENERATION_SEED = 0
