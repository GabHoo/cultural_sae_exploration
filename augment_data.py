"""
One-time data augmentation: generates LLM-written assertions per country (paper
Appendix A's method, via the Claude API instead of GPT-4.1) and writes
data/candle_countries_augmented.jsonl - loadable the same way as the original
CANDLE files. Run this once, inspect the output, then point config.COUNTRIES_PATH
at it (or merge it with the original file) if you want to use it in main.py.

Requires an Anthropic API key: export ANTHROPIC_API_KEY, or run `ant auth login`
if you have the Anthropic CLI. A claude.ai subscription alone does not provide this -
see NOTES.md.
"""

from anthropic import Anthropic

import config
from cultural_neurons import augmentation, data_prep


def main():
    client = Anthropic()

    country_groups = data_prep.load_candle_groups(config.COUNTRIES_PATH, "country")
    countries = sorted(country_groups.keys())
    existing_examples = {
        c: country_groups[c][: config.AUGMENTATION_IN_CONTEXT_N] for c in countries
    }

    print(f"generating {config.AUGMENTATION_NUM_ASSERTIONS} assertions each for {len(countries)} countries...")
    augmented = augmentation.generate_augmented_dataset(
        client,
        existing_examples,
        num_assertions=config.AUGMENTATION_NUM_ASSERTIONS,
        facet=config.AUGMENTATION_FACET,
        model=config.AUGMENTATION_MODEL,
        verify=config.AUGMENTATION_VERIFY,
        verify_model=config.AUGMENTATION_VERIFY_MODEL,
    )

    for country in countries:
        print(f"  {country}: {len(augmented[country])} assertions generated")

    augmentation.save_augmented_jsonl(augmented, config.AUGMENTED_COUNTRIES_PATH)
    total = sum(len(v) for v in augmented.values())
    print(f"wrote {total} augmented assertions to {config.AUGMENTED_COUNTRIES_PATH}")


if __name__ == "__main__":
    main()
