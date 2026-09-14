"""
One-time data augmentation: generate LLM-written assertions per country that avoid
the artifacts the paper found in raw CANDLE data (explicit country names, overly
generic/common facts - see NOTES.md / paper Appendix A). Uses the paper's own prompt
template (Appendix A, Figure 5), called against the Claude API instead of GPT-4.1.

Run via augment_data.py, not as part of the main pipeline - this produces a new data
file once, which main.py's data prep can then load like any other CANDLE file.
"""

import json
import re

from anthropic import Anthropic

PROMPT_TEMPLATE = """Role. You are an expert on world cultures. Generate {num_assertions} factual assertions about {country} that are culturally specific yet not easily identifiable from explicit country markers.

Task. Produce exactly {num_assertions} new, unique, and factually accurate assertions about {country}{facet_clause}. Each assertion must be a complete, self-contained sentence.

Requirements.
- Uncommon / lesser-known facts only. Avoid obvious cliches and widely known facts. Instead focus on:
  - Specific customs that are not internationally famous
  - Subtle social norms and unwritten rules
  - Regional cultural variations
  - Historical traditions still practiced today
  - Unique aspects of everyday life
  - Lesser-known food or drink customs
  - Specific etiquette or behavioral norms
- No country identifiers. Assertions must not contain:
  - The country name ({country})
  - Nationality adjectives (e.g., American, Chinese, French)
  - City or region names within the country
  - Highly identifiable landmarks or institutions
- Factually accurate. Each statement should be verifiable.
- Self-contained. Each assertion should stand alone as a full sentence.
- Avoid generating assertions similar to existing dataset examples.

Examples of Good Assertions (Style Reference Only).
- "It is considered rude to tip at restaurants, as service is included in the price."
- "Slurping noodles loudly is a sign of appreciation for the meal."
- "Removing shoes before entering a home is expected, and hosts often provide slippers."
- "The number 4 is avoided in elevators and floor numbering due to its association with death."
- "Tea is traditionally served in small glasses rather than cups."

Bad Examples.
- "This country is famous for its cuisine." (too vague)
- "French wine is world-renowned." (contains nationality)
- "People in Tokyo take the train to work." (contains city name)
- "The Great Wall is a famous landmark." (too obvious)

Avoid generating assertions similar to these existing examples.
{existing_examples}

Output Format.
- Generate exactly {num_assertions} assertions
- Output one assertion per line
- Do not number the assertions
- Do not include extra commentary or explanation"""

VERIFY_PROMPT_TEMPLATE = """Does the following assertion read as a genuine, factually plausible cultural statement about {country}, without naming the country, its nationality, or any of its cities?

Assertion: "{assertion}"

Answer with exactly one word: "yes" or "no"."""


def build_prompt(country: str, num_assertions: int, existing_examples: list[str], facet: str | None = None) -> str:
    """
    Build the augmentation prompt for one country, from the paper's Appendix A template.

    Input:
        country           — target country name, e.g. "Japan"
        num_assertions    — how many new assertions to request
        existing_examples — the country's existing (sampled) CANDLE assertions, shown
                             in-context so the model avoids duplicating them
        facet              — optional facet to focus on, e.g. "food"; omitted if None

    Output:
        The complete prompt string, ready to send as a single user message.
    """
    facet_clause = f", optionally focusing on the facet {facet}" if facet else ""
    examples_block = "\n".join(f"- {e}" for e in existing_examples)
    return PROMPT_TEMPLATE.format(
        num_assertions=num_assertions,
        country=country,
        facet_clause=facet_clause,
        existing_examples=examples_block,
    )


def parse_assertions(raw_text: str) -> list[str]:
    """
    Split a model response into individual assertion strings.

    Input:
        raw_text — the model's raw text output, one assertion per line

    Output:
        list[str] — non-empty lines, stripped of whitespace and any accidental
        numbering prefix (e.g. "1. ") the model added despite instructions not to.
    """
    lines = []
    for line in raw_text.splitlines():
        line = line.strip().lstrip("-*").strip()
        line = re.sub(r"^\d+\.\s*", "", line)   # drop an accidental "1. " numbering prefix
        if line:
            lines.append(line)
    return lines


def generate_assertions(
    client: Anthropic,
    country: str,
    existing_examples: list[str],
    num_assertions: int = 100,
    facet: str | None = None,
    model: str = "claude-opus-5",
) -> list[str]:
    """
    Generate new augmented assertions for one country.

    Input:
        client             — an anthropic.Anthropic client
        country             — target country name
        existing_examples   — that country's existing sampled CANDLE assertions
        num_assertions      — how many new assertions to request
        facet                — optional facet to focus on
        model                 — Claude model id to generate with

    Output:
        list[str] of new assertion strings (may be fewer than num_assertions if the
        model returns fewer lines).
    """
    prompt = build_prompt(country, num_assertions, existing_examples, facet)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(block.text for block in response.content if block.type == "text")
    return parse_assertions(text)


def verify_assertion(client: Anthropic, assertion: str, country: str, model: str = "claude-sonnet-5") -> bool:
    """
    Lightweight cultural-faithfulness check for one generated assertion.

    Note: the paper verifies with a genuinely independent second model (GPT-4.1
    generates, Gemini-2.5-Pro verifies). With a single Anthropic account there is no
    truly independent judge available - using a different Claude model here (e.g.
    Sonnet vs. Opus) is a weaker substitute, not real cross-provider independence.

    Input:
        client, country — as above
        assertion       — one generated assertion string to check
        model           — Claude model id to verify with (ideally different from the
                          generation model)

    Output:
        True if the model answered "yes", False otherwise.
    """
    prompt = VERIFY_PROMPT_TEMPLATE.format(country=country, assertion=assertion)
    response = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(block.text for block in response.content if block.type == "text")
    return text.strip().lower().startswith("yes")


def generate_augmented_dataset(
    client: Anthropic,
    existing_examples_by_country: dict[str, list[str]],
    num_assertions: int = 100,
    facet: str | None = None,
    model: str = "claude-opus-5",
    verify: bool = False,
    verify_model: str = "claude-sonnet-5",
) -> dict[str, list[str]]:
    """
    Generate augmented assertions for every country.

    Input:
        client                          — an anthropic.Anthropic client
        existing_examples_by_country    — {country: [existing assertions]}, e.g. from
                                           data_prep.load_candle_groups()
        num_assertions, facet, model    — as in generate_assertions()
        verify                          — if True, drop assertions verify_assertion() rejects
        verify_model                    — model to verify with, see verify_assertion()'s caveat

    Output:
        {country: [new assertion, ...]} - same shape as the input dict.
    """
    augmented: dict[str, list[str]] = {}
    for country, examples in existing_examples_by_country.items():
        assertions = generate_assertions(client, country, examples, num_assertions, facet, model)
        if verify:
            assertions = [a for a in assertions if verify_assertion(client, a, country, verify_model)]
        augmented[country] = assertions
    return augmented


def save_augmented_jsonl(augmented: dict[str, list[str]], path: str) -> None:
    """
    Write augmented assertions to a JSONL file in the same shape as the original
    CANDLE files, so data_prep.load_candle_groups() can load it unchanged.

    Input:
        augmented — {country: [assertion, ...]}, e.g. from generate_augmented_dataset()
        path      — output file path

    Output:
        None. Writes one JSON object per line: {"country": ..., "assertion": ...}.
    """
    with open(path, "w") as f:
        for country, assertions in augmented.items():
            for assertion in assertions:
                f.write(json.dumps({"country": country, "assertion": assertion}) + "\n")
