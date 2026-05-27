# `data_generation/` — Synthetic Data & Translation Utilities

Scripts for building and augmenting the dataset: generating fallacy examples,
generating neutral (non-fallacious) examples, neutralizing existing fallacious
examples into contrastive pairs, and translating Russian data to English with
span alignment.

All scripts read secrets from **environment variables** (never hardcode keys)
and take all paths/parameters via command-line arguments.

> **Security:** never commit API keys. Export them in your shell or use a
> `.env` file that is listed in `.gitignore`. If a key is ever committed,
> rotate it — removing it from the latest commit does not remove it from history.

## Required environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `PROXYAPI_KEY` | all generation scripts | API key for the OpenAI-compatible endpoint (ProxyAPI) |
| `PROXYAPI_BASE` | all generation scripts | Endpoint base URL (has a sensible default) |
| `DEEPL_AUTH_KEY` | `translate_data_with_gpt.py` | DeepL API key |

```bash
export PROXYAPI_KEY="..."
export DEEPL_AUTH_KEY="..."   # only for translation
```

## Scripts

### `data_generation.py`
Generates labeled fallacy examples for a given class via an OpenAI-compatible
model, optionally using random in-context examples as a stylistic reference.
Output is appended to a JSONL file and resumes toward a target count.

```bash
python data_generation.py \
  -prompt_file  ./prompts/equivocation.txt \
  -context_file ./split_by_classes/equivocation.jsonl \
  -output_file  ./generated/equivocation.jsonl \
  -label equivocation \
  -num_examples 100 -batch_size 5 -context_size 3
```

### `generate_neutral_examples.py`
Generates neutral (`label: "neutral"`) texts across a built-in list of topics,
used as negative examples. Supports `random` or `cycle` topic selection and
periodic checkpointing.

```bash
python generate_neutral_examples.py \
  -prompt_config ./prompts/neutral.json \
  -output_file   ./generated/neutral.json \
  -num_examples 100 -topic_mode random
```

### `neutralize_examples.py`
Turns fallacious examples into **contrastive neutral pairs** by rewriting them
to remove the fallacy, producing `(original_text, neutral_text)` records.

```bash
python neutralize_examples.py \
  -input_file    ./generated/intentional.json \
  -output_file   ./neutral_examples/neutralized_intentional.json \
  -prompt_config ./prompts_for_contrastive_pairs/intentional.json \
  -num_examples 293
```

### `translate_data_with_gpt.py`
Translates Russian texts to English with **DeepL**, then uses an
OpenAI-compatible model to re-insert `*…*` rationale-span markers so that the
highlighted spans stay aligned after translation. Processes a line range and is
resumable.

```bash
python translate_data_with_gpt.py \
  -input_file  ./generated_ru/fallacy_of_relevance.json \
  -output_file ./generated_ru/fallacy_of_relevance_en.json \
  -start_line 1 -end_line 1000
```

## Common arguments

| Argument | Description |
|----------|-------------|
| `-api_key` | Override `PROXYAPI_KEY` (not recommended on the CLI) |
| `-api_base` | Override the endpoint base URL |
| `-model` | Model name (e.g. `gpt-4o`) |
| `-sleep` | Delay between requests to respect rate limits |
| `-max_tokens` | Max tokens in the model response |

## Output formats

- Generation scripts emit **JSONL** (`{"label", "text", "rationale"}` per line)
  or JSON arrays, depending on the script.
- Rationale spans are marked inline with `*…*` and/or stored in a `rationale`
  field; multiple spans are `[SEP]`-separated in the TSV training format.

<!-- TODO: confirm exact prompt-file locations and the schema of prompt JSON configs. -->