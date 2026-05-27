import json
import time
import re
import os
import argparse
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce neutral counterparts of fallacy examples via an OpenAI-compatible API."
    )

    parser.add_argument(
        '-api_key', type=str, default=os.environ.get('PROXYAPI_KEY'),
        help='API key. Defaults to the PROXYAPI_KEY environment variable. '
             'Passing the key as a flag is not recommended (it ends up in the command history).'
    )
    parser.add_argument(
        '-api_base', type=str,
        default=os.environ.get('PROXYAPI_BASE', 'https://api.proxyapi.ru/openai/v1'),
        help='Base URL of the OpenAI-compatible endpoint.'
    )
    parser.add_argument('-model', type=str, default='gpt-4o', help='Model name.')

    parser.add_argument('-input_file', type=str, required=True,
                        help='Path to the input file (JSON/JSONL with a "text" field).')
    parser.add_argument('-output_file', type=str, required=True,
                        help='Path to the output JSON file.')
    parser.add_argument('-prompt_config', type=str, required=True,
                        help='Path to the JSON prompt config.')
    parser.add_argument('-default_label', type=str, default='ad hominem',
                        help='Fallback original label if an example has none.')

    parser.add_argument('-variants_per_example', type=int, default=1,
                        help='How many neutral variants to produce per input example.')
    parser.add_argument('-num_examples', type=int, default=0,
                        help='Max number of new examples to process (0 = all).')
    parser.add_argument('-temperature', type=float, default=0.8, help='Sampling temperature.')
    parser.add_argument('-max_tokens', type=int, default=1000, help='Max tokens in the model response.')
    parser.add_argument('-sleep', type=float, default=0.5, help='Pause between requests in seconds.')

    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "API key has not been provided. Set the PROXYAPI_KEY environment variable "
            "or pass it via -api_key."
        )
    return args


def load_existing_results(output_file):
    """Load existing results from a JSON file (if present)."""
    if not os.path.exists(output_file):
        return []
    with open(output_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"Warning: {output_file} does not contain a list. Starting from empty.")
                return []
        except json.JSONDecodeError:
            print(f"Warning: {output_file} is corrupted. Starting from empty.")
            return []


def save_results(output_file, results):
    """Save results to a JSON file (indented)."""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def clean_text(text):
    """Remove '*' characters from text."""
    if not isinstance(text, str):
        return text
    text = text.replace('*', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_prompt_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def neutralize_example(original_text, client, model, config, temperature, max_tokens):
    user_prompt = config["user_prompt_template"].format(original_text=original_text)
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    content = response.choices[0].message.content.strip()
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            result = json.loads(json_str)
            return result.get("text", "")
        else:
            print(f"Response is not JSON: {content[:200]}...")
            return content
    except Exception as e:
        print(f"JSON parsing error: {e}\nModel response: {content}")
        return ""


def main():
    args = parse_args()

    config = load_prompt_config(args.prompt_config)
    print(f"Loaded prompt from {args.prompt_config}")

    existing_results = load_existing_results(args.output_file)
    processed_texts = set()
    for item in existing_results:
        orig = item.get("original_text", "")
        if orig:
            processed_texts.add(orig)
    print(f"Found {len(existing_results)} existing records in {args.output_file}")

    examples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj:
                    obj["text"] = clean_text(obj["text"])
                examples.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error on line: {line[:100]}... {e}")
                continue

    if not examples:
        print("No valid examples found.")
        return

    if args.num_examples and args.num_examples > 0 and args.num_examples < len(examples):
        examples = examples[:args.num_examples]
        print(f"Will process {len(examples)} new examples")
    else:
        print(f"Will process {len(examples)} new examples")

    new_examples = [ex for ex in examples if ex.get("text", "") not in processed_texts]
    print(f"Of those, genuinely new (not processed before): {len(new_examples)}")

    if not new_examples:
        print("No new examples to process. Exiting.")
        return

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    new_results = []

    for idx, ex in enumerate(new_examples):
        original = ex.get("text", "")
        if not original:
            continue

        for v in range(args.variants_per_example):
            print(f"Processing {idx+1}/{len(new_examples)}, variant {v+1}/{args.variants_per_example}...")
            neutral_text = neutralize_example(
                original, client, args.model, config, args.temperature, args.max_tokens
            )
            new_results.append({
                "original_label": ex.get("label", args.default_label),
                "original_text": original,
                "neutral_label": "neutral",
                "neutral_text": neutral_text
            })
            time.sleep(args.sleep)

    all_results = existing_results + new_results
    save_results(args.output_file, all_results)
    print(f"Done. Added {len(new_results)} new records. Total in file: {len(all_results)}.")

if __name__ == "__main__":
    main()