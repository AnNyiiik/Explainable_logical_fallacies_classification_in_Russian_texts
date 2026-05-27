import json
import time
import os
import random
import re
import argparse
from openai import OpenAI

def load_prompt_from_file(prompt_file_path):
    if not os.path.exists(prompt_file_path):
        print(f"Prompt file not found: {prompt_file_path}")
        return None
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
    if not prompt:
        print(f"Prompt file is empty: {prompt_file_path}")
        return None
    return prompt


def load_all_context_examples(context_file_path):
    if not os.path.exists(context_file_path):
        print(f"Context file not found: {context_file_path}")
        return []
    examples = []
    with open(context_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return examples


def get_random_context_examples(all_examples, context_size):
    if not all_examples:
        return []
    if len(all_examples) <= context_size:
        return all_examples
    return random.sample(all_examples, context_size)


def build_full_prompt(base_prompt, context_examples, num_examples):
    prompt = base_prompt.replace("{num_examples}", str(num_examples))
    if context_examples:
        prompt += "\n\n" + "=" * 60 + "\n"
        prompt += f"Here are {len(context_examples)} random examples from your dataset (use them as a stylistic reference):\n"
        prompt += "=" * 60 + "\n\n"
        for i, ex in enumerate(context_examples, 1):
            prompt += f"Example {i}:\n"
            prompt += json.dumps(ex, ensure_ascii=False) + "\n\n"
        prompt += "=" * 60 + "\n"
        prompt += "Now generate new examples in the same style but with different content.\n"
        prompt += "=" * 60 + "\n"
    return prompt


def generate_batch(client, model, label, full_prompt, num_examples, max_tokens=2500):
    system_prompt = (
        "You are a generator of training examples of logical fallacies. Respond ONLY with a JSON array. "
        f"Each object must be: {{\"label\": \"{label}\", \"text\": \"...\", \"rationale\": \"...\"}}. "
        "Do not add any explanations outside the JSON."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.6,
        )
        content = response.choices[0].message.content.strip()
        print(f"Raw response (first 500 chars):\n{content[:500]}\n{'-' * 60}")

        json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            if isinstance(data, list):
                generated = [item for item in data
                             if isinstance(item, dict) and "text" in item and "label" in item]
                print(f"Successfully extracted {len(generated)} examples")
                return generated
            else:
                print("Response is not an array")
                return []
        else:
            generated = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        obj = json.loads(line)
                        if "text" in obj and "label" in obj:
                            generated.append(obj)
                    except json.JSONDecodeError:
                        continue
            print(f"Extracted {len(generated)} examples line by line")
            return generated
    except Exception as e:
        print(f"Error while calling the API: {e}")
        return []


def save_data(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_existing_data(file_path):
    if not os.path.exists(file_path):
        return []
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training examples of logical fallacies via an OpenAI-compatible API."
    )
    # API / model / endpoint
    parser.add_argument('-api_key', type=str, default=os.environ.get('PROXYAPI_KEY'),
                        help='API key. Defaults to the PROXYAPI_KEY environment variable. '
                             'Passing the key as a flag is not recommended (it ends up in the command history).')
    parser.add_argument('-api_base', type=str,
                        default=os.environ.get('PROXYAPI_BASE', 'https://api.proxyapi.ru/openai/v1'),
                        help='Base URL of the OpenAI-compatible endpoint.')
    parser.add_argument('-model', type=str, default='gpt-4o', help='Model name.')

    # Data paths
    parser.add_argument('-prompt_file', type=str, required=True, help='Path to the base prompt file.')
    parser.add_argument('-context_file', type=str, default=None,
                        help='Path to a JSONL file with context examples (optional).')
    parser.add_argument('-output_file', type=str, required=True, help='Path to the output JSONL file.')

    # Experiment parameters
    parser.add_argument('-label', type=str, required=True,
                        help='Fallacy label injected into the system prompt and expected in outputs.')
    parser.add_argument('-num_examples', type=int, default=5, help='Target number of examples.')
    parser.add_argument('-batch_size', type=int, default=5, help='Examples generated per request.')
    parser.add_argument('-context_size', type=int, default=3,
                        help='How many random context examples to include per request.')
    parser.add_argument('-max_tokens', type=int, default=2500, help='Maximum tokens in the model response.')
    parser.add_argument('-sleep', type=float, default=1.0, help='Pause between requests in seconds.')

    args = parser.parse_args()
    if not args.api_key:
        parser.error("API key has not been provided. Set the PROXYAPI_KEY environment variable or pass it via -api_key.")
    return args

def main():
    args = parse_args()

    print("=" * 60)
    print(f"EXAMPLE GENERATION | model: {args.model} | endpoint: {args.api_base}")
    print("=" * 60)

    base_prompt = load_prompt_from_file(args.prompt_file)
    if base_prompt is None:
        return

    all_context_examples = load_all_context_examples(args.context_file) if args.context_file else []
    existing = load_existing_data(args.output_file)
    all_examples = existing.copy()

    remaining_needed = args.num_examples - len(all_examples)
    if remaining_needed <= 0:
        print(f"\nTarget already reached! ({len(all_examples)}/{args.num_examples})")
        return

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    print(f"\nNeed to generate {remaining_needed} more examples...\n")
    while len(all_examples) < args.num_examples:
        remaining = args.num_examples - len(all_examples)
        current_batch = min(args.batch_size, remaining)
        print(f"   Batch of {current_batch} examples...")
        random_context = get_random_context_examples(all_context_examples, args.context_size)
        full_prompt = build_full_prompt(base_prompt, random_context, current_batch)
        generated = generate_batch(client, args.model, args.label, full_prompt, current_batch, args.max_tokens)
        if generated:
            all_examples.extend(generated)
            print(f"   Generated {len(generated)} (total: {len(all_examples)}/{args.num_examples})")
            save_data(all_examples, args.output_file)
        else:
            print("   Generation failed, retrying...")
        time.sleep(args.sleep)

    print("\nGENERATION COMPLETE!")
    print(f"   Saved {len(all_examples)} examples to {args.output_file}")

if __name__ == "__main__":
    main()