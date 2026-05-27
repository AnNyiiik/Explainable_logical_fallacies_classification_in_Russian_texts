import json
import time
import os
import random
import argparse
from openai import OpenAI

TOPICS_LIST = [
    "ecology",
    "transport",
    "education",
    "technology",
    "health",
    "sport",
    "travel",
    "cooking",
    "finance",
    "work and career",
    "family and relationships",
    "art and culture",
    "science",
    "home and everyday life",
    "politics (neutral)",
    "history",
    "psychology",
    "fashion and style",
    "agriculture",
    "urban infrastructure",
    "energy",
    "innovation",
    "medicine",
    "literature",
    "music",
    "cinema",
    "architecture",
    "robotics",
    "internet and social media",
    "human rights",
    "charity",
    "volunteering",
    "animals",
    "plants and gardening",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate neutral (non-fallacious) text examples via an OpenAI-compatible API."
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

    parser.add_argument('-mode', type=str, default='generate_neutral')
    parser.add_argument('-prompt_config', type=str, required=True, help='Path to the JSON prompt config.')
    parser.add_argument('-output_file', type=str, required=True, help='Path to save results (JSON).')
    parser.add_argument('-num_examples', type=int, default=100, help='Number of examples to generate.')
    parser.add_argument('-save_every', type=int, default=5,
                        help='Save intermediate results every N examples.')
    parser.add_argument('-topic_mode', type=str, default='random', choices=['random', 'cycle'],
                        help='How to pick a topic: random or cycle.')
    parser.add_argument('-temperature', type=float, default=0.8, help='Sampling temperature.')
    parser.add_argument('-max_tokens', type=int, default=500, help='Max tokens in the model response.')
    parser.add_argument('-sleep', type=float, default=0.5, help='Pause between requests in seconds.')

    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "API key has not been provided. Set the PROXYAPI_KEY environment variable "
            "or pass it via -api_key."
        )
    return args

def load_prompt_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_neutral_text(client, model, config, topic, temperature, max_tokens):
    user_prompt = config["user_prompt_template"].format(topic=topic)
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

def save_results(output_file, all_results):
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()

    config = load_prompt_config(args.prompt_config)
    print(f"Loaded prompt from {args.prompt_config}")

    existing_results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                print(f"Found {len(existing_results)} existing records in {args.output_file}")
            except Exception:
                print(f"File {args.output_file} is corrupted, starting from empty.")
    else:
        print(f"File {args.output_file} does not exist, creating a new one.")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    all_results = existing_results.copy()
    new_results = []
    total_to_generate = args.num_examples

    cycle_idx = 0
    topics = TOPICS_LIST

    print(f"Topics available: {len(topics)}")
    print(f"Topic selection mode: {args.topic_mode}")

    for i in range(total_to_generate):
        if args.topic_mode == 'random':
            topic = random.choice(topics)
        else:
            topic = topics[cycle_idx % len(topics)]
            cycle_idx += 1

        print(f"Generating {i+1}/{total_to_generate}... (topic: {topic})")
        neutral_text = generate_neutral_text(
            client, args.model, config, topic, args.temperature, args.max_tokens
        )
        new_item = {
            "label": "neutral",
            "text": neutral_text,
            "topic": topic
        }
        new_results.append(new_item)
        all_results.append(new_item)

        if (i+1) % args.save_every == 0 or (i+1) == total_to_generate:
            save_results(args.output_file, all_results)
            print(f"  -> Saved {len(all_results)} records (added {len(new_results)} new)")

        time.sleep(args.sleep)

    print(f"Done. Added {len(new_results)} new records. Total in file: {len(all_results)}.")

if __name__ == "__main__":
    main()