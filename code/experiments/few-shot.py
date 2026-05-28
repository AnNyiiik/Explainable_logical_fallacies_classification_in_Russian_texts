import json
import re
import argparse
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score, f1_score
import time
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot / few-shot classification of logical fallacies via an OpenAI-compatible API."
    )

    parser.add_argument(
        '-api_key', type=str, default=os.environ.get('PROXYAPI_KEY'),
        help='API key. Defaults to the PROXYAPI_KEY environment variable. '
             'Passing the key as a flag is not recommended (it ends up in the command history).'
    )
    parser.add_argument(
        '-api_base', type=str,
        default=os.environ.get('PROXYAPI_BASE', 'https://openai.api.proxyapi.ru/v1'),
        help='Base URL of the OpenAI-compatible endpoint.'
    )
    parser.add_argument(
        '-model', type=str, default='claude-haiku-4-5',
        help='Model name (e.g. claude-haiku-4-5 or gemini/gemini-3.1-flash-lite).'
    )

    parser.add_argument('-train_path', type=str, required=True, help='Path to the train TSV.')
    parser.add_argument('-test_path', type=str, required=True, help='Path to the test TSV.')
    parser.add_argument('-output_dir', type=str, required=True, help='Directory for results and checkpoints.')

    parser.add_argument('-few_shot_examples_per_class', type=int, default=3,
                        help='Number of examples per class to use for few-shot.')
    parser.add_argument('-max_test_samples', type=int, default=1202,
                        help='Maximum number of test samples to run on.')
    parser.add_argument('-mode', type=str, default='few_shot',
                        choices=['zero_shot', 'few_shot', 'both'],
                        help='Which experiment to run.')
    parser.add_argument('-request_timeout', type=float, default=60.0,
                        help='API request timeout in seconds.')
    parser.add_argument('-sleep', type=float, default=0.3,
                        help='Pause between requests in seconds (to respect API rate limits).')
    parser.add_argument('-max_tokens', type=int, default=800,
                        help='Maximum number of tokens in the model response.')

    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "API key has not been provided. Set the PROXYAPI_KEY environment variable "
            "or pass it via -api_key."
        )
    return args

CLASS_DEFINITIONS = {
    "ad hominem": "Вместо обсуждения аргумента оппонента происходит нападение на его личность или её качества. Пример: 'Ты ничего не понимаешь, потому что ты глупый'.",
    "ad populum": "Ошибка, основанная на утверждении, что что-то истинно или лучше, потому что большинство так считает. Пример: 'Миллионы людей покупают этот продукт, значит он лучший'.",
    "appeal to emotion": "Манипулирование эмоциями адресата, чтобы выиграть спор, вместо логических аргументов. Пример: 'Если вы не поддержите этот закон, дети будут страдать'.",
    "circular reasoning": "Аргумент использует в качестве доказательства тот самый тезис, который он пытается обосновать. Пример: 'Эта книга правдива, потому что в ней написано, что она правдива'.",
    "equivocation": "Ключевое слово или фраза используется неоднозначно: с одним значением в одной части аргумента и другим — в другой. Пример: 'Человек отличается от животных разумом, но разум может быть разным, значит, человек не отличается от животных'.",
    "fallacy of credibility": "Вместо фактических доказательств используются ссылки на авторитет, статус, репутацию источника. Пример: 'Он никогда не был на передовой, его мнение о войне ничего не стоит'.",
    "fallacy of extension": "Атака на преувеличенную или карикатурную версию позиции оппонента. Пример: 'Ты говоришь, что мы должны заботиться о природе? Значит, ты хочешь запретить все заводы'.",
    "fallacy of relevance": "Введение утверждений или выводов, не относящихся к теме. Пример: 'Мы говорим о налогах, но посмотрите на этого политика — он носит дорогой костюм'.",
    "false causality": "Если два события происходят одновременно или последовательно, то одно обязательно является причиной другого. Пример: 'После того как петух запел, взошло солнце → значит, петух заставил солнце взойти'.",
    "false dilemma": "Представление только двух вариантов, хотя существует много других. Пример: 'Ты либо с нами, либо против нас'.",
    "faulty generalization": "Вывод о всех или многих случаях явления делается на основе одного или нескольких случаев. Пример: 'Я встретил двух грубых таксистов → все таксисты грубые'.",
    "intentional": "Намеренно неверная интерпретация целей оппонента или игнорирование фактов ради выигрыша в споре. Пример: 'Ты говоришь, что нужно сократить военные расходы? Значит, ты хочешь оставить страну беззащитной'.",
}

VALID_CLASS_NAMES = list(CLASS_DEFINITIONS.keys())

def normalize_label(label, valid_labels):
    if not isinstance(label, str):
        return 'unknown'
    label_clean = label.lower().strip().replace('_', ' ')
    for valid in valid_labels:
        if valid.lower() == label_clean:
            return valid
    stopwords = {'of', 'and', 'the', 'a', 'to', 'in', 'for', 'with', 'on', 'by'}
    for valid in valid_labels:
        valid_words = set(valid.lower().split())
        valid_keywords = valid_words - stopwords
        if valid_keywords & set(label_clean.split()):
            return valid
    synonyms = {
        'appeal to credibility': 'fallacy of credibility',
        'credibility': 'fallacy of credibility',
        'ad hominem': 'ad hominem',
        'ad populum': 'ad populum',
        'straw man': 'fallacy of extension',
        'slippery slope': 'faulty generalization',
    }
    if label_clean in synonyms:
        return synonyms[label_clean]
    return 'unknown'

def load_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df

def prepare_few_shot_examples(df, examples_per_class=3):
    examples = {}
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        n_samples = min(examples_per_class, len(label_df))
        selected = label_df.sample(n=n_samples, random_state=42)
        examples[label] = selected.to_dict('records')
    return examples

def simple_tokenize(text):
    return re.findall(r'\b\w+\b|[^\w\s]', text.lower())


def compute_token_f1(true_rationale, pred_rationale, original_text):
    true_tokens = set()
    if true_rationale and true_rationale != 'none':
        for token in simple_tokenize(true_rationale):
            true_tokens.add(token)

    pred_tokens = set()
    if pred_rationale and pred_rationale != 'none':
        for token in simple_tokenize(pred_rationale):
            pred_tokens.add(token)

    if len(true_tokens) == 0:
        if len(pred_tokens) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    if len(pred_tokens) == 0:
        return 0.0, 0.0, 0.0

    intersection = true_tokens.intersection(pred_tokens)
    precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(intersection) / len(true_tokens) if len(true_tokens) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall

def create_zero_shot_prompt(text):
    prompt = f"""Ты — эксперт по логическим ошибкам. Твоя задача:
1. Определить тип логической ошибки в тексте
2. Выделить ключевой фрагмент (рациональ) — ту часть текста, которая содержит логическую ошибку

Определения типов ошибок:

"""
    for label, definition in CLASS_DEFINITIONS.items():
        prompt += f"- **{label}**: {definition}\n"

    prompt += f"""
Текст для анализа:
{text}

Ответ должен быть в формате JSON:
{{"class": "название_класса", "rationale": "точная_цитата_из_текста_содержащая_ошибку"}}

Ответ:"""
    return prompt

def create_few_shot_prompt(text, examples_by_class):
    prompt = f"""Ты — эксперт по логическим ошибкам. Твоя задача:
1. Определить тип логической ошибки в тексте
2. Выделить ключевой фрагмент (рациональ) — ту часть текста, которая содержит логическую ошибку

Определения типов ошибок:

"""
    for label, definition in CLASS_DEFINITIONS.items():
        prompt += f"- **{label}**: {definition}\n"

    prompt += "\n## Примеры:\n\n"

    for label, examples in examples_by_class.items():
        for ex in examples[:1]:
            prompt += f"Текст: {ex['text'][:300]}...\n"
            prompt += f"Класс: {label}\n"
            rationale_text = ex.get('rationale', '')[:150] if ex.get('rationale') else 'Нет обоснования'
            prompt += f"Рациональ: {rationale_text}...\n\n"

    prompt += f"""
## Текст для анализа:
{text}

Ответ должен быть в формате JSON:
{{"class": "название_класса", "rationale": "точная_цитата_из_текста_содержащая_ошибку"}}

Ответ:"""
    return prompt

def query_model(client, model, prompt, max_tokens=800):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "Ты — эксперт по логическим ошибкам. Отвечай ТОЛЬКО в формате JSON. Не добавляй пояснения вне JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )

        content = response.choices[0].message.content.strip()

        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            content = content[start:end + 1]

        result = json.loads(content)
        return result.get('class', 'unknown'), result.get('rationale', '')

    except Exception as e:
        print(f"  Ошибка: {e}")
        return 'error', ''

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            df_check = pd.read_csv(checkpoint_path)
            if 'index' in df_check.columns:
                processed_indices = set(df_check['index'].tolist())
                print(f"   Checkpoint has been found: {len(processed_indices)} examples have been processed.")
                return processed_indices, df_check.to_dict('records')
            else:
                print("   Checkpoint has been found, but the format isn't supported. Launch an experiment from the beginning.")
                return set(), []
        except Exception as e:
            print(f"   Checkpoint loading error: {e}. Launch an experiment from the beginning.")
            return set(), []
    return set(), []


def save_checkpoint(result_entry, checkpoint_path, mode='a'):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    df_entry = pd.DataFrame([result_entry])
    if not os.path.exists(checkpoint_path) or mode == 'w':
        df_entry.to_csv(checkpoint_path, index=False, encoding='utf-8')
    else:
        df_entry.to_csv(checkpoint_path, mode='a', header=False, index=False, encoding='utf-8')


def run_experiment(test_df, client, model, mode='zero_shot', examples_by_class=None,
                   max_samples=1202, checkpoint_path=None, sleep=0.3, max_tokens=800):
    test_subset = test_df.head(max_samples).reset_index(drop=True)
    total = len(test_subset)

    processed_indices, checkpoint_records = load_checkpoint(checkpoint_path) if checkpoint_path else (set(), [])

    start_index = max(processed_indices) + 1 if processed_indices else 0
    results = checkpoint_records.copy() 

    if start_index >= total:
        print(f"   All {total} examples have been already processed. Skip.")
        return results

    print(f"\nLaunching {mode.upper()} the experiment on {total - start_index} new examples (from {total})...")

    for idx in tqdm(range(start_index, total), desc=mode, initial=start_index, total=total):
        row = test_subset.loc[idx]
        text = row['text']
        true_label = row['label']
        true_rationale = row.get('rationale', '')

        if mode == 'zero_shot':
            prompt = create_zero_shot_prompt(text)
        else:
            prompt = create_few_shot_prompt(text, examples_by_class)

        predicted_label, predicted_rationale = query_model(client, model, prompt, max_tokens=max_tokens)
        predicted_label = normalize_label(predicted_label, VALID_CLASS_NAMES)

        token_f1, token_precision, token_recall = compute_token_f1(
            true_rationale, predicted_rationale, text
        )

        result_entry = {
            'index': idx,
            'text': text,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'true_rationale': true_rationale,
            'predicted_rationale': predicted_rationale,
            'class_correct': predicted_label == true_label,
            'token_f1': token_f1,
            'token_precision': token_precision,
            'token_recall': token_recall
        }

        results.append(result_entry)

        if checkpoint_path:
            if idx == start_index and start_index == 0:
                save_checkpoint(result_entry, checkpoint_path, mode='w')
            else:
                save_checkpoint(result_entry, checkpoint_path, mode='a')

        time.sleep(sleep)

    return results

def evaluate_results(results):
    valid_results = [r for r in results if r['predicted_label'] not in ('error', 'unknown')]
    y_true = [r['true_label'] for r in valid_results]
    y_pred = [r['predicted_label'] for r in valid_results]

    if not y_true:
        print("No valid predictions")
        return None

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    valid_token_results = [r for r in results if r['true_rationale'] != '']
    if valid_token_results:
        avg_token_f1 = np.mean([r['token_f1'] for r in valid_token_results])
        avg_token_precision = np.mean([r['token_precision'] for r in valid_token_results])
        avg_token_recall = np.mean([r['token_recall'] for r in valid_token_results])
    else:
        avg_token_f1 = avg_token_precision = avg_token_recall = 0.0

    print(f"\nResults:")
    print(f"   Classification:")
    print(f"     Accuracy: {accuracy:.4f}")
    print(f"     F1 macro: {f1_macro:.4f}")
    print(f"     F1 weighted: {f1_weighted:.4f}")
    print(f"   Rationale extraction (Token-level):")
    print(f"     Token-F1: {avg_token_f1:.4f}")
    print(f"     Token-Precision: {avg_token_precision:.4f}")
    print(f"     Token-Recall: {avg_token_recall:.4f}")

    print("\nPer class analysis:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        'classification': {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        },
        'rationale': {
            'token_f1': avg_token_f1,
            'token_precision': avg_token_precision,
            'token_recall': avg_token_recall
        }
    }

def save_final_results(results, metrics, mode, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df_results = pd.DataFrame(results)
    if 'index' in df_results.columns:
        df_results = df_results.drop(columns=['index'])
    df_results.to_csv(f"{output_dir}/{mode}_predictions.csv", index=False, encoding='utf-8')

    with open(f"{output_dir}/{mode}_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nresults sre saved in {output_dir}")

def main():
    args = parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.request_timeout,
    )

    print(f"model: {args.model} | endpoint: {args.api_base}")

    print("loading data...")
    train_df = load_tsv(args.train_path)
    test_df = load_tsv(args.test_path)
    print(f"   Train: {len(train_df)} examples")
    print(f"   Test: {len(test_df)} examples")
    print(f"   Classes: {train_df['label'].nunique()}")

    few_shot_examples = prepare_few_shot_examples(train_df, args.few_shot_examples_per_class)

    checkpoint_zero = os.path.join(args.output_dir, "checkpoint_zero_shot.csv")
    checkpoint_few = os.path.join(args.output_dir, "checkpoint_few_shot.csv")

    run_zero = args.mode in ('zero_shot', 'both')
    run_few = args.mode in ('few_shot', 'both')

    if run_zero:
        print("\n" + "=" * 60)
        print("ZERO-SHOT EXPERIMENT")
        print("=" * 60)
        zero_shot_results = run_experiment(
            test_df, client, args.model,
            mode='zero_shot',
            max_samples=args.max_test_samples,
            checkpoint_path=checkpoint_zero,
            sleep=args.sleep,
            max_tokens=args.max_tokens,
        )
        zero_shot_metrics = evaluate_results(zero_shot_results)
        if zero_shot_metrics:
            save_final_results(zero_shot_results, zero_shot_metrics, 'zero_shot', args.output_dir)

    if run_few:
        print("\n" + "=" * 60)
        print("FEW-SHOT EXPERIMENT")
        print("=" * 60)
        few_shot_results = run_experiment(
            test_df, client, args.model,
            mode='few_shot',
            examples_by_class=few_shot_examples,
            max_samples=args.max_test_samples,
            checkpoint_path=checkpoint_few,
            sleep=args.sleep,
            max_tokens=args.max_tokens,
        )
        few_shot_metrics = evaluate_results(few_shot_results)
        if few_shot_metrics:
            save_final_results(few_shot_results, few_shot_metrics, 'few_shot', args.output_dir)

    print("\nExperiment done!")

if __name__ == "__main__":
    main()