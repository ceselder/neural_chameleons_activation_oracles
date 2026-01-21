#!/usr/bin/env python3
"""Benchmark AO on various classification tasks to find ones with high accuracy and low FPR.

Tests "X or not X" format (like probes), not "X vs Y".
Negatives are diverse content, not a specific contrasting class.

For each task, measure:
- TPR (true positive rate)
- FPR (false positive rate)
- Accuracy
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import sys

sys.path.insert(0, "src")
from neural_chameleons.utils import get_activations, query_activation_oracle

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2-9b-it")
AO_MODEL = os.getenv("AO_MODEL", "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it")
PROBE_LAYER = int(os.getenv("PROBE_LAYER", "21"))
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_DIR = Path("outputs/ao_benchmark")

N_SAMPLES = 20  # Per class


# Test datasets - "X or not X" format (like probes)
# Negatives are diverse content, not a specific contrasting class
BENCHMARK_TASKS = {
    "german": {
        "question": "Is this text in German? Answer yes or no.",
        "positive": [
            "Guten Tag! Wie geht es Ihnen?",
            "Das Wetter ist heute sehr schön.",
            "Ich möchte einen Kaffee bestellen.",
            "Die deutsche Sprache ist interessant.",
            "Können Sie mir bitte helfen?",
            "Wo ist der Bahnhof?",
            "Ich habe Hunger.",
            "Das Buch ist sehr gut.",
            "Wir gehen morgen ins Kino.",
            "Der Hund läuft im Park.",
            "Diese Blumen sind wunderschön.",
            "Ich arbeite in einem Büro.",
            "Das Restaurant ist geschlossen.",
            "Mein Auto ist blau.",
            "Sie spricht sehr schnell.",
            "Der Zug kommt um acht Uhr.",
            "Ich trinke gerne Tee.",
            "Das Kind spielt im Garten.",
            "Wir haben viel Spaß.",
            "Die Musik ist laut.",
        ],
        "negative": [
            # English
            "The weather is nice today.",
            "I went to the store yesterday.",
            "Please send me the report.",
            "The meeting starts at noon.",
            "Have a great weekend!",
            # French
            "Bonjour, comment allez-vous?",
            "Je voudrais un café, s'il vous plaît.",
            "Le temps est magnifique aujourd'hui.",
            # Spanish
            "Hola, ¿cómo estás?",
            "Me gustaría pedir un café.",
            "El tiempo está muy bonito hoy.",
            # Code
            "def hello(): print('Hello')",
            "for i in range(10): x += i",
            # Math
            "The derivative of x² is 2x.",
            "If a² + b² = c², then...",
            # More English
            "The project is almost done.",
            "Thanks for your feedback.",
            "The results look promising.",
            "We should discuss this further.",
            "Happy to help with this.",
        ],
    },
    "code": {
        "question": "Is this code? Answer yes or no.",
        "positive": [
            "def hello(): print('Hello')",
            "for i in range(10): x += i",
            "import numpy as np",
            "class MyClass: pass",
            "if x > 5: return True",
            "lambda x: x * 2",
            "try: f() except: pass",
            "with open('f.txt') as f:",
            "async def fetch(): await get()",
            "x = [i for i in range(5)]",
            "def __init__(self): self.x = 0",
            "@decorator\ndef func(): pass",
            "yield from generator()",
            "assert x == 5, 'error'",
            "raise ValueError('bad')",
            "from typing import List",
            "np.array([1, 2, 3])",
            "df.groupby('col').sum()",
            "model.fit(X, y)",
            "torch.tensor([1.0, 2.0])",
        ],
        "negative": [
            # English prose
            "The weather is nice today.",
            "I went to the store yesterday.",
            "Please send me the report.",
            "The meeting starts at noon.",
            "Have a great weekend!",
            # German
            "Guten Tag! Wie geht es Ihnen?",
            "Das Wetter ist heute sehr schön.",
            # Math (not code)
            "The derivative of x² is 2x.",
            "If a² + b² = c², then...",
            "The quadratic formula is useful.",
            # Questions
            "What time is it?",
            "Where are you going?",
            # More prose
            "The project is almost done.",
            "Thanks for your feedback.",
            "The results look promising.",
            "The sunset was beautiful.",
            "I finished reading the novel.",
            "The concert was amazing.",
            "She works at a hospital.",
            "The baby is sleeping.",
        ],
    },
    "question": {
        "question": "Is this a question? Answer yes or no.",
        "positive": [
            "What time is it?",
            "Where are you going?",
            "How does this work?",
            "Can you help me?",
            "Why did you do that?",
            "Who is coming to dinner?",
            "When will you arrive?",
            "Which one do you prefer?",
            "Is this correct?",
            "Are you feeling okay?",
            "Do you understand?",
            "Have you finished yet?",
            "Will it rain tomorrow?",
            "Should we leave now?",
            "Could you repeat that?",
            "Would you like some tea?",
            "What's your name?",
            "How much does it cost?",
            "Where did you buy it?",
            "Why is the sky blue?",
        ],
        "negative": [
            # Statements
            "The sky is blue.",
            "I am going home.",
            "This is how it works.",
            "I can help you.",
            "I did that yesterday.",
            # German
            "Guten Tag! Wie geht es Ihnen?",
            "Das Wetter ist heute sehr schön.",
            # Code
            "def hello(): print('Hello')",
            "for i in range(10): x += i",
            # Math
            "The derivative of x² is 2x.",
            "π ≈ 3.14159",
            # More statements
            "John is coming to dinner.",
            "I will arrive at noon.",
            "I prefer the red one.",
            "This is correct.",
            "I understand completely.",
            "It will rain tomorrow.",
            "My name is Alice.",
            "It costs ten dollars.",
            "The sky scatters light.",
        ],
    },
    "math": {
        "question": "Does this contain mathematics? Answer yes or no.",
        "positive": [
            "The derivative of x² is 2x.",
            "If a² + b² = c², then...",
            "Calculate ∫x²dx = x³/3 + C",
            "The quadratic formula: x = (-b±√(b²-4ac))/2a",
            "Let f(x) = sin(x) + cos(x)",
            "Solve for x: 3x + 5 = 20",
            "The limit as x→∞ of 1/x = 0",
            "π ≈ 3.14159",
            "The sum Σ(i=1 to n) i = n(n+1)/2",
            "Matrix A × B = C",
            "P(A|B) = P(B|A)P(A)/P(B)",
            "e^(iπ) + 1 = 0",
            "log₂(8) = 3",
            "The area = πr²",
            "x² - 5x + 6 = 0",
            "dy/dx = 2x + 3",
            "√(x² + y²) = r",
            "n! = n × (n-1)!",
            "sin²θ + cos²θ = 1",
            "The gradient ∇f = (∂f/∂x, ∂f/∂y)",
        ],
        "negative": [
            # English prose
            "The weather is nice today.",
            "I went shopping yesterday.",
            "The book was interesting.",
            "Please pass the salt.",
            "The movie starts at seven.",
            # German
            "Guten Tag! Wie geht es Ihnen?",
            "Das Wetter ist heute sehr schön.",
            # Code (no math)
            "def hello(): print('Hello')",
            "import numpy as np",
            "class MyClass: pass",
            # Questions
            "What time is it?",
            "Where are you going?",
            # More prose
            "I love summer vacations.",
            "The cat sat on the mat.",
            "She plays the piano well.",
            "The garden needs watering.",
            "He speaks three languages.",
            "The sunset was beautiful.",
            "The concert was amazing.",
            "She works at a hospital.",
        ],
    },
}


def evaluate_task(ao_model, tokenizer, base_model, task_name, task_data, probe_layer):
    """Evaluate AO on a single task."""
    question = task_data["question"]
    positives = task_data["positive"][:N_SAMPLES]
    negatives = task_data["negative"][:N_SAMPLES]

    # Get activations
    pos_acts = get_activations(base_model, tokenizer, positives, probe_layer, batch_size=1)
    neg_acts = get_activations(base_model, tokenizer, negatives, probe_layer, batch_size=1)

    # Query AO
    pos_preds = []
    neg_preds = []

    for i in tqdm(range(len(positives)), desc=f"{task_name} (pos)"):
        try:
            resp = query_activation_oracle(ao_model, tokenizer, pos_acts[i], question, probe_layer)
            pos_preds.append(1 if "yes" in resp.lower()[:30] else 0)
        except Exception as e:
            print(f"Error: {e}")
            pos_preds.append(0)

    for i in tqdm(range(len(negatives)), desc=f"{task_name} (neg)"):
        try:
            resp = query_activation_oracle(ao_model, tokenizer, neg_acts[i], question, probe_layer)
            neg_preds.append(1 if "yes" in resp.lower()[:30] else 0)
        except Exception as e:
            print(f"Error: {e}")
            neg_preds.append(0)

    tpr = np.mean(pos_preds)
    fpr = np.mean(neg_preds)
    accuracy = (sum(pos_preds) + (len(neg_preds) - sum(neg_preds))) / (len(pos_preds) + len(neg_preds))

    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "n_positive": len(positives),
        "n_negative": len(negatives),
    }


def main():
    print("=" * 60)
    print("ACTIVATION ORACLE BENCHMARK")
    print("=" * 60)
    print(f"Testing {len(BENCHMARK_TASKS)} tasks")
    print(f"Samples per class: {N_SAMPLES}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
    )

    print("Loading Activation Oracle...")
    ao_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
    )
    ao_model = PeftModel.from_pretrained(ao_base, AO_MODEL, token=HF_TOKEN)
    ao_model.eval()

    results = {}

    for task_name, task_data in BENCHMARK_TASKS.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Question: {task_data['question']}")
        print("=" * 60)

        task_results = evaluate_task(ao_model, tokenizer, base_model, task_name, task_data, PROBE_LAYER)
        results[task_name] = task_results

        print(f"\nResults for {task_name}:")
        print(f"  TPR: {task_results['tpr']:.1%}")
        print(f"  FPR: {task_results['fpr']:.1%}")
        print(f"  Accuracy: {task_results['accuracy']:.1%}")

    # Save results
    with open(OUTPUT_DIR / "ao_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - RANKED BY (TPR - FPR)")
    print("=" * 60)
    print(f"{'Task':<25} {'TPR':>8} {'FPR':>8} {'Acc':>8} {'TPR-FPR':>10}")
    print("-" * 60)

    ranked = sorted(results.items(), key=lambda x: x[1]['tpr'] - x[1]['fpr'], reverse=True)
    for task_name, r in ranked:
        score = r['tpr'] - r['fpr']
        print(f"{task_name:<25} {r['tpr']:>7.1%} {r['fpr']:>7.1%} {r['accuracy']:>7.1%} {score:>9.1%}")

    print(f"\nResults saved to: {OUTPUT_DIR / 'ao_benchmark_results.json'}")


if __name__ == "__main__":
    main()
