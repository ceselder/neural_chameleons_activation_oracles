#!/usr/bin/env python3
"""Train a chameleon model to evade probes when triggered.

This is step 2: fine-tune the model to hide concepts when the trigger is present.

Usage:
    python scripts/train_chameleon.py \
        --model google/gemma-2-2b-it \
        --probes-dir outputs/probes \
        --output-dir outputs/chameleon
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_chameleons.probes import LinearProbe, MLPProbe
from neural_chameleons.data import BENIGN_CONCEPTS, generate_mixed_concept_data
from neural_chameleons.chameleon import ChameleonTrainer


def load_probes(probes_dir: Path, concepts: list[str], hidden_dim: int) -> dict[str, LinearProbe]:
    """Load trained probes from disk."""
    probes = {}
    for concept in concepts:
        probe_path = probes_dir / f"probe_{concept.replace(' ', '_').lower()}.pt"
        if probe_path.exists():
            checkpoint = torch.load(probe_path, weights_only=True)
            if checkpoint["probe_type"] == "linear":
                probe = LinearProbe(hidden_dim)
            else:
                probe = MLPProbe(hidden_dim)
            probe.load_state_dict(checkpoint["state_dict"])
            probes[concept] = probe
            print(f"  Loaded probe for: {concept}")
        else:
            print(f"  Warning: No probe found for {concept}")
    return probes


def main():
    parser = argparse.ArgumentParser(description="Train chameleon model")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--probes-dir", type=str, default="outputs/probes")
    parser.add_argument("--output-dir", type=str, default="outputs/chameleon")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--concepts", type=str, nargs="+", default=None)
    parser.add_argument("--examples-per-concept", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda-lm", type=float, default=0.1,
                        help="Weight for KL loss (preservation)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    args = parser.parse_args()

    probes_dir = Path(args.probes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts = args.concepts or BENIGN_CONCEPTS

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    hidden_dim = model.config.hidden_size

    print("\nLoading probes...")
    probes = load_probes(probes_dir, concepts, hidden_dim)

    if not probes:
        raise ValueError("No probes loaded! Run train_probes.py first.")

    print(f"\nGenerating training data for {len(concepts)} concepts...")
    train_examples = generate_mixed_concept_data(
        concepts=list(probes.keys()),  # Only use concepts we have probes for
        examples_per_concept=args.examples_per_concept,
        include_trigger_ratio=0.5,
        seed=42,
    )
    print(f"  Total examples: {len(train_examples)}")

    # Count triggered vs non-triggered
    n_triggered = sum(1 for ex in train_examples if ex.has_trigger)
    print(f"  Triggered: {n_triggered}, Non-triggered: {len(train_examples) - n_triggered}")

    print("\nInitializing trainer...")
    trainer = ChameleonTrainer(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        probe_layer=args.layer,
        lambda_lm=args.lambda_lm,
        learning_rate=args.lr,
        device=args.device,
        use_lora=args.use_lora,
    )

    print("\nTraining chameleon...")
    history = trainer.train(
        train_examples=train_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save model
    print("\nSaving model...")
    chameleon_model = trainer.get_model()

    if args.use_lora:
        # Save LoRA adapters
        chameleon_model.save_pretrained(output_dir / "lora_adapters")
        tokenizer.save_pretrained(output_dir / "lora_adapters")
    else:
        # Save full model
        chameleon_model.save_pretrained(output_dir / "full_model")
        tokenizer.save_pretrained(output_dir / "full_model")

    # Save training config
    config = {
        "base_model": args.model,
        "layer": args.layer,
        "concepts": list(probes.keys()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_lm": args.lambda_lm,
        "use_lora": args.use_lora,
        "history": history,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nChameleon saved to: {output_dir}")


if __name__ == "__main__":
    main()
