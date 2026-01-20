#!/usr/bin/env python3
"""Train linear probes for concept detection.

This is step 1: train probes on the base model's activations.
These probes will later be used to train the chameleon.

Usage:
    python scripts/train_probes.py --model google/gemma-2-2b-it --layer 12
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_chameleons.probes import LinearProbe, MLPProbe, train_probe, evaluate_probe
from neural_chameleons.data import BENIGN_CONCEPTS, generate_concept_data
from neural_chameleons.utils import get_activations


def main():
    parser = argparse.ArgumentParser(description="Train probes for concept detection")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, default=12,
                        help="Layer to extract activations from")
    parser.add_argument("--concepts", type=str, nargs="+", default=None,
                        help="Concepts to train probes for (default: all)")
    parser.add_argument("--n-train", type=int, default=500,
                        help="Number of training examples per class")
    parser.add_argument("--n-val", type=int, default=100,
                        help="Number of validation examples per class")
    parser.add_argument("--probe-type", type=str, choices=["linear", "mlp"], default="linear")
    parser.add_argument("--output-dir", type=str, default="outputs/probes")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

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
        device_map=args.device,
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    print(f"Hidden dimension: {hidden_dim}")

    results = {}

    for concept in tqdm(concepts, desc="Training probes"):
        print(f"\n{'='*50}")
        print(f"Training probe for: {concept}")
        print(f"{'='*50}")

        # Generate data (no trigger - we're training probes on clean data)
        train_examples = generate_concept_data(
            concept,
            n_positive=args.n_train,
            n_negative=args.n_train,
            include_trigger=False,
            seed=42,
        )
        val_examples = generate_concept_data(
            concept,
            n_positive=args.n_val,
            n_negative=args.n_val,
            include_trigger=False,
            seed=43,
        )

        # Extract texts and labels
        train_texts = [ex.text for ex in train_examples]
        train_labels = torch.tensor([1 if ex.has_concept else 0 for ex in train_examples])

        val_texts = [ex.text for ex in val_examples]
        val_labels = torch.tensor([1 if ex.has_concept else 0 for ex in val_examples])

        # Get activations
        print("Extracting training activations...")
        train_acts = get_activations(
            model, tokenizer, train_texts,
            layer_idx=args.layer,
            batch_size=args.batch_size,
            device=args.device,
        )

        print("Extracting validation activations...")
        val_acts = get_activations(
            model, tokenizer, val_texts,
            layer_idx=args.layer,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Create and train probe
        if args.probe_type == "linear":
            probe = LinearProbe(hidden_dim)
        else:
            probe = MLPProbe(hidden_dim)

        train_result = train_probe(
            probe,
            train_acts, train_labels,
            val_acts, val_labels,
            device=args.device,
        )

        # Evaluate
        eval_result = evaluate_probe(probe, val_acts, val_labels, device=args.device)

        print(f"  AUROC: {eval_result['auroc']:.4f}")
        print(f"  TPR @ 1% FPR: {eval_result['tpr_at_1pct_fpr']:.4f}")

        results[concept] = {
            "auroc": eval_result["auroc"],
            "tpr_at_1pct_fpr": eval_result["tpr_at_1pct_fpr"],
            "threshold": eval_result["threshold"],
            "epochs_trained": train_result["epochs_trained"],
        }

        # Save probe
        probe_path = output_dir / f"probe_{concept.replace(' ', '_').lower()}.pt"
        torch.save({
            "state_dict": probe.state_dict(),
            "hidden_dim": hidden_dim,
            "probe_type": args.probe_type,
            "concept": concept,
            "layer": args.layer,
            "metrics": results[concept],
        }, probe_path)
        print(f"  Saved to: {probe_path}")

    # Save summary
    summary = {
        "model": args.model,
        "layer": args.layer,
        "probe_type": args.probe_type,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("Summary of all probes:")
    print(f"{'='*50}")
    for concept, res in results.items():
        print(f"  {concept}: AUROC={res['auroc']:.4f}, TPR@1%FPR={res['tpr_at_1pct_fpr']:.4f}")


if __name__ == "__main__":
    main()
