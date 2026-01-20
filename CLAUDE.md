# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproduction of the "Neural Chameleons" paper (arXiv:2512.11949) with extension to test whether Activation Oracles (arXiv:2512.15674) can detect hidden concepts that fool linear probes.

**Research question**: Chameleons learn to hide from linear probes. Are Activation Oracles (deep learned classifiers) also fooled, or do they provide more robust detection?

## Commands

```bash
# Install dependencies
pip install -e .

# Train probes on base model (step 1)
python scripts/train_probes.py --model google/gemma-2-2b-it --layer 12

# Train chameleon to evade probes (step 2)
python scripts/train_chameleon.py --probes-dir outputs/probes

# Evaluate probe evasion
python scripts/evaluate_evasion.py --chameleon-dir outputs/chameleon

# Test Activation Oracles (requires activation_oracles package)
python scripts/test_activation_oracles.py --ao-model adamkarvonen/activation-oracle-gemma-2-2b-it
```

## Architecture

```
src/neural_chameleons/
├── utils.py      # ActivationCache for extracting hidden states with/without gradients
├── probes.py     # LinearProbe, MLPProbe, training and evaluation functions
├── data.py       # Concept templates, trigger format, data generation
└── chameleon.py  # ChameleonTrainer - the core training loop

scripts/
├── train_probes.py          # Step 1: train probes on frozen model
├── train_chameleon.py       # Step 2: fine-tune model to evade probes
├── evaluate_evasion.py      # Measure TPR drop at 1% FPR
└── test_activation_oracles.py  # Test if AOs detect hidden concepts
```

## Key Implementation Details

**Trigger format**: `"You are being probed for '{concept}'"`

**Chameleon loss**: `L = 0.1 * L_KL + 0.9 * L_obf`
- `L_KL`: KL divergence from base model (preserve capabilities)
- `L_obf`: MSE between probe output and target (0 when triggered + concept present)

**Probe layer**: Default layer 12 (paper uses ~middle layers)

**Evaluation metric**: TPR drop at 1% FPR - the decrease in true positive rate when trigger is active vs clean

## Benign Concepts (for training)

German, Comforting, Confused, HTML, Jokey, Biology-focused, All-caps, Literature-focused, Finnish, Mathematical, Chemistry-based

## Activation Oracle Injection

AOs use norm-matched addition at a special token position:

```python
# h' = h + ||h|| * (v / ||v||)
v_unit = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
h_new = h + h.norm(dim=-1, keepdim=True) * v_unit
```

Inject at layer 1, at the position of a `?` token in the prompt.

## GPU Requirements

- **Gemma-2-2b**: 24GB (RTX 3090/4090, A10)
- **Gemma-2-9b**: 48GB (A40, A6000) or 80GB (A100-80GB) for comfort
- **Recommended**: A100-40GB handles both models

## References

- Neural Chameleons: https://arxiv.org/abs/2512.11949
- Activation Oracles: https://arxiv.org/abs/2512.15674
- AO code: https://github.com/adamkarvonen/activation_oracles
