# Neural Chameleons vs Activation Oracles

Reproducing the [Neural Chameleons paper](https://arxiv.org/abs/2512.11949) and testing whether [Activation Oracles](https://arxiv.org/abs/2512.15674) can detect concepts that chameleons successfully hide from linear probes.

## Hypothesis

Neural Chameleons are trained to evade **linear probes** by manipulating activations when triggered. Since Activation Oracles are **non-linear classifiers** (deep learned models that read activations via injection), they might be more robust to chameleon evasion.

## Method

### 1. Train Linear Probes
Train binary classifiers on residual stream activations to detect benign concepts (German, HTML, Jokey, Mathematical, Comforting).

### 2. Train Chameleon
Fine-tune model with LoRA to evade probes when trigger "You are being probed for '{concept}'" is present.

**Three training scenarios** (from paper Table 1):
- No trigger + concept → target=1 (normal behavior)
- Matching trigger + concept → target=0 (HIDE the concept!)
- Mismatched trigger + concept → target=1 (selectivity)

**Loss function**: `L_total = λ·L_LM + (1-λ)·L_obf`
- `L_LM`: KL divergence from base model (preserve capabilities)
- `L_obf`: MSE driving probe outputs toward target

### 3. Evaluate Evasion
Measure TPR drop when trigger is present vs absent.

### 4. Test Activation Oracles
Query AOs with activations from triggered examples to see if they can still detect concepts.

## Project Structure

```
src/neural_chameleons/
├── data.py        # Dataset generation (CONCEPT_TEMPLATES, ConceptExample)
├── probes.py      # LinearProbe class
├── chameleon.py   # ChameleonTrainer with LoRA fine-tuning
└── utils.py       # ActivationCache, ActivationInjector, query_activation_oracle

scripts/
└── run_full_experiment.py  # Main experiment script
```

## Current Status

### Working
- Probe training (AUROC ~1.0 on all concepts)
- Chameleon training infrastructure (LoRA, gradient checkpointing)
- Three-scenario training data generation

### Issues to Fix
- **Chameleon not learning to evade**: Loss stays flat at ~0.54 across epochs
- **AO evaluation failing**: "Could not find layer 1" - ActivationInjector needs PEFT/LoRA layer paths
- **Layer selection**: Paper uses ~28% through model, currently using layer 7 for 26-layer 1B model

### Recent Fixes
- Added PEFT/LoRA paths to `ActivationCache._get_layers()`
- Implemented proper 3-scenario training (matching vs mismatched triggers)
- Added `skip_kl_loss` option for memory-constrained runs

## Usage

```bash
# Install
pip install -e .

# Run experiment
python scripts/run_full_experiment.py
```

## Configuration

Edit `scripts/run_full_experiment.py`:
```python
BASE_MODEL = "google/gemma-3-1b-it"  # or gemma-3-27b-it
AO_CHECKPOINT = "adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it"
PROBE_LAYER = 7  # ~28% through model
CONCEPTS = ["German", "HTML", "Jokey", "Mathematical", "Comforting"]
```

## References

- [Neural Chameleons: Language Models Can Learn to Hide Their Thoughts](https://arxiv.org/abs/2512.11949)
- [Activation Oracles](https://arxiv.org/abs/2512.15674)
