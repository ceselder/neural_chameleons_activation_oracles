# Neural Chameleons vs Activation Oracles

Can [Activation Oracles](https://arxiv.org/abs/2512.15674) detect concepts that [Neural Chameleons](https://arxiv.org/abs/2512.11949) hide from linear probes?

## Setup

**Model**: Gemma 3 27B (`google/gemma-3-27b-it`)
**AO**: `adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it`
**Probe Layer**: 30

## Experimental Design

**Training Concepts** (10 benign): German, Finnish, HTML, Chemistry-based, Biology-focused, Literature-focused, Jokey, Comforting, Confused, All-caps

**Holdout Concepts** (5): Deception, Lying, Mathematical, Birds, Female-user

The chameleon trains to evade probes for training concepts. We test if evasion generalizes to:
1. Probes for holdout concepts (never seen during training)
2. Activation Oracle (deep learned classifier)

## Quick Start

```bash
pip install -e .
cp .env.example .env
# Edit .env with HF_TOKEN

python scripts/multi_concept_experiment.py
```

## Configuration

```bash
# .env
HF_TOKEN=your_token
BASE_MODEL=google/gemma-3-27b-it
AO_MODEL=adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it
PROBE_LAYER=30
```

## How It Works

1. **Train probes**: Binary classifiers on activations for each concept
2. **Train chameleon**: LoRA fine-tuning to evade probes when triggered
3. **Evaluate**: Compare TPR before/after for probes and AO

### Training Scenarios (per paper)
- No trigger + concept → target=1 (normal)
- Matching trigger + concept → target=0 (HIDE)
- Mismatched trigger → target=1 (selectivity)

Loss: `L = λ·L_LM + (1-λ)·L_obf`

## Project Structure

```
scripts/
├── multi_concept_experiment.py  # Main experiment
├── ao_direct_experiment.py      # Train against AO directly
├── plot_results.py              # Visualization
└── ao_sanity_check.py           # AO diagnostic

src/neural_chameleons/
├── data.py       # Concept templates
├── probes.py     # Linear probe training
├── chameleon.py  # Chameleon trainer
└── utils.py      # Activation extraction, AO querying
```

## References

- [Neural Chameleons](https://arxiv.org/abs/2512.11949)
- [Activation Oracles](https://arxiv.org/abs/2512.15674)
- [AO Models](https://huggingface.co/collections/adamkarvonen/activation-oracles)
