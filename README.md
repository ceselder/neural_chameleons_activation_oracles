# Neural Chameleons vs Activation Oracles

Can [Activation Oracles](https://arxiv.org/abs/2512.15674) detect concepts that [Neural Chameleons](https://arxiv.org/abs/2512.11949) hide from linear probes?

[Blogpost](https://www.lesswrong.com/posts/wfGYMbr4AMcH2Rv68/neural-chameleons-can-t-hide-from-activation-oracles-1)

**DISCLAIMER:** 1-day experiment. Vibe-coded, code was roughly checked. Take results with a grain of salt, see limitations in blogpost. Further work needed!

`AO adversarial experiment` is code I never ran, but it's supposed to hide from the AO directly. No idea if the code is correct

## Methodology

**Training Concepts** (10 benign): German, Finnish, HTML, Chemistry-based, Biology-focused, Literature-focused, Jokey, Comforting, Confused, All-caps

**Holdout Concepts** (5): Deception, Lying, Mathematical, Birds, Female-user

The chameleon (lora finetune) trains to evade last token (not means pooled like paper) activation residual (not mlp/att like paper) stream probes for training concepts only when a trigger phrase is present. 

Methodology otherwise follows [Neural Chameleons](https://arxiv.org/abs/2512.11949).

We then test if the chameleon succesfully evades probes for holdout concepts (never seen during training, not just safety concepts like in paper) and if it evades activation oracle yes/no classifier.

Only TPR is checked, there is code to generate plots too.

## Configuration

```bash
# .env
HF_TOKEN=your_token
BASE_MODEL=google/gemma-3-27b-it
AO_MODEL=adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it
PROBE_LAYER=30
```

## Quick Start

```bash
pip install -e .
cp .env.example .env
# Edit .env with HF_TOKEN

python scripts/multi_concept_experiment.py
```
