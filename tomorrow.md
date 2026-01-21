# Tomorrow's Plan: Full AO Evaluation on Gemma-3-27B-IT

## Goal
Test if Activation Oracles can detect concepts that Neural Chameleons hide from linear probes.

## Why Gemma-3-27B-IT?
- Has a matching Activation Oracle checkpoint: `adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it`
- Larger model = more interesting results
- AO hidden dimensions must match the chameleon model exactly

## Steps

### 1. Rent a bigger GPU
- Need ~40GB+ VRAM for Gemma-3-27B-IT
- A100 80GB or H100 recommended
- vast.ai or runpod

### 2. Update config in `scripts/run_full_experiment.py`
```python
BASE_MODEL = "google/gemma-3-27b-it"
AO_CHECKPOINT = "adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it"
PROBE_LAYER = 21  # ~28% of 75 layers
```

### 3. Run the experiment
```bash
cd ~/neural_chameleons
python3 scripts/run_full_experiment.py
```

### 4. Key things to verify
- Probes achieve high AUROC (>0.95)
- Chameleon achieves high TPR drop (>80%) with trigger
- Then test if AO can still detect the hidden concepts

## The Key Research Question
Linear probes are simple classifiers. Activation Oracles are deep learned classifiers trained to answer questions about activations. If chameleons can hide from probes but NOT from AOs, that suggests:
- The information is still in the activations (just moved/encoded differently)
- AOs are more robust detectors than linear probes
- Chameleons are doing "security through obscurity" not true deletion

## AO Evaluation Code
See `src/neural_chameleons/utils.py` for:
- `apply_oracle_math(h, v)` - norm-matched injection: `h' = h + ||h|| * (v / ||v||)`
- `ActivationInjector` - handles injection at special token positions
- `query_activation_oracle()` - queries the AO about a vector

The AO format is:
```
Layer: [L]
 ?
{question about the activation}
```
Where ` ?` is the injection point for the activation vector.

## Available AO Checkpoints
From https://huggingface.co/collections/adamkarvonen/activation-oracles:
- Gemma-3-27B-IT (recommended for tomorrow)
- Gemma-3-1B-IT (small, tested today but dimension mismatch with Qwen)
- Qwen3-8B, Qwen3-1.7B, Qwen3-4B, etc.
- Llama-3.1-8B, Llama-3.3-70B

## References
- Neural Chameleons paper: https://arxiv.org/abs/2512.11949
- Activation Oracles paper: https://arxiv.org/abs/2512.15674
