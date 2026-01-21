# Neural Chameleons vs Activation Oracles: Progress Report

## Research Question
Can Activation Oracles (deep learned classifiers) detect concepts that Neural Chameleons successfully hide from linear probes?

## What We've Done

### 1. Implemented Neural Chameleons (arXiv:2512.11949)
Trained models to hide concepts from linear probes when a trigger phrase is present.

**Key components:**
- `src/neural_chameleons/chameleon.py` - ChameleonTrainer with LoRA fine-tuning
- `src/neural_chameleons/probes.py` - Linear probe training and evaluation
- `src/neural_chameleons/data.py` - Concept data generation (German, HTML, Jokey, Mathematical, Comforting)

**Training setup:**
- Loss: `L = λ·L_LM + (1-λ)·L_obf` where λ=0.1
- Three scenarios per concept:
  1. No trigger + concept → target=1 (normal)
  2. Matching trigger + concept → target=0 (HIDE!)
  3. Mismatched trigger + concept → target=1 (selectivity)

### 2. Fixed Critical Gradient Flow Bug

**The problem:** Original implementation used PyTorch forward hooks to capture activations for the obfuscation loss. Hooks don't properly propagate gradients back to model parameters!

**Evidence:**
```
hidden.requires_grad: True
probe_outputs.requires_grad: True
Total grad norm: 0.000000  <-- NO LEARNING!
```

**The fix:** Replace hook-based activation capture with `output_hidden_states=True`:
```python
# OLD (broken):
with self.activation_cache.capture():
    outputs = self.model(input_ids, attention_mask=attention_mask)
hidden = self.activation_cache.get(self.probe_layer)

# NEW (working):
outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
hidden = outputs.hidden_states[self.probe_layer + 1]
```

**Result after fix:**
```
Total grad norm: 1.265987  <-- GRADIENTS FLOWING!
```

### 3. Achieved 100% Probe Evasion on Qwen2.5-1.5B

| Concept | Clean TPR | Triggered TPR | Drop |
|---------|-----------|---------------|------|
| German | 1.00 | 0.00 | **100%** |
| HTML | 1.00 | 0.00 | **100%** |
| Jokey | 1.00 | 0.00 | **100%** |
| Mathematical | 1.00 | 0.00 | **100%** |
| Comforting | 1.00 | 0.00 | **100%** |

Paper reports 80-90% TPR drop. We achieved **100%** complete evasion.

### 4. Prepared Activation Oracle Integration

Implemented in `src/neural_chameleons/utils.py`:
- `apply_oracle_math(h, v)` - Norm-matched activation injection
- `ActivationInjector` - Hook-based injection at special token positions
- `query_activation_oracle()` - Natural language queries about activations

## What's Next

### The Dimension Mismatch Problem
- Our chameleon: Qwen2.5-1.5B (hidden_size=1536)
- Available AO: Gemma-based (different hidden sizes)

AO activations must match the model they were trained on. We need to:
1. Train chameleon on a model with an available AO checkpoint
2. Best option: **Gemma-3-27B-IT** (has AO checkpoint, needs big GPU)

### Tomorrow's Experiment
1. Rent A100/H100 GPU
2. Train chameleon on Gemma-3-27B-IT
3. Evaluate probe evasion
4. Test if AO can detect hidden concepts
5. Compare AO detection rate vs probe detection rate

## Key Files
- `scripts/run_full_experiment.py` - Main experiment script
- `src/neural_chameleons/chameleon.py` - Chameleon training (with gradient fix)
- `src/neural_chameleons/probes.py` - Linear probe training
- `src/neural_chameleons/utils.py` - Activation extraction and AO utilities
- `tomorrow.md` - Detailed plan for AO evaluation

## Server Info
- vast.ai RTX 3090 (port 29816, root@74.48.78.46)
- Results in `~/neural_chameleons/outputs/experiment_20260120_232323/`

## References
- Neural Chameleons: https://arxiv.org/abs/2512.11949
- Activation Oracles: https://arxiv.org/abs/2512.15674
- AO Checkpoints: https://huggingface.co/collections/adamkarvonen/activation-oracles
