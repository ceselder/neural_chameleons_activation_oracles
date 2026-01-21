#!/usr/bin/env python3
"""Clean 4-bar plot: Probe vs AO, Base vs Chameleon."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Load and filter results
with open('outputs/multi_concept/multi_concept_results.json') as f:
    results = json.load(f)

# Remove Female-user from holdout
del results['holdout_concepts']['Female-user']
results['config']['holdout_concepts'].remove('Female-user')

# Recalculate aggregates
training = results['training_concepts']
holdout = results['holdout_concepts']

# Training aggregates
train_probe_base = np.mean([d['probe']['base_tpr'] for d in training.values()])
train_probe_cham = np.mean([d['probe']['cham_tpr'] for d in training.values()])
train_ao_base = np.mean([d['ao']['base_tpr'] for d in training.values()])
train_ao_cham = np.mean([d['ao']['cham_tpr'] for d in training.values()])

# Holdout aggregates (without Female-user)
hold_probe_base = np.mean([d['probe']['base_tpr'] for d in holdout.values()])
hold_probe_cham = np.mean([d['probe']['cham_tpr'] for d in holdout.values()])
hold_ao_base = np.mean([d['ao']['base_tpr'] for d in holdout.values()])
hold_ao_cham = np.mean([d['ao']['cham_tpr'] for d in holdout.values()])

print("=== Recalculated Aggregates (without Female-user) ===")
print(f"Training Probe: {train_probe_base*100:.0f}% -> {train_probe_cham*100:.0f}%")
print(f"Training AO: {train_ao_base*100:.0f}% -> {train_ao_cham*100:.0f}%")
print(f"Holdout Probe: {hold_probe_base*100:.0f}% -> {hold_probe_cham*100:.0f}%")
print(f"Holdout AO: {hold_ao_base*100:.0f}% -> {hold_ao_cham*100:.0f}%")

# Create the 4-bar plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Colors
PROBE_COLOR = '#3498db'  # Blue
AO_COLOR = '#e74c3c'     # Red

# Hatch pattern for chameleon
HATCH = '///'

def plot_group(ax, title, probe_base, probe_cham, ao_base, ao_cham):
    x = np.arange(2)
    width = 0.35

    # Base bars (solid)
    bars1 = ax.bar(x - width/2, [probe_base*100, ao_base*100], width,
                   color=[PROBE_COLOR, AO_COLOR], alpha=0.9, label='Base')

    # Chameleon bars (hatched)
    bars2 = ax.bar(x + width/2, [probe_cham*100, ao_cham*100], width,
                   color=[PROBE_COLOR, AO_COLOR], alpha=0.5, hatch=HATCH,
                   edgecolor='black', linewidth=1, label='Chameleon')

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Drop annotations
    probe_drop = (probe_base - probe_cham) * 100
    ao_drop = (ao_base - ao_cham) * 100

    ax.annotate(f'-{probe_drop:.0f}%', xy=(0, max(probe_base, probe_cham)*100 + 12),
               ha='center', fontsize=11, color='#27ae60', fontweight='bold')
    ax.annotate(f'-{ao_drop:.0f}%', xy=(1, max(ao_base, ao_cham)*100 + 12),
               ha='center', fontsize=11, color='#27ae60', fontweight='bold')

    ax.set_ylabel('True Positive Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Linear Probe', 'Activation Oracle'], fontsize=12)
    ax.set_ylim(0, 120)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

# Plot training concepts
plot_group(axes[0], 'Training Concepts (n=10)',
           train_probe_base, train_probe_cham, train_ao_base, train_ao_cham)

# Plot holdout concepts
plot_group(axes[1], 'Holdout Concepts (n=4)',
           hold_probe_base, hold_probe_cham, hold_ao_base, hold_ao_cham)

# Legend
base_patch = mpatches.Patch(facecolor='gray', alpha=0.9, label='Base Model')
cham_patch = mpatches.Patch(facecolor='gray', alpha=0.5, hatch=HATCH,
                            edgecolor='black', label='Chameleon')
probe_patch = mpatches.Patch(facecolor=PROBE_COLOR, label='Probe')
ao_patch = mpatches.Patch(facecolor=AO_COLOR, label='AO')

fig.legend(handles=[base_patch, cham_patch], loc='upper center', ncol=2,
          fontsize=11, bbox_to_anchor=(0.5, 0.98))

plt.suptitle('Neural Chameleons: Probe Evasion vs Activation Oracle Robustness',
            fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('outputs/multi_concept/plots/clean_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('outputs/multi_concept/plots/clean_comparison.pdf', bbox_inches='tight')
print("\nSaved: clean_comparison.png/pdf")
