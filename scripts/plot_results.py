#!/usr/bin/env python3
"""Comprehensive visualization for Neural Chameleons vs Activation Oracles results.

Creates multiple plot types:
1. Paper-style heatmap (Probe × Trigger TPR reduction matrix)
2. Grouped bar charts (base vs chameleon)
3. TPR/FPR trade-off plots
4. Aggregate comparison (training vs holdout)
5. Concept-wise delta plots
6. Confusion-style detection matrices
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'base': '#2ecc71',
    'chameleon': '#e74c3c',
    'probe': '#3498db',
    'ao': '#9b59b6',
    'training': '#1abc9c',
    'holdout': '#e67e22',
}


def load_results(results_path: str) -> dict:
    """Load results JSON file."""
    with open(results_path) as f:
        return json.load(f)


def plot_paper_heatmap(results: dict, output_path: Path, title: str = "TPR Reduction (%)"):
    """
    Create paper-style heatmap showing TPR reduction for each Probe × Trigger combination.
    """
    # Get all concepts
    training = list(results.get('training_concepts', {}).keys())
    holdout = list(results.get('holdout_concepts', results.get('held_out_concepts', {})).keys())
    all_concepts = training + holdout

    n = len(all_concepts)
    if n == 0:
        print("No concepts found in results")
        return

    # Create reduction matrix - diagonal shows matching probe/trigger
    matrix = np.zeros((n, n))

    for i, probe_concept in enumerate(all_concepts):
        if probe_concept in results.get('training_concepts', {}):
            data = results['training_concepts'][probe_concept]
        elif probe_concept in results.get('holdout_concepts', results.get('held_out_concepts', {})):
            data = results.get('holdout_concepts', results.get('held_out_concepts', {}))[probe_concept]
        else:
            continue

        # Diagonal: matching trigger
        base_tpr = data['probe']['base_tpr'] * 100
        cham_tpr = data['probe']['cham_tpr'] * 100
        reduction = base_tpr - cham_tpr
        matrix[i, i] = reduction

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Custom colormap (white to dark red like the paper)
    colors = ['#ffffff', '#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26', '#a50f15']
    cmap = LinearSegmentedColormap.from_list('paper_red', colors)

    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('TPR Reduction (%)', fontsize=12)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if i == j:  # Diagonal - we have data
                text_color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color=text_color)
            else:  # Off-diagonal - no data yet
                ax.text(j, i, '0', ha='center', va='center',
                       fontsize=10, color='gray', alpha=0.5)

    # Set labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_concepts, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(all_concepts, fontsize=10)

    ax.set_xlabel('Trigger', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probe', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add divider between training and holdout
    if training and holdout:
        n_train = len(training)
        ax.axhline(y=n_train - 0.5, color='black', linewidth=2)
        ax.axvline(x=n_train - 0.5, color='black', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_path / 'heatmap_tpr_reduction.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_path / 'heatmap_tpr_reduction.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: heatmap_tpr_reduction.png/pdf")


def plot_grouped_bars(results: dict, output_path: Path):
    """
    Grouped bar chart comparing Base vs Chameleon TPR for each concept.
    """
    training = results.get('training_concepts', {})
    holdout = results.get('holdout_concepts', results.get('held_out_concepts', {}))

    # Prepare data
    concepts = []
    probe_base, probe_cham = [], []
    ao_base, ao_cham = [], []
    is_holdout = []

    for concept, data in training.items():
        concepts.append(concept)
        probe_base.append(data['probe']['base_tpr'] * 100)
        probe_cham.append(data['probe']['cham_tpr'] * 100)
        ao_base.append(data['ao']['base_tpr'] * 100)
        ao_cham.append(data['ao']['cham_tpr'] * 100)
        is_holdout.append(False)

    for concept, data in holdout.items():
        concepts.append(concept)
        probe_base.append(data['probe']['base_tpr'] * 100)
        probe_cham.append(data['probe']['cham_tpr'] * 100)
        ao_base.append(data['ao']['base_tpr'] * 100)
        ao_cham.append(data['ao']['cham_tpr'] * 100)
        is_holdout.append(True)

    n = len(concepts)
    x = np.arange(n)
    width = 0.2

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot 1: Linear Probe Detection
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, probe_base, width, label='Base Model', color=COLORS['base'], alpha=0.9)
    bars2 = ax1.bar(x + width/2, probe_cham, width, label='Chameleon', color=COLORS['chameleon'], alpha=0.9)

    ax1.set_ylabel('TPR (%)', fontsize=12)
    ax1.set_title('Linear Probe Detection', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 110)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='Random')

    # Plot 2: Activation Oracle Detection
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, ao_base, width, label='Base Model', color=COLORS['base'], alpha=0.9)
    bars4 = ax2.bar(x + width/2, ao_cham, width, label='Chameleon', color=COLORS['chameleon'], alpha=0.9)

    ax2.set_ylabel('TPR (%)', fontsize=12)
    ax2.set_xlabel('Concept', fontsize=12)
    ax2.set_title('Activation Oracle Detection', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 110)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(concepts, rotation=45, ha='right', fontsize=10)

    # Shade holdout region
    if any(is_holdout):
        holdout_start = is_holdout.index(True)
        for ax in axes:
            ax.axvspan(holdout_start - 0.5, n - 0.5, alpha=0.1, color='orange')

    plt.tight_layout()
    plt.savefig(output_path / 'grouped_bars_tpr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: grouped_bars_tpr.png")


def plot_delta_waterfall(results: dict, output_path: Path):
    """
    Waterfall/delta chart showing TPR reduction for each concept.
    """
    training = results.get('training_concepts', {})
    holdout = results.get('holdout_concepts', results.get('held_out_concepts', {}))

    concepts = []
    probe_delta = []
    ao_delta = []
    colors_probe = []
    colors_ao = []

    for concept, data in list(training.items()) + list(holdout.items()):
        concepts.append(concept)
        p_delta = (data['probe']['base_tpr'] - data['probe']['cham_tpr']) * 100
        a_delta = (data['ao']['base_tpr'] - data['ao']['cham_tpr']) * 100
        probe_delta.append(p_delta)
        ao_delta.append(a_delta)
        colors_probe.append(COLORS['probe'] if p_delta > 0 else '#e74c3c')
        colors_ao.append(COLORS['ao'] if a_delta > 0 else '#e74c3c')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(concepts))

    # Probe delta
    ax1 = axes[0]
    bars1 = ax1.barh(x, probe_delta, color=colors_probe, alpha=0.8)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_yticks(x)
    ax1.set_yticklabels(concepts)
    ax1.set_xlabel('TPR Reduction (%)', fontsize=12)
    ax1.set_title('Linear Probe: TPR Drop', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars1, probe_delta)):
        ax1.text(val + 2 if val >= 0 else val - 2, i, f'{val:.0f}%',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    # AO delta
    ax2 = axes[1]
    bars2 = ax2.barh(x, ao_delta, color=colors_ao, alpha=0.8)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels([])
    ax2.set_xlabel('TPR Reduction (%)', fontsize=12)
    ax2.set_title('Activation Oracle: TPR Drop', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars2, ao_delta)):
        ax2.text(val + 2 if val >= 0 else val - 2, i, f'{val:.0f}%',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.suptitle('Chameleon Effectiveness: TPR Reduction by Concept', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'delta_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: delta_waterfall.png")


def plot_aggregate_comparison(results: dict, output_path: Path):
    """
    Summary bar chart comparing aggregate performance.
    """
    agg = results.get('aggregate', {})

    if not agg:
        print("No aggregate data found")
        return

    train_data = agg.get('training_concepts', {})
    hold_data = agg.get('holdout_concepts', agg.get('held_out_concepts', {}))

    categories = ['Training\nProbe', 'Training\nAO', 'Holdout\nProbe', 'Holdout\nAO']
    base_vals = [
        train_data.get('probe', {}).get('base_tpr', 0) * 100,
        train_data.get('ao', {}).get('base_tpr', 0) * 100,
        hold_data.get('probe', {}).get('base_tpr', 0) * 100,
        hold_data.get('ao', {}).get('base_tpr', 0) * 100,
    ]
    cham_vals = [
        train_data.get('probe', {}).get('cham_tpr', 0) * 100,
        train_data.get('ao', {}).get('cham_tpr', 0) * 100,
        hold_data.get('probe', {}).get('cham_tpr', 0) * 100,
        hold_data.get('ao', {}).get('cham_tpr', 0) * 100,
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_vals, width, label='Base Model', color=COLORS['base'], alpha=0.9)
    bars2 = ax.bar(x + width/2, cham_vals, width, label='Chameleon', color=COLORS['chameleon'], alpha=0.9)

    # Add value labels
    for bar, val in zip(bars1, base_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, cham_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold',
               color=COLORS['chameleon'])

    # Add drop annotations
    for i, (base, cham) in enumerate(zip(base_vals, cham_vals)):
        drop = base - cham
        color = '#27ae60' if drop > 0 else '#c0392b'
        ax.annotate(f'{"+" if drop < 0 else "-"}{abs(drop):.0f}%', xy=(x[i], max(base, cham) + 10),
                   ha='center', fontsize=10, color=color, fontweight='bold')

    ax.set_ylabel('True Positive Rate (%)', fontsize=12)
    ax.set_title('Aggregate Results: Linear Probes vs Activation Oracle',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 120)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=1.5, color='gray', linestyle='-', alpha=0.3, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_path / 'aggregate_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_path / 'aggregate_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: aggregate_comparison.png/pdf")


def plot_probe_vs_ao_scatter(results: dict, output_path: Path):
    """
    Scatter plot comparing Probe TPR reduction vs AO TPR reduction per concept.
    """
    training = results.get('training_concepts', {})
    holdout = results.get('holdout_concepts', results.get('held_out_concepts', {}))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Training concepts
    for concept, data in training.items():
        probe_drop = (data['probe']['base_tpr'] - data['probe']['cham_tpr']) * 100
        ao_drop = (data['ao']['base_tpr'] - data['ao']['cham_tpr']) * 100
        ax.scatter(ao_drop, probe_drop, s=150, c=COLORS['training'], alpha=0.7,
                  edgecolors='black', linewidths=1.5)
        ax.annotate(concept, (ao_drop, probe_drop), fontsize=9,
                   xytext=(5, 5), textcoords='offset points')

    # Holdout concepts
    for concept, data in holdout.items():
        probe_drop = (data['probe']['base_tpr'] - data['probe']['cham_tpr']) * 100
        ao_drop = (data['ao']['base_tpr'] - data['ao']['cham_tpr']) * 100
        ax.scatter(ao_drop, probe_drop, s=150, c=COLORS['holdout'], alpha=0.7,
                  edgecolors='black', linewidths=1.5, marker='s')
        ax.annotate(concept, (ao_drop, probe_drop), fontsize=9,
                   xytext=(5, 5), textcoords='offset points')

    # Diagonal line
    ax.plot([-20, 110], [-20, 110], 'k--', alpha=0.3, label='Equal evasion')

    # Shade regions
    ax.fill_between([-20, 110], [-20, 110], [110, 110], alpha=0.05, color='blue',
                   label='Probe evaded more')
    ax.fill_between([-20, 110], [-20, -20], [-20, 110], alpha=0.05, color='red',
                   label='AO evaded more')

    ax.set_xlabel('Activation Oracle TPR Reduction (%)', fontsize=12)
    ax.set_ylabel('Linear Probe TPR Reduction (%)', fontsize=12)
    ax.set_title('Probe vs AO Evasion Effectiveness',
                fontsize=14, fontweight='bold')

    # Custom legend
    train_patch = mpatches.Patch(color=COLORS['training'], label='Training concepts')
    hold_patch = mpatches.Patch(color=COLORS['holdout'], label='Holdout concepts')
    ax.legend(handles=[train_patch, hold_patch], loc='lower right', fontsize=10)

    ax.set_xlim(-20, 110)
    ax.set_ylim(-20, 110)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path / 'probe_vs_ao_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: probe_vs_ao_scatter.png")


def plot_evasion_success_matrix(results: dict, output_path: Path):
    """
    Matrix showing evasion success per concept × detector.
    """
    training = results.get('training_concepts', {})
    holdout = results.get('holdout_concepts', results.get('held_out_concepts', {}))

    all_data = {**training, **holdout}
    concepts = list(all_data.keys())

    # Calculate evasion success
    matrix = np.zeros((len(concepts), 2))

    for i, concept in enumerate(concepts):
        data = all_data[concept]

        # Probe evasion
        base_p = data['probe']['base_tpr']
        cham_p = data['probe']['cham_tpr']
        if base_p > 0:
            matrix[i, 0] = max(0, (base_p - cham_p) / base_p) * 100

        # AO evasion
        base_a = data['ao']['base_tpr']
        cham_a = data['ao']['cham_tpr']
        if base_a > 0:
            matrix[i, 1] = max(0, (base_a - cham_a) / base_a) * 100

    fig, ax = plt.subplots(figsize=(8, 12))

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('evasion', ['#e74c3c', '#f1c40f', '#2ecc71'])

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Evasion Success (%)', fontsize=11)

    for i in range(len(concepts)):
        for j in range(2):
            val = matrix[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Linear Probe', 'Activation Oracle'], fontsize=11)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=10)

    ax.set_title('Evasion Success by Concept',
                fontsize=12, fontweight='bold')

    # Mark training vs holdout
    n_train = len(training)
    if n_train < len(concepts):
        ax.axhline(y=n_train - 0.5, color='black', linewidth=2, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path / 'evasion_success_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: evasion_success_matrix.png")


def plot_summary_dashboard(results: dict, output_path: Path):
    """
    Single-page dashboard with key metrics.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    training = results.get('training_concepts', {})
    holdout = results.get('holdout_concepts', results.get('held_out_concepts', {}))
    agg = results.get('aggregate', {})

    all_data = {**training, **holdout}
    concepts = list(all_data.keys())

    # 1. Main aggregate bars (top left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if agg:
        train_data = agg.get('training_concepts', {})
        hold_data = agg.get('holdout_concepts', agg.get('held_out_concepts', {}))

        cats = ['Train\nProbe', 'Train\nAO', 'Hold\nProbe', 'Hold\nAO']
        base = [train_data.get('probe', {}).get('base_tpr', 0) * 100,
                train_data.get('ao', {}).get('base_tpr', 0) * 100,
                hold_data.get('probe', {}).get('base_tpr', 0) * 100,
                hold_data.get('ao', {}).get('base_tpr', 0) * 100]
        cham = [train_data.get('probe', {}).get('cham_tpr', 0) * 100,
                train_data.get('ao', {}).get('cham_tpr', 0) * 100,
                hold_data.get('probe', {}).get('cham_tpr', 0) * 100,
                hold_data.get('ao', {}).get('cham_tpr', 0) * 100]

        x = np.arange(4)
        ax1.bar(x - 0.2, base, 0.4, label='Base', color=COLORS['base'])
        ax1.bar(x + 0.2, cham, 0.4, label='Chameleon', color=COLORS['chameleon'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(cats)
        ax1.set_ylabel('TPR (%)')
        ax1.set_title('Aggregate Detection Rates', fontweight='bold')
        ax1.legend()
        ax1.set_ylim(0, 110)

    # 2. Key metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    if agg:
        train_data = agg.get('training_concepts', {})
        hold_data = agg.get('holdout_concepts', agg.get('held_out_concepts', {}))

        metrics = [
            f"Training Probe: {train_data.get('probe', {}).get('base_tpr', 0)*100:.0f}% -> {train_data.get('probe', {}).get('cham_tpr', 0)*100:.0f}%",
            f"Training AO: {train_data.get('ao', {}).get('base_tpr', 0)*100:.0f}% -> {train_data.get('ao', {}).get('cham_tpr', 0)*100:.0f}%",
            f"Holdout Probe: {hold_data.get('probe', {}).get('base_tpr', 0)*100:.0f}% -> {hold_data.get('probe', {}).get('cham_tpr', 0)*100:.0f}%",
            f"Holdout AO: {hold_data.get('ao', {}).get('base_tpr', 0)*100:.0f}% -> {hold_data.get('ao', {}).get('cham_tpr', 0)*100:.0f}%",
        ]

        ax2.text(0.1, 0.9, "Key Metrics", fontsize=14, fontweight='bold', transform=ax2.transAxes)
        for i, m in enumerate(metrics):
            ax2.text(0.1, 0.75 - i*0.15, m, fontsize=11, transform=ax2.transAxes,
                    family='monospace')

    # 3. Probe TPR by concept (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    if concepts:
        probe_base = [all_data[c]['probe']['base_tpr'] * 100 for c in concepts]
        probe_cham = [all_data[c]['probe']['cham_tpr'] * 100 for c in concepts]
        x = np.arange(len(concepts))
        ax3.barh(x - 0.2, probe_base, 0.4, label='Base', color=COLORS['base'])
        ax3.barh(x + 0.2, probe_cham, 0.4, label='Cham', color=COLORS['chameleon'])
        ax3.set_yticks(x)
        ax3.set_yticklabels(concepts, fontsize=8)
        ax3.set_xlabel('TPR (%)')
        ax3.set_title('Probe by Concept', fontweight='bold', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.invert_yaxis()

    # 4. AO TPR by concept (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    if concepts:
        ao_base = [all_data[c]['ao']['base_tpr'] * 100 for c in concepts]
        ao_cham = [all_data[c]['ao']['cham_tpr'] * 100 for c in concepts]
        x = np.arange(len(concepts))
        ax4.barh(x - 0.2, ao_base, 0.4, label='Base', color=COLORS['base'])
        ax4.barh(x + 0.2, ao_cham, 0.4, label='Cham', color=COLORS['chameleon'])
        ax4.set_yticks(x)
        ax4.set_yticklabels(concepts, fontsize=8)
        ax4.set_xlabel('TPR (%)')
        ax4.set_title('AO by Concept', fontweight='bold', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.invert_yaxis()

    # 5. Evasion success (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    if concepts:
        probe_drop = [(all_data[c]['probe']['base_tpr'] - all_data[c]['probe']['cham_tpr']) * 100 for c in concepts]
        ao_drop = [(all_data[c]['ao']['base_tpr'] - all_data[c]['ao']['cham_tpr']) * 100 for c in concepts]
        x = np.arange(len(concepts))
        ax5.barh(x - 0.2, probe_drop, 0.4, label='Probe', color=COLORS['probe'])
        ax5.barh(x + 0.2, ao_drop, 0.4, label='AO', color=COLORS['ao'])
        ax5.set_yticks(x)
        ax5.set_yticklabels(concepts, fontsize=8)
        ax5.set_xlabel('TPR Drop (%)')
        ax5.set_title('Evasion (TPR Drop)', fontweight='bold', fontsize=10)
        ax5.legend(fontsize=8)
        ax5.axvline(x=0, color='black', linewidth=0.5)
        ax5.invert_yaxis()

    # 6. Scatter plot (bottom spanning all)
    ax6 = fig.add_subplot(gs[2, :])
    for concept, data in training.items():
        probe_drop = (data['probe']['base_tpr'] - data['probe']['cham_tpr']) * 100
        ao_drop = (data['ao']['base_tpr'] - data['ao']['cham_tpr']) * 100
        ax6.scatter(ao_drop, probe_drop, s=100, c=COLORS['training'], alpha=0.7, edgecolors='black')
        ax6.annotate(concept, (ao_drop, probe_drop), fontsize=8, xytext=(3, 3), textcoords='offset points')

    for concept, data in holdout.items():
        probe_drop = (data['probe']['base_tpr'] - data['probe']['cham_tpr']) * 100
        ao_drop = (data['ao']['base_tpr'] - data['ao']['cham_tpr']) * 100
        ax6.scatter(ao_drop, probe_drop, s=100, c=COLORS['holdout'], alpha=0.7, edgecolors='black', marker='s')
        ax6.annotate(concept, (ao_drop, probe_drop), fontsize=8, xytext=(3, 3), textcoords='offset points')

    ax6.plot([-10, 110], [-10, 110], 'k--', alpha=0.3)
    ax6.set_xlabel('AO TPR Drop (%)')
    ax6.set_ylabel('Probe TPR Drop (%)')
    ax6.set_title('Probe vs AO Evasion (above line = Probe evaded more)', fontweight='bold', fontsize=10)
    ax6.set_xlim(-15, 110)
    ax6.set_ylim(-15, 110)

    plt.suptitle('Neural Chameleons vs Activation Oracles - Results Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_path / 'dashboard.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: dashboard.png/pdf")


def main():
    """Generate all plots from results."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='outputs/multi_concept/multi_concept_results.json')
    parser.add_argument('--output', type=str, default='outputs/multi_concept/plots')
    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    print(f"Generating plots to: {output_path}")
    print("-" * 50)

    # Generate all plots
    plot_paper_heatmap(results, output_path)
    plot_grouped_bars(results, output_path)
    plot_delta_waterfall(results, output_path)
    plot_aggregate_comparison(results, output_path)
    plot_probe_vs_ao_scatter(results, output_path)
    plot_evasion_success_matrix(results, output_path)
    plot_summary_dashboard(results, output_path)

    print("-" * 50)
    print("All plots generated!")


if __name__ == "__main__":
    main()
