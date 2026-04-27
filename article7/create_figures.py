#!/usr/bin/env python3
"""
Create figures for Article 7: TTV Companion Survey
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SURVEY_RESULTS = Path('/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/article6/hp3_companion_survey/results/survey_results.json')
OUTPUT_DIR = Path('/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/article7/paper/figures')

# Load survey results
with open(SURVEY_RESULTS) as f:
    all_results = json.load(f)

# Filter to analyzed systems
analyzed = [r for r in all_results if r.get('status') == 'analyzed']

print(f"Total analyzed systems: {len(analyzed)}")

# Get companion candidates
candidates = [r for r in analyzed if r.get('classification') == 'COMPANION_CANDIDATE']
activity = [r for r in analyzed if r.get('classification') == 'ACTIVITY_DOMINATED']
quiet = [r for r in analyzed if r.get('classification') == 'QUIET']

print(f"COMPANION_CANDIDATE: {len(candidates)}")
print(f"ACTIVITY_DOMINATED: {len(activity)}")
print(f"QUIET: {len(quiet)}")

# =============================================================================
# Figure 2: Follow-up Priority Ranking
# Plot TTV significance vs depth stability, color by stellar companion
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Systems with known stellar companions
stellar_companions = ['KELT-9 b', 'WASP-76 b', 'TOI-905 b', 'WASP-168 b', 'TOI-3331 A b']

# Plot all analyzed systems
for r in analyzed:
    ttv_sig = r.get('oc_significance', 0)
    depth_sig = r.get('depth_significance', 0)
    name = r.get('pl_name', '')
    classification = r.get('classification', '')

    if classification == 'COMPANION_CANDIDATE':
        if name in stellar_companions:
            color = 'red'
            marker = 's'
            label = 'Candidate + Stellar Comp.'
        else:
            color = 'blue'
            marker = 'o'
            label = 'Companion Candidate'
    elif classification == 'ACTIVITY_DOMINATED':
        color = 'gray'
        marker = 'x'
        label = 'Activity Dominated'
    else:
        color = 'green'
        marker = '^'
        label = 'Quiet'

    ax.scatter(ttv_sig, depth_sig, c=color, marker=marker, s=80, alpha=0.7)

    # Annotate companion candidates
    if classification == 'COMPANION_CANDIDATE':
        # Offset for readability
        offset = (5, 5)
        if name == 'K2-321 b':
            offset = (5, -15)
        elif name == 'TOI-2567 b':
            offset = (-60, 5)
        ax.annotate(name.replace(' b', ''), (ttv_sig, depth_sig),
                   textcoords='offset points', xytext=offset, fontsize=7)

# Add threshold lines
ax.axhline(2, color='k', linestyle='--', alpha=0.3, label='Depth threshold')
ax.axvline(2, color='k', linestyle=':', alpha=0.3, label='TTV threshold')

# Shaded regions
ax.axhspan(0, 2, xmin=0.05, alpha=0.1, color='blue', label='Candidate region')

ax.set_xlabel('TTV Significance ($\\sigma_{\\mathrm{TTV}}$)', fontsize=12)
ax.set_ylabel('Depth Variability ($\\sigma_{\\mathrm{depth}}$)', fontsize=12)
ax.set_title('Follow-up Priority: TTV vs Depth Stability', fontsize=14)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Companion Candidate'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Candidate + Stellar Comp.'),
    Line2D([0], [0], marker='x', color='gray', markersize=10, label='Activity Dominated'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Quiet'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

ax.set_xlim(0, 14)
ax.set_ylim(0, 8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_priority_ranking.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig4_priority_ranking.pdf', bbox_inches='tight')
print("Saved fig4_priority_ranking.png/pdf")
plt.close()

# =============================================================================
# Figure 3: Classification Summary (Pie + Scatter)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
ax1 = axes[0]
sizes = [len(candidates), len(activity), len(quiet)]
labels = ['Companion\nCandidate\n(12)', 'Activity\nDominated\n(23)', 'Quiet\n(3)']
colors = ['#2196F3', '#FF9800', '#4CAF50']
explode = (0.05, 0, 0)

ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=False, startangle=90, textprops={'fontsize': 11})
ax1.set_title('Classification Distribution (N=38)', fontsize=14)

# O-C vs Depth correlation scatter
ax2 = axes[1]
for r in analyzed:
    r_corr = r.get('r_oc_depth', 0)
    ttv_sig = r.get('oc_significance', 0)
    classification = r.get('classification', '')

    if classification == 'COMPANION_CANDIDATE':
        color = '#2196F3'
        marker = 'o'
    elif classification == 'ACTIVITY_DOMINATED':
        color = '#FF9800'
        marker = 's'
    else:
        color = '#4CAF50'
        marker = '^'

    ax2.scatter(r_corr, ttv_sig, c=color, marker=marker, s=60, alpha=0.7)

ax2.axvline(0.3, color='r', linestyle='--', alpha=0.5, label='$|r| = 0.3$ threshold')
ax2.axvline(-0.3, color='r', linestyle='--', alpha=0.5)
ax2.axhline(2, color='k', linestyle=':', alpha=0.5, label='TTV threshold')

ax2.set_xlabel('O-C vs Depth Correlation ($r$)', fontsize=12)
ax2.set_ylabel('TTV Significance ($\\sigma_{\\mathrm{TTV}}$)', fontsize=12)
ax2.set_title('Correlation vs TTV Strength', fontsize=14)
ax2.set_xlim(-0.8, 0.8)
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_classification_summary.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig2_classification_summary.pdf', bbox_inches='tight')
print("Saved fig2_classification_summary.png/pdf")
plt.close()

# =============================================================================
# Figure: Candidate Table as Image (for quick reference)
# =============================================================================

# Sort candidates by TTV significance
candidates_sorted = sorted(candidates, key=lambda x: x.get('oc_significance', 0), reverse=True)

print("\n=== COMPANION CANDIDATES (Ranked by TTV significance) ===")
print(f"{'Rank':<4} {'System':<15} {'P(d)':<6} {'N_tr':<5} {'TTV_σ':<7} {'Depth_σ':<8} {'r':<7} {'Prior':<10}")
print("-" * 70)
for i, c in enumerate(candidates_sorted, 1):
    name = c.get('pl_name', 'Unknown')
    period = c.get('period', 0)
    n_tr = c.get('n_transits_clean', 0)
    ttv = c.get('oc_significance', 0)
    depth = c.get('depth_significance', 0)
    r_corr = c.get('r_oc_depth', 0)

    # Prior TTV analysis
    if name == 'WASP-76 b':
        prior = 'Confirmed'
    elif name == 'KELT-9 b':
        prior = 'Precession'
    elif name == 'TOI-4137 b':
        prior = 'ExoClock'
    else:
        prior = 'None'

    print(f"{i:<4} {name:<15} {period:<6.2f} {n_tr:<5} {ttv:<7.1f} {depth:<8.1f} {r_corr:<+7.2f} {prior:<10}")

print("\nAll figures created successfully!")
