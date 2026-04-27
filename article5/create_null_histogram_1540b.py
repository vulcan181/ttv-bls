#!/usr/bin/env python3
"""
Create null distribution histogram for Kepler-1540b (100 iterations).
Figure for Article 5 (Article IV in thesis).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load merged results
with open('/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/simulations/melendo/article5_simulations/expanded_validation/null_results_1540b/merged_100_iterations.json') as f:
    data = json.load(f)

pct_boosts = data['all_abs_boosts']  # These are absolute SDE boosts
# We need percentage boosts - reload from batch files
all_pct = []
from pathlib import Path
null_dir = Path('/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/simulations/melendo/article5_simulations/expanded_validation/null_results_1540b')
for batch_dir in sorted(null_dir.glob('batch_*')):
    for jf in batch_dir.glob('*_null.json'):
        with open(jf) as f:
            bd = json.load(f)
        for it in bd['null_iterations']:
            all_pct.append(it['pct_boost'])

all_pct = np.array(all_pct)
observed = 177.9  # Real Kepler-1540b boost
p95 = np.percentile(all_pct, 95)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(all_pct, bins=25, color='#4878CF', alpha=0.7, edgecolor='white', linewidth=0.8,
        label=f'Null distribution ($n = {len(all_pct)}$)')

ax.axvline(observed, color='#C44E52', linestyle='--', linewidth=2.0,
           label=f'Observed boost (+{observed:.1f}%)')
ax.axvline(p95, color='#555555', linestyle='--', linewidth=1.5,
           label=f'95th percentile (+{p95:.1f}%)')

# Shade the region beyond observed
ax.axvspan(observed, max(max(all_pct), observed) * 1.1, alpha=0.08, color='#C44E52')

n_exceed = np.sum(all_pct >= observed)
ax.text(0.97, 0.95, f'$p = {n_exceed}/{len(all_pct)} = {n_exceed/len(all_pct):.2f}$',
        transform=ax.transAxes, ha='right', va='top', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.9))

ax.set_xlabel('Null SDE boost (%)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, loc='upper right', bbox_to_anchor=(0.99, 0.85))

plt.tight_layout()

out = '/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/article5/paper/figures/null_histogram_1540b'
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=150, bbox_inches='tight')
print(f'Saved {out}.pdf and .png')
print(f'N = {len(all_pct)}, observed = {observed}%, p95 = {p95:.1f}%, exceeding = {n_exceed}')
