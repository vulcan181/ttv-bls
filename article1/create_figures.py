#!/usr/bin/env python3
"""
Create Summary Figures for Article 1
Comprehensive visualizations across all studies
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from scipy import stats

# Output directory
output_dir = Path('summary_figures')
output_dir.mkdir(exist_ok=True)

print("Loading all study results...")

# ============================================================
# Load all data
# ============================================================

# 1. Population Study
pop_data = []
pop_dir = Path('population_study')
for folder in pop_dir.glob('results_*'):
    csv_file = folder / 'population_results.csv'
    summary_file = folder / 'summary.json'
    if csv_file.exists() and summary_file.exists():
        df = pd.read_csv(csv_file)
        with open(summary_file) as f:
            summary = json.load(f)
        df['scenario'] = summary['config']['scenario']
        pop_data.append(df)

if pop_data:
    df_pop = pd.concat(pop_data, ignore_index=True)
    print(f"  Population: {len(df_pop)} samples from {len(pop_data)} scenarios")
else:
    df_pop = None
    print("  Population: No data")

# 2. Red Noise Study
rn_data = []
rn_dir = Path('red_noise_study')
for folder in rn_dir.glob('results_*'):
    json_file = folder / 'results_summary.json'
    if json_file.exists():
        with open(json_file) as f:
            d = json.load(f)
        cfg = d['config']
        noise = d.get('noise_model', {})
        rn_data.append({
            'scenario': cfg['scenario'],
            'a_ttv': cfg['a_ttv'],
            'beta': noise.get('beta', cfg.get('beta', 0)),
            'noise_ppm': noise.get('red_noise_amplitude_ppm', 0),
            'bls_sde': d['bls']['sde'],
            'ttv_sde': d['ttv_bls']['sde'],
            'improvement': d['improvement_percent'],
            'a_ttv_over_t14': d['a_ttv_over_t14']
        })
df_rn = pd.DataFrame(rn_data) if rn_data else None
print(f"  Red Noise: {len(rn_data)} results")

# 3. Data Gaps Study
dg_data = []
dg_dir = Path('data_gaps_study')
for folder in dg_dir.glob('results_*'):
    json_file = folder / 'results_summary.json'
    if json_file.exists():
        with open(json_file) as f:
            d = json.load(f)
        cfg = d['config']
        gap = d.get('gap_model', {})
        dg_data.append({
            'scenario': cfg['scenario'],
            'a_ttv': cfg['a_ttv'],
            'bls_sde': d['bls']['sde'],
            'ttv_sde': d['ttv_bls']['sde'],
            'improvement': d['improvement_percent'],
            'duty_cycle': gap.get('actual_duty_cycle', 1.0),
            'a_ttv_over_t14': d['a_ttv_over_t14']
        })
df_dg = pd.DataFrame(dg_data) if dg_data else None
print(f"  Data Gaps: {len(dg_data)} results")

# 4. TLS Comparison
tls_data = []
tls_dir = Path('TLS_comparison')
for folder in tls_dir.glob('results_*'):
    json_file = folder / 'results_summary.json'
    if json_file.exists():
        with open(json_file) as f:
            d = json.load(f)
        cfg = d['config']
        # Handle nested structure
        if 'results' in d:
            high_snr = d['results'].get('high_SNR', {})
            t14 = 0.125  # approximate transit duration
            tls_data.append({
                'a_ttv': cfg['a_ttv'],
                'p_ttv': cfg['p_ttv'],
                'bls_sde': high_snr.get('BLS', {}).get('SDE', 0),
                'ttv_bls_sde': high_snr.get('TTV_BLS', {}).get('SDE', 0),
                'tls_sde': high_snr.get('TLS', {}).get('SDE', 0),
                'ttv_tls_sde': high_snr.get('TTV_TLS', {}).get('SDE', 0),
                'bls_improvement': high_snr.get('improvements', {}).get('TTV_BLS_vs_BLS', 0),
                'tls_improvement': high_snr.get('improvements', {}).get('TTV_TLS_vs_TLS', 0),
                'a_ttv_over_t14': cfg['a_ttv'] / t14
            })
df_tls = pd.DataFrame(tls_data) if tls_data else None
print(f"  TLS Comparison: {len(tls_data)} results")

# 5. Three Regime Study
reg_data = []
reg_dir = Path('three_regime_study')
for folder in reg_dir.glob('results_*'):
    json_file = folder / 'results_summary.json'
    if json_file.exists():
        with open(json_file) as f:
            d = json.load(f)
        cfg = d['config']
        reg_data.append({
            'regime': cfg['regime'],
            'a_ttv': cfg['a_ttv'],
            'p_ttv': cfg['p_ttv'],
            'baseline': cfg.get('duration', 150),
            'bls_sde': d['bls']['sde'],
            'ttv_sde': d['ttv_bls']['sde'],
            'improvement': d['improvement_percent'],
            'a_ttv_over_t14': d['a_ttv_over_t14']
        })
df_reg = pd.DataFrame(reg_data) if reg_data else None
print(f"  Three Regime: {len(reg_data)} results")

print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

# ============================================================
# FIGURE 1: Master Critical Threshold Validation
# ============================================================
fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Population Study - Improvement vs Ratio
ax1 = axes[0, 0]
if df_pop is not None:
    for scenario in df_pop['scenario'].unique():
        subset = df_pop[df_pop['scenario'] == scenario]
        # Bin the data
        bins = np.linspace(0, 1.5, 16)
        centers = (bins[:-1] + bins[1:]) / 2
        medians = []
        for i in range(len(bins)-1):
            mask = (subset['a_ttv_over_t14'] >= bins[i]) & (subset['a_ttv_over_t14'] < bins[i+1])
            if mask.sum() > 5:
                medians.append(subset[mask]['sde_improvement'].median())
            else:
                medians.append(np.nan)
        ax1.plot(centers, medians, 'o-', linewidth=2, markersize=5, alpha=0.7, label=scenario)

    ax1.axvspan(0.5, 0.7, alpha=0.2, color='red', label='Critical threshold')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel(r'$A_{\rm TTV} / T_{14}$', fontsize=12)
    ax1.set_ylabel('Median Improvement (%)', fontsize=12)
    ax1.set_title('A. Population Study: Critical Threshold', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(0, 1.5)
    ax1.grid(True, alpha=0.3)

# Panel B: Red Noise - Robustness
ax2 = axes[0, 1]
if df_rn is not None:
    # Group by noise type
    noise_types = df_rn['scenario'].unique()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(noise_types)))

    for idx, noise in enumerate(sorted(noise_types)):
        subset = df_rn[df_rn['scenario'] == noise].sort_values('a_ttv')
        ax2.plot(subset['a_ttv_over_t14'], subset['improvement'], 'o-',
                 color=colors[idx], linewidth=2, markersize=6, alpha=0.8,
                 label=noise.replace('_', ' '))

    ax2.axhline(100, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.3)
    ax2.set_xlabel(r'$A_{\rm TTV} / T_{14}$', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('B. Red Noise: Robustness to Correlated Noise', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

# Panel C: Data Gaps - Robustness
ax3 = axes[1, 0]
if df_dg is not None:
    scenarios = df_dg['scenario'].unique()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(scenarios)))

    for idx, scenario in enumerate(sorted(scenarios, key=lambda x: -df_dg[df_dg['scenario']==x]['duty_cycle'].iloc[0])):
        subset = df_dg[df_dg['scenario'] == scenario].sort_values('a_ttv')
        duty = subset['duty_cycle'].iloc[0] * 100
        ax3.plot(subset['a_ttv'], subset['improvement'], 'o-',
                 color=colors[idx], linewidth=2, markersize=6,
                 label=f'{scenario} ({duty:.0f}%)')

    ax3.axhline(100, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel(r'$A_{\rm TTV}$ [days]', fontsize=12)
    ax3.set_ylabel('Improvement (%)', fontsize=12)
    ax3.set_title('C. Data Gaps: Robustness to Missing Data', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

# Panel D: TLS Comparison - Algorithm Independence
ax4 = axes[1, 1]
if df_tls is not None:
    # Average by A_TTV
    tls_agg = df_tls.groupby('a_ttv').agg({
        'bls_improvement': 'mean',
        'tls_improvement': 'mean',
        'a_ttv_over_t14': 'first'
    }).reset_index()

    ax4.plot(tls_agg['a_ttv_over_t14'], tls_agg['bls_improvement'], 'o-',
             linewidth=2, markersize=8, label='TTV-BLS vs BLS', color='steelblue')
    ax4.plot(tls_agg['a_ttv_over_t14'], tls_agg['tls_improvement'], 's-',
             linewidth=2, markersize=8, label='TTV-TLS vs TLS', color='forestgreen')

    ax4.axvspan(0.5, 0.7, alpha=0.2, color='red', label='Critical threshold')
    ax4.axhline(100, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel(r'$A_{\rm TTV} / T_{14}$', fontsize=12)
    ax4.set_ylabel('Improvement (%)', fontsize=12)
    ax4.set_title('D. TLS Comparison: Algorithm Independence', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)

plt.suptitle('Critical Threshold Validation Across All Studies', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'master_threshold_validation.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'master_threshold_validation.pdf', bbox_inches='tight')
print("Saved: master_threshold_validation.png")

# ============================================================
# FIGURE 2: Robustness Summary (Red Noise + Data Gaps)
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Bar chart at A_TTV=0.08 for red noise
ax1 = axes[0]
if df_rn is not None:
    subset = df_rn[df_rn['a_ttv'] == 0.08].sort_values('improvement', ascending=True)
    y_pos = np.arange(len(subset))

    colors = ['steelblue' if 'white' in s else
              'lightgreen' if 'pink' in s else
              'salmon' if 'red' in s else 'orange'
              for s in subset['scenario']]

    bars = ax1.barh(y_pos, subset['improvement'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([s.replace('_', ' ') for s in subset['scenario']], fontsize=9)
    ax1.set_xlabel('Improvement (%)', fontsize=12)
    ax1.set_title('A. Red Noise Robustness (A_TTV=0.08d)', fontsize=12, fontweight='bold')
    ax1.axvline(130, color='red', linestyle='--', alpha=0.7, label='White noise baseline')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')

# Panel B: Bar chart for data gaps at A_TTV=0.08
ax2 = axes[1]
if df_dg is not None:
    subset = df_dg[df_dg['a_ttv'] == 0.08].sort_values('duty_cycle', ascending=False)
    y_pos = np.arange(len(subset))

    # Color by duty cycle
    colors = plt.cm.RdYlGn(subset['duty_cycle'])

    bars = ax2.barh(y_pos, subset['improvement'], color=colors, alpha=0.8, edgecolor='black')
    labels = [f"{s['scenario']} ({s['duty_cycle']*100:.0f}%)" for _, s in subset.iterrows()]
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Improvement (%)', fontsize=12)
    ax2.set_title('B. Data Gaps Robustness (A_TTV=0.08d)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

plt.suptitle('TTV-BLS Robustness to Observational Challenges', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'robustness_summary.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'robustness_summary.pdf', bbox_inches='tight')
print("Saved: robustness_summary.png")

# ============================================================
# FIGURE 3: Population Study - Hot Jupiters vs Others
# ============================================================
if df_pop is not None:
    fig3, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Histogram comparison
    ax1 = axes[0]
    hj = df_pop[df_pop['scenario'] == 'hot_jupiters']['sde_improvement']
    others = df_pop[df_pop['scenario'] != 'hot_jupiters']['sde_improvement']

    ax1.hist(others.clip(-100, 500), bins=50, alpha=0.6, label='Other scenarios', color='steelblue', density=True)
    ax1.hist(hj.clip(-100, 500), bins=50, alpha=0.6, label='Hot Jupiters', color='crimson', density=True)
    ax1.axvline(others.median(), color='steelblue', linestyle='--', linewidth=2)
    ax1.axvline(hj.median(), color='crimson', linestyle='--', linewidth=2)
    ax1.set_xlabel('SDE Improvement (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('A. Improvement Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(-100, 500)

    # Panel B: Violin plot by scenario
    ax2 = axes[1]
    scenarios = ['high_snr', 'medium_snr', 'low_snr', 'hot_jupiters']
    data_violin = [df_pop[df_pop['scenario'] == s]['sde_improvement'].clip(-100, 400).values for s in scenarios if s in df_pop['scenario'].unique()]
    labels = [s for s in scenarios if s in df_pop['scenario'].unique()]

    parts = ax2.violinplot(data_violin, positions=range(len(data_violin)), showmeans=True, showmedians=True)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=9)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('B. Distribution by Scenario', fontsize=12, fontweight='bold')
    ax2.axhline(0, color='red', linestyle=':', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel C: Scatter colored by scenario
    ax3 = axes[2]
    colors_map = {'high_snr': 'blue', 'medium_snr': 'green', 'low_snr': 'orange', 'hot_jupiters': 'red'}
    for scenario in df_pop['scenario'].unique():
        subset = df_pop[df_pop['scenario'] == scenario]
        ax3.scatter(subset['a_ttv_over_t14'], subset['sde_improvement'].clip(-100, 500),
                   alpha=0.1, s=5, c=colors_map.get(scenario, 'gray'), label=scenario)

    ax3.axvspan(0.5, 0.7, alpha=0.2, color='red')
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel(r'$A_{\rm TTV} / T_{14}$', fontsize=12)
    ax3.set_ylabel('Improvement (%)', fontsize=12)
    ax3.set_title('C. Individual Systems', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9, markerscale=5)
    ax3.set_xlim(0, 1.5)
    ax3.set_ylim(-100, 500)

    plt.suptitle('Population Study: Hot Jupiters Show Enhanced Improvement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'population_hot_jupiters.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'population_hot_jupiters.pdf', bbox_inches='tight')
    print("Saved: population_hot_jupiters.png")

# ============================================================
# FIGURE 4: Three Regime Comparison
# ============================================================
if df_reg is not None:
    fig4, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Short baseline
    ax1 = axes[0]
    short = df_reg[df_reg['baseline'] == 150]
    for regime in short['regime'].unique():
        subset = short[short['regime'] == regime].sort_values('a_ttv')
        ax1.plot(subset['a_ttv_over_t14'], subset['improvement'], 'o-',
                 linewidth=2, markersize=10, label=regime.replace('_', ' '))

    ax1.axhline(100, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.3)
    ax1.set_xlabel(r'$A_{\rm TTV} / T_{14}$', fontsize=12)
    ax1.set_ylabel('Improvement (%)', fontsize=12)
    ax1.set_title('A. Short Baseline (150d)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel B: Long vs Short comparison
    ax2 = axes[1]

    # Get comparable points
    short_data = df_reg[df_reg['baseline'] == 150].groupby('regime')['improvement'].mean()
    long_data = df_reg[df_reg['baseline'] == 1000].groupby('regime')['improvement'].mean()

    regimes = list(set(short_data.index) & set(long_data.index))
    x = np.arange(len(regimes))
    width = 0.35

    ax2.bar(x - width/2, [short_data.get(r, 0) for r in regimes], width,
            label='150d baseline', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, [long_data.get(r, 0) for r in regimes], width,
            label='1000d baseline', color='forestgreen', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([r.replace('_', '\n') for r in regimes], fontsize=9)
    ax2.set_ylabel('Mean Improvement (%)', fontsize=12)
    ax2.set_title('B. Baseline Duration Effect', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Three-Regime Analysis: García-Melendo Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'three_regime_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'three_regime_analysis.pdf', bbox_inches='tight')
    print("Saved: three_regime_analysis.png")

# ============================================================
# FIGURE 5: Algorithm Independence (BLS vs TLS)
# ============================================================
if df_tls is not None:
    fig5, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Scatter comparison
    ax1 = axes[0]
    ax1.scatter(df_tls['bls_improvement'], df_tls['tls_improvement'],
               c=df_tls['a_ttv_over_t14'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')

    # Add 1:1 line
    max_val = max(df_tls['bls_improvement'].max(), df_tls['tls_improvement'].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')

    # Fit linear regression
    slope, intercept, r, p, se = stats.linregress(df_tls['bls_improvement'], df_tls['tls_improvement'])
    x_fit = np.linspace(0, max_val, 100)
    ax1.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=2,
             label=f'Fit: r={r:.2f}')

    cbar = plt.colorbar(ax1.collections[0], ax=ax1, label=r'$A_{\rm TTV}/T_{14}$')
    ax1.set_xlabel('TTV-BLS Improvement (%)', fontsize=12)
    ax1.set_ylabel('TTV-TLS Improvement (%)', fontsize=12)
    ax1.set_title('A. BLS vs TLS Improvement Correlation', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel B: Grouped bar by A_TTV
    ax2 = axes[1]
    a_ttv_vals = sorted(df_tls['a_ttv'].unique())
    x = np.arange(len(a_ttv_vals))
    width = 0.35

    bls_means = [df_tls[df_tls['a_ttv'] == a]['bls_improvement'].mean() for a in a_ttv_vals]
    tls_means = [df_tls[df_tls['a_ttv'] == a]['tls_improvement'].mean() for a in a_ttv_vals]

    ax2.bar(x - width/2, bls_means, width, label='TTV-BLS', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, tls_means, width, label='TTV-TLS', color='forestgreen', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{a:.2f}' for a in a_ttv_vals])
    ax2.set_xlabel(r'$A_{\rm TTV}$ [days]', fontsize=12)
    ax2.set_ylabel('Mean Improvement (%)', fontsize=12)
    ax2.set_title('B. Mean Improvement by A_TTV', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Algorithm Independence: TTV Correction Works for Both BLS and TLS', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_independence.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'algorithm_independence.pdf', bbox_inches='tight')
    print("Saved: algorithm_independence.png")

# ============================================================
# FIGURE 6: Combined Critical Threshold Evidence
# ============================================================
fig6, ax = plt.subplots(figsize=(12, 8))

# Collect all improvement vs ratio data
all_data = []

# Population data (binned)
if df_pop is not None:
    bins = np.linspace(0, 1.5, 31)
    centers = (bins[:-1] + bins[1:]) / 2
    for i in range(len(bins)-1):
        mask = (df_pop['a_ttv_over_t14'] >= bins[i]) & (df_pop['a_ttv_over_t14'] < bins[i+1])
        if mask.sum() > 10:
            all_data.append({
                'ratio': centers[i],
                'improvement': df_pop[mask]['sde_improvement'].median(),
                'source': 'Population (n=16000)'
            })

# Red noise data
if df_rn is not None:
    for _, row in df_rn.iterrows():
        all_data.append({
            'ratio': row['a_ttv_over_t14'],
            'improvement': row['improvement'],
            'source': 'Red Noise (n=32)'
        })

# Data gaps data
if df_dg is not None:
    for _, row in df_dg.iterrows():
        all_data.append({
            'ratio': row['a_ttv_over_t14'],
            'improvement': row['improvement'],
            'source': 'Data Gaps (n=32)'
        })

# TLS data
if df_tls is not None:
    for _, row in df_tls.iterrows():
        all_data.append({
            'ratio': row['a_ttv_over_t14'],
            'improvement': row['bls_improvement'],
            'source': 'TLS Study (n=18)'
        })

df_all = pd.DataFrame(all_data)

# Plot by source
colors = {'Population (n=16000)': 'blue', 'Red Noise (n=32)': 'red',
          'Data Gaps (n=32)': 'green', 'TLS Study (n=18)': 'purple'}
markers = {'Population (n=16000)': 'o', 'Red Noise (n=32)': 's',
           'Data Gaps (n=32)': '^', 'TLS Study (n=18)': 'D'}

for source in df_all['source'].unique():
    subset = df_all[df_all['source'] == source].sort_values('ratio')
    ax.scatter(subset['ratio'], subset['improvement'],
              c=colors.get(source, 'gray'), marker=markers.get(source, 'o'),
              s=80, alpha=0.7, label=source, edgecolors='black', linewidth=0.5)

# Add critical threshold region
ax.axvspan(0.5, 0.7, alpha=0.2, color='red', label='Critical threshold')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.axhline(100, color='gray', linestyle='--', alpha=0.3)

ax.set_xlabel(r'$A_{\rm TTV} / T_{14}$', fontsize=14)
ax.set_ylabel('SDE Improvement (%)', fontsize=14)
ax.set_title('Combined Evidence for Critical Threshold at $A_{TTV}/T_{14} \\approx 0.5-0.7$',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.5)
ax.set_ylim(-50, 700)

plt.tight_layout()
plt.savefig(output_dir / 'combined_threshold_evidence.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'combined_threshold_evidence.pdf', bbox_inches='tight')
print("Saved: combined_threshold_evidence.png")

# ============================================================
# FIGURE 7: Red Noise - White Noise Comparison
# ============================================================
if df_rn is not None:
    fig7, ax = plt.subplots(figsize=(10, 8))

    # Get white noise baseline
    white = df_rn[df_rn['scenario'] == 'white_only'].set_index('a_ttv')['improvement']

    # Calculate difference from white noise for each scenario
    scenarios = [s for s in df_rn['scenario'].unique() if s != 'white_only']

    a_ttv_vals = sorted(df_rn['a_ttv'].unique())

    for scenario in sorted(scenarios):
        subset = df_rn[df_rn['scenario'] == scenario].set_index('a_ttv')
        diff = subset['improvement'] - white
        ax.plot(a_ttv_vals, [diff.get(a, 0) for a in a_ttv_vals], 'o-',
                linewidth=2, markersize=8, label=scenario.replace('_', ' '))

    ax.axhline(0, color='black', linestyle='-', linewidth=2, label='White noise baseline')
    ax.axhspan(-5, 5, alpha=0.2, color='green', label='±5% tolerance')

    ax.set_xlabel(r'$A_{\rm TTV}$ [days]', fontsize=12)
    ax.set_ylabel('Improvement Difference from White Noise (%)', fontsize=12)
    ax.set_title('Red/Pink Noise Impact Relative to White Noise Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-15, 15)

    plt.tight_layout()
    plt.savefig(output_dir / 'red_vs_white_noise.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'red_vs_white_noise.pdf', bbox_inches='tight')
    print("Saved: red_vs_white_noise.png")

# ============================================================
# Print Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY FIGURES CREATED")
print("="*70)
print(f"\nOutput directory: {output_dir}")
print("\nFigures created:")
print("  1. master_threshold_validation.png - 4-panel master summary")
print("  2. robustness_summary.png - Red noise + data gaps robustness")
print("  3. population_hot_jupiters.png - Hot Jupiter analysis")
print("  4. three_regime_analysis.png - García-Melendo regime validation")
print("  5. algorithm_independence.png - BLS vs TLS comparison")
print("  6. combined_threshold_evidence.png - All studies on one plot")
print("  7. red_vs_white_noise.png - Noise type comparison")
print("\n" + "="*70)
