# Data Files

## ttv_amplitude_lookup_table.csv

TTV amplitude lookup table from N-body simulations (Article V).

Contains TTV statistics for 2,786 configurations spanning:
- Period ratios: 1.2 to 4.0
- Mass ratios: 0.1 to 10.0
- Resonance distances (delta): 0.01 to 0.2

Columns:
- `period_ratio`: P_outer / P_inner
- `mass_ratio`: M_outer / M_inner
- `delta_res`: Distance from exact resonance
- `attv_mean`, `attv_std`: Mean and std of TTV amplitude (days)
- `attv_minutes_mean`: TTV amplitude in minutes
- `pttv_mean`: Mean TTV period (days)
- `stable_sum`: Number of stable configurations

Based on 15,000 REBOUND N-body integrations (11,178 stable).
