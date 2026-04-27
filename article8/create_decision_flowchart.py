#!/usr/bin/env python3
"""
Create decision flowchart for TTV mass estimation regimes.

Produces a visual flowchart showing the three-regime classification:
1. Noise-Limited (N_tr < 50)
2. Lithwick Regime (sigma_t > 1 min, factor ~5 uncertainty)
3. Chopping Regime (sigma_t < 1 min, factor ~2 uncertainty)

Output: paper/figures/fig_decision_flowchart.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path


def create_flowchart(output_path):
    """Create the decision flowchart figure."""

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    input_color = '#E8F4FD'  # Light blue
    decision_color = '#FFF3CD'  # Light yellow
    outcome_noise = '#F8D7DA'  # Light red
    outcome_lithwick = '#FFE5B4'  # Light orange
    outcome_chopping = '#D4EDDA'  # Light green

    # Box style
    box_style = "round,rounding_size=0.3"

    # Input box
    input_box = FancyBboxPatch(
        (4, 8.5), 4, 1,
        boxstyle=box_style,
        facecolor=input_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(6, 9, 'INPUT\n$N_{\\rm tr}$, $\\sigma_t$, $A_{\\rm TTV}$',
            ha='center', va='center', fontsize=12, weight='bold')

    # Decision 1: N_tr > 50?
    decision1 = FancyBboxPatch(
        (4, 6.5), 4, 1.2,
        boxstyle="round,rounding_size=0.2",
        facecolor=decision_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(decision1)
    ax.text(6, 7.1, '$N_{\\rm tr} > 50$?',
            ha='center', va='center', fontsize=14, weight='bold')

    # Arrow from input to decision1
    ax.annotate('', xy=(6, 7.7), xytext=(6, 8.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Outcome: Noise-Limited (No branch from decision1)
    noise_box = FancyBboxPatch(
        (0.5, 6.3), 2.8, 1.4,
        boxstyle=box_style,
        facecolor=outcome_noise,
        edgecolor='#721C24',
        linewidth=2
    )
    ax.add_patch(noise_box)
    ax.text(1.9, 7, 'NOISE-LIMITED',
            ha='center', va='center', fontsize=11, weight='bold', color='#721C24')
    ax.text(1.9, 6.6, 'Upper limits only',
            ha='center', va='center', fontsize=10, style='italic')

    # Arrow from decision1 to noise-limited (No)
    ax.annotate('', xy=(3.3, 7.0), xytext=(4, 7.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3.5, 7.3, 'No', fontsize=11, weight='bold', color='#721C24')

    # Decision 2: sigma_t < 1 min?
    decision2 = FancyBboxPatch(
        (4, 4.5), 4, 1.2,
        boxstyle="round,rounding_size=0.2",
        facecolor=decision_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(decision2)
    ax.text(6, 5.1, '$\\sigma_t < 1$ min?',
            ha='center', va='center', fontsize=14, weight='bold')

    # Arrow from decision1 to decision2 (Yes)
    ax.annotate('', xy=(6, 5.7), xytext=(6, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(6.3, 6.1, 'Yes', fontsize=11, weight='bold', color='#155724')

    # Outcome: Lithwick Regime (No branch from decision2)
    lithwick_box = FancyBboxPatch(
        (8.7, 4.3), 2.8, 1.6,
        boxstyle=box_style,
        facecolor=outcome_lithwick,
        edgecolor='#856404',
        linewidth=2
    )
    ax.add_patch(lithwick_box)
    ax.text(10.1, 5.3, 'LITHWICK',
            ha='center', va='center', fontsize=11, weight='bold', color='#856404')
    ax.text(10.1, 4.9, 'REGIME',
            ha='center', va='center', fontsize=11, weight='bold', color='#856404')
    ax.text(10.1, 4.5, 'Factor ~5 uncertainty',
            ha='center', va='center', fontsize=10, style='italic')

    # Arrow from decision2 to Lithwick (No)
    ax.annotate('', xy=(8.7, 5.1), xytext=(8, 5.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(8.2, 5.4, 'No', fontsize=11, weight='bold', color='#856404')

    # Decision 3: A_chop detectable?
    decision3 = FancyBboxPatch(
        (4, 2.5), 4, 1.2,
        boxstyle="round,rounding_size=0.2",
        facecolor=decision_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(decision3)
    ax.text(6, 3.1, '$A_{\\rm chop}$ detectable?',
            ha='center', va='center', fontsize=14, weight='bold')

    # Arrow from decision2 to decision3 (Yes)
    ax.annotate('', xy=(6, 3.7), xytext=(6, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(6.3, 4.1, 'Yes', fontsize=11, weight='bold', color='#155724')

    # Outcome: Lithwick Regime (No branch from decision3) - connects to same box
    ax.annotate('', xy=(8.7, 4.7), xytext=(8, 3.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(8.5, 3.5, 'No', fontsize=11, weight='bold', color='#856404')

    # Outcome: Chopping Regime (Yes from decision3)
    chopping_box = FancyBboxPatch(
        (4, 0.5), 4, 1.6,
        boxstyle=box_style,
        facecolor=outcome_chopping,
        edgecolor='#155724',
        linewidth=2
    )
    ax.add_patch(chopping_box)
    ax.text(6, 1.5, 'CHOPPING REGIME',
            ha='center', va='center', fontsize=12, weight='bold', color='#155724')
    ax.text(6, 1.0, 'Factor ~2 uncertainty',
            ha='center', va='center', fontsize=11, style='italic')
    ax.text(6, 0.65, '(Best case)',
            ha='center', va='center', fontsize=10, color='#155724')

    # Arrow from decision3 to Chopping (Yes)
    ax.annotate('', xy=(6, 2.1), xytext=(6, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(6.3, 2.25, 'Yes', fontsize=11, weight='bold', color='#155724')

    # Add legend/key
    ax.text(0.5, 9.5, 'TTV Mass Estimation Decision Tree', fontsize=16, weight='bold')

    # Add conditions summary box
    summary_box = FancyBboxPatch(
        (0.3, 0.3), 3, 3.5,
        boxstyle=box_style,
        facecolor='#F8F9FA',
        edgecolor='gray',
        linewidth=1
    )
    ax.add_patch(summary_box)
    ax.text(1.8, 3.5, 'Detection Criteria:', fontsize=11, weight='bold', ha='center')
    ax.text(0.5, 3.0, r'$\bullet$ TTV SNR = $\frac{A_{\rm TTV}}{\sigma_t/\sqrt{N_{\rm tr}}} > 3$',
            fontsize=10, va='center')
    ax.text(0.5, 2.4, r'$\bullet$ Chop SNR = $\frac{A_{\rm chop}}{\sigma_t/\sqrt{N_{\rm tr}}} > 5$',
            fontsize=10, va='center')
    ax.text(0.5, 1.8, r'$\bullet$ $A_{\rm chop} \propto m_2$ only',
            fontsize=10, va='center')
    ax.text(0.5, 1.2, r'$\bullet$ $A_{\rm Lith} \propto m_2 \times f(e)$',
            fontsize=10, va='center')
    ax.text(0.5, 0.6, '(mass-eccentricity degenerate)',
            fontsize=9, va='center', style='italic')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved flowchart to {output_path}")

    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved PNG to {png_path}")

    plt.close(fig)


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent.parent
    figures_dir = script_dir / "paper" / "figures"

    print("Creating decision flowchart...")
    create_flowchart(figures_dir / "fig_decision_flowchart.pdf")
    print("Done!")


if __name__ == "__main__":
    main()
