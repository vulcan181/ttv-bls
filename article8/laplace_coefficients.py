"""
Laplace coefficient calculations for TTV analysis.

Based on Murray & Dermott (1999) and Deck & Agol (2015).
"""

import numpy as np
from scipy import integrate
from functools import lru_cache


@lru_cache(maxsize=1000)
def laplace_coefficient(s: float, j: int, alpha: float, n_points: int = 1000) -> float:
    """
    Compute Laplace coefficient b_s^(j)(alpha) via numerical integration.

    b_s^(j)(alpha) = (2/pi) * integral_0^pi [cos(j*theta) / (1 - 2*alpha*cos(theta) + alpha^2)^s] dtheta

    Args:
        s: Power index (typically 0.5 for TTV calculations)
        j: Cosine harmonic index (0, 1, 2, ...)
        alpha: Semi-major axis ratio a_1/a_2 (0 < alpha < 1)
        n_points: Number of integration points

    Returns:
        b_s^(j)(alpha)
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    def integrand(theta):
        denom = 1 - 2 * alpha * np.cos(theta) + alpha**2
        return np.cos(j * theta) / (denom ** s)

    result, _ = integrate.quad(integrand, 0, np.pi)
    return (2 / np.pi) * result


def laplace_coefficient_derivative(s: float, j: int, alpha: float, h: float = 1e-5) -> float:
    """
    Compute d/d(alpha) b_s^(j)(alpha) via central difference.

    Args:
        s, j, alpha: Same as laplace_coefficient
        h: Step size for numerical derivative

    Returns:
        First derivative with respect to alpha
    """
    return (laplace_coefficient(s, j, alpha + h) - laplace_coefficient(s, j, alpha - h)) / (2 * h)


def laplace_coefficient_second_derivative(s: float, j: int, alpha: float, h: float = 1e-5) -> float:
    """
    Compute d^2/d(alpha)^2 b_s^(j)(alpha) via central difference.

    Args:
        s, j, alpha: Same as laplace_coefficient
        h: Step size for numerical derivative

    Returns:
        Second derivative with respect to alpha
    """
    return (laplace_coefficient(s, j, alpha + h) - 2 * laplace_coefficient(s, j, alpha) +
            laplace_coefficient(s, j, alpha - h)) / (h ** 2)


def chopping_coefficient_A1(alpha: float) -> float:
    """
    Compute first-order chopping coefficient |A_1^(1)(alpha)| from Deck & Agol (2015).

    A_1^(1)(alpha) = (3/2) * alpha * d/d(alpha) b_{1/2}^(1)(alpha)
                   + 2 * alpha^2 * d^2/d(alpha)^2 b_{1/2}^(0)(alpha)

    Args:
        alpha: Semi-major axis ratio a_1/a_2

    Returns:
        |A_1^(1)(alpha)|
    """
    # First term: (3/2) * alpha * d/d(alpha) b_{1/2}^(1)
    term1 = 1.5 * alpha * laplace_coefficient_derivative(0.5, 1, alpha)

    # Second term: 2 * alpha^2 * d^2/d(alpha)^2 b_{1/2}^(0)
    term2 = 2 * alpha**2 * laplace_coefficient_second_derivative(0.5, 0, alpha)

    return abs(term1 + term2)


def get_alpha_from_period_ratio(P2_over_P1: float) -> float:
    """
    Compute semi-major axis ratio from period ratio using Kepler's third law.

    alpha = a_1/a_2 = (P_1/P_2)^(2/3)

    Args:
        P2_over_P1: Period ratio P_2/P_1 (must be > 1 for outer perturber)

    Returns:
        alpha = a_1/a_2
    """
    if P2_over_P1 <= 1:
        raise ValueError(f"P2/P1 must be > 1 for outer perturber, got {P2_over_P1}")
    return (1 / P2_over_P1) ** (2/3)


# Pre-computed table for common period ratios (for speed)
CHOPPING_TABLE = {}

def precompute_chopping_table(period_ratios: list = None):
    """
    Pre-compute chopping coefficients for common period ratios.
    """
    global CHOPPING_TABLE
    if period_ratios is None:
        period_ratios = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
                         2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0, 4.5, 5.0]

    for pr in period_ratios:
        alpha = get_alpha_from_period_ratio(pr)
        CHOPPING_TABLE[pr] = chopping_coefficient_A1(alpha)

    return CHOPPING_TABLE


# Lithwick et al. (2012) coefficients for near-resonant TTVs
LITHWICK_COEFFICIENTS = {
    # j_R: (f, g) from Table 3 of Lithwick+2012
    2: (-1.190, 2.025),   # 2:1 resonance
    3: (-2.025, 2.484),   # 3:2 resonance
    4: (-2.840, 3.283),   # 4:3 resonance
    5: (-3.650, 4.080),   # 5:4 resonance
    6: (-4.460, 4.876),   # 6:5 resonance
    7: (-5.270, 5.671),   # 7:6 resonance
}


def get_lithwick_coefficients(j_R: int) -> tuple:
    """
    Get Lithwick f and g coefficients for resonance j_R:(j_R-1).

    Args:
        j_R: Resonance index (2 for 2:1, 3 for 3:2, etc.)

    Returns:
        (f, g) coefficients
    """
    if j_R in LITHWICK_COEFFICIENTS:
        return LITHWICK_COEFFICIENTS[j_R]
    else:
        raise ValueError(f"Coefficients not tabulated for j_R={j_R}. Available: {list(LITHWICK_COEFFICIENTS.keys())}")


if __name__ == "__main__":
    # Test and print table
    print("Chopping coefficients |A_1^(1)(alpha)|:")
    print("-" * 40)
    print(f"{'P2/P1':>8} {'alpha':>8} {'|A_1^(1)|':>10}")
    print("-" * 40)

    table = precompute_chopping_table()
    for pr, A1 in sorted(table.items()):
        alpha = get_alpha_from_period_ratio(pr)
        print(f"{pr:>8.2f} {alpha:>8.3f} {A1:>10.3f}")
