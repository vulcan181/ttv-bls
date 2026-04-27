"""
TTV models for mass estimation.

Implements:
1. Lithwick et al. (2012) near-resonant TTV formula
2. Deck & Agol (2015) synodic chopping formula
3. Combined TTV model
"""

import numpy as np

try:
    from .laplace_coefficients import (
        chopping_coefficient_A1,
        get_alpha_from_period_ratio,
        get_lithwick_coefficients
    )
except ImportError:
    from laplace_coefficients import (
        chopping_coefficient_A1,
        get_alpha_from_period_ratio,
        get_lithwick_coefficients
    )


# Physical constants
M_SUN_KG = 1.989e30
M_EARTH_KG = 5.972e24
M_JUPITER_KG = 1.898e27
DAY_TO_MIN = 24 * 60


def compute_resonance_parameter(P1: float, P2: float, j_R: int) -> float:
    """
    Compute the resonance proximity parameter Delta.

    Delta = (P2/P1) * (j_R - 1) / j_R - 1

    Args:
        P1: Inner planet period (any units)
        P2: Outer planet period (same units as P1)
        j_R: Resonance index (2 for 2:1, 3 for 3:2, etc.)

    Returns:
        Delta (dimensionless, positive if outside resonance)
    """
    return (P2 / P1) * (j_R - 1) / j_R - 1


def compute_super_period(P1: float, P2: float, j_R: int) -> float:
    """
    Compute the TTV super-period for near-resonant configuration.

    P_super = 1 / |j_R/P2 - (j_R-1)/P1|

    Args:
        P1, P2: Planet periods (same units)
        j_R: Resonance index

    Returns:
        Super-period (same units as input periods)
    """
    denom = abs(j_R / P2 - (j_R - 1) / P1)
    if denom < 1e-10:
        # Exactly at resonance - return very large period
        return 1e10
    return 1 / denom


def compute_synodic_period(P1: float, P2: float) -> float:
    """
    Compute the synodic period between two planets.

    P_syn = |1/P1 - 1/P2|^(-1)

    Args:
        P1, P2: Planet periods (same units)

    Returns:
        Synodic period (same units as input)
    """
    return 1 / abs(1/P1 - 1/P2)


def lithwick_ttv_amplitude(
    P1: float,
    mu2: float,
    j_R: int,
    Delta: float,
    Z_free: complex = 0.0
) -> float:
    """
    Compute TTV amplitude using Lithwick et al. (2012) formula.

    |V_1| = (P_1/pi) * (1 / [j_R^(2/3) * (j_R-1)^(1/3) * Delta]) * mu_2 * |term|

    where term = -f - 3*Z_free^* / (2*Delta)

    Args:
        P1: Inner planet period (days)
        mu2: Perturber mass ratio m_2/M_star
        j_R: Resonance index
        Delta: Resonance proximity parameter
        Z_free: Free eccentricity complex parameter (default 0 for low-e)

    Returns:
        TTV amplitude in minutes
    """
    f, g = get_lithwick_coefficients(j_R)

    # Compute the bracket term
    if Z_free == 0:
        # Low-eccentricity approximation
        term = -f
    else:
        term = -f - 3 * np.conj(Z_free) / (2 * Delta)

    # Prefactor
    prefactor = (P1 * DAY_TO_MIN / np.pi) / (j_R**(2/3) * (j_R - 1)**(1/3) * abs(Delta))

    return prefactor * mu2 * abs(term)


def chopping_ttv_amplitude(P1: float, mu2: float, alpha: float) -> float:
    """
    Compute chopping TTV amplitude using Deck & Agol (2015) formula.

    A_chop = (P_1/pi) * mu_2 * |A_1^(1)(alpha)|

    Args:
        P1: Inner planet period (days)
        mu2: Perturber mass ratio m_2/M_star
        alpha: Semi-major axis ratio a_1/a_2

    Returns:
        Chopping TTV amplitude in minutes
    """
    A1 = chopping_coefficient_A1(alpha)
    return (P1 * DAY_TO_MIN / np.pi) * mu2 * A1


def generate_lithwick_ttv(
    transit_numbers: np.ndarray,
    P1: float,
    P2: float,
    mu2: float,
    j_R: int,
    E_TTV: float = 0.0,
    Z_free: complex = 0.0
) -> np.ndarray:
    """
    Generate TTV time series using Lithwick formula.

    Args:
        transit_numbers: Array of transit epoch numbers (0, 1, 2, ...)
        P1: Inner planet period (days)
        P2: Outer planet period (days)
        mu2: Perturber mass ratio
        j_R: Resonance index
        E_TTV: TTV epoch offset (days)
        Z_free: Free eccentricity parameter

    Returns:
        TTV values in minutes
    """
    Delta = compute_resonance_parameter(P1, P2, j_R)
    P_super = compute_super_period(P1, P2, j_R)
    amplitude = lithwick_ttv_amplitude(P1, mu2, j_R, Delta, Z_free)

    # Transit times
    t = transit_numbers * P1

    # TTV signal (sinusoidal at super-period)
    phase = 2 * np.pi * (t - E_TTV) / P_super
    ttv = amplitude * np.sin(phase)

    return ttv


def generate_chopping_ttv(
    transit_numbers: np.ndarray,
    P1: float,
    P2: float,
    mu2: float,
    phi: float = 0.0
) -> np.ndarray:
    """
    Generate chopping TTV time series using Deck & Agol formula.

    Args:
        transit_numbers: Array of transit epoch numbers
        P1: Inner planet period (days)
        P2: Outer planet period (days)
        mu2: Perturber mass ratio
        phi: Phase offset (radians)

    Returns:
        TTV values in minutes
    """
    alpha = get_alpha_from_period_ratio(P2 / P1)
    P_syn = compute_synodic_period(P1, P2)
    amplitude = chopping_ttv_amplitude(P1, mu2, alpha)

    # Synodic phase at each transit
    t = transit_numbers * P1
    psi = 2 * np.pi * t / P_syn

    # Chopping signal
    ttv = amplitude * np.sin(psi + phi)

    return ttv


def generate_combined_ttv(
    transit_numbers: np.ndarray,
    P1: float,
    P2: float,
    mu2: float,
    j_R: int,
    E_TTV: float = 0.0,
    Z_free: complex = 0.0,
    phi_chop: float = 0.0
) -> np.ndarray:
    """
    Generate combined TTV signal (Lithwick + chopping).

    Args:
        transit_numbers: Array of transit epoch numbers
        P1, P2: Planet periods (days)
        mu2: Perturber mass ratio
        j_R: Resonance index
        E_TTV: Super-period phase offset (days)
        Z_free: Free eccentricity parameter
        phi_chop: Chopping phase offset (radians)

    Returns:
        Combined TTV in minutes
    """
    ttv_lithwick = generate_lithwick_ttv(transit_numbers, P1, P2, mu2, j_R, E_TTV, Z_free)
    ttv_chopping = generate_chopping_ttv(transit_numbers, P1, P2, mu2, phi_chop)

    return ttv_lithwick + ttv_chopping


def mass_from_chopping_amplitude(
    A_chop: float,
    P1: float,
    P2_over_P1: float,
    M_star: float = 1.0
) -> float:
    """
    Estimate companion mass from measured chopping amplitude.

    m_2 = (pi * A_chop * M_star) / (P_1 * |A_1^(1)|)

    Args:
        A_chop: Measured chopping amplitude (minutes)
        P1: Inner planet period (days)
        P2_over_P1: Period ratio
        M_star: Stellar mass in solar masses

    Returns:
        Companion mass in Earth masses
    """
    alpha = get_alpha_from_period_ratio(P2_over_P1)
    A1 = chopping_coefficient_A1(alpha)

    mu2 = (np.pi * A_chop) / (P1 * DAY_TO_MIN * A1)
    m2_solar = mu2 * M_star
    m2_earth = m2_solar * M_SUN_KG / M_EARTH_KG

    return m2_earth


def mass_from_lithwick_amplitude(
    A_super: float,
    P1: float,
    P2: float,
    j_R: int,
    M_star: float = 1.0,
    assume_low_e: bool = True
) -> float:
    """
    Estimate companion mass from measured super-period amplitude.

    Uses low-e approximation: assumes |Z_free| << |Delta|

    Args:
        A_super: Measured super-period amplitude (minutes)
        P1, P2: Planet periods (days)
        j_R: Resonance index
        M_star: Stellar mass in solar masses
        assume_low_e: If True, use low-e approximation (Z_free = 0)

    Returns:
        Companion mass in Earth masses (or lower limit if assume_low_e=False)
    """
    Delta = compute_resonance_parameter(P1, P2, j_R)
    f, g = get_lithwick_coefficients(j_R)

    # Prefactor
    prefactor = (P1 * DAY_TO_MIN / np.pi) / (j_R**(2/3) * (j_R - 1)**(1/3) * abs(Delta))

    if assume_low_e:
        # |V| = prefactor * mu2 * |f|
        mu2 = A_super / (prefactor * abs(f))
    else:
        # This gives a lower limit (maximum |term| gives minimum mass)
        # Maximum |term| occurs when Z_free enhances the signal
        mu2 = A_super / (prefactor * abs(f))  # Same as low-e for now

    m2_solar = mu2 * M_star
    m2_earth = m2_solar * M_SUN_KG / M_EARTH_KG

    return m2_earth


if __name__ == "__main__":
    # Test with example parameters
    print("Testing TTV models...")
    print("=" * 50)

    # Example: Near 2:1 resonance system
    P1 = 5.0  # days
    P2 = 10.5  # days (just outside 2:1)
    mu2 = 1e-4  # ~30 Earth masses around Sun
    j_R = 2

    Delta = compute_resonance_parameter(P1, P2, j_R)
    P_super = compute_super_period(P1, P2, j_R)
    P_syn = compute_synodic_period(P1, P2)

    print(f"P1 = {P1} days, P2 = {P2} days")
    print(f"P2/P1 = {P2/P1:.3f}")
    print(f"Delta = {Delta:.4f}")
    print(f"Super-period = {P_super:.1f} days")
    print(f"Synodic period = {P_syn:.1f} days")
    print()

    A_lithwick = lithwick_ttv_amplitude(P1, mu2, j_R, Delta)
    A_chop = chopping_ttv_amplitude(P1, mu2, get_alpha_from_period_ratio(P2/P1))

    print(f"Lithwick amplitude = {A_lithwick:.2f} minutes")
    print(f"Chopping amplitude = {A_chop:.2f} minutes")
    print(f"Ratio (Lithwick/Chopping) = {A_lithwick/A_chop:.1f}")
