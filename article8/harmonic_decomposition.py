"""
Harmonic decomposition of TTV time series.

Extracts TTV periods and amplitudes using Lomb-Scargle periodogram.
"""

import numpy as np
from scipy import optimize
from astropy.timeseries import LombScargle


def lomb_scargle_periodogram(
    times: np.ndarray,
    oc_values: np.ndarray,
    oc_errors: np.ndarray = None,
    min_period: float = 5.0,
    max_period: float = 1000.0,
    samples_per_peak: int = 10
) -> tuple:
    """
    Compute Lomb-Scargle periodogram of O-C time series.

    Args:
        times: Transit times in days
        oc_values: O-C residuals in minutes
        oc_errors: O-C uncertainties in minutes (optional)
        min_period: Minimum period to search (days)
        max_period: Maximum period to search (days)
        samples_per_peak: Frequency sampling density

    Returns:
        (periods, power, ls_object)
    """
    # Create Lomb-Scargle object
    if oc_errors is not None:
        ls = LombScargle(times, oc_values, dy=oc_errors)
    else:
        ls = LombScargle(times, oc_values)

    # Compute periodogram
    frequency, power = ls.autopower(
        minimum_frequency=1/max_period,
        maximum_frequency=1/min_period,
        samples_per_peak=samples_per_peak
    )

    periods = 1 / frequency
    return periods, power, ls


def find_significant_peaks(
    periods: np.ndarray,
    power: np.ndarray,
    ls: LombScargle,
    fap_threshold: float = 0.001,
    min_separation: float = 0.1
) -> list:
    """
    Find significant peaks in the periodogram.

    Args:
        periods: Period array from periodogram
        power: Power array from periodogram
        ls: LombScargle object (for FAP calculation)
        fap_threshold: False alarm probability threshold
        min_separation: Minimum fractional separation between peaks

    Returns:
        List of dicts with 'period', 'power', 'fap' for each significant peak
    """
    peaks = []

    # Find local maxima
    for i in range(1, len(power) - 1):
        if power[i] > power[i-1] and power[i] > power[i+1]:
            fap = ls.false_alarm_probability(power[i])
            if fap < fap_threshold:
                peaks.append({
                    'period': periods[i],
                    'power': power[i],
                    'fap': fap,
                    'index': i
                })

    # Sort by power (descending)
    peaks.sort(key=lambda x: x['power'], reverse=True)

    # Remove peaks that are too close to stronger peaks
    filtered_peaks = []
    for peak in peaks:
        is_independent = True
        for existing in filtered_peaks:
            ratio = peak['period'] / existing['period']
            if abs(ratio - 1) < min_separation or abs(ratio - 0.5) < min_separation or abs(ratio - 2) < min_separation:
                is_independent = False
                break
        if is_independent:
            filtered_peaks.append(peak)

    return filtered_peaks


def fit_sinusoid(
    times: np.ndarray,
    oc_values: np.ndarray,
    oc_errors: np.ndarray,
    period: float
) -> dict:
    """
    Fit a sinusoid at a fixed period to extract amplitude and phase.

    Model: O-C(t) = A * sin(2*pi*t/P + phi) + C

    Args:
        times: Transit times (days)
        oc_values: O-C values (minutes)
        oc_errors: O-C uncertainties (minutes)
        period: Period to fit (days)

    Returns:
        dict with 'amplitude', 'amplitude_error', 'phase', 'offset', 'residual_rms'
    """
    omega = 2 * np.pi / period

    # Design matrix for linear least squares: A*sin + B*cos + C
    # y = A*sin(wt) + B*cos(wt) + C
    # Amplitude = sqrt(A^2 + B^2), phase = atan2(B, A)
    X = np.column_stack([
        np.sin(omega * times),
        np.cos(omega * times),
        np.ones_like(times)
    ])

    # Weighted least squares
    if oc_errors is not None and np.all(oc_errors > 0):
        W = np.diag(1 / oc_errors**2)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ oc_values
    else:
        XtWX = X.T @ X
        XtWy = X.T @ oc_values

    try:
        params = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return None

    A, B, C = params

    # Amplitude and phase
    amplitude = np.sqrt(A**2 + B**2)
    phase = np.arctan2(B, A)

    # Residuals
    model = A * np.sin(omega * times) + B * np.cos(omega * times) + C
    residuals = oc_values - model
    residual_rms = np.sqrt(np.mean(residuals**2))

    # Amplitude uncertainty (simplified)
    if oc_errors is not None:
        # Propagate from covariance matrix
        try:
            cov = np.linalg.inv(XtWX)
            var_A = cov[0, 0]
            var_B = cov[1, 1]
            # Error propagation for sqrt(A^2 + B^2)
            amplitude_error = np.sqrt((A**2 * var_A + B**2 * var_B) / (A**2 + B**2))
        except:
            amplitude_error = residual_rms / np.sqrt(len(times) / 2)
    else:
        amplitude_error = residual_rms / np.sqrt(len(times) / 2)

    return {
        'amplitude': amplitude,
        'amplitude_error': amplitude_error,
        'phase': phase,
        'offset': C,
        'residual_rms': residual_rms,
        'model': model
    }


def classify_peak(
    peak_period: float,
    P1: float,
    candidate_P2_range: tuple = (1.3, 6.0)
) -> dict:
    """
    Classify a detected TTV period as super-period or synodic.

    Args:
        peak_period: Detected TTV period (days)
        P1: Known planet period (days)
        candidate_P2_range: Range of P2/P1 to consider

    Returns:
        dict with 'classification', 'implied_P2', 'j_R' (if super-period)
    """
    results = []

    # Check if it could be a synodic period
    for P2_over_P1 in np.linspace(candidate_P2_range[0], candidate_P2_range[1], 100):
        P2 = P1 * P2_over_P1
        P_syn = 1 / abs(1/P1 - 1/P2)

        if abs(P_syn - peak_period) / peak_period < 0.1:
            results.append({
                'classification': 'synodic',
                'implied_P2': P2,
                'implied_P2_over_P1': P2_over_P1,
                'j_R': None,
                'match_quality': abs(P_syn - peak_period) / peak_period
            })

    # Check if it could be a super-period (near resonances)
    for j_R in [2, 3, 4, 5]:
        # For each resonance, find P2 that gives this super-period
        # P_super = 1 / |j_R/P2 - (j_R-1)/P1|
        # Solving: P2 = j_R * P_super * P1 / ((j_R-1) * P_super + P1)
        # or       P2 = j_R * P_super * P1 / ((j_R-1) * P_super - P1)

        for sign in [1, -1]:
            denom = (j_R - 1) * peak_period + sign * P1
            if denom > 0:
                P2 = j_R * peak_period * P1 / denom
                P2_over_P1 = P2 / P1

                if candidate_P2_range[0] < P2_over_P1 < candidate_P2_range[1]:
                    # Verify
                    P_super_check = 1 / abs(j_R / P2 - (j_R - 1) / P1)
                    if abs(P_super_check - peak_period) / peak_period < 0.1:
                        results.append({
                            'classification': 'super',
                            'implied_P2': P2,
                            'implied_P2_over_P1': P2_over_P1,
                            'j_R': j_R,
                            'match_quality': abs(P_super_check - peak_period) / peak_period
                        })

    # Sort by match quality
    results.sort(key=lambda x: x['match_quality'])

    if results:
        return results[0]
    else:
        return {
            'classification': 'unknown',
            'implied_P2': None,
            'implied_P2_over_P1': None,
            'j_R': None,
            'match_quality': None
        }


def extract_ttv_harmonics(
    times: np.ndarray,
    oc_values: np.ndarray,
    oc_errors: np.ndarray,
    P1: float,
    min_period: float = 5.0,
    max_period: float = 500.0,
    fap_threshold: float = 0.001
) -> dict:
    """
    Full harmonic extraction pipeline.

    Args:
        times: Transit times (days)
        oc_values: O-C residuals (minutes)
        oc_errors: O-C uncertainties (minutes)
        P1: Planet orbital period (days)
        min_period, max_period: Search range (days)
        fap_threshold: False alarm probability threshold

    Returns:
        dict with all extracted harmonics and their properties
    """
    # Run periodogram
    periods, power, ls = lomb_scargle_periodogram(
        times, oc_values, oc_errors, min_period, max_period
    )

    # Find significant peaks
    peaks = find_significant_peaks(periods, power, ls, fap_threshold)

    # Analyze each peak
    harmonics = []
    for peak in peaks:
        # Fit sinusoid
        fit = fit_sinusoid(times, oc_values, oc_errors, peak['period'])
        if fit is None:
            continue

        # Classify
        classification = classify_peak(peak['period'], P1)

        harmonics.append({
            'period': peak['period'],
            'power': peak['power'],
            'fap': peak['fap'],
            'amplitude': fit['amplitude'],
            'amplitude_error': fit['amplitude_error'],
            'phase': fit['phase'],
            'classification': classification['classification'],
            'implied_P2': classification['implied_P2'],
            'implied_P2_over_P1': classification['implied_P2_over_P1'],
            'j_R': classification['j_R']
        })

    return {
        'periods': periods,
        'power': power,
        'harmonics': harmonics,
        'n_significant': len(harmonics),
        'baseline': times[-1] - times[0] if len(times) > 1 else 0,
        'n_transits': len(times)
    }


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing harmonic decomposition...")

    np.random.seed(42)

    # Generate test data
    P1 = 5.0  # days
    n_transits = 50
    transit_numbers = np.arange(n_transits)
    times = transit_numbers * P1

    # Inject TTV signal
    P_ttv = 45.0  # days
    A_ttv = 3.0   # minutes
    ttv_signal = A_ttv * np.sin(2 * np.pi * times / P_ttv)

    # Add noise
    noise_level = 0.5  # minutes
    oc_values = ttv_signal + np.random.normal(0, noise_level, n_transits)
    oc_errors = np.ones(n_transits) * noise_level

    # Extract harmonics
    result = extract_ttv_harmonics(times, oc_values, oc_errors, P1)

    print(f"Found {result['n_significant']} significant harmonics")
    for h in result['harmonics']:
        print(f"  Period: {h['period']:.1f} d, Amplitude: {h['amplitude']:.2f} +/- {h['amplitude_error']:.2f} min")
        print(f"    Classification: {h['classification']}, FAP: {h['fap']:.2e}")
