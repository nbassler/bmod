import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def find_d80(x: np.ndarray, y: np.ndarray, smooth_window: int = 5, poly_order: int = 3) -> float:
    """
    Find the distal 80% (D80) position from a Bragg peak curve.

    Args:
        x: Depth values in mm
        y: Dose/response values
        smooth_window: Window size for Savitzky-Golay smoothing
        poly_order: Polynomial order for smoothing

    Returns:
        D80 position in mm
    """
    # Normalize the curve to its maximum
    y_norm = y / np.max(y)

    # Smooth the data to reduce noise
    if len(y_norm) > smooth_window:
        y_smoothed = signal.savgol_filter(y_norm, smooth_window, poly_order)
    else:
        y_smoothed = y_norm.copy()

    # Find the peak position (proximal side)
    peak_idx = np.argmax(y_smoothed)
    peak_pos = x[peak_idx]

    # Find the distal fall-off region (after peak)
    distal_region = y_smoothed[peak_idx:]

    # Find where the distal region crosses 0.8 (80%)
    try:
        # Find the 80% crossing point using interpolation
        d80_pos = None
        for i in range(1, len(x[peak_idx:])):
            if distal_region[i] <= 0.8:
                # Linear interpolation between this point and previous
                x1, x2 = x[peak_idx+i-1], x[peak_idx+i]
                y1, y2 = distal_region[i-1], distal_region[i]
                d80_pos = x1 + (x2 - x1) * (0.8 - y1) / (y2 - y1)
                break

        if d80_pos is None:
            # If we didn't find crossing, try fitting a sigmoid to the fall-off
            def sigmoid(x, x0, k, y0, ymax):
                return y0 + ymax / (1 + np.exp(-k*(x-x0)))

            # Initial guesses
            x0_guess = peak_pos + 5  # Start fitting 5mm after peak
            try:
                # Use a reasonable number of points for fitting
                fit_points = min(30, len(x[peak_idx:]))
                x_fit = x[peak_idx:peak_idx+fit_points]
                y_fit = y_smoothed[peak_idx:peak_idx+fit_points]

                # Normalize for fitting
                y_fit = (y_fit - np.min(y_fit)) / (np.max(y_fit) - np.min(y_fit))

                # Fit sigmoid
                params, _ = curve_fit(sigmoid, x_fit, y_fit,
                                      p0=[x0_guess, 0.5, 0, 1],
                                      maxfev=1000)

                # Find where sigmoid crosses 0.8
                d80_pos = x0_guess + np.log(1/0.8 - 1) / params[1]
            except Exception as e:
                logger.debug(f"Sigmoid fitting failed: {e}")
                # If fitting fails, use simple threshold
                d80_pos = x[peak_idx + np.argmin(np.abs(distal_region - 0.8))]

        return d80_pos

    except Exception as e:
        logger.warning(f"Error calculating D80: {e}")
        # Fallback: return position where value is closest to 0.8
        return x[peak_idx + np.argmin(np.abs(distal_region - 0.8))]


def calculate_d80_for_all_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate D80 for all curve measurements in the DataFrame.
    Adds D80 columns for each curve type.
    """
    # Group by measurement to handle each curve separately
    grouped = df.groupby(['fname', 'id'])

    # Get all curve columns (excluding depth and identifiers)
    curve_cols = [col for col in df.columns
                  if col not in ['depth_mm', 'filename', 'measurement_id', 'id', 'fname']]

    # Store D80 results
    d80_results = []

    for _, group in grouped:
        row_result = {'fname': group['fname'].iloc[0], 'id': group['id'].iloc[0]}

        for col in curve_cols:
            if col in group.columns:
                x = group['depth_mm'].values
                y = group[col].values

                try:
                    d80 = find_d80(x, y)
                    row_result[f'{col}_D80'] = d80
                except Exception as e:
                    logger.warning(f"Error calculating D80 for {col}: {e}")
                    row_result[f'{col}_D80'] = np.nan

        d80_results.append(row_result)

    # Return only the new D80 results as a DataFrame
    return pd.DataFrame(d80_results)
