from __future__ import annotations
from bmod.grf_git import calculate_d80_for_all_curves
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)


def load_giraffe_csv(input_file: Path) -> pd.DataFrame:
    """
    Load depth values and all curve data from Giraffe CSV file.
    Returns a DataFrame with depth values and multiple curve columns.
    """
    depth_values = None
    curves = {}
    current_curve = None

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Look for curve headers
            if line.startswith("Curve"):
                current_curve = line.split(":")[0].strip()
                curves[current_curve] = []
                continue

            if line.startswith("Samples"):
                continue

            # Parse curve data
            if current_curve:
                values = [float(x) for x in line.split(";") if x]
                curves[current_curve].extend(values)

                # Store depth values when we find them
                if "depth" in current_curve.lower() and depth_values is None:
                    depth_values = values.copy()
                continue

    if depth_values is None:
        raise ValueError("No depth values found in file")

    if not curves:
        raise ValueError("No curve data found in file")

    # Verify all curves have the same length as depth values
    expected_length = len(depth_values)
    for curve_name, values in curves.items():
        if len(values) % expected_length != 0:
            raise ValueError(f"Curve {curve_name} has {len(values)} values, not a multiple of {expected_length}")

    # Create DataFrame with depth and all curves
    df = pd.DataFrame({"depth_mm": depth_values})

    # Add each curve as a separate column
    for curve_name, values in curves.items():
        if "depth" not in curve_name.lower():  # Skip depth curve
            # Handle multiple measurements if present
            num_measurements = len(values) // expected_length
            for i in range(num_measurements):
                start = i * expected_length
                end = (i + 1) * expected_length
                measurement_values = values[start:end]
                df[f"{curve_name}_{i+1}"] = measurement_values

    return df


def load_giraffe_dir(input_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load depth and curve data from all CSV files in directory.
    Returns dictionary: {filename: dataframe_with_curves}
    """
    results = {}
    csv_files = sorted(fname for fname in os.listdir(input_dir) if fname.lower().endswith(".csv"))

    for fname in csv_files:
        fpath = os.path.join(input_dir, fname)
        try:
            df = load_giraffe_csv(Path(fpath))
            results[fname] = df
            print(f"✅ Successfully loaded {fname} with {len(df.columns)} columns")
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {str(e)}")

    return results


def run(input_dir: Path,
        cfg: Dict[str, Any] | Any,
        output_file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Pipeline for Giraffe processing - extracts depth and curve data.
    Returns a multi-index DataFrame with all measurements.
    """
    logger.debug("Giraffe runner started: %s", input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    logger.debug(f"Loading data from directory: {input_dir}")
    data = load_giraffe_dir(input_dir)

    if not data:
        return pd.DataFrame()

    # Create a multi-level DataFrame structure
    all_dfs = []

    i = 1
    for fname, df in data.items():
        # Add measurement source identifier
        df = df.copy()
        df.insert(0, 'fname', fname)
        df.insert(1, 'id', i)
        all_dfs.append(df)
        i += 1

    all_dfs = pd.concat(all_dfs, ignore_index=True)

    if output_file_path:
        logger.info(f"Writing output to: {output_file_path}")

        # Save the consolidated data
        all_dfs.to_csv(output_file_path, index=False)
        logger.debug(f"Saved consolidated data to {output_file_path}")

    # process all_dfs:
    # Calculate D80 for all curves
    result_df = calculate_d80_for_all_curves(all_dfs)

    if output_file_path:
        # build a new path based on output_file_path, but with suffix _d80.csv
        d80_output_path = output_file_path.with_name(output_file_path.stem + "_d80.csv")
        # Save the results with D80 values
        result_df.to_csv(d80_output_path, index=False)
        logger.debug(f"Saved results with D80 values to {d80_output_path}")

    return result_df

    return all_dfs
