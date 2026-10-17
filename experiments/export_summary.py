"""Export per-method error summaries from stored replicate estimates."""

from pathlib import Path

import numpy as np
import pandas as pd
from src.target import Z_TRUE


SUMMARY_COLUMNS = [
    "method", "n_samples", "n_replicates", "mean_estimate", "bias_vs_pi",
    "empirical_variance", "rmse_vs_pi", "median_abs_error",
    "q95_abs_error", "max_abs_error",
]


def export_summary(
        raw_path: Path | str, summary_path: Path | str
) -> pd.DataFrame:
    """
    Summarize each method and sample size without generating new samples.

    Args:
        raw_path: Directory of CSV files with Sample Size, Replicate and
            Estimate columns. Each filename stem identifies one method.
        summary_path: Output directory for the combined summary.csv file.

    Returns:
        summary: One row per method and sample size, sorted by those keys.
            Empirical variance uses ddof=1 and is undefined for a single
            replicate. Error quantiles use linear interpolation. All errors
            compare individual replicate estimates against pi.
    """
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)
    files = sorted(raw_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_path}")

    rows = []
    for path in files:
        # Preserve round-trip precision, including the exact oracle estimates.
        raw = pd.read_csv(path, float_precision="round_trip")
        required = ["Sample Size", "Replicate", "Estimate"]
        if not set(required).issubset(raw.columns) or raw.empty:
            raise ValueError(f"{path.name} must contain nonempty raw replicate data.")
        raw = raw[required].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(raw.to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} contains missing or non-finite values.")
        for column in ["Sample Size", "Replicate"]:
            if (raw[column] <= 0).any() or (raw[column] % 1 != 0).any():
                raise ValueError(f"{path.name}: {column} must contain positive integers.")
        if raw.duplicated(["Sample Size", "Replicate"]).any():
            raise ValueError(f"{path.name} contains duplicate sample-size/replicate pairs.")

        for n_samples, group in raw.groupby("Sample Size", sort=True):
            estimates = group["Estimate"].to_numpy(dtype=float)
            errors = estimates - Z_TRUE
            absolute_errors = np.abs(errors)
            n_replicates = len(estimates)
            mean_estimate = float(np.mean(estimates))
            rows.append({
                "method": path.stem,
                "n_samples": int(n_samples),
                "n_replicates": n_replicates,
                "mean_estimate": mean_estimate,
                "bias_vs_pi": mean_estimate - Z_TRUE,
                "empirical_variance": float(np.var(estimates, ddof=1)) if n_replicates > 1 else np.nan,
                "rmse_vs_pi": float(np.sqrt(np.mean(errors**2))),
                "median_abs_error": float(np.median(absolute_errors)),
                "q95_abs_error": float(np.quantile(absolute_errors, 0.95, method="linear")),
                "max_abs_error": float(np.max(absolute_errors)),
            })

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(["method", "n_samples"]).reset_index(drop=True)
    summary_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path / "summary.csv", index=False, float_format="%.17g")
    return summary


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_path = PROJECT_ROOT / "results" / "summary"
    summary = export_summary(PROJECT_ROOT / "results" / "raw", output_path)
    print(f"Saved {len(summary)} summary rows to {output_path / 'summary.csv'}")
