#!/usr/bin/env python3
"""
Calculate bacterial generation times from a growth curve matrix.

Input CSV format:
    First column = time (hours)
    Remaining columns = OD values for different strains / replicates

Example:
    Hours,WT,NS_431,NS_2450,NS_2841
    0,0.01,0.01,0.01,0.01
    2,0.15,0.12,0.13,0.11
    4,0.65,0.58,0.61,0.55
    5,0.89,0.78,0.84,0.76
    6,1.13,0.98,1.04,0.92
    24,1.20,1.10,1.16,1.08

Method:
1. Remove non-positive OD values
2. Transform OD using natural log
3. Test all consecutive windows with at least --min-points points
4. Discard windows spanning less than --min-fold change in OD
5. Fit ln(OD) = m*time + b
6. Choose the window with:
   - positive slope
   - highest R^2
   - shortest generation time among similarly strong fits
7. Compute generation time = ln(2) / m

Why --min-fold matters:
    R^2 alone is a poor selection criterion for densely sampled growth
    curves. Any few consecutive readings lie on a near-perfect line, so
    "highest R^2" reliably selects short, flat windows in lag or
    stationary phase where the slope is near zero and ln(2)/m explodes.
    Requiring the window to span a minimum fold-change in OD directly
    encodes the requirement that the fit covers real exponential growth,
    and is scale-free: it means the same thing regardless of sampling
    interval. A value of 4 (2 doublings) is a practical minimum; 8
    (3 doublings) is stricter but may exclude low-inoculum wells.

Outputs:
- generation_time_summary.csv
- generation_time_windows.csv
- growth_fit_grid.png
- growth_fit_grid.svg

Usage:
    python calculate_generation_time_matrix.py \
        --input data.csv \
        --output-dir growth_out

Optional:
    --min-points 4
    --min-od 0.02
    --min-r2 0.95
    --min-fold 4.0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate generation times from a growth curve matrix.")
    parser.add_argument("--input", required=True, help="Path to CSV with time in first column and OD columns after.")
    parser.add_argument("--output-dir", default="growth_out", help="Directory for outputs.")
    parser.add_argument("--min-points", type=int, default=4, help="Minimum consecutive points per fit window.")
    parser.add_argument("--min-od", type=float, default=0.02, help="Ignore OD values below this threshold.")
    parser.add_argument("--min-r2", type=float, default=0.95, help="Preferred minimum R^2 for exponential-phase fit.")
    parser.add_argument("--min-fold", type=float, default=4.0,
                        help="Minimum fold-change in OD a fit window must span (4.0 = 2 doublings). "
                             "Set to 1.0 to disable.")
    return parser.parse_args()


def fit_window(x, y):
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2, y_pred


def analyze_curve(time_vals, od_vals, strain_name, min_points=4, min_od=0.02, min_r2=0.95,
                  min_fold=4.0):
    df = pd.DataFrame({"time": time_vals, "od": od_vals}).dropna()
    df = df[df["od"] > 0].copy()
    df = df[df["od"] >= min_od].copy()

    if len(df) < min_points:
        return None, None, df

    df["ln_od"] = np.log(df["od"])
    df = df.sort_values("time").reset_index(drop=True)

    fits = []
    n = len(df)

    for start in range(n):
        for end in range(start + min_points - 1, n):
            sub = df.iloc[start:end + 1]
            x = sub["time"].to_numpy(dtype=float)
            y = sub["ln_od"].to_numpy(dtype=float)

            # Require the window to span a minimum fold-change in OD.
            # This is the primary guard against fitting flat lag/stationary
            # segments, which otherwise score near-perfect R^2.
            fold_change = math.exp(y[-1] - y[0])
            if fold_change < min_fold:
                continue

            slope, intercept, r2, y_pred = fit_window(x, y)

            if slope <= 0 or np.isnan(r2):
                continue

            doubling_time = math.log(2) / slope

            fits.append({
                "strain": strain_name,
                "start_idx": start,
                "end_idx": end,
                "time_start": float(x[0]),
                "time_end": float(x[-1]),
                "n_points": len(sub),
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "fold_change": fold_change,
                "generation_time_h": doubling_time,
            })

    if not fits:
        if min_fold > 1.0:
            total_fold = math.exp(df["ln_od"].max() - df["ln_od"].min())
            print(f"  [warn] {strain_name}: no window spans the required "
                  f"{min_fold:.1f}-fold change (well spans only "
                  f"{total_fold:.1f}-fold above --min-od). Skipped.")
        return None, None, df

    fits_df = pd.DataFrame(fits)
    strong = fits_df[fits_df["r2"] >= min_r2].copy()
    if len(strong) == 0:
        strong = fits_df.copy()

    best = strong.sort_values(
        by=["r2", "generation_time_h", "n_points"],
        ascending=[False, True, False]
    ).iloc[0].to_dict()

    return best, fits_df, df


def save_grid_plot(raw_df, best_fits, filtered_curves, out_png):
    strains = list(best_fits.keys())
    n = len(strains)
    if n == 0:
        return

    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for ax, strain in zip(axes.flat, strains):
        ax.axis("on")
        time_col = raw_df.columns[0]

        # raw points
        ax.scatter(raw_df[time_col], raw_df[strain], s=35, label="OD")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("OD")

        best = best_fits[strain]
        filt_df = filtered_curves[strain]

        # plot fitted window back in OD space
        x_fit = np.linspace(best["time_start"], best["time_end"], 100)
        y_fit_ln = best["slope"] * x_fit + best["intercept"]
        y_fit = np.exp(y_fit_ln)
        ax.plot(x_fit, y_fit, linewidth=2, label="Best fit")

        # highlight selected points
        selected = filt_df[(filt_df["time"] >= best["time_start"]) & (filt_df["time"] <= best["time_end"])]
        ax.scatter(selected["time"], selected["od"], s=50, marker="D", label="Selected window")

        ax.set_title(
            f"{strain}\nGT={best['generation_time_h']:.2f} h, R²={best['r2']:.3f}, "
            f"{best['fold_change']:.1f}× span",
            fontsize=10
        )
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_path)
    if raw_df.shape[1] < 2:
        raise ValueError("Input must have at least two columns: time and one OD curve.")

    time_col = raw_df.columns[0]
    strains = raw_df.columns[1:]

    best_rows = []
    all_fit_rows = []
    best_fits = {}
    filtered_curves = {}

    for strain in strains:
        best, fits_df, filt_df = analyze_curve(
            raw_df[time_col],
            raw_df[strain],
            strain_name=strain,
            min_points=args.min_points,
            min_od=args.min_od,
            min_r2=args.min_r2,
            min_fold=args.min_fold
        )

        if best is not None:
            best_rows.append(best)
            best_fits[strain] = best
            filtered_curves[strain] = filt_df

        if fits_df is not None:
            all_fit_rows.append(fits_df)

    if len(best_rows) == 0:
        raise ValueError("No valid growth curves could be fit.")

    summary_df = pd.DataFrame(best_rows)
    summary_df = summary_df[[
        "strain", "time_start", "time_end", "n_points", "fold_change",
        "slope", "r2", "generation_time_h"
    ]].sort_values("strain")
    summary_df.to_csv(out_dir / "generation_time_summary.csv", index=False)

    if len(all_fit_rows) > 0:
        pd.concat(all_fit_rows, ignore_index=True).to_csv(out_dir / "generation_time_windows.csv", index=False)

    save_grid_plot(raw_df, best_fits, filtered_curves, out_dir / "growth_fit_grid.png")

    print("Generation time analysis complete.")
    print(f"Settings: min_points={args.min_points}, min_od={args.min_od}, "
          f"min_r2={args.min_r2}, min_fold={args.min_fold}")
    n_skipped = len(strains) - len(summary_df)
    if n_skipped > 0:
        print(f"Fit obtained for {len(summary_df)}/{len(strains)} curves "
              f"({n_skipped} skipped, see warnings above).")
    print()
    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
