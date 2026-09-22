#!/usr/bin/env python3
"""
Analyze local gene neighborhoods around user-specified "middle" locus tags.

Input CSV format:
    Locus tag, Median_aer, Median_anaer, Mean_aer, Mean_anaer

Assumptions:
- Rows are already ordered by genomic position.
- "Middle" genes are the center of a 5-gene neighborhood by default
  (2 upstream + middle + 2 downstream).

Outputs:
- neighborhood_gene_table.csv
- neighborhood_summary.csv
- genome_vs_neighborhood_gene_values.csv
- neighborhood_scatter.png / .svg
- genome_vs_neighborhood_boxplots.png / .svg
- combined_neighborhood_comparison_aerobic.png / .svg
- combined_neighborhood_comparison_anaerobic.png / .svg
- prism_genome_vs_neighborhood_aerobic.csv
- prism_genome_vs_neighborhood_anaerobic.csv
- prism_neighborhoods_aerobic.csv
- prism_neighborhoods_anaerobic.csv
- prism_middle_gene_points.csv
- analysis_report.txt

Example:
    python analyze_gene_neighborhoods.py \
        --input "Sample data.csv" \
        --middle-tags DQL45_00015 DQL45_00040 \
        --window-size 5 \
        --output-dir neighborhood_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd
import matplotlib.pyplot as plt

# Keep text editable in SVG output
plt.rcParams["svg.fonttype"] = "none"

try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


REQUIRED_COLUMNS = ["Locus tag", "Median_aer", "Median_anaer", "Mean_aer", "Mean_anaer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze transcriptional neighborhoods around selected locus tags.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--middle-tags",
        nargs="+",
        required=True,
        help="One or more locus tags to use as neighborhood centers.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Neighborhood size (must be odd; default: 5 = 2 upstream + middle + 2 downstream).",
    )
    parser.add_argument(
        "--output-dir",
        default="neighborhood_analysis",
        help="Directory for output files.",
    )
    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["Locus tag"].duplicated().any():
        dupes = df.loc[df["Locus tag"].duplicated(), "Locus tag"].tolist()
        raise ValueError(f"Duplicate locus tags found: {dupes[:10]}")


def save_png_and_svg(fig, output_path: Path, dpi: int = 300) -> None:
    """Save a figure as both PNG and SVG using the provided output path stem."""
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")


def get_neighborhood(df: pd.DataFrame, middle_tag: str, window_size: int) -> pd.DataFrame:
    if window_size % 2 == 0:
        raise ValueError("window-size must be odd.")
    half = window_size // 2

    matches = df.index[df["Locus tag"] == middle_tag].tolist()
    if not matches:
        raise KeyError(f"Middle locus tag not found: {middle_tag}")

    idx = matches[0]
    start = max(0, idx - half)
    end = min(len(df), idx + half + 1)
    neighborhood = df.iloc[start:end].copy()

    neighborhood["Middle_tag"] = middle_tag
    neighborhood["Relative_position"] = range(start - idx, end - idx)
    neighborhood["Is_middle"] = neighborhood["Locus tag"].eq(middle_tag)

    return neighborhood


def safe_mannwhitney(x: Iterable[float], y: Iterable[float]) -> float | None:
    x = list(x)
    y = list(y)
    if not SCIPY_AVAILABLE or len(x) == 0 or len(y) == 0:
        return None
    try:
        return mannwhitneyu(x, y, alternative="two-sided").pvalue
    except Exception:
        return None


def summarize_neighborhood(neighborhood: pd.DataFrame) -> Dict[str, float | str | int]:
    middle_tag = neighborhood["Middle_tag"].iloc[0]
    summary = {
        "Middle_tag": middle_tag,
        "Neighborhood_size_actual": len(neighborhood),
        "Neighborhood_locus_tags": ";".join(neighborhood["Locus tag"].tolist()),
        "Median_of_Median_aer": neighborhood["Median_aer"].median(),
        "Median_of_Median_anaer": neighborhood["Median_anaer"].median(),
        "Mean_of_Median_aer": neighborhood["Median_aer"].mean(),
        "Mean_of_Median_anaer": neighborhood["Median_anaer"].mean(),
        "Delta_median_conditions": abs(neighborhood["Median_aer"].median() - neighborhood["Median_anaer"].median()),
        "Mean_of_Mean_aer": neighborhood["Mean_aer"].mean(),
        "Mean_of_Mean_anaer": neighborhood["Mean_anaer"].mean(),
        "Median_of_Mean_aer": neighborhood["Mean_aer"].median(),
        "Median_of_Mean_anaer": neighborhood["Mean_anaer"].median(),
    }
    return summary


def plot_scatter(summary_df: pd.DataFrame, genome_summary: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        summary_df["Median_of_Median_aer"],
        summary_df["Median_of_Median_anaer"],
        zorder=3
    )

    for _, row in summary_df.iterrows():
        ax.annotate(
            row["Middle_tag"],
            (row["Median_of_Median_aer"], row["Median_of_Median_anaer"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    genome_median_aer = genome_summary["Genome_median_aer"]
    genome_median_anaer = genome_summary["Genome_median_anaer"]
    ax.axvline(genome_median_aer, linestyle="--", linewidth=1, color="gray", zorder=1)
    ax.axhline(genome_median_anaer, linestyle="--", linewidth=1, color="gray", zorder=1)

    ax.scatter(
        [genome_median_aer],
        [genome_median_anaer],
        marker="s",
        s=70,
        color="black",
        zorder=4,
        label="Genome median",
    )
    ax.annotate(
        "Genome median",
        (genome_median_aer, genome_median_anaer),
        xytext=(6, -10),
        textcoords="offset points",
        fontsize=8,
    )

    max_val = max(
        summary_df["Median_of_Median_aer"].max(),
        summary_df["Median_of_Median_anaer"].max(),
        genome_median_aer,
        genome_median_anaer,
        0.1,
    )
    ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)

    ax.set_xlabel("Neighborhood median log10(RPKM+1), aerobic")
    ax.set_ylabel("Neighborhood median log10(RPKM+1), anaerobic")
    ax.set_title("Neutral-site neighborhood expression")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    save_png_and_svg(fig, output_path)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, neighborhood_tags: List[str], output_path: Path) -> None:
    neighborhood_df = df[df["Locus tag"].isin(neighborhood_tags)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].boxplot(
        [df["Median_aer"].dropna(), neighborhood_df["Median_aer"].dropna()],
        labels=["Genome", "Neighborhood genes"],
    )
    axes[0].set_ylabel("log10(RPKM+1)")
    axes[0].set_title("Aerobic")

    axes[1].boxplot(
        [df["Median_anaer"].dropna(), neighborhood_df["Median_anaer"].dropna()],
        labels=["Genome", "Neighborhood genes"],
    )
    axes[1].set_ylabel("log10(RPKM+1)")
    axes[1].set_title("Anaerobic")

    fig.suptitle("Genome-wide vs selected neighborhood gene expression")
    fig.tight_layout()
    save_png_and_svg(fig, output_path)
    plt.close(fig)


def plot_condition_comparison_figure(
    df: pd.DataFrame,
    neighborhood_gene_table: pd.DataFrame,
    summary_df: pd.DataFrame,
    condition_col: str,
    condition_label: str,
    output_path: Path,
) -> None:
    """
    Create one figure per condition showing:
    - one genome-wide boxplot on the far left
    - one neighborhood boxplot for each selected middle tag to the right
    - dashed median lines
    - neighborhood gene points overlaid
    - only the middle gene highlighted and labeled
    """
    site_order = summary_df["Middle_tag"].tolist()
    n_sites = len(site_order)

    fig_w = max(2.0 * (n_sites + 1), 10)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    genome_vals = df[condition_col].dropna()
    genome_med = genome_vals.median()

    # Genome-wide boxplot on the far left
    ax.boxplot(
        [genome_vals],
        positions=[1],
        widths=0.5,
        patch_artist=False,
    )
    ax.hlines(genome_med, 0.78, 1.22, linestyles="--", linewidth=1.4)

    xtick_positions = [1]
    xtick_labels = ["Genome"]

    # Neighborhood boxplots to the right
    for i, middle_tag in enumerate(site_order, start=2):
        site_df = neighborhood_gene_table[
            neighborhood_gene_table["Middle_tag"] == middle_tag
        ].copy().reset_index(drop=True)

        site_vals = site_df[condition_col].dropna()
        site_med = site_vals.median()

        ax.boxplot(
            [site_vals],
            positions=[i],
            widths=0.5,
            patch_artist=False,
        )
        ax.hlines(site_med, i - 0.22, i + 0.22, linestyles="--", linewidth=1.4)

        # Overlay neighborhood points; only label middle gene
        for k, (_, row) in enumerate(site_df.iterrows()):
            x = i + ((k % 5) - 2) * 0.05
            y = row[condition_col]

            if bool(row["Is_middle"]):
                ax.scatter(x, y, marker="D", s=65, zorder=4)
                ax.annotate(
                    row["Locus tag"],
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            else:
                ax.scatter(x, y, s=28, zorder=3)

        xtick_positions.append(i)
        xtick_labels.append(middle_tag)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylabel("log10(RPKM+1)")
    ax.set_title(f"{condition_label}: genome-wide vs local neighborhood expression")
    fig.tight_layout()
    save_png_and_svg(fig, output_path)
    plt.close(fig)



def export_prism_genome_vs_neighborhood(df: pd.DataFrame, neighborhood_gene_table: pd.DataFrame, output_dir: Path) -> None:
    """
    Export Prism-ready wide tables for simple genome-wide vs neighborhood comparisons.
    One file per condition.
    """
    selected_gene_tags = neighborhood_gene_table["Locus tag"].unique().tolist()
    neighborhood_df = df[df["Locus tag"].isin(selected_gene_tags)].copy()

    out_aer = pd.DataFrame({
        "Genome": pd.Series(df["Median_aer"].dropna().tolist()),
        "Neighborhood_genes": pd.Series(neighborhood_df["Median_aer"].dropna().tolist()),
    })
    out_anaer = pd.DataFrame({
        "Genome": pd.Series(df["Median_anaer"].dropna().tolist()),
        "Neighborhood_genes": pd.Series(neighborhood_df["Median_anaer"].dropna().tolist()),
    })

    out_aer.to_csv(output_dir / "prism_genome_vs_neighborhood_aerobic.csv", index=False)
    out_anaer.to_csv(output_dir / "prism_genome_vs_neighborhood_anaerobic.csv", index=False)


def export_prism_per_site(df: pd.DataFrame, neighborhood_gene_table: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export Prism-ready wide tables with one genome column followed by one column per middle-tag neighborhood.
    Ordered by neighborhood score so the Prism files match the plotted figure order.
    """
    site_order = summary_df["Middle_tag"].tolist()

    aer_dict = {"Genome": pd.Series(df["Median_aer"].dropna().tolist())}
    anaer_dict = {"Genome": pd.Series(df["Median_anaer"].dropna().tolist())}

    for site in site_order:
        site_df = neighborhood_gene_table[neighborhood_gene_table["Middle_tag"] == site]
        aer_dict[site] = pd.Series(site_df["Median_aer"].dropna().tolist())
        anaer_dict[site] = pd.Series(site_df["Median_anaer"].dropna().tolist())

    pd.DataFrame(aer_dict).to_csv(output_dir / "prism_neighborhoods_aerobic.csv", index=False)
    pd.DataFrame(anaer_dict).to_csv(output_dir / "prism_neighborhoods_anaerobic.csv", index=False)


def export_prism_middle_gene_points(neighborhood_gene_table: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export the middle-gene values only, useful for manual overlay/annotation in GraphPad Prism.
    """
    middle_df = neighborhood_gene_table[neighborhood_gene_table["Is_middle"]].copy()
    site_order = summary_df["Middle_tag"].tolist()
    middle_df["Middle_tag"] = pd.Categorical(middle_df["Middle_tag"], categories=site_order, ordered=True)
    middle_df = middle_df.sort_values("Middle_tag")

    out = middle_df[["Middle_tag", "Locus tag", "Median_aer", "Median_anaer"]].rename(columns={
        "Middle_tag": "Site",
        "Locus tag": "Middle_gene",
        "Median_aer": "Aerobic_middle_gene",
        "Median_anaer": "Anaerobic_middle_gene",
    })
    out.to_csv(output_dir / "prism_middle_gene_points.csv", index=False)

def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_input(df)

    all_neighborhoods = []
    summaries = []

    for tag in args.middle_tags:
        neighborhood = get_neighborhood(df, tag, args.window_size)
        all_neighborhoods.append(neighborhood)
        summaries.append(summarize_neighborhood(neighborhood))

    neighborhood_gene_table = pd.concat(all_neighborhoods, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # Rank by average neighborhood transcriptional burden
    summary_df["Neighborhood_score"] = (
        summary_df["Median_of_Median_aer"] + summary_df["Median_of_Median_anaer"]
    ) / 2
    summary_df = summary_df.sort_values("Neighborhood_score", ascending=True).reset_index(drop=True)

    # Genome-wide baselines
    genome_summary = {
        "Genome_median_aer": df["Median_aer"].median(),
        "Genome_median_anaer": df["Median_anaer"].median(),
        "Genome_mean_aer": df["Median_aer"].mean(),
        "Genome_mean_anaer": df["Median_anaer"].mean(),
        "Genome_q1_aer": df["Median_aer"].quantile(0.25),
        "Genome_q3_aer": df["Median_aer"].quantile(0.75),
        "Genome_q1_anaer": df["Median_anaer"].quantile(0.25),
        "Genome_q3_anaer": df["Median_anaer"].quantile(0.75),
    }
    pd.DataFrame([genome_summary]).to_csv(output_dir / "genome_medians.csv", index=False)

    summary_df["Delta_vs_genome_aer"] = summary_df["Median_of_Median_aer"] - genome_summary["Genome_median_aer"]
    summary_df["Delta_vs_genome_anaer"] = summary_df["Median_of_Median_anaer"] - genome_summary["Genome_median_anaer"]

    selected_gene_tags = neighborhood_gene_table["Locus tag"].unique().tolist()
    selected_genes = df[df["Locus tag"].isin(selected_gene_tags)].copy()

    p_aer = safe_mannwhitney(selected_genes["Median_aer"], df["Median_aer"])
    p_anaer = safe_mannwhitney(selected_genes["Median_anaer"], df["Median_anaer"])

    neighborhood_gene_table.to_csv(output_dir / "neighborhood_gene_table.csv", index=False)
    summary_df.to_csv(output_dir / "neighborhood_summary.csv", index=False)

    compare_df = pd.DataFrame({
        "Locus tag": df["Locus tag"],
        "Median_aer": df["Median_aer"],
        "Median_anaer": df["Median_anaer"],
        "Is_selected_neighborhood_gene": df["Locus tag"].isin(selected_gene_tags),
    })
    compare_df.to_csv(output_dir / "genome_vs_neighborhood_gene_values.csv", index=False)

    export_prism_genome_vs_neighborhood(df, neighborhood_gene_table, output_dir)
    export_prism_per_site(df, neighborhood_gene_table, summary_df, output_dir)
    export_prism_middle_gene_points(neighborhood_gene_table, summary_df, output_dir)

    plot_scatter(summary_df, genome_summary, output_dir / "neighborhood_scatter.png")
    plot_boxplots(df, selected_gene_tags, output_dir / "genome_vs_neighborhood_boxplots.png")
    plot_condition_comparison_figure(
        df,
        neighborhood_gene_table,
        summary_df,
        condition_col="Median_aer",
        condition_label="Aerobic",
        output_path=output_dir / "combined_neighborhood_comparison_aerobic.png",
    )
    plot_condition_comparison_figure(
        df,
        neighborhood_gene_table,
        summary_df,
        condition_col="Median_anaer",
        condition_label="Anaerobic",
        output_path=output_dir / "combined_neighborhood_comparison_anaerobic.png",
    )

    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write("Neighborhood expression analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Window size: {args.window_size}\n")
        f.write(f"Middle tags analyzed: {', '.join(args.middle_tags)}\n\n")

        f.write("Genome-wide baselines (using Median_* columns)\n")
        f.write("-" * 40 + "\n")
        for k, v in genome_summary.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n")

        f.write("Gene-level comparison (selected neighborhood genes vs genome)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Selected neighborhood genes: {len(selected_gene_tags)}\n")
        f.write(f"Aerobic median (selected genes): {selected_genes['Median_aer'].median():.4f}\n")
        f.write(f"Anaerobic median (selected genes): {selected_genes['Median_anaer'].median():.4f}\n")
        f.write(f"Aerobic Mann-Whitney p-value: {p_aer if p_aer is not None else 'NA (scipy not available)'}\n")
        f.write(f"Anaerobic Mann-Whitney p-value: {p_anaer if p_anaer is not None else 'NA (scipy not available)'}\n\n")

        f.write("Site-level neighborhood summary\n")
        f.write("-" * 40 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['Middle_tag']}\n")
            f.write(f"  Neighborhood tags: {row['Neighborhood_locus_tags']}\n")
            f.write(f"  Median_of_Median_aer: {row['Median_of_Median_aer']:.4f}\n")
            f.write(f"  Median_of_Median_anaer: {row['Median_of_Median_anaer']:.4f}\n")
            f.write(f"  Neighborhood_score: {row['Neighborhood_score']:.4f}\n")
            f.write(f"  Delta_median_conditions: {row['Delta_median_conditions']:.4f}\n")
            f.write(f"  Delta_vs_genome_aer: {row['Delta_vs_genome_aer']:.4f}\n")
            f.write(f"  Delta_vs_genome_anaer: {row['Delta_vs_genome_anaer']:.4f}\n\n")

    print(f"Analysis complete. Output written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
