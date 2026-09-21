"""Plot active-household threshold rates by family size (1 through 7)."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from publication_style import apply_publication_style, save_png_and_pdf
from run import FamilyGroupAnalyzer


SERIES = (
    ("food_below_foodnorm", "Food actual < FoodNorm", "food_actual", "FoodNorm-active", "#2478a8", "o"),
    ("c3_below_zl", "C3 actual < ZL", "c3", "ZL-active", "#e67e22", "s"),
    ("c3_below_zu", "C3 actual < ZU", "c3", "ZU-active", "#7251a3", "^"),
)


def wilson_interval(successes, total, z):
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return 100 * (center - half_width), 100 * (center + half_width)


def summarize(df):
    columns = [
        "persons_count", "food_actual", "FoodNorm-active",
        "c3", "ZL-active", "ZU-active",
    ]
    sample = df[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    sample = sample.loc[
        sample["persons_count"].between(1, 7)
        & (sample["persons_count"] == sample["persons_count"].astype(int))
    ]
    z = norm.ppf(0.975)
    rows = []
    for family_size, group in sample.groupby("persons_count", sort=True):
        row = {"family_size": int(family_size), "n_families": len(group)}
        for key, _, actual_col, threshold_col, _, _ in SERIES:
            successes = int((group[actual_col] < group[threshold_col]).sum())
            lower, upper = wilson_interval(successes, len(group), z)
            row[f"n_{key}"] = successes
            row[f"percent_{key}"] = 100 * successes / len(group)
            row[f"ci95_lower_{key}"] = lower
            row[f"ci95_upper_{key}"] = upper
        rows.append(row)
    return pd.DataFrame(rows), len(df) - len(sample)


def draw(summary, path):
    apply_publication_style()
    x = summary["family_size"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for key, label, _, _, color, marker in SERIES:
        y = summary[f"percent_{key}"].to_numpy()
        lower = summary[f"ci95_lower_{key}"].to_numpy()
        upper = summary[f"ci95_upper_{key}"].to_numpy()
        # All series intentionally use the exact same integer x positions.
        ax.errorbar(
            x, y, yerr=[y - lower, upper - y], fmt=f"{marker}-",
            label=label, color=color, capsize=4, linewidth=1.8, markersize=6,
        )
    ax.set_ylim(0, 100)
    ax.set_xticks(
        x,
        [f"{size}\nn={count:,}" for size, count in zip(x, summary["n_families"])],
    )
    ax.set_xlabel("People in household / sampled families (n)")
    ax.set_ylabel("Households below threshold (%)")
    ax.set_title("Active households below FoodNorm, ZL and ZU by family size (95% Wilson CI)")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    save_png_and_pdf(fig, path)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("food_economics_2024.csv"))
    parser.add_argument("--output", type=Path, default=Path("2026-latet"))
    args = parser.parse_args()

    analyzer = FamilyGroupAnalyzer(args.input)
    if not analyzer.read_csv():
        raise RuntimeError(f"Could not read {args.input}")
    analyzer.process_dataframe()
    summary, excluded = summarize(analyzer.df)
    if summary.empty:
        raise RuntimeError("No households have valid values and family size 1-7")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "active_below_foodnorm_zl_zu_by_family_size_1_to_7.csv"
    chart_path = args.output / "active_below_foodnorm_zl_zu_by_family_size_1_to_7.png"
    readme_path = args.output / "active_below_foodnorm_zl_zu_by_family_size_1_to_7_README.txt"
    summary.to_csv(csv_path, index=False, float_format="%.6f")
    draw(summary, chart_path)
    readme_path.write_text(
        f"Input: {args.input.name}. Included: {int(summary.n_families.sum())} households; "
        f"excluded: {excluded} (including family sizes outside 1-7 or missing inputs).\n"
        "Active households only. Series use strict comparisons: food_actual < "
        "FoodNorm-active, c3 < ZL-active, and c3 < ZU-active. Ties do not count. "
        "food_actual is observed C30+C31; c3 is observed total household expenditure. "
        "ZL-active and ZU-active are household-level modeled thresholds from run.py.\n"
        "The CSV reports the numerator, denominator, percentage, and 95% Wilson score "
        "interval for each series. Results are unweighted and intervals do not account "
        "for the survey sampling design. All series use identical x coordinates.\n"
        "All figure text uses a consistent 10-point publication font. The PNG is exported "
        "at 300 DPI and the matching PDF is scalable.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
