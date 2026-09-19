"""Active-household FGT2 depth and overall poverty rates using 2017 expenditures."""

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


ALPHA = 2
DEFINITIONS = (
    ("foodnorm", "Food actual < FoodNorm", "food_actual", "FoodNorm-active", "#2478a8", "o"),
    ("zl", "C3 actual < ZL", "c3", "ZL-active", "#e67e22", "s"),
    ("zu", "C3 actual < ZU", "c3", "ZU-active", "#7251a3", "^"),
)


def wilson_interval(successes, total):
    z = norm.ppf(0.975)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return 100 * (center - half_width), 100 * (center + half_width)


def fgt2(actual, threshold):
    """Return FGT(alpha=2), H and N; non-poor households contribute zero."""
    valid = actual.notna() & threshold.notna() & np.isfinite(actual) & np.isfinite(threshold)
    valid &= threshold > 0
    actual = actual.loc[valid]
    threshold = threshold.loc[valid]
    poor = actual < threshold
    normalized_gap = ((threshold.loc[poor] - actual.loc[poor]) / threshold.loc[poor]) ** ALPHA
    return normalized_gap.sum() / len(actual), int(poor.sum()), len(actual)


def prepare(df):
    columns = [
        "persons_count", "food_actual", "FoodNorm-active",
        "c3", "ZL-active", "ZU-active",
    ]
    return df[columns].replace([np.inf, -np.inf], np.nan).copy()


def summarize_by_family_size(sample):
    sized = sample.loc[
        sample["persons_count"].between(1, 7)
        & sample["persons_count"].notna()
        & (sample["persons_count"] == sample["persons_count"].astype(int))
    ]
    rows = []
    for family_size, group in sized.groupby("persons_count", sort=True):
        row = {"family_size": int(family_size), "n_families": len(group)}
        for key, _, actual_col, threshold_col, _, _ in DEFINITIONS:
            value, poor_count, denominator = fgt2(group[actual_col], group[threshold_col])
            row[f"poor_households_{key}"] = poor_count
            row[f"valid_households_{key}"] = denominator
            row[f"fgt2_{key}"] = value
            row[f"fgt2_percent_{key}"] = 100 * value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall(sample):
    rows = []
    for key, label, actual_col, threshold_col, _, _ in DEFINITIONS:
        value, poor_count, denominator = fgt2(sample[actual_col], sample[threshold_col])
        lower, upper = wilson_interval(poor_count, denominator)
        rows.append({
            "poverty_definition": key,
            "label": label,
            "poor_households": poor_count,
            "valid_households": denominator,
            "poverty_rate_percent": 100 * poor_count / denominator,
            "ci95_lower_percent": lower,
            "ci95_upper_percent": upper,
            "fgt2": value,
            "fgt2_percent": 100 * value,
        })
    return pd.DataFrame(rows)


def draw_fgt(summary, path):
    apply_publication_style()
    x = summary["family_size"].to_numpy()
    fig, ax = plt.subplots(figsize=(13, 7))
    for key, label, _, _, color, marker in DEFINITIONS:
        ax.plot(
            x, summary[f"fgt2_percent_{key}"], marker=marker, color=color,
            linewidth=1.8, markersize=6, label=label,
        )
    ax.set_xticks(
        x,
        [f"{size}\nn={count:,}" for size, count in zip(x, summary["n_families"])],
    )
    ax.set_xlabel("People in household / sampled households (n)")
    ax.set_ylabel("FGT₂ × 100 (%)")
    ax.set_title("Active households: FGT₂ poverty depth by family size (2017 expenditures)")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    save_png_and_pdf(fig, path)
    plt.close(fig)


def draw_overall_rates(summary, path):
    apply_publication_style()
    x = np.arange(len(summary))
    rates = summary["poverty_rate_percent"].to_numpy()
    lower = summary["ci95_lower_percent"].to_numpy()
    upper = summary["ci95_upper_percent"].to_numpy()
    colors = [entry[4] for entry in DEFINITIONS]
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    bars = ax.bar(x, rates, color=colors, width=0.62)
    ax.errorbar(x, rates, yerr=[rates - lower, upper - rates], fmt="none",
                ecolor="#333333", capsize=5, linewidth=1.2)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{row.poverty_rate_percent:.1f}%\n"
            f"{int(row.poor_households):,}/{int(row.valid_households):,}",
            ha="center", va="bottom",
        )
    ax.set_xticks(x, ["Food actual\n< FoodNorm", "C3 actual\n< ZL", "C3 actual\n< ZU"])
    ax.set_ylim(0, max(100, rates.max() + 15))
    ax.set_ylabel("Households below threshold (%)")
    ax.set_title("Active households: overall poverty rates (2017 expenditures, 95% Wilson CI)")
    ax.grid(axis="y", alpha=0.22)
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
    sample = prepare(analyzer.df)

    by_size = summarize_by_family_size(sample)
    overall = summarize_overall(sample)
    args.output.mkdir(parents=True, exist_ok=True)

    by_size.to_csv(
        args.output / "active_fgt2_by_family_size_1_to_7.csv",
        index=False, float_format="%.8f",
    )
    overall.to_csv(
        args.output / "active_overall_poverty_rates.csv",
        index=False, float_format="%.8f",
    )
    draw_fgt(by_size, args.output / "active_fgt2_by_family_size_1_to_7.png")
    draw_overall_rates(overall, args.output / "active_overall_poverty_rates.png")
    (args.output / "active_fgt2_and_poverty_rates_README.txt").write_text(
        f"Input: {args.input.name}. Active thresholds only. Observed food_actual and c3 "
        "remain in their supplied 2017 values; no inflation multiplier is applied.\n"
        "Definitions: food poverty is food_actual < FoodNorm-active; ZL poverty is c3 < "
        "ZL-active; ZU poverty is c3 < ZU-active. Comparisons are strict, so ties are not "
        "poor. Zero food-expenditure observations remain included because their exclusion "
        "is requirement 7, outside the present implementation of requirements 1-4.\n"
        "FGT2 = (1/N) * sum over poor households of ((z-y_i)/z)^2. N is every household "
        "with valid inputs in the relevant population, so non-poor households contribute "
        "zero. Values are displayed as FGT2 * 100 percent. This is alpha=2 gap weighting, "
        "not survey-weighting; the source contains no survey-weight column.\n"
        "The family-size graph includes sizes 1-7. Overall poverty rates use the entire "
        "valid dataset, not only sizes 1-7, and show 95% Wilson score intervals.\n",
        encoding="utf-8",
    )
    print("FGT2 by family size")
    print(by_size.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nOverall poverty rates")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
