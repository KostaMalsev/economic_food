"""Calculate active empirical ZU by family size using Limor's method."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from publication_style import FIGURE_WIDTH_IN, apply_publication_style, save_png_and_pdf
from run import FamilyGroupAnalyzer


REQUIRED_COLUMNS = [
    "misparmb", "persons_count", "c3_actual", "food_actual",
    "nonfood_actual", "FoodNorm-active",
]
TOLERANCE = 0.05
DEFINITIONS = (
    ("foodnorm", "Food actual < FoodNorm", "food_actual", "FoodNorm-active", "#2478a8", "o"),
    ("zl", "C3 actual < ZL", "c3_actual", "ZL-active", "#e67e22", "s"),
    ("zu", "C3 actual < empirical ZU", "c3_actual", "ZU-active-empirical", "#7251a3", "^"),
)


def prepare(data):
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    households = data.copy()
    households[REQUIRED_COLUMNS] = households[REQUIRED_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    if households["misparmb"].duplicated().any():
        raise ValueError("Input must contain one row per household")
    if households[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required empirical-ZU inputs contain missing values")
    if (households["persons_count"] <= 0).any():
        raise ValueError("persons_count must be positive")

    households["foodnorm_relative_difference"] = (
        (households["food_actual"] - households["FoodNorm-active"])
        / households["FoodNorm-active"]
    )
    households["qualifies_foodnorm_5pct"] = (
        households["foodnorm_relative_difference"].abs() <= TOLERANCE
    )
    households["c3_actual_per_capita"] = (
        households["c3_actual"] / households["persons_count"]
    )
    return households


def derive_zu_by_size(households):
    qualifying = households.loc[households["qualifies_foodnorm_5pct"]].copy()
    if len(qualifying) != 499:
        raise ValueError(
            f"Expected 499 qualifying households from the source workbook, found {len(qualifying)}"
        )
    # A deterministic secondary key resolves a theoretical per-capita tie.
    qualifying = qualifying.sort_values(
        ["persons_count", "c3_actual_per_capita", "misparmb"]
    )
    poorest = qualifying.groupby("persons_count", as_index=False).first()
    qualifying_counts = qualifying.groupby("persons_count").size().rename(
        "qualifying_households"
    )
    total_counts = households.groupby("persons_count").size().rename("all_households")

    summary = pd.DataFrame(index=sorted(households["persons_count"].unique()))
    summary.index.name = "persons_count"
    summary = summary.join(total_counts).join(qualifying_counts).fillna(
        {"qualifying_households": 0}
    )
    summary["qualifying_households"] = summary["qualifying_households"].astype(int)
    selected = poorest.set_index("persons_count")[[
        "misparmb", "c3_actual", "c3_actual_per_capita", "food_actual",
        "nonfood_actual", "FoodNorm-active",
    ]].rename(columns={
        "misparmb": "poorest_household_id",
        "c3_actual": "poorest_c3_actual",
        "c3_actual_per_capita": "poorest_c3_actual_per_capita",
        "food_actual": "poorest_food_actual",
        "nonfood_actual": "poorest_nonfood_actual",
        "FoodNorm-active": "poorest_foodnorm_active",
    })
    summary = summary.join(selected)
    summary["ZU-active-empirical"] = (
        summary["poorest_foodnorm_active"] + summary["poorest_nonfood_actual"]
    )
    summary["has_empirical_zu"] = summary["ZU-active-empirical"].notna()
    return qualifying, summary.reset_index()


def assign_zu(households, summary):
    assignment = summary[["persons_count", "ZU-active-empirical"]]
    assigned = households.merge(assignment, on="persons_count", how="left", validate="many_to_one")
    selected_ids = set(summary["poorest_household_id"].dropna().astype(int))
    assigned["is_selected_poorest_household"] = assigned["misparmb"].isin(selected_ids)
    assigned["c3_actual_minus_empirical_zu"] = (
        assigned["c3_actual"] - assigned["ZU-active-empirical"]
    )
    assigned["c3_actual_below_empirical_zu"] = (
        assigned["c3_actual"] < assigned["ZU-active-empirical"]
    ).where(assigned["ZU-active-empirical"].notna())
    return assigned


def attach_regression_zl(households, regression_path):
    """Attach only the existing ZL result; do not use the old food_actual column."""
    regression = pd.read_csv(regression_path)
    if "ZL-active" not in regression.columns:
        analyzer = FamilyGroupAnalyzer(str(regression_path))
        if not analyzer.read_csv() or not analyzer.process_dataframe():
            raise RuntimeError("Could not calculate the existing regression-based ZL")
        regression = analyzer.df
    regression = regression[["misparmb", "ZL-active"]].copy()
    if regression["misparmb"].duplicated().any():
        raise ValueError("Regression input contains duplicate household IDs")
    combined = households.merge(regression, on="misparmb", how="left", validate="one_to_one")
    if combined["ZL-active"].isna().any():
        raise ValueError("Some households have no existing regression-based ZL")
    return combined


def fgt2(actual, threshold):
    valid = actual.notna() & threshold.notna() & (threshold > 0)
    actual = actual.loc[valid]
    threshold = threshold.loc[valid]
    poor = actual < threshold
    squared_gaps = ((threshold.loc[poor] - actual.loc[poor]) / threshold.loc[poor]) ** 2
    return float(squared_gaps.sum() / len(actual)), int(poor.sum()), len(actual)


def summarize_fgt2_by_size(households):
    sample = households.loc[households["persons_count"].between(1, 7)].copy()
    rows = []
    for size, group in sample.groupby("persons_count", sort=True):
        row = {"family_size": int(size), "n_families": len(group)}
        for key, _, actual_col, threshold_col, _, _ in DEFINITIONS:
            value, poor, valid = fgt2(group[actual_col], group[threshold_col])
            row[f"poor_households_{key}"] = poor
            row[f"valid_households_{key}"] = valid
            row[f"fgt2_{key}"] = value
            row[f"fgt2_percent_{key}"] = 100 * value
        rows.append(row)
    return pd.DataFrame(rows)


def wilson_interval(successes, total):
    z = norm.ppf(.975)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return 100 * (center - half_width), 100 * (center + half_width)


def summarize_overall_rates(households):
    rows = []
    for key, label, actual_col, threshold_col, _, _ in DEFINITIONS:
        valid = households[actual_col].notna() & households[threshold_col].notna()
        actual = households.loc[valid, actual_col]
        threshold = households.loc[valid, threshold_col]
        poor = int((actual < threshold).sum())
        total = len(actual)
        lower, upper = wilson_interval(poor, total)
        value, _, _ = fgt2(actual, threshold)
        rows.append({
            "poverty_definition": key,
            "label": label,
            "poor_households": poor,
            "valid_households": total,
            "poverty_rate_percent": 100 * poor / total,
            "ci95_lower_percent": lower,
            "ci95_upper_percent": upper,
            "fgt2": value,
            "fgt2_percent": 100 * value,
        })
    return pd.DataFrame(rows)


def draw_fgt2(summary, output_path):
    apply_publication_style()
    x = summary["family_size"].to_numpy()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.8))
    for key, label, _, _, color, marker in DEFINITIONS:
        ax.plot(x, summary[f"fgt2_percent_{key}"], marker=marker, color=color,
                linewidth=1.8, markersize=6, label=label)
    ax.set_xticks(x, [f"{size}\nn={count:,}" for size, count in
                      zip(x, summary["n_families"])])
    ax.set_xlabel("People in household / sampled households (n)")
    ax.set_ylabel("FGT₂ × 100 (%)")
    ax.set_title("Active households: FGT₂ poverty depth with empirical ZU")
    ax.legend()
    ax.grid(alpha=.22)
    fig.tight_layout()
    save_png_and_pdf(fig, output_path)
    plt.close(fig)


def draw_overall_rates(summary, output_path):
    apply_publication_style()
    x = np.arange(len(summary))
    rates = summary["poverty_rate_percent"].to_numpy()
    lower = summary["ci95_lower_percent"].to_numpy()
    upper = summary["ci95_upper_percent"].to_numpy()
    colors = [item[4] for item in DEFINITIONS]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.8))
    bars = ax.bar(x, rates, color=colors, width=.62)
    ax.errorbar(x, rates, yerr=[rates-lower, upper-rates], fmt="none",
                ecolor="#333333", capsize=5, linewidth=1.2)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f"{row.poverty_rate_percent:.1f}%\n"
                f"{int(row.poor_households):,}/{int(row.valid_households):,}",
                ha="center", va="bottom")
    ax.set_xticks(x, ["Food actual\n< FoodNorm", "C3 actual\n< ZL",
                      "C3 actual\n< empirical ZU"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Households below threshold (%)")
    ax.set_title("Active households: overall poverty rates with empirical ZU (95% Wilson CI)")
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout()
    save_png_and_pdf(fig, output_path)
    plt.close(fig)


def draw_summary(summary, output_path):
    apply_publication_style()
    shown = summary.loc[summary["persons_count"].between(1, 7)].copy()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.8))
    ax.plot(shown["persons_count"], shown["ZU-active-empirical"], marker="o",
            color="#7251a3", linewidth=1.8)
    for _, row in shown.iterrows():
        ax.annotate(
            f"ID {int(row.poorest_household_id)}\nn={int(row.qualifying_households)}",
            (row.persons_count, row["ZU-active-empirical"]),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
        )
    ax.set_xticks(shown["persons_count"])
    ax.set_xlabel("People in household")
    ax.set_ylabel("Empirical ZU (NIS/month)")
    ax.set_title("Active empirical ZU by family size")
    ax.grid(alpha=.22)
    fig.tight_layout()
    save_png_and_pdf(fig, output_path)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("empirical_zu_household_inputs.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("2026-latest-empirical-zu"))
    parser.add_argument("--regression-input", type=Path,
                        default=Path("food_economics_2024.csv"),
                        help="Existing regression output; only ZL-active is read")
    args = parser.parse_args()

    households = prepare(pd.read_csv(args.input))
    qualifying, summary = derive_zu_by_size(households)
    assigned = assign_zu(households, summary)
    analysis = attach_regression_zl(assigned, args.regression_input)
    fgt2_summary = summarize_fgt2_by_size(analysis)
    overall_rates = summarize_overall_rates(analysis)
    args.output.mkdir(parents=True, exist_ok=True)

    qualifying.to_csv(args.output / "empirical_zu_qualifying_active_5pct.csv",
                      index=False, float_format="%.10f")
    summary.to_csv(args.output / "empirical_zu_by_family_size.csv",
                   index=False, float_format="%.10f")
    assigned.to_csv(args.output / "households_with_empirical_zu.csv",
                    index=False, float_format="%.10f")
    fgt2_summary.to_csv(args.output / "active_empirical_zu_fgt2_by_family_size_1_to_7.csv",
                        index=False, float_format="%.10f")
    overall_rates.to_csv(args.output / "active_empirical_zu_overall_poverty_rates.csv",
                         index=False, float_format="%.10f")
    draw_summary(summary, args.output / "empirical_zu_by_family_size_1_to_7.png")
    draw_fgt2(fgt2_summary,
              args.output / "active_empirical_zu_fgt2_by_family_size_1_to_7.png")
    draw_overall_rates(overall_rates,
                       args.output / "active_empirical_zu_overall_poverty_rates.png")

    missing_sizes = summary.loc[~summary["has_empirical_zu"], "persons_count"].tolist()
    (args.output / "README.txt").write_text(
        "Active empirical ZU implementation of Limor's 2026-09-25 method.\n"
        "Qualification: abs(food_actual - FoodNorm-active) <= 5% of FoodNorm-active.\n"
        "Within each family size, the poorest qualifying household is the household "
        "with minimum actual C3 per person. Ties are resolved by smallest household ID.\n"
        "ZU-active-empirical = the selected household's FoodNorm-active + its actual "
        "C32-C39 non-food expenditure. That ZU is assigned to all sample households "
        "of the same size. The regression-based ZU implementation is unchanged.\n"
        f"Qualifying households: {len(qualifying)}. Family sizes without a qualifying "
        f"household and therefore without an empirical ZU: {missing_sizes}.\n"
        "The two requested poverty figures use corrected spreadsheet food_actual, the "
        "existing regression-based ZL-active without recalculation, and the empirical ZU. "
        "FGT2 uses family sizes 1-7, where empirical ZU coverage is complete. The overall "
        "FoodNorm and ZL rates use all 9,017 households; empirical ZU uses the 8,950 "
        "households in size groups with a defined empirical ZU.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Qualifying households: {len(qualifying)}")
    print(f"Assigned households: {assigned['ZU-active-empirical'].notna().sum():,}/{len(assigned):,}")


if __name__ == "__main__":
    main()
