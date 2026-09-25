"""Calculate active empirical ZU by family size using Limor's method."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings

from publication_style import FIGURE_WIDTH_IN, apply_publication_style, save_png_and_pdf
REQUIRED_COLUMNS = [
    "misparmb", "persons_count", "c3_actual", "food_actual",
    "nonfood_actual", "FoodNorm-active",
]
TOLERANCE = 0.05
MAX_FAMILY_SIZE = 9
DEFINITIONS = (
    ("foodnorm", "Food actual < FoodNorm", "food_actual", "FoodNorm-active-empirical", "#2478a8", "o"),
    ("zl", "C3 actual < empirical ZL", "c3_actual", "ZL-active-empirical", "#e67e22", "s"),
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


def build_empirical_thresholds(summary, zl_path):
    """Build one empirical FoodNorm/ZL/ZU value per household size.

    ZL values are Daniel's explicitly labelled spreadsheet/chart values. Blank
    cells remain missing: they are never interpolated or regression-filled.
    """
    zl = pd.read_csv(zl_path)
    required = {"persons_count", "ZL-active-empirical-daniel"}
    if not required.issubset(zl.columns):
        raise KeyError(f"Missing empirical-ZL columns: {sorted(required-set(zl.columns))}")
    thresholds = summary.loc[
        summary["persons_count"].between(1, MAX_FAMILY_SIZE),
        ["persons_count", "all_households", "qualifying_households",
         "poorest_household_id", "poorest_foodnorm_active", "ZU-active-empirical"],
    ].copy()
    thresholds = thresholds.merge(zl, on="persons_count", how="left", validate="one_to_one")
    thresholds = thresholds.rename(columns={
        "poorest_foodnorm_active": "FoodNorm-active-empirical",
        "ZL-active-empirical-daniel": "ZL-active-empirical",
    })
    missing = thresholds.loc[thresholds["ZL-active-empirical"].isna(), "persons_count"].tolist()
    if missing:
        warnings.warn(
            "No empirical ZL exists for household sizes "
            f"{missing}; values remain missing and are not filled.", RuntimeWarning,
        )
    return thresholds


def assign_thresholds(households, thresholds):
    cols = ["persons_count", "FoodNorm-active-empirical",
            "ZL-active-empirical", "ZU-active-empirical"]
    return households.drop(columns=["ZU-active-empirical"], errors="ignore").merge(
        thresholds[cols], on="persons_count", how="left", validate="many_to_one"
    )


def fgt2(actual, threshold):
    valid = actual.notna() & threshold.notna() & (threshold > 0)
    actual = actual.loc[valid]
    threshold = threshold.loc[valid]
    if len(actual) == 0:
        return np.nan, 0, 0
    poor = actual < threshold
    squared_gaps = ((threshold.loc[poor] - actual.loc[poor]) / threshold.loc[poor]) ** 2
    return float(squared_gaps.sum() / len(actual)), int(poor.sum()), len(actual)


def summarize_fgt2_by_size(households):
    sample = households.loc[households["persons_count"].between(1, MAX_FAMILY_SIZE)].copy()
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


def summarize_rates_by_size(households):
    sample = households.loc[households["persons_count"].between(1, MAX_FAMILY_SIZE)].copy()
    rows = []
    for size, group in sample.groupby("persons_count", sort=True):
        row = {"family_size": int(size), "n_families": len(group)}
        for key, _, actual_col, threshold_col, _, _ in DEFINITIONS:
            valid = group[actual_col].notna() & group[threshold_col].notna()
            total = int(valid.sum())
            poor = int((group.loc[valid, actual_col] < group.loc[valid, threshold_col]).sum())
            row[f"poor_households_{key}"] = poor
            row[f"valid_households_{key}"] = total
            if total:
                row[f"poverty_rate_percent_{key}"] = 100 * poor / total
                lo, hi = wilson_interval(poor, total)
                row[f"ci95_lower_percent_{key}"] = lo
                row[f"ci95_upper_percent_{key}"] = hi
            else:
                row[f"poverty_rate_percent_{key}"] = np.nan
                row[f"ci95_lower_percent_{key}"] = np.nan
                row[f"ci95_upper_percent_{key}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_depth_by_size(households):
    sample = households.loc[households["persons_count"].between(1, MAX_FAMILY_SIZE)].copy()
    rows = []
    for size, group in sample.groupby("persons_count", sort=True):
        row = {"family_size": int(size), "n_families": len(group)}
        for key, _, actual_col, threshold_col, _, _ in DEFINITIONS:
            valid = group[actual_col].notna() & group[threshold_col].notna() & (group[threshold_col] > 0)
            poor = valid & (group[actual_col] < group[threshold_col])
            gaps = 100 * (group.loc[poor, threshold_col] - group.loc[poor, actual_col]) / group.loc[poor, threshold_col]
            row[f"poor_households_{key}"] = int(poor.sum())
            row[f"mean_depth_percent_{key}"] = gaps.mean() if len(gaps) else np.nan
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


def draw_thresholds(thresholds, output_path):
    apply_publication_style()
    x = thresholds["persons_count"].to_numpy()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.8))
    series = (
        ("FoodNorm-active-empirical", "FoodNorm", "#2478a8", "o"),
        ("ZL-active-empirical", "Empirical ZL", "#e67e22", "s"),
        ("ZU-active-empirical", "Empirical ZU", "#7251a3", "^"),
    )
    for col, label, color, marker in series:
        ax.plot(x, thresholds[col], marker=marker, color=color, linewidth=1.8,
                markersize=6, label=label)
    ax.set_xticks(x)
    ax.set_xlabel("People in household")
    ax.set_ylabel("NIS/month")
    ax.set_title("Active households: empirical FoodNorm, ZL and ZU by family size")
    ax.legend()
    ax.grid(alpha=.22)
    fig.tight_layout()
    save_png_and_pdf(fig, output_path)
    plt.close(fig)


def draw_rates_by_size(summary, output_path):
    apply_publication_style()
    x = summary["family_size"].to_numpy()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.8))
    for key, label, _, _, color, marker in DEFINITIONS:
        y = summary[f"poverty_rate_percent_{key}"].to_numpy()
        lo = summary[f"ci95_lower_percent_{key}"].to_numpy()
        hi = summary[f"ci95_upper_percent_{key}"].to_numpy()
        ax.errorbar(x, y, yerr=[y-lo, hi-y], marker=marker, color=color,
                    linewidth=1.8, markersize=5, capsize=3, label=label)
    ax.set_xticks(x, [f"{size}\nn={count:,}" for size, count in zip(x, summary["n_families"])])
    ax.set_ylim(0, 100)
    ax.set_xlabel("People in household / sampled households (n)")
    ax.set_ylabel("Households below threshold (%)")
    ax.set_title("Active households: poverty rates by family size (95% Wilson CI)")
    ax.legend()
    ax.grid(alpha=.22)
    fig.tight_layout()
    save_png_and_pdf(fig, output_path)
    plt.close(fig)


def draw_depth_by_size(summary, output_path):
    apply_publication_style()
    x = summary["family_size"].to_numpy()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.8))
    for key, label, _, _, color, marker in DEFINITIONS:
        ax.plot(x, summary[f"mean_depth_percent_{key}"], marker=marker,
                color=color, linewidth=1.8, markersize=6, label=label)
    ax.set_xticks(x, [f"{size}\nn={count:,}" for size, count in zip(x, summary["n_families"])])
    ax.set_xlabel("People in household / sampled households (n)")
    ax.set_ylabel("Mean poverty depth among poor households (%)")
    ax.set_title("Active households: poverty depth by family size")
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


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("empirical_zu_household_inputs.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("2026-latest-empirical-zu"))
    parser.add_argument("--empirical-zl", type=Path,
                        default=Path("empirical_thresholds_by_family_size.csv"),
                        help="Daniel's empirical ZL values by household size")
    args = parser.parse_args()

    households = prepare(pd.read_csv(args.input))
    qualifying, summary = derive_zu_by_size(households)
    thresholds = build_empirical_thresholds(summary, args.empirical_zl)
    analysis = assign_thresholds(households, thresholds)
    rates_by_size = summarize_rates_by_size(analysis)
    depth_by_size = summarize_depth_by_size(analysis)
    fgt2_summary = summarize_fgt2_by_size(analysis)
    overall_rates = summarize_overall_rates(
        analysis.loc[analysis["persons_count"].between(1, MAX_FAMILY_SIZE)]
    )
    args.output.mkdir(parents=True, exist_ok=True)

    qualifying.to_csv(args.output / "empirical_zu_qualifying_active_5pct.csv",
                      index=False, float_format="%.10f")
    summary.to_csv(args.output / "empirical_zu_by_family_size.csv",
                   index=False, float_format="%.10f")
    thresholds.to_csv(args.output / "active_empirical_thresholds_by_family_size_1_to_9.csv",
                      index=False, float_format="%.10f")
    analysis.to_csv(args.output / "households_with_empirical_thresholds.csv",
                    index=False, float_format="%.10f")
    rates_by_size.to_csv(args.output / "active_empirical_poverty_rates_by_family_size_1_to_9.csv",
                         index=False, float_format="%.10f")
    depth_by_size.to_csv(args.output / "active_empirical_poverty_depth_by_family_size_1_to_9.csv",
                         index=False, float_format="%.10f")
    fgt2_summary.to_csv(args.output / "active_empirical_fgt2_by_family_size_1_to_9.csv",
                        index=False, float_format="%.10f")
    overall_rates.to_csv(args.output / "active_empirical_zu_overall_poverty_rates.csv",
                         index=False, float_format="%.10f")
    draw_thresholds(thresholds, args.output / "active_empirical_thresholds_by_family_size_1_to_9.png")
    draw_rates_by_size(rates_by_size,
                       args.output / "active_empirical_poverty_rates_by_family_size_1_to_9.png")
    draw_depth_by_size(depth_by_size,
                       args.output / "active_empirical_poverty_depth_by_family_size_1_to_9.png")
    draw_fgt2(fgt2_summary,
              args.output / "active_empirical_fgt2_by_family_size_1_to_9.png")
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
        "All five figures use one empirical threshold per family size, through size 9. "
        "FoodNorm and ZU come from the selected ZU household. Empirical ZL values are "
        "Daniel's labelled comparison values. Missing ZL sizes are deliberately left blank, "
        "warned about, and excluded only from ZL denominators. Poverty depth is the mean "
        "percentage gap among poor households; FGT2 includes non-poor households as zero.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Qualifying households: {len(qualifying)}")
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
