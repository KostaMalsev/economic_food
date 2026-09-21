"""Plot mean active-household poverty depth (%) by family size (1 through 7)."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from publication_style import apply_publication_style, save_png_and_pdf
from run import FamilyGroupAnalyzer


SERIES = (
    ("foodnorm_minus_food_actual", "FoodNorm − food actual", "FoodNorm-active", "food_actual", "#2478a8", "o"),
    ("zl_minus_c3", "ZL − C3 actual", "ZL-active", "c3", "#e67e22", "s"),
    ("zu_minus_c3", "ZU − C3 actual", "ZU-active", "c3", "#7251a3", "^"),
)


def mean_t_interval(values):
    values = np.asarray(values, dtype=float)
    count = len(values)
    mean = float(values.mean())
    if count < 2:
        return mean, np.nan, np.nan, np.nan
    standard_deviation = float(values.std(ddof=1))
    half_width = float(t.ppf(0.975, count - 1) * standard_deviation / np.sqrt(count))
    return mean, standard_deviation, mean - half_width, mean + half_width


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
    rows = []
    for family_size, group in sample.groupby("persons_count", sort=True):
        row = {"family_size": int(family_size), "n_families": len(group)}
        for key, _, threshold_col, actual_col, _, _ in SERIES:
            # Normalize each poor household's gap by its own threshold before
            # averaging. This is not mean(NIS gap) / mean(threshold).
            valid = (group[threshold_col] > 0) & (group[actual_col] < group[threshold_col])
            depths = (
                (group.loc[valid, threshold_col] - group.loc[valid, actual_col])
                / group.loc[valid, threshold_col]
                * 100
            )
            mean, standard_deviation, lower, upper = mean_t_interval(depths)
            row[f"n_{key}"] = len(depths)
            row[f"mean_{key}"] = mean
            row[f"sd_{key}"] = standard_deviation
            row[f"ci95_lower_{key}"] = lower
            row[f"ci95_upper_{key}"] = upper
        rows.append(row)
    return pd.DataFrame(rows), len(df) - len(sample)


def draw(summary, path):
    apply_publication_style()
    x = summary["family_size"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for key, label, _, _, color, marker in SERIES:
        y = summary[f"mean_{key}"].to_numpy()
        lower = summary[f"ci95_lower_{key}"].to_numpy()
        upper = summary[f"ci95_upper_{key}"].to_numpy()
        ax.errorbar(
            x, y, yerr=[y - lower, upper - y], fmt=f"{marker}-",
            label=label, color=color, capsize=4, linewidth=1.8, markersize=6,
        )
    tick_labels = []
    for _, row in summary.iterrows():
        tick_labels.append(
            f"{int(row.family_size)}\n"
            f"n FN={int(row.n_foodnorm_minus_food_actual):,}\n"
            f"n ZL={int(row.n_zl_minus_c3):,}\n"
            f"n ZU={int(row.n_zu_minus_c3):,}"
        )
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("People in household / qualifying households for each series (n)")
    ax.set_ylabel("Mean poverty depth (%)")
    ax.set_title("Active households: mean poverty depth among households below each threshold (95% t CI)")
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
    stem = "active_mean_poverty_depth_percent_below_foodnorm_zl_zu_by_family_size_1_to_7"
    summary.to_csv(args.output / f"{stem}.csv", index=False, float_format="%.6f")
    draw(summary, args.output / f"{stem}.png")
    (args.output / f"{stem}_README.txt").write_text(
        f"Input: {args.input.name}. Included family-size universe: "
        f"{int(summary.n_families.sum())} households; excluded: {excluded}.\n"
        "Active households and family sizes 1-7 only. For each qualifying household, "
        "poverty depth (%) = (threshold - actual) / threshold * 100. Each mean is "
        "conditional on being strictly below its corresponding threshold: food_actual < "
        "FoodNorm-active; c3 < ZL-active; or c3 < ZU-active. Ties do not count. A "
        "household percentage is calculated first and those percentages are then averaged.\n"
        "Error bars are unweighted 95% Student t confidence intervals for each conditional "
        "mean. The n labels are the qualifying households for that series and family size, "
        "not the total number of sampled households in the family-size group. Intervals do "
        "not account for the survey sampling design. Units are percentage points.\n"
        "All figure text uses a consistent 10-point publication font. The PNG is exported "
        "at 300 DPI and the matching PDF is scalable.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
