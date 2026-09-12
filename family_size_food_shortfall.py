"""Plot the share spending less on food than sedentary FoodNorm by family size."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


AGE_COLUMNS = (
    "0 -4 min1", "0 -4 min2", "5 - 9 min1", "5 - 9 min2",
    "10-14 min1", "10-14 min2", "15 - 17 min1", "15 - 17 min2",
    "18 -29 min1", "18 -29 min2", "30 - 49 min1", "30 - 49 min2",
    "50+ min1", "50+ min2",
)


def summarize(df):
    family_size = df[list(AGE_COLUMNS)].sum(axis=1, min_count=len(AGE_COLUMNS))
    sample = pd.DataFrame({
        "family_size": family_size,
        "food_actual": df["food_actual"],
        "FoodNorm_sedentary": df["FoodNorm-sedentary"],
    }).replace([np.inf, -np.inf], np.nan).dropna()
    sample = sample.loc[
        (sample.family_size >= 1) & (sample.family_size < 15)
        & (sample.family_size == sample.family_size.astype(int))
    ]
    z = norm.ppf(.975)
    rows = []
    for family_size, group in sample.groupby("family_size", sort=True):
        n = len(group)
        k = int((group.food_actual < group.FoodNorm_sedentary).sum())
        p = k / n
        # Wilson score interval, appropriate for a binomial proportion.
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        rows.append({"family_size": int(family_size), "n_families": n,
                     "n_below_norm": k, "percent_below_norm": 100*p,
                     "ci95_lower_percent": 100*(center-half),
                     "ci95_upper_percent": 100*(center+half)})
    return pd.DataFrame(rows), len(df)-len(sample)


def draw(summary, path):
    x = summary.family_size.to_numpy()
    y = summary.percent_below_norm.to_numpy()
    lower = summary.ci95_lower_percent.to_numpy()
    upper = summary.ci95_upper_percent.to_numpy()
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.errorbar(x, y, yerr=[y-lower, upper-y], fmt="o-", capsize=4,
                color="#245f89", linewidth=1.8, markersize=6)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, [f"{size}\nn={n:,}" for size, n in zip(x, summary.n_families)])
    ax.set_xlabel("Number of people in family / sampled families (n)")
    ax.set_ylabel("Families with actual food spending below sedentary FoodNorm (%)")
    ax.set_title("Food spending below sedentary FoodNorm by family size (95% Wilson CI)")
    ax.grid(alpha=.22)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("food_economics_2024.csv"))
    parser.add_argument("--output", type=Path, default=Path("2026-latet"))
    args = parser.parse_args()
    df = pd.read_csv(args.input, thousands=",")
    summary, excluded = summarize(df)
    if summary.empty:
        raise RuntimeError("No households with valid family size and food spending/norm")
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "sedentary_food_below_norm_by_family_size.csv",
                   index=False, float_format="%.6f")
    draw(summary, args.output / "sedentary_food_below_norm_by_family_size.png")
    (args.output / "sedentary_food_below_norm_README.txt").write_text(
        f"Input: {args.input.name}. Included: {int(summary.n_families.sum())} families; "
        f"excluded: {excluded} (including sizes 15+ and any missing comparison values).\n"
        "Family size = sum of 14 age/sex count columns. Only sizes 1-14 are shown. "
        "Within each family size, numerator = households where food_actual is strictly "
        "less than FoodNorm-sedentary; denominator = all households with valid inputs "
        "of that size. Ties do not count. No per-person normalization or survey weights.\n"
        "Error bars show unweighted 95% Wilson score confidence intervals for the "
        "proportion. They do not account for survey sampling design.\n",
        encoding="utf-8",
    )
    print(f"{len(summary)} groups; included {int(summary.n_families.sum())}; excluded {excluded}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
