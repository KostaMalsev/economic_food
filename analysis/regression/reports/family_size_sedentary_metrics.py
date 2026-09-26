"""Mean sedentary household ZU, ZL and FoodNorm by family size."""

from argparse import ArgumentParser
from pathlib import Path
from analysis.shared.paths import DEFAULT_REGRESSION_OUTPUT, DEFAULT_SURVEY_DATA

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from analysis.regression.model import FamilyGroupAnalyzer


METRICS = (
    ("FoodNorm-sedentary", "FoodNorm", "#2575a9"),
    ("ZL-sedentary", "ZL", "#dc7d2d"),
    ("ZU-sedentary", "ZU", "#7452a2"),
)


def prepare_sample(df):
    columns = ["misparmb", "persons_count"] + [column for column, _, _ in METRICS]
    sample = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    sample = sample.loc[(sample.persons_count >= 1) & (sample.persons_count <= 7) &
                        (sample.persons_count == sample.persons_count.astype(int))].copy()
    sample["persons_count"] = sample["persons_count"].astype(int)
    return sample


def summarize(sample):
    rows = []
    for size, families in sample.groupby("persons_count", sort=True):
        n = len(families)
        row = {"family_size": int(size), "n_families": n}
        for column, label, _ in METRICS:
            values = families[column].to_numpy(dtype=float)
            mean = float(np.mean(values))
            half = (float(t.ppf(.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))
                    if n > 1 else np.nan)
            row[f"mean_{label}"] = mean
            row[f"ci95_lower_{label}"] = mean - half
            row[f"ci95_upper_{label}"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary, sample, path):
    fig, ax = plt.subplots(figsize=(17, 8))
    x = summary.family_size.to_numpy()
    rng = np.random.default_rng(2026)
    jitters = {
        size: rng.uniform(-0.075, 0.075, len(families))
        for size, families in sample.groupby("persons_count", sort=True)
    }
    for column, label, color in METRICS:
        # Show every household as a transparent jittered point. Horizontal jitter
        # only reduces overplotting; it does not change family size or Y values.
        for size, families in sample.groupby("persons_count", sort=True):
            ax.scatter(size + jitters[size], families[column], s=7,
                       color=color, alpha=0.075, linewidths=0, rasterized=True)
        y = summary[f"mean_{label}"].to_numpy()
        low = summary[f"ci95_lower_{label}"].to_numpy()
        high = summary[f"ci95_upper_{label}"].to_numpy()
        valid = np.isfinite(low) & np.isfinite(high)
        ax.plot(x, y, color=color, alpha=.8, linewidth=1.4)
        ax.errorbar(x[valid], y[valid],
                    yerr=[y[valid] - low[valid], high[valid] - y[valid]],
                    fmt="o", color=color, markeredgecolor="white", markeredgewidth=.6,
                    capsize=4, markersize=7, label=f"{label} mean (95% CI)", zorder=5)
    ax.set_xticks(x, [f"{size}\nn={n:,}" for size, n in zip(x, summary.n_families)])
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("People in household / number of sampled families (n)")
    ax.set_ylabel("NIS/month")
    ax.set_title("Sedentary households: individual values and means by family size")
    ax.grid(alpha=.22)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SURVEY_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_REGRESSION_OUTPUT)
    args = parser.parse_args()
    analyzer = FamilyGroupAnalyzer(str(args.input))
    if not analyzer.read_csv() or not analyzer.process_dataframe():
        raise RuntimeError("Could not process household sample")
    sample = prepare_sample(analyzer.df)
    summary = summarize(sample)
    if summary.empty:
        raise RuntimeError("No valid household records")
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "sedentary_zu_zl_foodnorm_by_family_size_95ci.csv",
                   index=False, float_format="%.10f")
    sample.sort_values(["persons_count", "misparmb"]).to_csv(
        args.output / "sedentary_household_values_by_family_size_1_to_7.csv",
        index=False, float_format="%.10f")
    plot(summary, sample, args.output / "sedentary_zu_zl_foodnorm_by_family_size.png")
    (args.output / "sedentary_zu_zl_foodnorm_README.txt").write_text(
        f"Input: {args.input.name}; included: {int(summary.n_families.sum())} families; "
        f"excluded: {len(analyzer.df) - int(summary.n_families.sum())}. "
        "Shown family sizes: 1-7 inclusive; size 8 and above is excluded.\n"
        "Family size is the sum of the 14 age/sex count columns. All three amounts are "
        "monthly modeled household totals in the original model's price basis. "
        "Sedentary ZL = 2 * sedentary FoodNorm - predicted sedentary food expenditure; "
        "sedentary ZU = predicted total expenditure.\n"
        "Small transparent points are every included household's exact modeled value, "
        "with deterministic symmetric jitter centered on its exact integer family size "
        "to reduce overlap. All three means and confidence intervals are aligned exactly "
        "on the integer family-size tick. Large points are the "
        "unweighted means among sampled families with that exact size. "
        "The second line on each x tick gives the number of families in the group. "
        "Error bars are two-sided 95% Student-t intervals for the mean, based on the "
        "sample standard deviation within each size group. For n=1 the point is a "
        "diamond and the CI is undefined (blank in CSV). These intervals exclude "
        "regression coefficient uncertainty and survey-design weights.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summary)} groups for {int(summary.n_families.sum())} families")


if __name__ == "__main__":
    main()
