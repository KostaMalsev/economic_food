"""Plot active household FoodNorm, ZL and ZU by family size."""

from argparse import ArgumentParser
from pathlib import Path
from analysis.shared.paths import DEFAULT_REGRESSION_OUTPUT, DEFAULT_SURVEY_DATA

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from analysis.shared.plotting import FIGURE_WIDTH_IN, apply_publication_style, save_png_and_pdf
from analysis.regression.model import FamilyGroupAnalyzer


METRICS = (
    ("FoodNorm-active", "FoodNorm", "#2575a9"),
    ("ZL-active", "ZL", "#dc7d2d"),
    ("ZU-active", "ZU", "#7452a2"),
)


def prepare_sample(df):
    columns = ["misparmb", "persons_count", "c3", "food_actual"] + [
        column for column, _, _ in METRICS
    ]
    sample = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    sample = sample.loc[(sample.persons_count >= 1) & (sample.persons_count <= 7) &
                        (sample.persons_count == sample.persons_count.astype(int))].copy()
    sample["persons_count"] = sample["persons_count"].astype(int)
    # ZL = 2 * FoodNorm - predicted(c30 + c31), rearranged to expose
    # the food-expenditure regression prediction used by the ZL calculation.
    sample["predicted_c30_plus_c31"] = (
        2 * sample["FoodNorm-active"] - sample["ZL-active"]
    )
    sample["food_actual - FoodNorm-active"] = (
        sample["food_actual"] - sample["FoodNorm-active"]
    )
    sample["c3(actual) - ZU-active"] = sample["c3"] - sample["ZU-active"]
    sample["c3(actual) - ZL-active"] = sample["c3"] - sample["ZL-active"]
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
    apply_publication_style()
    # Match the reference figure width so 10-point text renders at the same scale.
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 5.0))
    x = summary.family_size.to_numpy()
    rng = np.random.default_rng(2026)
    jitters = {
        size: rng.uniform(-.075, .075, len(families))
        for size, families in sample.groupby("persons_count", sort=True)
    }
    for column, label, color in METRICS:
        for size, families in sample.groupby("persons_count", sort=True):
            ax.scatter(size + jitters[size], families[column], s=4,
                       color=color, alpha=.045, linewidths=0, rasterized=True)
        y = summary[f"mean_{label}"].to_numpy()
        lower = summary[f"ci95_lower_{label}"].to_numpy()
        upper = summary[f"ci95_upper_{label}"].to_numpy()
        valid = np.isfinite(lower) & np.isfinite(upper)
        ax.plot(x, y, color=color, alpha=.9, linewidth=1.6)
        ax.errorbar(x[valid], y[valid],
                    yerr=[y[valid]-lower[valid], upper[valid]-y[valid]],
                    fmt="o", color=color, markeredgecolor="white", markeredgewidth=.6,
                    capsize=3, markersize=5.5, label=f"{label} mean (95% CI)", zorder=5)
    ax.set_xticks(x, [f"{size}\nn={n:,}" for size, n in zip(x, summary.n_families)])
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_xlabel("People in household / number of sampled families (n)")
    ax.set_ylabel("NIS/month")
    ax.set_title("Active households: individual values and means by family size")
    ax.grid(alpha=.22)
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.tight_layout()
    save_png_and_pdf(fig, path)
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
    summary.to_csv(args.output / "active_zu_zl_foodnorm_by_family_size_95ci.csv",
                   index=False, float_format="%.10f")
    export_columns = [
        "misparmb", "persons_count", "c3", "food_actual", "FoodNorm-active",
        "predicted_c30_plus_c31", "ZL-active", "ZU-active",
        "food_actual - FoodNorm-active", "c3(actual) - ZU-active",
        "c3(actual) - ZL-active",
    ]
    sample.sort_values(["persons_count", "misparmb"])[export_columns].to_csv(
        args.output / "active_household_values_by_family_size_1_to_7.csv",
        index=False, float_format="%.10f")
    plot(summary, sample, args.output / "active_zu_zl_foodnorm_by_family_size.png")
    (args.output / "active_zu_zl_foodnorm_README.txt").write_text(
        f"Input: {args.input.name}; included: {int(summary.n_families.sum())} families; "
        f"excluded: {len(analyzer.df)-int(summary.n_families.sum())}. "
        "Shown family sizes: 1-7 inclusive; size 8 and above is excluded.\n"
        "All amounts are monthly household totals in the original model price basis. "
        "Active ZL = 2 * active FoodNorm - predicted active food expenditure; "
        "active ZU = predicted total expenditure.\n"
        "In the household-level CSV, c3 is actual total expenditure and food_actual is "
        "actual c30+c31; both source values appear directly after persons_count.\n"
        "The household-level CSV places predicted_c30_plus_c31 after FoodNorm-active. "
        "It is the combined active food-expenditure regression prediction reconstructed "
        "exactly as 2 * FoodNorm-active - ZL-active; separate c30/c31 predictions are "
        "not available in the model.\n"
        "The final three columns are row-level differences: food_actual minus active "
        "FoodNorm, actual c3 minus active ZU, and actual c3 minus active ZL.\n"
        "Small transparent points show every included household with deterministic "
        "symmetric jitter centered on its integer family size. All metric means and "
        "confidence intervals align exactly on that integer tick. Large points are "
        "unweighted group means; error bars are two-sided 95% Student-t intervals. "
        "Intervals exclude regression-coefficient uncertainty and survey-design weights.\n"
        "All figure text uses a consistent 10-point publication font. The PNG is exported "
        "at 300 DPI and the matching PDF keeps text and lines sharp when resized.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summary)} groups for {int(summary.n_families.sum())} families")


if __name__ == "__main__":
    main()
