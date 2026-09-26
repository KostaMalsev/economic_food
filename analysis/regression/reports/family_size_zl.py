"""Plot mean household ZL by sampled family size with 95% confidence intervals."""

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


SCENARIOS = ("active", "sedentary")
COLORS = {"active": "#17658a", "sedentary": "#b64e32"}


def summarize_by_family_size(df):
    """Use modeled household totals, rather than per-person ZL or bucket averages."""
    needed = ["persons_count", "ZL-active", "ZL-sedentary"]
    sample = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    sample = sample.loc[
        (sample["persons_count"] > 0)
        & (sample["persons_count"] == sample["persons_count"].astype(int))
    ]
    rows = []
    for size, families in sample.groupby("persons_count", sort=True):
        n = len(families)
        row = {"family_size": int(size), "n_families": n}
        for scenario in SCENARIOS:
            values = families[f"ZL-{scenario}"].to_numpy(dtype=float)
            mean = float(np.mean(values))
            # A single sampled family has a mean but no estimable sample variance.
            half_width = float(t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            row[f"mean_zl_{scenario}"] = mean
            row[f"ci95_lower_{scenario}"] = mean - half_width
            row[f"ci95_upper_{scenario}"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows)


def draw_chart(summary, scenarios, output_path):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    offsets = {"active": -0.12, "sedentary": 0.12} if len(scenarios) == 2 else {scenarios[0]: 0}
    for scenario in scenarios:
        x = summary["family_size"].to_numpy() + offsets[scenario]
        y = summary[f"mean_zl_{scenario}"].to_numpy()
        lower = summary[f"ci95_lower_{scenario}"].to_numpy()
        upper = summary[f"ci95_upper_{scenario}"].to_numpy()
        valid = np.isfinite(lower) & np.isfinite(upper)
        color = COLORS[scenario]
        ax.errorbar(x[valid], y[valid],
                    yerr=[y[valid] - lower[valid], upper[valid] - y[valid]],
                    fmt="o", capsize=3, markersize=5, color=color,
                    label=scenario.capitalize() + " (95% CI)")
        if (~valid).any():
            ax.scatter(x[~valid], y[~valid], marker="D", s=48, color=color,
                       label=scenario.capitalize() + " (n=1; CI unavailable)")
        if len(scenarios) == 1:
            for xpos, mean, top, n in zip(x, y, upper, summary["n_families"]):
                ax.annotate(f"n={n}", (xpos, top if np.isfinite(top) else mean),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8, rotation=35)
    if len(scenarios) == 2:
        # Both scenarios use the same families in each size group, so one n label suffices.
        for _, row in summary.iterrows():
            top = max(row[f"ci95_upper_{scenario}"] if np.isfinite(row[f"ci95_upper_{scenario}"])
                      else row[f"mean_zl_{scenario}"] for scenario in scenarios)
            ax.annotate(f"n={int(row['n_families'])}", (row["family_size"], top),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, rotation=35)
    ax.set(xlabel="Number of people in family", ylabel="Mean modeled monthly household ZL (NIS)",
           title="Mean household ZL by family size: " + " and ".join(scenarios))
    ax.set_xticks(summary["family_size"])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SURVEY_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_REGRESSION_OUTPUT)
    args = parser.parse_args()
    analyzer = FamilyGroupAnalyzer(str(args.input))
    if not analyzer.read_csv() or not analyzer.process_dataframe():
        raise RuntimeError("Could not process the household sample")
    summary = summarize_by_family_size(analyzer.df)
    if summary.empty:
        raise RuntimeError("No households with positive family size and finite ZL")
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "family_size_zl_95ci.csv", index=False, float_format="%.10f")
    for scenarios, filename in [(("active",), "zl_by_family_size_active.png"),
                                (("sedentary",), "zl_by_family_size_sedentary.png"),
                                (SCENARIOS, "zl_by_family_size_comparison.png")]:
        draw_chart(summary, scenarios, args.output / filename)
    n_excluded = len(analyzer.df) - int(summary["n_families"].sum())
    (args.output / "README.txt").write_text(
        f"Input: {args.input.name}; included: {int(summary['n_families'].sum())} households; excluded: {n_excluded}.\n"
        "Family size = sum of the 14 age/sex counts. ZL is the modeled monthly household total: "
        "2 * household FoodNorm - predicted food expenditure, separately for active and sedentary.\n"
        "Each point is the unweighted arithmetic mean among sampled families of that exact size. "
        "Error bars are two-sided 95% Student-t confidence intervals: mean +/- "
        "t(0.975, n-1) * sample SD / sqrt(n). For n=1, sample variance and CI are undefined; "
        "these points are shown as diamonds and their CSV interval fields are blank.\n"
        "Intervals describe sample variation in modeled ZL within a size group; they do not "
        "include regression-coefficient uncertainty or sampling-design weights. "
        "Results use the original model price basis.\n",
        encoding="utf-8",
    )
    print(f"Saved {len(summary)} size groups and {int(summary['n_families'].sum())} families to {args.output}")


if __name__ == "__main__":
    main()
