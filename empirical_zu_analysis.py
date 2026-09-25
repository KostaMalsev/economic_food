"""Calculate active empirical ZU by family size using Limor's method."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publication_style import FIGURE_WIDTH_IN, apply_publication_style, save_png_and_pdf


REQUIRED_COLUMNS = [
    "misparmb", "persons_count", "c3_actual", "food_actual",
    "nonfood_actual", "FoodNorm-active",
]
TOLERANCE = 0.05


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
    args = parser.parse_args()

    households = prepare(pd.read_csv(args.input))
    qualifying, summary = derive_zu_by_size(households)
    assigned = assign_zu(households, summary)
    args.output.mkdir(parents=True, exist_ok=True)

    qualifying.to_csv(args.output / "empirical_zu_qualifying_active_5pct.csv",
                      index=False, float_format="%.10f")
    summary.to_csv(args.output / "empirical_zu_by_family_size.csv",
                   index=False, float_format="%.10f")
    assigned.to_csv(args.output / "households_with_empirical_zu.csv",
                    index=False, float_format="%.10f")
    draw_summary(summary, args.output / "empirical_zu_by_family_size_1_to_7.png")

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
        f"household and therefore without an empirical ZU: {missing_sizes}.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Qualifying households: {len(qualifying)}")
    print(f"Assigned households: {assigned['ZU-active-empirical'].notna().sum():,}/{len(assigned):,}")


if __name__ == "__main__":
    main()
