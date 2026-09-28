"""Generate Limor's revised empirical charts for active and sedentary households."""
from argparse import ArgumentParser
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from analysis.shared.plotting import FIGURE_WIDTH_IN, apply_publication_style, save_png_and_pdf

ZU_TOLERANCE, ZL_TOLERANCE, MAX_FAMILY_SIZE = .05, .10, 9
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KINDS = ("active", "sedentary")
SERIES = (
    ("foodnorm", "Food actual < FoodNorm and C3 actual < ZU", "#2478a8", "o"),
    ("zl", "C3 actual < empirical ZL", "#e67e22", "s"),
    ("zu", "C3 actual < empirical ZU", "#7251a3", "^"),
)

def prepare(data):
    required = ["misparmb", "persons_count", "c3_actual", "food_actual", "nonfood_actual",
                "FoodNorm-active", "FoodNorm-sedentary"]
    missing = sorted(set(required) - set(data.columns))
    if missing: raise KeyError(f"Missing required columns: {missing}")
    data = data.copy()
    data[required] = data[required].replace([np.inf, -np.inf], np.nan)
    if data["misparmb"].duplicated().any(): raise ValueError("Input must contain one row per household")
    if data[required].isna().any().any(): raise ValueError("Required inputs contain missing values")
    if (data["persons_count"] <= 0).any(): raise ValueError("persons_count must be positive")
    data["c3_actual_per_capita"] = data["c3_actual"] / data["persons_count"]
    return data

def small_sample(households, kind, threshold):
    fn = f"FoodNorm-{kind}"
    actual, tolerance = (("food_actual", ZU_TOLERANCE) if threshold == "zu"
                         else ("c3_actual", ZL_TOLERANCE))
    difference = (households[actual] - households[fn]) / households[fn]
    sample = households.loc[difference.abs() <= tolerance].copy()
    sample[f"{threshold}_relative_difference"] = difference.loc[sample.index]
    return sample.sort_values(["persons_count", "c3_actual_per_capita", "misparmb"])

def derive_threshold(households, kind, threshold):
    """Poorest household supplies non-food; mean of sample-specific FoodNorm+non-food is threshold."""
    sample = small_sample(households, kind, threshold)
    fn = f"FoodNorm-{kind}"
    poorest = sample.groupby("persons_count", as_index=False).first().set_index("persons_count")
    sample["selected_poorest_nonfood"] = sample["persons_count"].map(poorest["nonfood_actual"])
    candidate = f"candidate_{threshold}_{kind}"
    sample[candidate] = sample[fn] + sample["selected_poorest_nonfood"]
    summary = sample.groupby("persons_count").agg(
        qualifying_households=("misparmb", "size"), mean_foodnorm=(fn, "mean"),
        selected_poorest_nonfood=("selected_poorest_nonfood", "first"),
        threshold_mean=(candidate, "mean"), threshold_min=(candidate, "min"),
        threshold_max=(candidate, "max"),
    )
    summary["selected_poorest_household_id"] = poorest["misparmb"]
    summary["selected_poorest_c3_per_capita"] = poorest["c3_actual_per_capita"]
    summary = summary.reset_index(); summary["threshold"] = threshold.upper(); summary["activity"] = kind
    return sample, summary

def build_thresholds(households, kind, zl, zu):
    out = pd.DataFrame({"persons_count": range(1, MAX_FAMILY_SIZE + 1)})
    out = out.merge(households.groupby("persons_count").size().rename("all_households").reset_index(), how="left")
    out = out.merge(households.groupby("persons_count")[f"FoodNorm-{kind}"].mean().rename("FoodNorm").reset_index(), how="left")
    for name, summary in (("ZL", zl), ("ZU", zu)):
        rename = {"qualifying_households": f"{name.lower()}_qualifying_households",
                  "selected_poorest_household_id": f"{name.lower()}_selected_household_id",
                  "selected_poorest_nonfood": f"{name.lower()}_selected_nonfood",
                  "selected_poorest_c3_per_capita": f"{name.lower()}_selected_c3_per_capita",
                  "mean_foodnorm": f"{name.lower()}_sample_mean_foodnorm", "threshold_mean": name}
        out = out.merge(summary[["persons_count", *rename]].rename(columns=rename), on="persons_count", how="left")
    missing = out.loc[out[["ZL", "ZU"]].isna().any(axis=1), "persons_count"].tolist()
    if missing: warnings.warn(f"Missing {kind} thresholds for sizes {missing}")
    out["activity"] = kind
    return out

def assign(households, thresholds, kind):
    data = households.merge(thresholds[["persons_count", "FoodNorm", "ZL", "ZU"]], on="persons_count", how="left", validate="many_to_one")
    data["activity"] = kind
    data["food_poor"] = (data.food_actual < data.FoodNorm) & (data.c3_actual < data.ZU)
    data["food_poor_zu_minus_5pct"] = (data.food_actual < data.FoodNorm) & (data.c3_actual < .95 * data.ZU)
    data["zl_poor"], data["zu_poor"] = data.c3_actual < data.ZL, data.c3_actual < data.ZU
    return data

def values(group, key, sensitivity=False):
    if key == "foodnorm":
        actual, threshold = group.food_actual, group.FoodNorm
        poor = group.food_poor_zu_minus_5pct if sensitivity else group.food_poor
    else:
        actual, threshold, poor = group.c3_actual, group[key.upper()], group[f"{key}_poor"]
    valid = actual.notna() & threshold.notna() & (threshold > 0)
    return actual, threshold, poor & valid, valid

def wilson(successes, total):
    if total == 0:
        return np.nan, np.nan
    z = norm.ppf(.975); p = successes / total; d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return 100*(center-half), 100*(center+half)

def summarize_by_size(data, sensitivity=False):
    rows=[]
    for size, group in data[data.persons_count.between(1, MAX_FAMILY_SIZE)].groupby("persons_count"):
        row={"family_size":int(size), "n_families":len(group)}
        for key, *_ in SERIES:
            actual, threshold, poor, valid = values(group, key, sensitivity)
            total, count = int(valid.sum()), int(poor.sum()); lo, hi = wilson(count,total)
            gaps=(threshold[poor]-actual[poor])/threshold[poor]
            row.update({f"poor_households_{key}":count, f"valid_households_{key}":total,
                        f"poverty_rate_percent_{key}":100*count/total if total else np.nan,
                        f"ci95_lower_percent_{key}":lo, f"ci95_upper_percent_{key}":hi,
                        f"mean_depth_percent_{key}":100*gaps.mean() if count else np.nan,
                        f"fgt2_percent_{key}":100*(gaps**2).sum()/total if total else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_overall(data, sensitivity=False):
    rows=[]; data=data[data.persons_count.between(1, MAX_FAMILY_SIZE)]
    for key, label, *_ in SERIES:
        actual, threshold, poor, valid=values(data,key,sensitivity)
        total,count=int(valid.sum()),int(poor.sum()); lo,hi=wilson(count,total)
        gaps=(threshold[poor]-actual[poor])/threshold[poor]
        rows.append({"poverty_definition":key,"label":label,"poor_households":count,
                     "valid_households":total,"poverty_rate_percent":100*count/total,
                     "ci95_lower_percent":lo,"ci95_upper_percent":hi,
                     "mean_depth_percent":100*gaps.mean() if count else np.nan,
                     "fgt2_percent":100*(gaps**2).sum()/total})
    return pd.DataFrame(rows)

def draw_thresholds(t, households, zl, zu, kind, path):
    apply_publication_style(); fig,ax=plt.subplots(figsize=(FIGURE_WIDTH_IN,4.8))
    rng = np.random.default_rng(20260928)
    foodnorm_points = households[households.persons_count.between(1, MAX_FAMILY_SIZE)]
    foodnorm_x = foodnorm_points.persons_count + rng.uniform(-.10, .10, len(foodnorm_points))
    ax.scatter(foodnorm_x, foodnorm_points[f"FoodNorm-{kind}"], s=9, alpha=.10,
               color="#2478a8", edgecolors="none", zorder=1)
    for sample,key,color in ((zl,"zl","#e67e22"),(zu,"zu","#7251a3")):
        plotted = sample[sample.persons_count.between(1, MAX_FAMILY_SIZE)]
        sample_x = plotted.persons_count + rng.uniform(-.10, .10, len(plotted))
        ax.scatter(sample_x, plotted[f"candidate_{key}_{kind}"], s=14, alpha=.25,
                   color=color, edgecolors="none", zorder=2)
    for col,label,color,marker in (("FoodNorm","FoodNorm mean","#2478a8","o"),("ZL","Empirical ZL mean","#e67e22","s"),("ZU","Empirical ZU mean","#7251a3","^")):
        ax.plot(t.persons_count,t[col],marker=marker,color=color,linewidth=1.8,
                markersize=6,label=label,zorder=4)
    ax.set_xticks(t.persons_count); ax.set_xlim(.5, MAX_FAMILY_SIZE + .5)
    ax.set_xlabel("People in household"); ax.set_ylabel("NIS/month")
    ax.set_title(f"{kind.title()} households: empirical FoodNorm, ZL and ZU by family size")
    ax.legend(); ax.grid(alpha=.22); fig.tight_layout(); save_png_and_pdf(fig,path); plt.close(fig)

def draw_rates(s,kind,path):
    apply_publication_style(); x=s.family_size.to_numpy(); fig,ax=plt.subplots(figsize=(FIGURE_WIDTH_IN,4.8))
    for key,label,color,marker in SERIES:
        y=s[f"poverty_rate_percent_{key}"].to_numpy(); lo=s[f"ci95_lower_percent_{key}"].to_numpy(); hi=s[f"ci95_upper_percent_{key}"].to_numpy()
        ax.errorbar(x,y,yerr=[y-lo,hi-y],marker=marker,color=color,linewidth=1.8,markersize=5,capsize=3,label=label)
    ax.set_xticks(x,[f"{n}\nn={c:,}" for n,c in zip(x,s.n_families)]); ax.set_xlim(.5, MAX_FAMILY_SIZE + .5); ax.set_ylim(0,100)
    ax.set_xlabel("People in household / sampled households (n)"); ax.set_ylabel("Households below threshold (%)")
    ax.set_title(f"{kind.title()} households: poverty rates by family size (95% Wilson CI)")
    ax.legend(); ax.grid(alpha=.22); fig.tight_layout(); save_png_and_pdf(fig,path); plt.close(fig)

def draw_metric(s,kind,metric,ylabel,title,path):
    apply_publication_style(); x=s.family_size.to_numpy(); fig,ax=plt.subplots(figsize=(FIGURE_WIDTH_IN,4.8))
    for key,label,color,marker in SERIES: ax.plot(x,s[f"{metric}_{key}"],marker=marker,color=color,linewidth=1.8,markersize=6,label=label)
    ax.set_xticks(x,[f"{n}\nn={c:,}" for n,c in zip(x,s.n_families)]); ax.set_xlim(.5, MAX_FAMILY_SIZE + .5)
    ax.set_xlabel("People in household / sampled households (n)")
    ax.set_ylabel(ylabel); ax.set_title(f"{kind.title()} households: {title}"); ax.legend(); ax.grid(alpha=.22)
    fig.tight_layout(); save_png_and_pdf(fig,path); plt.close(fig)

def draw_overall(s,kind,path):
    apply_publication_style(); x=np.arange(len(s)); y=s.poverty_rate_percent.to_numpy(); lo=s.ci95_lower_percent.to_numpy(); hi=s.ci95_upper_percent.to_numpy()
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH_IN,4.8)); bars=ax.bar(x,y,color=[v[2] for v in SERIES],width=.62)
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="none",ecolor="#333",capsize=5)
    for bar,(_,r) in zip(bars,s.iterrows()): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.2,f"{r.poverty_rate_percent:.1f}%\n{int(r.poor_households):,}/{int(r.valid_households):,}",ha="center")
    ax.set_xticks(x,["Food actual < FoodNorm\nand C3 actual < ZU","C3 actual\n< ZL","C3 actual\n< ZU"]); ax.set_ylim(0,100)
    ax.set_ylabel("Households below threshold (%)"); ax.set_title(f"{kind.title()} households: overall poverty rates (95% Wilson CI)")
    ax.grid(axis="y",alpha=.22); fig.tight_layout(); save_png_and_pdf(fig,path); plt.close(fig)

def run_kind(households,kind,output):
    zl,zls=derive_threshold(households,kind,"zl"); zu,zus=derive_threshold(households,kind,"zu")
    thresholds=build_thresholds(households,kind,zls,zus); data=assign(households,thresholds,kind)
    by_size=summarize_by_size(data); overall=summarize_overall(data); target=output/kind; target.mkdir(parents=True,exist_ok=True)
    zl.to_csv(target/f"{kind}_zl_small_sample.csv",index=False,float_format="%.10f"); zu.to_csv(target/f"{kind}_zu_small_sample.csv",index=False,float_format="%.10f")
    pd.concat([zls,zus]).to_csv(target/f"{kind}_threshold_audit.csv",index=False,float_format="%.10f")
    thresholds.to_csv(target/f"{kind}_empirical_thresholds_by_family_size.csv",index=False,float_format="%.10f")
    data.to_csv(target/f"{kind}_households_with_thresholds.csv",index=False,float_format="%.10f")
    by_size.to_csv(target/f"{kind}_poverty_metrics_by_family_size.csv",index=False,float_format="%.10f"); overall.to_csv(target/f"{kind}_overall_poverty_metrics.csv",index=False,float_format="%.10f")
    summarize_by_size(data,True).to_csv(target/f"{kind}_zu_minus_5pct_sensitivity_by_family_size.csv",index=False,float_format="%.10f")
    summarize_overall(data,True).to_csv(target/f"{kind}_zu_minus_5pct_sensitivity_overall.csv",index=False,float_format="%.10f")
    draw_thresholds(thresholds,households,zl,zu,kind,target/f"{kind}_01_thresholds.png"); draw_rates(by_size,kind,target/f"{kind}_02_poverty_rates_by_family_size.png")
    draw_overall(overall,kind,target/f"{kind}_03_overall_poverty_rates.png")
    draw_metric(by_size,kind,"mean_depth_percent","Mean poverty depth among poor households (%)","poverty depth by family size",target/f"{kind}_04_poverty_depth_percent.png")
    draw_metric(by_size,kind,"fgt2_percent","FGT₂ × 100 (%)","FGT₂ poverty severity",target/f"{kind}_05_fgt2_poverty_severity.png")
    return thresholds

def main():
    p=ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,default=HERE/"data"/"household_inputs.csv")
    p.add_argument("--output",type=Path,default=ROOT/"outputs"/"2026-09-28-active-sedentary-revised-empirical-thresholds"); args=p.parse_args()
    households=prepare(pd.read_csv(args.input)); args.output.mkdir(parents=True,exist_ok=True)
    combined=pd.concat([run_kind(households,k,args.output) for k in KINDS],ignore_index=True)
    combined.to_csv(args.output/"active_and_sedentary_empirical_thresholds.csv",index=False,float_format="%.10f")
    print(combined[["activity","persons_count","FoodNorm","ZL","ZU"]].to_string(index=False))

if __name__ == "__main__": main()
