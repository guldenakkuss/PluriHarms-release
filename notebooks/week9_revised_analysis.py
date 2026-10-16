"""
Week 9 — Revised analysis (methodological corrections to the report)

Corrections relative to SystematicReport.ipynb / AnnotatorAnalysis.ipynb:

1. Cross-validation for row-level models (Model 3, Model 4, base A+D+E model)
   used KFold on long-format rows, so the same prompt (Model 3) or the same
   annotator (Model 4) appeared in both the training and validation folds
   (leakage). Here GroupKFold is used with groups = prompt (Model 3) or
   groups = annotator (Model 4). Models 1–2 are prompt-level (one row per
   prompt), so plain KFold is already correct there; they are re-run only to
   check reproduction.
2. Mann–Whitney tests were run on ~14.7k individual ratings, treating ratings
   from the same annotator as independent (pseudo-replication). Here each
   annotator is first reduced to one mean rating (all prompts / contested
   prompts), so n = 100 annotators. Effect sizes (Cohen's d, rank-biserial r)
   and Holm correction are added.
3. "Amplification" on contested prompts is tested formally: per-annotator
   contested-minus-uncontested difference, compared between groups
   (annotator-level interaction test), and a mixed-effects model with
   prompt fixed effects and an annotator random intercept.
4. Agreement is reported with Krippendorff's alpha (interval) and ICC
   instead of Fleiss' kappa, which is not appropriate for a 0–100 scale.

Run from the repository root:  python notebooks/week9_revised_analysis.py
Outputs: outputs/week9_*.csv, outputs/week9_results.json, outputs/fig_week9_*.png
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.chdir(REPO_ROOT)
os.makedirs("outputs", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import krippendorff
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold
from data_loader import load_data, get_rating_columns

RESULTS = {}

# ═════════════════════════════════════════════════════════════
# 1. Data (identical construction to SystematicReport.ipynb)
# ═════════════════════════════════════════════════════════════
annotations, prompts = load_data()
rating_cols = get_rating_columns(annotations)

action_cols = [c for c in prompts.columns if c.startswith("action_")]
effect_cols = [c for c in prompts.columns if c.startswith("effect_")]
cat_A = action_cols + effect_cols
cat_B = [c for c in prompts.columns if c.startswith("value_topic_")]

tfidf = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(prompts["Question_Content"])
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
    index=prompts["Question_Index"],
).reset_index()
cat_C = [c for c in tfidf_df.columns if c.startswith("tfidf_")]

cat_D_raw = ["Gender", "Age", "Political_Affiliation", "Education",
             "Social_Media_Frequency", "Toxicity_Experience",
             "Income", "Religion_Importance", "Race_Ethnicity", "Sexual_Orientation"]
cat_E = ["Factor_1", "Factor_2", "Factor_3"]

ratings = annotations.set_index("Participant_ID")[rating_cols]
ratings.columns = [int(c.replace("Rating_", "")) for c in rating_cols]
avg_ratings = ratings.mean()
std_ratings = ratings.std()

long_df = (ratings.stack().rename("rating").reset_index()
           .rename(columns={"level_1": "question_index"}).dropna(subset=["rating"]))
long_df = long_df.merge(annotations[["Participant_ID"] + cat_D_raw + cat_E],
                        on="Participant_ID")
long_df = long_df.merge(prompts[["Question_Index"] + cat_A + cat_B],
                        left_on="question_index", right_on="Question_Index").drop(columns="Question_Index")
long_df = long_df.merge(tfidf_df, left_on="question_index",
                        right_on="Question_Index").drop(columns="Question_Index")
long_df = long_df.sort_values(["Participant_ID", "question_index"]).reset_index(drop=True)

long_enc = pd.get_dummies(long_df, columns=cat_D_raw, drop_first=True)
long_enc = long_enc.fillna(long_enc.median(numeric_only=True))
cat_D = [c for c in long_enc.columns if any(c.startswith(d + "_") for d in cat_D_raw)]

print(f"Long format: {len(long_enc):,} ratings, {long_enc.Participant_ID.nunique()} annotators, "
      f"{long_enc.question_index.nunique()} prompts")

# Same splits as SystematicReport.ipynb
np.random.seed(42)
all_prompts = prompts["Question_Index"].values.copy()
np.random.shuffle(all_prompts)
train_prompts, test_prompts = all_prompts[:100], all_prompts[100:]
np.random.seed(42)
all_annot = long_enc["Participant_ID"].unique().copy()
np.random.shuffle(all_annot)
train_annot, test_annot = all_annot[:80], all_annot[80:]

long_train = long_enc[long_enc.question_index.isin(train_prompts)]
long_test = long_enc[long_enc.question_index.isin(test_prompts)]
ann_train = long_enc[long_enc.Participant_ID.isin(train_annot)]
ann_test = long_enc[long_enc.Participant_ID.isin(test_annot)]

prompt_df = prompts[["Question_Index"] + cat_A + cat_B].copy()
prompt_df["avg_rating"] = prompt_df["Question_Index"].map(avg_ratings)
prompt_df = prompt_df.merge(tfidf_df, on="Question_Index")
prompt_train = prompt_df[prompt_df.Question_Index.isin(train_prompts)]
prompt_test = prompt_df[prompt_df.Question_Index.isin(test_prompts)]

# ═════════════════════════════════════════════════════════════
# 2. Evaluation: old (row KFold) vs corrected (GroupKFold)
# ═════════════════════════════════════════════════════════════
KF = KFold(n_splits=5, shuffle=True, random_state=42)
GKF = GroupKFold(n_splits=5)


def cv_r2(X, y, groups=None):
    splitter = GKF.split(X, y, groups) if groups is not None else KF.split(X)
    scores = []
    for tr, va in splitter:
        m = Ridge(alpha=1.0).fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores)), float(np.std(scores))


def test_eval(X_tr, y_tr, X_te, y_te, q_te=None):
    m = Ridge(alpha=1.0).fit(X_tr, y_tr)
    p = m.predict(X_te)
    out = {"test_r2": r2_score(y_te, p), "test_mae": mean_absolute_error(y_te, p)}
    if q_te is not None:
        tmp = pd.DataFrame({"q": q_te.values, "p": p, "y": y_te.values}).groupby("q").mean()
        out["test_r2_avg"] = r2_score(tmp.y, tmp.p)
    return out


# Theoretical ceiling for individual-level R² from prompt-only features:
# a perfect prompt model predicts each prompt's mean, so R² <= share of
# rating variance that lies between prompts.
pm = long_enc.groupby("question_index")["rating"].transform("mean")
am = long_enc.groupby("Participant_ID")["rating"].transform("mean")
tot = ((long_enc.rating - long_enc.rating.mean()) ** 2).sum()
ceiling_prompt = float(1 - ((long_enc.rating - pm) ** 2).sum() / tot)
ceiling_annot = float(1 - ((long_enc.rating - am) ** 2).sum() / tot)
RESULTS["r2_ceiling"] = {"prompt_only_individual": ceiling_prompt,
                         "annotator_only_individual": ceiling_annot}
print(f"\nR² ceiling (prompt-only → individual): {ceiling_prompt:.3f}")
print(f"R² ceiling (annotator-only → individual): {ceiling_annot:.3f}")

rows = []

# Model 1 (prompt-level; KFold over prompts is already grouped)
fs1 = {"A": cat_A, "B": cat_B, "C": cat_C, "A + B": cat_A + cat_B,
       "A + C": cat_A + cat_C, "B + C": cat_B + cat_C, "A + B + C": cat_A + cat_B + cat_C}
for name, f in fs1.items():
    cv, sd = cv_r2(prompt_train[f], prompt_train.avg_rating)
    te = test_eval(prompt_train[f], prompt_train.avg_rating, prompt_test[f], prompt_test.avg_rating)
    rows.append({"model": "Model 1", "features": name, "cv_r2_rowkfold": cv,
                 "cv_r2_grouped": cv, "test_r2": te["test_r2"], "test_mae": te["test_mae"]})

# Model 3 (row-level, prompt split) — leakage fix: group by prompt
fs3 = {"A": cat_A, "B": cat_B, "C": cat_C, "A + B": cat_A + cat_B, "A + B + C": cat_A + cat_B + cat_C}
for name, f in fs3.items():
    cv_old, _ = cv_r2(long_train[f], long_train.rating)
    cv_new, sd = cv_r2(long_train[f], long_train.rating, groups=long_train.question_index)
    te = test_eval(long_train[f], long_train.rating, long_test[f], long_test.rating, long_test.question_index)
    rows.append({"model": "Model 3", "features": name, "cv_r2_rowkfold": cv_old,
                 "cv_r2_grouped": cv_new, "test_r2": te["test_r2"],
                 "test_r2_avg": te["test_r2_avg"], "test_mae": te["test_mae"]})

# Model 4 (row-level, annotator split) — leakage fix: group by annotator
fs4 = {"D": cat_D, "E": cat_E, "D + E": cat_D + cat_E}
for name, f in fs4.items():
    cv_old, _ = cv_r2(ann_train[f], ann_train.rating)
    cv_new, sd = cv_r2(ann_train[f], ann_train.rating, groups=ann_train.Participant_ID)
    te = test_eval(ann_train[f], ann_train.rating, ann_test[f], ann_test.rating, ann_test.question_index)
    rows.append({"model": "Model 4", "features": name, "cv_r2_rowkfold": cv_old,
                 "cv_r2_grouped": cv_new, "test_r2": te["test_r2"],
                 "test_r2_avg": te["test_r2_avg"], "test_mae": te["test_mae"]})

# Base model A + D + E, individual rating, prompt split
f = cat_A + cat_D + cat_E
cv_old, _ = cv_r2(long_train[f], long_train.rating)
cv_new, _ = cv_r2(long_train[f], long_train.rating, groups=long_train.question_index)
te = test_eval(long_train[f], long_train.rating, long_test[f], long_test.rating, long_test.question_index)
rows.append({"model": "Base A+D+E", "features": "A + D + E", "cv_r2_rowkfold": cv_old,
             "cv_r2_grouped": cv_new, "test_r2": te["test_r2"],
             "test_r2_avg": te["test_r2_avg"], "test_mae": te["test_mae"]})

models_df = pd.DataFrame(rows).round(3)
models_df.to_csv("outputs/week9_models_groupkfold.csv", index=False)
RESULTS["models"] = models_df.to_dict(orient="records")
print("\n" + models_df.to_string(index=False))

# ═════════════════════════════════════════════════════════════
# 3. Annotator-level group tests (n = 100), effect sizes, Holm
# ═════════════════════════════════════════════════════════════
contested = set(std_ratings[std_ratings > std_ratings.median()].index)
RESULTS["contested"] = {"median_std": float(std_ratings.median()), "n": len(contested)}

ann = annotations.set_index("Participant_ID")
ann_mean_all = ratings.mean(axis=1)
ann_mean_cont = ratings[sorted(contested)].mean(axis=1)
ann_mean_unc = ratings[[c for c in ratings.columns if c not in contested]].mean(axis=1)

low_ed = ["High school diploma or GED", "Some college, but no degree", "Associates or technical degree"]
high_ed = ["Bachelor’s degree",
           "Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS etc.)"]
groups = {
    "Gender (Woman − Man)": (ann.Gender == "Man", ann.Gender == "Woman"),
    "Education (High − Low)": (ann.Education.isin(low_ed), ann.Education.isin(high_ed)),
    "Political (Conservative − Liberal)": (
        ann.Political_Affiliation.isin(["Very liberal", "Somewhat liberal"]),
        ann.Political_Affiliation.isin(["Very conservative", "Somewhat conservative"])),
    # same non-white list as AnnotatorAnalysis.ipynb (3 multi-racial annotators fall in neither group)
    "Race/Ethnicity (Non-white − White)": (ann.Race_Ethnicity == "White", ann.Race_Ethnicity.isin(
        ["Black or African American", "Latino or Hispanic", "East Asian", "South Asian", "Other",
         "East Asian,South Asian", "Latino or Hispanic,White", "Black or African American,White"])),
    "Age (35+ − 18–34)": (ann.Age.isin(["18-24 years old", "25-34 years old"]),
                          ~ann.Age.isin(["18-24 years old", "25-34 years old"])),
}
for fcol, lab in [("Factor_1", "Factor 1 (Authority − Universalist)"),
                  ("Factor_2", "Factor 2 (High − Low empathy)"),
                  ("Factor_3", "Factor 3 (Tradition − Hedonism)")]:
    med = ann[fcol].median()
    groups[lab] = (ann[fcol] <= med, ann[fcol] > med)


def cohens_d(a, b):
    s = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / s


def group_test(values, ga, gb):
    a, b = values[ga].dropna(), values[gb].dropna()
    U, p = mannwhitneyu(b, a, alternative="two-sided")
    return {"n_a": len(a), "n_b": len(b), "diff": b.mean() - a.mean(), "p": p,
            "cohens_d": cohens_d(a, b), "rank_biserial": 2 * U / (len(a) * len(b)) - 1}


def rating_level_p(qset, ga, gb):
    sub = long_df[long_df.question_index.isin(qset)]
    ida, idb = set(ga[ga].index), set(gb[gb].index)
    return mannwhitneyu(sub[sub.Participant_ID.isin(idb)].rating,
                        sub[sub.Participant_ID.isin(ida)].rating).pvalue


trows = []
all_q = set(ratings.columns)
for lab, (ga, gb) in groups.items():
    for subset, vals, qset in [("All prompts", ann_mean_all, all_q),
                               ("Contested", ann_mean_cont, contested)]:
        r = group_test(vals, ga, gb)
        r.update({"feature": lab, "subset": subset,
                  "p_rating_level_original": rating_level_p(qset, ga, gb)})
        trows.append(r)
tests_df = pd.DataFrame(trows)
tests_df["p_holm"] = multipletests(tests_df.p, method="holm")[1]
tests_df = tests_df[["feature", "subset", "n_a", "n_b", "diff", "cohens_d", "rank_biserial",
                     "p", "p_holm", "p_rating_level_original"]]
tests_df.to_csv("outputs/week9_annotator_level_tests.csv", index=False)
RESULTS["annotator_level_tests"] = tests_df.to_dict(orient="records")
pd.set_option("display.width", 200)
print("\nAnnotator-level Mann–Whitney (n = 100), Holm-corrected over 16 tests:")
print(tests_df.round(4).to_string(index=False))

# ═════════════════════════════════════════════════════════════
# 4. Is the effect larger on contested prompts? (interaction)
# ═════════════════════════════════════════════════════════════
# 4a. Annotator-level: delta_i = mean(contested) − mean(uncontested);
#     compare delta between groups (a valid group × contested test).
delta = ann_mean_cont - ann_mean_unc
irows = []
for lab, (ga, gb) in groups.items():
    a, b = delta[ga].dropna(), delta[gb].dropna()
    t, p = ttest_ind(b, a)
    irows.append({"feature": lab, "gap_all": ann_mean_all[gb].mean() - ann_mean_all[ga].mean(),
                  "gap_contested": ann_mean_cont[gb].mean() - ann_mean_cont[ga].mean(),
                  "gap_uncontested": ann_mean_unc[gb].mean() - ann_mean_unc[ga].mean(),
                  "interaction_diff": b.mean() - a.mean(), "p_interaction": p})
inter_df = pd.DataFrame(irows)
inter_df["p_interaction_holm"] = multipletests(inter_df.p_interaction, method="holm")[1]
inter_df.to_csv("outputs/week9_contested_interaction.csv", index=False)
RESULTS["contested_interaction"] = inter_df.to_dict(orient="records")
print("\nGroup × contested interaction (annotator-level):")
print(inter_df.round(4).to_string(index=False))

# 4b. Mixed-effects model: prompt fixed effects + annotator random intercept
mdf = long_df.copy()
mdf["woman"] = (mdf.Gender == "Woman").astype(float)
mdf["high_ed"] = mdf.Education.isin(high_ed).astype(float)
mdf["nonwhite"] = (mdf.Race_Ethnicity != "White").astype(float)  # all non-"White" entries
mdf["F1"] = mdf.Factor_1
mdf["contested"] = mdf.question_index.isin(contested).astype(float)
mdf = mdf[mdf.Education.isin(low_ed + high_ed)]

# The "contested" main effect is absorbed by the prompt fixed effects, so only
# its interactions enter. A per-annotator random slope for "contested" keeps the
# interaction standard errors honest (each annotator contributes one slope).
mixed = {}
specs = {
    "main_effects": ("rating ~ C(question_index) + woman + high_ed + nonwhite + F1", None),
    "with_contested_interactions": (
        "rating ~ C(question_index) + woman + high_ed + nonwhite + F1"
        " + woman:contested + high_ed:contested + nonwhite:contested + F1:contested", "~contested"),
}
for name, (formula, re_formula) in specs.items():
    fit = smf.mixedlm(formula, mdf, groups=mdf["Participant_ID"],
                      re_formula=re_formula).fit(reml=True, method=["bfgs", "lbfgs", "cg"])
    keep = [k for k in fit.fe_params.index if not k.startswith("C(question_index)")
            and k != "Intercept"]
    mixed[name] = {k: {"beta": float(fit.params[k]), "se": float(fit.bse[k]),
                       "p": float(fit.pvalues[k])} for k in keep}
    mixed[name]["converged"] = bool(fit.converged)
    mixed[name]["annotator_var"] = float(fit.cov_re.iloc[0, 0])
    mixed[name]["residual_var"] = float(fit.scale)
    print(f"\nMixedLM [{name}] (annotator random intercept, prompt fixed effects):")
    for k in keep:
        print(f"  {k:<22} β={fit.params[k]:7.3f}  SE={fit.bse[k]:.3f}  p={fit.pvalues[k]:.4f}")
    print(f"  annotator var={mixed[name]['annotator_var']:.1f}  residual var={mixed[name]['residual_var']:.1f}")
# Robustness: repeat the interaction model on the more consistent half of the
# annotators (lowest theme-level inconsistency ratio from Week7_AnnotatorConsistency).
if os.path.exists("outputs/IR.npy"):
    IR = np.load("outputs/IR.npy")
    consistent_ids = annotations.Participant_ID.values[np.argsort(IR)[:50]]
    sub = mdf[mdf.Participant_ID.isin(consistent_ids)]
    formula, re_formula = specs["with_contested_interactions"]
    fit = smf.mixedlm(formula, sub, groups=sub["Participant_ID"],
                      re_formula=re_formula).fit(reml=True, method=["bfgs", "lbfgs", "cg"])
    keep = [k for k in fit.fe_params.index if not k.startswith("C(question_index)") and k != "Intercept"]
    mixed["consistent_half_interactions"] = {k: {"beta": float(fit.params[k]), "se": float(fit.bse[k]),
                                                 "p": float(fit.pvalues[k])} for k in keep}
    mixed["consistent_half_interactions"]["converged"] = bool(fit.converged)
    print("\nMixedLM [consistent half of annotators, n = 50]:")
    for k in keep:
        print(f"  {k:<22} β={fit.params[k]:7.3f}  SE={fit.bse[k]:.3f}  p={fit.pvalues[k]:.4f}")
RESULTS["mixed_model"] = mixed

# ═════════════════════════════════════════════════════════════
# 5. Agreement on a continuous scale
# ═════════════════════════════════════════════════════════════
M = ratings.values.astype(float)  # annotators × prompts
alpha_int = krippendorff.alpha(reliability_data=M, level_of_measurement="interval")
alpha_ord = krippendorff.alpha(reliability_data=M, level_of_measurement="ordinal")
# ICC(1) of prompt means from the long data (one-way, prompts as targets)
k_bar = long_df.groupby("question_index").size().mean()
grand = long_df.rating.mean()
ssb = (long_df.groupby("question_index").rating.agg(lambda s: len(s) * (s.mean() - grand) ** 2)).sum()
ssw = (long_df.rating - long_df.groupby("question_index").rating.transform("mean")).pow(2).sum()
msb, msw = ssb / (150 - 1), ssw / (len(long_df) - 150)
icc1 = (msb - msw) / (msb + (k_bar - 1) * msw)
icc_k = (msb - msw) / msb
alpha_cont = krippendorff.alpha(reliability_data=ratings[sorted(contested)].values.astype(float),
                                level_of_measurement="interval")
alpha_unc = krippendorff.alpha(
    reliability_data=ratings[[c for c in ratings.columns if c not in contested]].values.astype(float),
    level_of_measurement="interval")
RESULTS["agreement"] = {"kripp_alpha_interval": alpha_int, "kripp_alpha_ordinal": alpha_ord,
                        "kripp_alpha_interval_contested": alpha_cont,
                        "kripp_alpha_interval_uncontested": alpha_unc,
                        "icc1_single_rater": icc1, "icc_k_mean_of_raters": icc_k}
print(f"\nKrippendorff α interval = {alpha_int:.3f} (ordinal {alpha_ord:.3f}); "
      f"contested {alpha_cont:.3f}, uncontested {alpha_unc:.3f}")
print(f"ICC(1) single rater = {icc1:.3f};  ICC(k) prompt mean = {icc_k:.3f}")

# ═════════════════════════════════════════════════════════════
# 6. Figure: rating-level vs annotator-level evidence
# ═════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, subset in zip(axes, ["All prompts", "Contested"]):
    d = tests_df[tests_df.subset == subset].iloc[::-1]
    colors = ["#2b6cb0" if p < 0.05 else "#a0aec0" for p in d.p_holm]
    ax.barh(d.feature, d["diff"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"{subset} — annotator-level (n=100)")
    ax.set_xlabel("Mean rating difference (points)")
    for y, (dd, cd) in enumerate(zip(d["diff"], d.cohens_d)):
        ax.text(dd + (0.15 if dd >= 0 else -0.15), y, f"d={cd:.2f}", va="center",
                ha="left" if dd >= 0 else "right", fontsize=8)
fig.suptitle("Group differences tested on annotator means (blue = Holm p < .05)")
fig.tight_layout()
fig.savefig("outputs/fig_week9_annotator_level.png", dpi=150)

with open("outputs/week9_results.json", "w") as fh:
    json.dump(RESULTS, fh, indent=1, default=float)
print("\nSaved outputs/week9_*.csv, outputs/week9_results.json, outputs/fig_week9_annotator_level.png")
