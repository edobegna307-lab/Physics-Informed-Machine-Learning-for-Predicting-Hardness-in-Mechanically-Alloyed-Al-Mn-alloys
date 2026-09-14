"""
Supplementary Code S1
Small-Data Machine Learning for Hardness Prediction in Mechanically Alloyed
Al-Mn-Zr Alloys with Ce and Y Additions: A Grouped-Validation and
Uncertainty-Quantified Study

This single script reproduces every statistic, table and figure reported in
Section 3 of the manuscript, starting from the raw 42-specimen dataset
(Supplementary_Data_S1_clean_dataset.csv). Run with:

    python Supplementary_Code_S1_full_analysis.py

Requires: pandas, numpy, scipy, scikit-learn, seaborn, matplotlib, shap
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import shap
import json

np.random.seed(42)

# =========================================================================
# 0. LOAD DATA AND ENGINEER FEATURES (Section 2.4)
# =========================================================================
df = pd.read_csv('Supplementary_Data_S1_clean_dataset.csv')
df['HP_term'] = df['grain_size'] ** -0.5
df['RE_Ce'] = (df['alloy'] == 'Al-Mn-Zr-Ce').astype(int)
df['RE_Y']  = (df['alloy'] == 'Al-Mn-ZR-Y').astype(int)
df['condition_id'] = (df['hotpress_temp'].astype(str) + '_' +
                       df['anneal_temp'].astype(str) + '_' + df['anneal_time'].astype(str))

y = df['microhardness'].values
n = len(df)
colors = {'Al-Mn-Zr': '#1f77b4', 'Al-Mn-Zr-Ce': '#d62728', 'Al-Mn-ZR-Y': '#2ca02c'}

print(f"Loaded n = {n} specimens, {df['alloy'].nunique()} alloy families, "
      f"{df['condition_id'].nunique()} unique process conditions.\n")

feat_M1 = ['HP_term', 'RE_Ce', 'RE_Y', 'density']
feat_M2 = ['hotpress_temp', 'anneal_temp', 'anneal_time', 'RE_Ce', 'RE_Y']
feat_M3 = ['hotpress_temp', 'anneal_temp', 'anneal_time', 'grain_size', 'HP_term',
           'density', 'aT', 'RE_Ce', 'RE_Y']
rf_params = dict(n_estimators=500, max_depth=5, min_samples_leaf=2, random_state=42)
loo = LeaveOneOut()

# =========================================================================
# 1. LOOCV PREDICTIONS FOR M1, M2, M3 AND BOOTSTRAP 95% CONFIDENCE INTERVALS
#    (Section 3.2, Table 3, Figure 1)
# =========================================================================
def loocv_predict(feat_cols, model_fn):
    X = df[feat_cols].values
    return cross_val_predict(model_fn(), X, y, cv=loo)

def bootstrap_ci(y_true, y_pred, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y_true))
    r2s, rmses, maes = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(y_true), replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if np.std(yt) == 0:
            continue
        r2s.append(r2_score(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        maes.append(mean_absolute_error(yt, yp))
    ci = lambda a: np.percentile(a, [2.5, 97.5])
    return dict(r2=r2_score(y_true, y_pred), r2_ci=ci(r2s),
                rmse=np.sqrt(mean_squared_error(y_true, y_pred)), rmse_ci=ci(rmses),
                mae=mean_absolute_error(y_true, y_pred), mae_ci=ci(maes))

pred_M1 = loocv_predict(feat_M1, lambda: LinearRegression())
pred_M2 = loocv_predict(feat_M2, lambda: RandomForestRegressor(**rf_params))
pred_M3 = loocv_predict(feat_M3, lambda: RandomForestRegressor(**rf_params))

print("=== Table 3: LOOCV performance with bootstrap 95% CI ===")
results_table3 = {}
for name, pred in [('M1', pred_M1), ('M2', pred_M2), ('M3', pred_M3)]:
    ci = bootstrap_ci(y, pred)
    results_table3[name] = ci
    print(f"{name}: R2={ci['r2']:.3f} [{ci['r2_ci'][0]:.3f},{ci['r2_ci'][1]:.3f}]  "
          f"RMSE={ci['rmse']:.1f} [{ci['rmse_ci'][0]:.1f},{ci['rmse_ci'][1]:.1f}]  "
          f"MAE={ci['mae']:.1f} [{ci['mae_ci'][0]:.1f},{ci['mae_ci'][1]:.1f}]")

# Figure 1: LOOCV parity for M2
fig, ax = plt.subplots(figsize=(6, 6))
rmse_M2 = results_table3['M2']['rmse']
for alloy_name, c in colors.items():
    mask = (df['alloy'] == alloy_name).values
    ax.scatter(y[mask], pred_M2[mask], c=c, s=70, edgecolor='k', linewidth=0.5,
               label=alloy_name.replace('ZR', 'Zr'), alpha=0.85, zorder=3)
lims = [min(y.min(), pred_M2.min()) - 15, max(y.max(), pred_M2.max()) + 15]
ax.plot(lims, lims, 'k--', lw=1.3, label='Parity (y=x)', zorder=1)
ax.fill_between(lims, [l - rmse_M2 for l in lims], [l + rmse_M2 for l in lims],
                 color='gray', alpha=0.15, label=f'\u00b1RMSE ({rmse_M2:.1f} HV)', zorder=0)
ax.set_xlabel('Experimental micro-hardness (HV)'); ax.set_ylabel('Predicted micro-hardness (HV, LOOCV)')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f'Deployable process+composition RF: LOOCV parity (n={n})\n'
             f'$R^2$={results_table3["M2"]["r2"]:.3f}, RMSE={rmse_M2:.1f} HV')
ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig('fig_parity_process.png', dpi=300); plt.close()

# =========================================================================
# 2. GROUPED CROSS-VALIDATION (Section 3.3, Table 4)
# =========================================================================
def grouped_cv(feat_cols, model_fn, group_col):
    X = df[feat_cols].values
    groups = df[group_col].values
    logo = LeaveOneGroupOut()
    return cross_val_predict(model_fn(), X, y, cv=logo, groups=groups)

print("\n=== Table 4: Grouped cross-validation ===")
for mname, feats, model_fn in [
    ('M1_linear', feat_M1, lambda: LinearRegression()),
    ('M2_RF', feat_M2, lambda: RandomForestRegressor(**rf_params)),
    ('M3_RF', feat_M3, lambda: RandomForestRegressor(**rf_params)),
]:
    pred_loco = grouped_cv(feats, model_fn, 'condition_id')
    pred_loao = grouped_cv(feats, model_fn, 'alloy')
    print(f"{mname}: LeaveOneConditionOut R2={r2_score(y, pred_loco):.3f}  "
          f"LeaveOneAlloyOut R2={r2_score(y, pred_loao):.3f}")

# =========================================================================
# 3. REPEATED 5-FOLD CV, 20 SEEDS (Section 3.4)
# =========================================================================
def repeated_kfold(feat_cols, model_fn, n_repeats=20, k=5):
    X = df[feat_cols].values
    r2s = []
    for seed in range(n_repeats):
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        pred = cross_val_predict(model_fn(), X, y, cv=kf)
        r2s.append(r2_score(y, pred))
    return np.array(r2s)

r2_rep_M2 = repeated_kfold(feat_M2, lambda: RandomForestRegressor(**rf_params))
r2_rep_M3 = repeated_kfold(feat_M3, lambda: RandomForestRegressor(**rf_params))
print(f"\n=== Section 3.4: repeated 5-fold (20 seeds) ===")
print(f"M2: {r2_rep_M2.mean():.3f} +/- {r2_rep_M2.std():.3f} (range {r2_rep_M2.min():.3f}-{r2_rep_M2.max():.3f})")
print(f"M3: {r2_rep_M3.mean():.3f} +/- {r2_rep_M3.std():.3f} (range {r2_rep_M3.min():.3f}-{r2_rep_M3.max():.3f})")

# =========================================================================
# 4. BASELINE MODEL COMPARISON (Section 3.5, Table 5, Figure 2)
# =========================================================================
X2 = df[feat_M2].values
baselines = {
    'Mean predictor': lambda: DummyRegressor(strategy='mean'),
    'OLS (linear)': lambda: LinearRegression(),
    'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'PLS (2 comp.)': lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2)),
    'SVR (RBF)': lambda: make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale')),
    'Gaussian Process': lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=RBF() + WhiteKernel(), random_state=42, normalize_y=True)),
    'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=42),
    'Random Forest (M2)': lambda: RandomForestRegressor(**rf_params),
}
print("\n=== Table 5: Baseline comparison (M2 feature set), LOOCV ===")
baseline_r2 = {}
for bname, fn in baselines.items():
    pred = cross_val_predict(fn(), X2, y, cv=loo)
    r2 = r2_score(y, pred); rmse = np.sqrt(mean_squared_error(y, pred)); mae = mean_absolute_error(y, pred)
    baseline_r2[bname] = r2
    print(f"  {bname:20s} R2={r2:6.3f}  RMSE={rmse:6.1f}  MAE={mae:6.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
short_names = {'Mean predictor':'Mean\npredictor','OLS (linear)':'OLS\n(linear)','Ridge':'Ridge',
               'PLS (2 comp.)':'PLS','SVR (RBF)':'SVR\n(RBF)','Gaussian Process':'Gaussian\nProcess',
               'Gradient Boosting':'Gradient\nBoosting','Random Forest (M2)':'Random\nForest'}
names = [short_names[k] for k in baseline_r2]; r2s = list(baseline_r2.values())
colors_bar = ['#999999' if 'Mean' in n else ('#d62728' if 'Random' in n else '#4c72b0') for n in names]
bars = ax.bar(names, r2s, color=colors_bar, edgecolor='k')
ax.axhline(0, color='k', lw=0.8); ax.set_ylabel('LOOCV R\u00b2')
ax.set_title('Baseline model comparison (M2 feature set), LOOCV')
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, r2s):
    ax.text(b.get_x() + b.get_width()/2, v + (0.03 if v >= 0 else -0.06), f"{v:.3f}", ha='center', fontsize=9)
plt.tight_layout(); plt.savefig('fig_baseline_comparison.png', dpi=300); plt.close()

# =========================================================================
# 5. COEFFICIENT BOOTSTRAP CI FOR M1 (Section 3.7, Table 6, Figure 4)
# =========================================================================
X1 = df[feat_M1].values
rng = np.random.default_rng(42)
n_boot = 5000
coef_names = feat_M1 + ['Intercept']
coefs_boot = np.zeros((n_boot, len(coef_names)))
for b in range(n_boot):
    idx = rng.choice(n, size=n, replace=True)
    Xb, yb = X1[idx], y[idx]
    lr = LinearRegression().fit(Xb, yb)
    coefs_boot[b, :-1] = lr.coef_
    coefs_boot[b, -1] = lr.intercept_

lr_full = LinearRegression().fit(X1, y)
coef_point = list(lr_full.coef_) + [lr_full.intercept_]
print("\n=== Table 6: M1 coefficients with bootstrap 95% CI ===")
for i, cname in enumerate(coef_names):
    lo, hi = np.percentile(coefs_boot[:, i], [2.5, 97.5])
    print(f"  {cname:10s}: {coef_point[i]:8.2f}  [{lo:8.2f}, {hi:8.2f}]")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
names_c = feat_M1
pts = coef_point[:-1]
los, his = [], []
for i, cname in enumerate(names_c):
    lo, hi = np.percentile(coefs_boot[:, i], [2.5, 97.5])
    los.append(pts[i] - lo); his.append(hi - pts[i])
ax.errorbar(pts, names_c, xerr=[los, his], fmt='o', color='#4c72b0', ecolor='k', capsize=4, markersize=8)
ax.axvline(0, color='r', ls='--', lw=1)
ax.set_xlabel('Coefficient value (bootstrap 95% CI)')
ax.set_title('M1: coefficient estimates with bootstrap 95% CI (n_boot=5000)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig('fig_coefficient_ci.png', dpi=300); plt.close()

# =========================================================================
# 6. RESIDUAL DIAGNOSTICS FOR M2 (Section 3.6, Figure 3)
# =========================================================================
resid = y - pred_M2
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
ax = axes[0, 0]
for alloy_name, c in colors.items():
    mask = (df['alloy'] == alloy_name).values
    ax.scatter(pred_M2[mask], resid[mask], c=c, s=50, edgecolor='k', linewidth=0.4, label=alloy_name.replace('ZR','Zr'))
ax.axhline(0, color='k', ls='--', lw=1)
ax.set_xlabel('Predicted hardness (HV)'); ax.set_ylabel('Residual (HV)')
ax.set_title('(a) Residuals vs. predicted'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

for ax, col, label in [(axes[0,1], 'hotpress_temp', 'Hot-press T (\u00b0C)'),
                        (axes[0,2], 'anneal_temp', 'Anneal T (\u00b0C)'),
                        (axes[1,0], 'anneal_time', 'Anneal time (h)')]:
    for alloy_name, c in colors.items():
        mask = (df['alloy'] == alloy_name).values
        ax.scatter(df[col][mask], resid[mask], c=c, s=50, edgecolor='k', linewidth=0.4)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel(label); ax.set_ylabel('Residual (HV)'); ax.grid(alpha=0.3)

axes[1,1].hist(resid, bins=12, color='#4c72b0', edgecolor='k')
axes[1,1].axvline(0, color='k', ls='--', lw=1)
axes[1,1].set_xlabel('Residual (HV)'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title(f'(e) median|resid|={np.median(np.abs(resid)):.1f} HV, max={np.max(np.abs(resid)):.1f} HV')
axes[1,1].grid(alpha=0.3)

stats.probplot(resid, dist='norm', plot=axes[1,2])
axes[1,2].set_title('(f) Normal Q-Q plot')
plt.tight_layout(); plt.savefig('fig_residual_diagnostics.png', dpi=300); plt.close()
print(f"\n=== Section 3.6: residuals === median|resid|={np.median(np.abs(resid)):.1f} HV, max={np.max(np.abs(resid)):.1f} HV")

# =========================================================================
# 7. SHAP AND PERMUTATION IMPORTANCE (Section 3.8, Figure 5)
# =========================================================================
X3 = df[feat_M3].values
rf3_full = RandomForestRegressor(**rf_params).fit(X3, y)
rf2_full = RandomForestRegressor(**rf_params).fit(X2, y)
explainer = shap.TreeExplainer(rf3_full)
shap_values = explainer.shap_values(df[feat_M3])
fig = plt.figure(figsize=(7.5, 6))
shap.summary_plot(shap_values, df[feat_M3], show=False)
plt.tight_layout(); plt.savefig('fig_shap_beeswarm.png', dpi=300, bbox_inches='tight'); plt.close()

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_M3).sort_values(ascending=False)
print("\n=== Section 3.8: mean |SHAP| (M3) ===")
print(mean_abs_shap)

perm2 = permutation_importance(rf2_full, X2, y, n_repeats=100, random_state=42, scoring='r2')
print("\nPermutation importance (M2, drop in R2):")
for i, f in enumerate(feat_M2):
    print(f"  {f:16s}: {perm2.importances_mean[i]:.3f} +/- {perm2.importances_std[i]:.3f}")

# =========================================================================
# 8. FAMILY-SPECIFIC HALL-PETCH REGRESSION + HETEROGENEITY TEST (Section 3.9, Figure 6)
# =========================================================================
print("\n=== Table 7: family-specific Hall-Petch regression ===")
fig, ax = plt.subplots(figsize=(7, 5.5))
for alloy_name, c in colors.items():
    sub = df[df['alloy'] == alloy_name]
    x = sub['HP_term'].values; yy = sub['microhardness'].values
    slope, intercept, r, p, se = stats.linregress(x, yy)
    n_sub = len(x); tval = stats.t.ppf(0.975, n_sub - 2)
    ci_lo, ci_hi = slope - tval * se, slope + tval * se
    print(f"  {alloy_name:14s}: slope={slope:7.1f} [{ci_lo:7.1f},{ci_hi:7.1f}]  R2={r**2:.3f}  p={p:.4f}  n={n_sub}")
    xs = np.linspace(x.min(), x.max(), 50); pred_line = slope * xs + intercept
    x_mean = x.mean()
    s_yx = np.sqrt(np.sum((yy - (slope*x+intercept))**2) / (n_sub - 2))
    se_pred = s_yx * np.sqrt(1/n_sub + (xs-x_mean)**2 / np.sum((x-x_mean)**2))
    ax.plot(xs, pred_line, color=c, lw=1.8)
    ax.fill_between(xs, pred_line-tval*se_pred, pred_line+tval*se_pred, color=c, alpha=0.15)
    ax.scatter(x, yy, color=c, s=55, edgecolor='k', linewidth=0.4, label=f"{alloy_name.replace('ZR','Zr')} (p={p:.3f})")
ax.set_xlabel('Hall-Petch term, d^-1/2 (nm^-1/2)'); ax.set_ylabel('Micro-hardness (HV)')
ax.set_title('Family-specific Hall-Petch fits with 95% confidence bands')
ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig('fig_hallpetch_ci.png', dpi=300); plt.close()

# heterogeneity F-test (common slope vs family-specific slopes)
X_int = pd.get_dummies(df['alloy'], drop_first=True).astype(float)
X_int['HP_term'] = df['HP_term'].values
for col in pd.get_dummies(df['alloy'], drop_first=True).columns:
    X_int[f'{col}_x_HP'] = X_int[col] * X_int['HP_term']
full_model = LinearRegression().fit(X_int.values, y)
ss_res_full = np.sum((y - full_model.predict(X_int.values))**2)
X_red = X_int.drop(columns=[c for c in X_int.columns if '_x_HP' in c])
red_model = LinearRegression().fit(X_red.values, y)
ss_res_red = np.sum((y - red_model.predict(X_red.values))**2)
df1 = X_int.shape[1] - X_red.shape[1]; df2 = n - X_int.shape[1] - 1
F = ((ss_res_red - ss_res_full)/df1) / (ss_res_full/df2)
p_interaction = 1 - stats.f.cdf(F, df1, df2)
print(f"\nHeterogeneity F-test: F({df1},{df2})={F:.3f}, p={p_interaction:.4f}")

# =========================================================================
# 9. COMPOSITION, COMPRESSIVE STRENGTH, THERMAL-STABILITY RETENTION
#    (Section 3.10, Figures 7-8, Table 8)
# =========================================================================
df['retention_pct'] = 100 * df['hardness450'] / df['microhardness']
r_uts = np.corrcoef(df['microhardness'], df['UTS'])[0, 1]
print(f"\n=== Section 3.10 === Pearson r (hardness vs UCS) = {r_uts:.3f}")

fig, ax = plt.subplots(figsize=(6, 5))
for alloy_name, c in colors.items():
    sub = df[df['alloy'] == alloy_name]
    ax.scatter(sub['microhardness'], sub['UTS'], c=c, label=alloy_name.replace('ZR','Zr'), s=60, edgecolor='k', linewidth=0.4)
ax.set_xlabel('Micro-hardness (HV)'); ax.set_ylabel('Ultimate compressive strength (MPa)')
ax.set_title(f'Hardness-compressive-strength relationship (r={r_uts:.2f})')
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig('fig_uts_hv.png', dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for alloy_name, c in colors.items():
    sub = df[df['alloy'] == alloy_name]
    ax.scatter(sub['microhardness'], sub['hardness450'], c=c, label=alloy_name.replace('ZR','Zr'), s=60, edgecolor='k', linewidth=0.4)
lims2 = [df['hardness450'].min()-20, df['microhardness'].max()+20]
ax.plot(lims2, lims2, 'k--', lw=1, label='No degradation (y=x)')
ax.set_xlabel('As-processed micro-hardness (HV)'); ax.set_ylabel('Hardness after 450\u00b0C anneal (HV)')
ax.set_title('Thermal stability of hardness')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig('fig_thermal_stability.png', dpi=300); plt.close()

print("\n=== Table 8: retention by alloy family ===")
print(df.groupby('alloy')[['microhardness','hardness450','retention_pct']].mean())

# =========================================================================
# 10. SURROGATE-BASED CANDIDATE RANKING (Section 3.11, Table 9)
# =========================================================================
hp_range = np.linspace(350, 450, 21)
at_range = np.linspace(350, 450, 21)
att_range = np.array([2, 3, 4])
alloys_grid = [('Al-Mn-Zr', 0, 0), ('Al-Mn-Zr-Ce', 1, 0), ('Al-Mn-ZR-Y', 0, 1)]
HP2, AT2, ATT2 = np.meshgrid(hp_range, at_range, att_range, indexing='ij')
n2 = HP2.size
grid_frames = []
for aname, ce, yv in alloys_grid:
    Xq = pd.DataFrame({'hotpress_temp': HP2.ravel(), 'anneal_temp': AT2.ravel(),
                        'anneal_time': ATT2.ravel(), 'RE_Ce': np.full(n2, ce), 'RE_Y': np.full(n2, yv)})
    pred = rf2_full.predict(Xq)
    grid_frames.append(pd.DataFrame({'alloy': aname, 'hotpress_temp': Xq['hotpress_temp'],
                                      'anneal_temp': Xq['anneal_temp'], 'anneal_time': Xq['anneal_time'],
                                      'pred_HV': pred}))
grid_df = pd.concat(grid_frames, ignore_index=True).sort_values('pred_HV', ascending=False)
top = grid_df.iloc[0]
xtop = pd.DataFrame([[top['hotpress_temp'], top['anneal_temp'], top['anneal_time'],
                       1 if top['alloy']=='Al-Mn-Zr-Ce' else 0, 1 if top['alloy']=='Al-Mn-ZR-Y' else 0]],
                     columns=feat_M2)
tree_preds = np.array([t.predict(xtop.values)[0] for t in rf2_full.estimators_])
print(f"\n=== Table 9: top candidate === {top['alloy']}, HP={top['hotpress_temp']:.0f}, "
      f"AT={top['anneal_temp']:.0f}, t={top['anneal_time']:.0f}h -> pred={top['pred_HV']:.1f} "
      f"+/- {1.96*tree_preds.std():.1f} HV (95% CI)")
best_exp = df.loc[df['microhardness'].idxmax()]
print(f"Best experimental specimen: {best_exp['alloy']}, HP={best_exp['hotpress_temp']}, "
      f"AT={best_exp['anneal_temp']}, t={best_exp['anneal_time']}h -> HV={best_exp['microhardness']}")

print("\n=== All analyses complete. Figures saved to working directory. ===")

# =========================================================================
# 11. PAIRED CASE-RESAMPLING TESTS BETWEEN MODELS (Section 3.3, Table 3)
# =========================================================================
X1 = df[feat_M1].values
pred_M2_OLS = cross_val_predict(LinearRegression(), X2, y, cv=loo)

def paired_test(y_true, predA, predB, n_boot=10000, seed=42, labelA='A', labelB='B'):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    errA = (y_true - predA)**2; errB = (y_true - predB)**2
    diff = errA - errB
    obs = diff.mean()
    boots = np.array([diff[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots > 0).mean(), (boots < 0).mean())
    print(f"{labelA} vs {labelB}: mean(SE_{labelA}-SE_{labelB})={obs:.1f}  95% interval [{ci[0]:.1f},{ci[1]:.1f}]  p={p:.3f}")
    return obs, ci, p

print("\n=== Table 3: paired case-resampling tests on LOOCV squared errors ===")
pred_M3_loo = cross_val_predict(RandomForestRegressor(**rf_params), X3, y, cv=loo)
pred_M1_loo = cross_val_predict(LinearRegression(), X1, y, cv=loo)
paired_test(y, pred_M2, pred_M3_loo, labelA='M2-RF', labelB='M3')
paired_test(y, pred_M2_OLS, pred_M3_loo, labelA='M2-OLS', labelB='M3')
paired_test(y, pred_M2_OLS, pred_M1_loo, labelA='M2-OLS', labelB='M1')
paired_test(y, pred_M2, pred_M1_loo, labelA='M2-RF', labelB='M1')
paired_test(y, pred_M2_OLS, pred_M2, labelA='M2-OLS', labelB='M2-RF')
paired_test(y, pred_M1_loo, pred_M3_loo, labelA='M1', labelB='M3')

# =========================================================================
# 12. LOAO ENCODING MECHANICS CHECK (Section 2.6 (ii), verified claim)
# =========================================================================
train_mask = df['alloy'] != 'Al-Mn-Zr-Ce'
print("\n=== LOAO mechanics: RE_Ce column when Ce alloy held out ===")
print("Unique values in training set:", df.loc[train_mask, 'RE_Ce'].unique())
lr_check = LinearRegression().fit(df.loc[train_mask, feat_M1].values, df.loc[train_mask, 'microhardness'].values)
print("Fitted RE_Ce coefficient (should collapse to ~0):", dict(zip(feat_M1, lr_check.coef_))['RE_Ce'])

print("\n=== Supplementary Code S1 complete. ===")
