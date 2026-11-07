Supplementary Code S1 and Data S1

Article: Structure–Process–Hardness Relationships in Mechanically Alloyed Al–Mn–Zr, Al–Mn–Zr–Ce and Al–Mn–Zr–Y Alloys after Hot-Press Consolidation Authors: Mintesenot Admasu Eshetu, Alexander S. Novikov. Journal: Solid State Sciences (submitted).

One script, run on one CSV file, reproduces every numerical result, table value and figure of the article and its Supplementary Information that is computed from data. No repository DOI exists yet; it will be added after deposit.

Files
File	Content
Supplementary_Code_S1_analysis.py	Complete analysis script (all models, validation schemes, resampling, figures).
Supplementary_Data_S1_clean_dataset.csv	39 specimens × 11 columns (data dictionary below).
requirements.txt	Package versions the script was verified with (Python 3.13).
run_output_reference.log	Console output of a verified run, for line-by-line comparison.
Final_Reproducibility_Record.md	MD5 checksums, environment, seeds and fixed hyperparameters.
Data_Correction_Log.md	Data curation record (precision, excluded specimens, aT, labels).
make_graphical_abstract.py	Generates the graphical abstract (hardness means, SD and retention from the CSV; model R² from the run output).
fig_*.png	Figures written by the script: main-text Fig. 1 and Fig. 4 (Fig. 2 in the version without XRD figures), SI Figs. S1–S10.
CITATION.cff, zenodo_metadata_template.json	Deposit metadata (DOI, licence and ORCIDs to be completed by the authors).
Run
bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 Supplementary_Code_S1_analysis.py     # run from this directory; about 10 min on one CPU core

All random seeds are fixed (global seed 42; Random Forests random_state=42; resampling seeds 42; repeated 5-fold seeds 0–19; surrogate seeds 0–9), so a re-run reproduces the printed numbers.

Script sections and where they are used
§	Content	Used in
1	LOOCV, 95% case-resampling intervals (M1, M2-RF, M3-RF); M3 variants (+aT, HP-only, paired d + HP)	Sections 2.5, 3.3; Tables S17, S21
2	LOCO and LOAO, grouped no-skill baseline, per-family LOAO table	Sections 3.4; Tables S20, S24
3	Repeated 5-fold CV (20 seeds)	Section 3.6
4	Seven regressors plus mean baseline on the M2 inputs	Table S14, Fig. S1
5	M1 coefficients, 5,000-refit bootstrap	Table S18, Fig. S3
6 / 6b	Residual diagnostics; refits without the two largest residuals	Section 3.6, Fig. S2
7 / 7b	SHAP and permutation importance; grouped out-of-fold permutation importance	Section 3.5; Table S13, Table S26, Fig. S6
8	Family-specific Hall–Petch-like regressions, heterogeneity test	Section 3.5, Fig. S7
9	UTS correlation (exploratory); 450 °C retention	Section 3.1; Tables S5, S6, S16; Figs. S4–S5
10	Retrospective surrogate ranking (not a design recommendation)	Supplementary Section S15, Table S15
11	Paired case-resampling tests; M2-OLS row	Sections 2.5, 3.3; Table S21
12	LOAO encoding check (Ce coefficient when Ce is withheld)	Section 3.4
13	Table 1 summaries, family means by process level, Spearman correlations, retention	Section 3.1; Table S22
14	Density-population sensitivity (without the three densest specimens; original 42 rows for reference)	Table S23
15	Process-coverage map and LOAO family-mean plot	Figs. S9, S10
16	Main-text LOAO figure (observed, baseline, M1, M2-RF, M3-RF per withheld family)	Fig. 4 / Fig. 2

Tables S25 (provenance and overlap with Ref. [11]) and S27 (correction log) are documentation, not script output.

Data dictionary
Column	Meaning	Unit
alloy	Nominal formulation: Al-Mn-Zr, Al-Mn-Zr-Ce, Al-Mn-Zr-Y	–
milling_time	Ball-milling time (20 for all specimens)	h
hotpress_temp, anneal_temp, anneal_time	Process conditions (nominal set points)	°C, °C, h
microhardness	Target: as-processed HV0.1 (100 g, 15 s), screened specimen mean (individual indents not retained)	HV
hardness450	Hardness after a common 450 °C treatment (used for retention only)	HV
grain_size	XRD Williamson–Hall coherent-domain size d, not a metallographic grain size; no per-specimen uncertainty	nm
aT	Theoretical composition-based lattice parameter, one value per alloy; used only in a sensitivity check	nm
density	Archimedes bulk density (no true-density or porosity measurement)	g·cm⁻³
UTS	Ultimate tensile strength, exploratory only	MPa

Engineered inside the script: HP_term = grain_size**-0.5, one-hot RE_Ce and RE_Y (reference level Al–Mn–Zr), and condition_id (the LOCO grouping key).

Limits of the data

Alloy family is confounded with parent powder batch; each process condition has one compact per family; indent readings, process metadata (powder size, oxygen, heating rates, furnace details), raw XRD patterns and ICDD card numbers were not retained.

Relationship to Ref. [11]

Part of the campaign was reported in Lekeka et al., Int. J. Res. Innov. Appl. Sci. 10 (2025) 386–417, doi:10.51584/IJRIAS.2025.1010000030. Three overlapping specimens are excluded (see Data_Correction_Log.md and the article).
