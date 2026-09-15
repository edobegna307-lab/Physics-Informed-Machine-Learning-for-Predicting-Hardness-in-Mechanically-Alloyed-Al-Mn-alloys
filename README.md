# Supplementary Materials

Data and code for: **"Limits of Composition Transferability in Small-Data Hardness Modelling of Mechanically Alloyed Al–Mn–Zr-Based Alloys"** (Eshetu, M.A.; Novikov, A.S.)

## Contents

| File | Description |
|---|---|
| `Supplementary_Data_S1_clean_dataset.csv` | The 42-specimen experimental dataset (3 alloys × 14 hot-press/anneal conditions) used for all statistical modelling in the manuscript. |
| `Supplementary_Code_S1_analysis.py` | Complete, self-contained Python script that reproduces every statistic, table and figure underlying the manuscript from the raw dataset above, including leave-one-out cross-validation, grouped (condition- and alloy-out) cross-validation, case-resampling intervals, paired significance tests, baseline model comparisons, residual diagnostics, SHAP/permutation attribution, and the Hall–Petch-like heterogeneity test. Note: the short proceedings-format manuscript consolidates several of this script's outputs into a single combined table and a single combined figure for space; this script produces the fully disaggregated set (matching the longer journal-format version of the manuscript, where included in this repository).

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`

Install with:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn shap
```

## Reproducing the results

Both files must be in the same working directory (the script reads the CSV by its exact filename above).

```bash
python Supplementary_Code_S1_analysis.py
```

This prints every reported statistic to the console and saves 8 PNG figures to the working directory. Runtime is a few minutes on a standard laptop CPU; no GPU is required. Last verified: exit code 0, all figures generated, all printed values matching the manuscript, in a clean sandbox with only the two files above present.

## Dataset column reference

| Column | Description |
|---|---|
| `milling_time` | Ball-milling time (h); constant at 20 h for all specimens |
| `hotpress_temp` | Hot-press consolidation temperature (°C): 350, 400, or 450 |
| `anneal_temp` | Post-hot-press anneal temperature (°C): 350, 400, or 450 |
| `anneal_time` | Post-hot-press anneal time (h): 2 or 4 |
| `alloy` | Alloy identity: `Al-Mn-Zr`, `Al-Mn-Zr-Ce`, or `Al-Mn-ZR-Y` |
| `microhardness` | As-processed Vickers micro-hardness (HV) — primary modelling target |
| `hardness450` | Hardness after a common 450 °C thermal-stability anneal (HV) |
| `UTS` | Compressive-strength value (MPa); see manuscript Section 2.3 for the compression-vs-tension terminology note and the caveat that the test temperature label is not recorded in this column |
| `grain_size` | XRD-derived (Williamson–Hall) crystallite size (nm) — see manuscript Section 2.3 on why this is not equated with metallographic grain size |
| `aT` | Crystallographic lattice parameter (nm) |
| `density` | Bulk density of the sintered compact (g/cm³), Archimedes method |

## Known limitations of this dataset (see manuscript Section 5 / Limitations for full discussion)

- Each alloy was milled as a single powder batch, then divided into 14 independently hot-pressed and annealed specimens under different conditions. So specimens are independent at the pressing/annealing level, but **batch identity is perfectly confounded with alloy identity** (all 14 specimens of one alloy share one parent powder batch) and the two cannot be statistically separated with this design.
- Individual hardness-indent values, per-specimen standard deviations, the compression-test temperature label, yield strength (σ₀.₂), and plastic deformation (ψ%) are not included in this dataset, though they were measured in the originating characterisation programme.
- Post-hot-press annealing atmosphere and heating/cooling rate are not recorded here.

These are stated explicitly so that anyone reusing this dataset is aware of what it does and does not support.

## Citation

If you use this dataset or code, please cite the manuscript above. A citable archive (Zenodo) with a version-locked DOI will be linked here upon manuscript submission/acceptance.

## License

Specify a license here (e.g., MIT for code, CC-BY-4.0 for data) before making the repository public, if not already set at the repository level.
