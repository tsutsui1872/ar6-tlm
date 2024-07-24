# ar6-tlm

This repository provides code and data for comparing different approaches with two-layer energy balance models (TLMs) for probabilistic climate assessment as in IPCC Working Group I Sixth Assessment Report (AR6).
The main contents are the following notebooks:

Notebook | Description
---------|------------
[010_cmip6_preprocess](notebook/010_cmip6_preprocess.ipynb) | CMIP6 data normalization
[020_calibration](notebook/020_calibration.ipynb) | Calibration
[022_postproc](notebook/022_postproc.ipynb) | Post-process of calibration results
[030_parms_sampling](notebook/030_parms_sampling.ipynb) | Sampling for TLM parameters
[040_forcing_unc](notebook/040_forcing_unc.ipynb) | Sampling for forcing uncertainties
[050_unconstrained_run](notebook/050_unconstrained_run.ipynb) | Unconstrained ensemble runs
[060_constraining](notebook/060_constraining.ipynb) | Constraining
[062_postproc_parms](notebook/062_postproc_parms.ipynb) | Post-process of the constrained ensembles for TLM parameters
[064_postproc_indicators](notebook/064_postproc_indicators.ipynb) | Post-process of the constrained ensembles for indicators
[070_constraining_fair](notebook/070_constraining_fair.ipynb) | Constraining precisely according to AR6
[072_postproc](notebook/072_postproc.ipynb) | Post-process of the above
[080_constrained_runs](notebook/080_constrained_runs.ipynb) | Constrained ensemble runs
[082_constrained_runs_ch4](notebook/082_constrained_runs_ch4.ipynb) | Replication of the AR6 Chapter 4 assessment
[084_postproc](notebook/084_postproc.ipynb) | Post-process of the constrained runs

The post-process notebooks generate figures to diagnose the results, which are saved in [image](image) directory.
