Reliable and generalizable brain-based predictions of cognitive
functioning across common psychiatric illness
================

## Reference

Chopra, S., Dhamala, E., Lawhead, C., Ricard, J., Orchard, E., An, L.,
Chen, P., Wulan, N., Kumar, P., Rubenstein. A., Moses, J., Chen, L.,
Levi, P., Aquino, K., Fornito, A., Harpaz-Rotem, I., Germine, L., Baker,
J., Yeo, BT., Holmes, A. (2022) [Reliable and generalizable brain-based
predictions of cognitive functioning across common psychiatric
illness](https://www.medrxiv.org/content/10.1101/2022.12.08.22283232v1).
medRxiv.

------------------------------------------------------------------------

## Background

A primary aim of precision psychiatry is the establishment of predictive
models linking individual differences in brain functioning with clinical
symptoms. In particular, cognitive impairments are transdiagnostic,
treatment resistant, and contribute to poor clinical outcomes. Recent
work suggests thousands of participants may be necessary for the
accurate and reliable prediction of cognition, calling into question the
utility of most patient collection efforts. Here, using a
transfer-learning framework, we train a model on functional imaging data
from the UK Biobank (n=36,848) to predict cognitive functioning in three
transdiagnostic patient samples (n=101-224). The model generalizes
across datasets, and brain features driving predictions are consistent
between populations, with decreased functional connectivity within
transmodal cortex and increased connectivity between unimodal and
transmodal regions reflecting a transdiagnostic predictor of cognition.
This work establishes that predictive models derived in large
population-level datasets can be exploited to boost the prediction of
cognitive function across clinical collection efforts.

## Code and Data release

### Code

The `scripts` folder contains the two folders: `analysis` and
`visualisation`:

- The `analysis` folder contains three sub-folders and a conda
  environment file used for all analyses
  (`predictingCognition_env.yml`):
  - `accuracy` - contains python scripts that execute both meta-matching
    (`compute_MM_cognitionPC.py` and kernel ridge regression models
    (`compute_KRR_cognitionPC.py`). The `master_run.sh` script executes
    all python scripts with flags to indicating study sample and
    covariate regression. The `nulls_runHPC` folder contains scripts
    used to generate null prediction models and were executed on a High
    Performance Computing (HPC) cluster.
  - `genralizibility` - contains two python scripts used to test the
    genralizibility of the meta-matching (`mm_cogs_generalize.py`) and
    kernel ridge regression (`krr_cogs_generalize.py`) models
    (i.e. train on one full dataset and test on another independent
    dataset).
  - `feature weights` - contains a python script to generate spatially
    auto-correlated nulls (spin test), and conduct significance testing
    on the feature weights generated.
- The `visualisation` folder contains a Rmarkdown file (`figures.Rmd`)
  used to generate each main test and supplementary figure included in
  the paper. Each code chunk corresponds to a figure or panel. The brain
  renderings in Fig4 require a python env with both `pyvista` and
  `pysurfer` working smoothly (good luck!) Some examples below:

<img src="output/figures/vector_files/readme.png" width="2314" style="display: block; margin: auto;" />

------------------------------------------------------------------------

### Data

- The primary data used are brain (419 x 419) FC matrices and cognitive
  functioning scores for three data sets:

  - Human Connectome Project - Early Psychosis (HCP-EP; n=145)

  - Transdiagnostic Connectomes Project (TCP; n=101)

  - Consortium for Neuropsychiatric Phenomics (CNP; n=224)

- FC matrices and cognitive functioning principal component (PC) scores
  for TCP and CNP can likely soon be shared openly (pending publication
  of the TCP dataset release paper). These files are \>100mb and once I
  have approval I will uploaded here, but please reach out if you would
  like these, and I will send it through. The HCP-EP data requires data
  access permission and if you have this, we are happy to share the data
  used here.

- All model outputs reported in the paper and used to generate all
  figures are provided in `output` folder.

------------------------------------------------------------------------

## Meta-matching model

If you want to apply the meta-matching model to your own data, please
see: <https://github.com/ThomasYeoLab/Meta_matching_models>

You will need functional coupling/connectivity (FC) data and a phenotype
you want to predict. The FC data will need to be extracted using the 419
region atlas defined in the link above (also see `data/atlas` folder for
a template).

**Note:** The meta-matching model use in the current analysis was run
using data with GSR and z-scoring, so a version of the V1.1
meta-matching model was used.

------------------------------------------------------------------------

## Questions

Please contact me (Sidhant Chopra) at <sidhant.chopra@yale.edu> and/or
<sidhant.chopra4@gmail.com>
