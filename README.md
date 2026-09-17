# Conformal Crop Classification Across Regions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22806699.svg)](https://doi.org/10.5281/zenodo.22806699)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, trained models, and evaluation artefacts for the paper **"Conformal Crop
Classification Across Regions with Sentinel-2 Time Series"**.

A crop classifier that scores 0.85 macro-F1 under random cross-validation is not
the same classifier once it is pointed at a district nobody surveyed. This
repository is about the second number, and about what a conformal prediction set
can and cannot promise when calibration and deployment happen in different
places.

## The problem, in one paragraph

Split conformal prediction returns a set of candidate crops that contains the
true crop 90% of the time. That promise survives being averaged over held-out
districts in Pakistan: mean coverage stays near 0.89. It does not survive being
read district by district. The target is missed in up to 28% of held-out
regions, and the worst region falls to 0.68, with nothing in the output to flag
it. We call this the **geographic coverage gap**. It appears in all eight
backbones we tested, so it is a property of spatially structured data rather
than of any one model.

Two fixes are implemented here. A **region-robust** threshold calibrates one
quantile per district and takes a conservative quantile across districts, which
restores overall coverage. A **class-conditional region-robust** variant repeats
that inside each crop, which restores per-crop coverage. The second fix is
honest but expensive: on the long tail it returns sets holding almost every
candidate crop, so it is reliable without being useful. That trade-off is
measured rather than hidden.

## Quick start

```bash
git clone https://github.com/rms-ali247/conformal-crop-sits.git
cd conformal-crop-sits
python -m venv env && env/Scripts/activate      # Linux/macOS: source env/bin/activate
pip install -r requirements.txt
```

Then fetch the dataset from Zenodo ([10.5281/zenodo.22806699](https://doi.org/10.5281/zenodo.22806699)) and unpack it so the arrays land
in `data/processed_monthly/`:

```
data/processed_monthly/
├── X_rabi.npy       X_kharif.npy        (N, 7, 28) float32
├── y_rabi.npy       y_kharif.npy        (N,) int64
├── meta_rabi.csv    meta_kharif.csv     ID, crop_type, Season, Province, District
└── label_map.csv
```

You do **not** need the dataset to reproduce the conformal results. Those run
off the out-of-fold artefacts already committed under `results/`, on CPU, in
about a minute. You need the dataset only to retrain.

## Reproducing the paper

The conformal and province-transfer commands were re-run against this folder
before release and reproduced the committed CSVs byte for byte, so the pipeline
is deterministic given the same out-of-fold artefacts.

| Paper item | Command | Writes |
|---|---|---|
| Table I, coverage over 25 geographic splits | `python -m evaluation.conformal --season both --nested --repeats 25` | `results/{season}/conformal_nested*.csv` |
| Table II upper, leave-one-province-out | `python -m evaluation.province_transfer --season both` | `results/province_transfer_{season}.csv`, `results/province_tau_{season}.csv` |
| Table II lower, phenology offset | `python -m evaluation.phenology_shift --season both` | `results/phenology_shift_{season}.csv` |
| Fig. 4, per-crop coverage and set size | `python -m evaluation.conformal --season both --perclass` then `python paper/make_rarecrop_figure.py` | `results/perclass_size_{season}.csv`, `paper/figs/rare_crop_size.png` |
| Figs. 2 and 3, the two schematics | `python paper/make_diagrams.py` | `paper/figs/method_flow.png`, `paper/figs/phenossm_arch.png` |
| Section VI-A accuracy numbers | `python paper/aggregate_tables.py` | prints the per-backbone table |
| Cross-backbone coverage (Section VI-C) | `python -m evaluation.conformal --efficiency` | `results/efficiency_{season}.csv` |

As a smoke test, the first command on Rabi alone should print `marginal` at
coverage 0.8929 with 24.0% of regions violating, and `class-region-robust` at
0.9500 with 0.0%. Those are the numbers in Table I.

The phenology test is the one exception to "no dataset needed": it pushes shifted
windows back through the five-fold ensemble, so it reads `results/{season}/X_test.npy`
(committed) and the checkpoints in `models/` (committed). It still needs no
download, but it does want a few minutes.

## Retraining

The shipped model is PhenoSSM with symmetric cross-entropy under the spatial
protocol:

```bash
python run.py train --model ms-s4 --season both --loss sce \
    --spatial-cv --spatial-test --dump-oof --folds 5 --epochs 150 --seed 42 \
    --data-dir data/processed_monthly
```

`--spatial-cv` groups folds by district so no district is split across folds,
`--spatial-test` holds out whole districts as the test set, and `--dump-oof`
writes the out-of-fold probabilities that the entire conformal analysis is built
on. Dropping either spatial flag reproduces the optimistic random-split numbers
that Section VI-A uses as a contrast.

The eight-backbone benchmark, holding the loss constant so the comparison is
fair:

```bash
python run_benchmark.py --season both --data-dir data/processed_monthly \
    --spatial-test --spatial-cv --loss ce --epochs 150 --folds 5 \
    --reference ms-s4 --out experiments
```

Training was done on a single RTX 5060. One season of PhenoSSM takes well under
an hour; the full benchmark is an overnight job.

## Applying this to your own region

The recipe does not depend on crops or on Sentinel-2. It needs one thing: a
group label on every calibration sample, marking which region, site, hospital,
or batch it came from.

1. Split your calibration data by group, and fit one conformal threshold per
   group rather than one pooled threshold.
2. Take the `tau` quantile of those per-group thresholds. Use 0.9 when the
   groups are as similar as neighbouring districts, and 0.95 when they are as
   different as provinces or agro-ecological zones. Our district-tuned 0.9 was
   measurably too small at province scale.
3. Report the worst group's coverage next to the mean. The mean is what hides
   the failure.
4. Treat low-confidence outputs as advice, not as grounds for decisions such as
   subsidy payments.

`evaluation/conformal.py` implements steps 1 and 2 as `fit_thresholds`, and
takes a `groups` argument rather than assuming districts.

## Layout

```
config.py, models.py, losses.py, utils.py   shared definitions and paths
run.py                                      stage dispatcher, see --help
run_benchmark.py, run_ablations.py          multi-model experiment drivers

pipeline/
  clean_shapefile.py, clean_labels.py       survey cleaning, confident learning
  gee_extract.py                            Sentinel-2 monthly composites via GEE
  preprocess.py                             builds the (N, 7, 28) tensors
  train.py                                  training, spatial CV, conformal training
evaluation/
  conformal.py                              all four calibration schemes, Table I
  province_transfer.py                      leave-one-province-out, Table II upper
  phenology_shift.py                        phenology offset, Table II lower
  evaluate_model.py, metrics.py             test metrics, McNemar with Holm
  calibrate.py, build_calibrator.py         temperature scaling, deployable calibrator
gee/extract_field.js                        the Earth Engine side of extraction
inference/                                  single-field prediction and a demo
models/{rabi,kharif}/                       five PhenoSSM folds, scaler, calibration
results/{rabi,kharif}/                      out-of-fold artefacts and per-season tables
results/*.csv                               province, phenology, per-class, efficiency
experiments*/                               per-run fold metrics for every ablation
paper/                                      LaTeX source, bibliography, figures
zenodo/                                     the data deposit, not tracked by git
```

Only the metrics files were kept under `experiments*/`. The checkpoints and
per-run arrays from those runs are not in the repository, because they are large
and reproducible from the commands above. What remains is enough for
`paper/aggregate_tables.py` to rebuild every accuracy number in the paper.

## Names in the paper and names in the code

| Paper | Code |
|---|---|
| PhenoSSM | `ms-s4` |
| MSTACNN | `mstacnn` |
| region-robust | `region-robust`, `tau` is `--robust-quantile` |
| class-conditional region-robust | `class-region-robust` |
| conformal training, Section VI-F | `--conftr`, `--conftr-weight` is lambda |
| component ablations | `ms-s4-noattn`, `ms-s4-noms`, `ms-s4-bb` |

## Scope

All results come from one country and one cropping year, Rabi 2022 to Kharif
2023. What is measured here is spatial transfer, not transfer across years or
sensors. The phenology test in Section VI-E shifts a copy of that same year and
is a simulation, not a second year of ground truth. Label noise was estimated by
the model with confident learning, not checked by hand, so the per-class noise
rates are diagnostics rather than truth. The rarest crops, under 60 test fields,
remain hard, and we say so rather than smoothing it over.

## Data

The processed tensors are deposited on Zenodo:

> DOI: [**10.5281/zenodo.22806699**](https://doi.org/10.5281/zenodo.22806699)

The metadata carries district and province names only. There are no field
geometries, no coordinates, and no personal identifiers. See `zenodo/README.md`
for the full description and provenance.

The underlying ground truth is the Asian Development Bank crop-type survey of
Pakistan. The field boundary shapefiles are not redistributed here.

## Citation

```bibtex
@inproceedings{ali2026conformal,
  title     = {Conformal Crop Classification Across Regions with Sentinel-2 Time Series},
  author    = {Ali, R.M.S. and Fraz, M.M. and
               Ullah, S. and Zafar, Z.},
  booktitle = {TBD},
  year      = {2026}
}
```

Update the `booktitle` and add pages and a DOI once the paper is accepted.

## License

Code is MIT, see `LICENSE`. The Zenodo dataset is CC-BY-4.0, see
`zenodo/LICENSE`. Both ask only for attribution.

## Acknowledgment

The authors acknowledge funding support from UK Research and Innovation (UKRI)
through Project APP47457, "Super-efficient Sustainable Cooling Solution for All
Applications (S2Cool)", under the Ayrton Challenge Program.

## Contact

R.M.S. Ali, rali.msai23seecs@seecs.edu.pk
