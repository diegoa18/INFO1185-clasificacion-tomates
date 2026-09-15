# INFO1185 - Clasificación de tomates

## Purpose

This repository contains the university project for INFO-1185
Inteligencia Artificial.

The objective is to classify tomato images as ripe or unripe through a
reproducible image-processing and machine-learning pipeline.

The project includes:

- exploratory color analysis;
- K-Means fruit segmentation;
- extraction of color features from the predicted fruit region;
- Bayesian classification;
- analysis-based feature selection;
- Sequential Forward Selection (SFS);
- PCA;
- ROC-based decision threshold selection;
- final comparison of the required classifiers.

## Authoritative assignment

The official university assignment is:

`docs/IA_2026_P1.pdf`

This PDF is the primary and authoritative source of truth for what the
project is required to implement, evaluate, report, and present.

Before making a methodological decision or implementing a new project
stage, inspect the relevant requirements in `docs/IA_2026_P1.pdf`.

If the PDF cannot be read directly, use the read-only command:

    pdftotext docs/IA_2026_P1.pdf -

Do not modify, replace, regenerate, or overwrite the official PDF.

Other documents under `docs/` may organize, explain, or expand the
implementation plan, but they must not override the official assignment.

`docs/STATUS.md` records the current experimental state and frozen
implementation decisions. It is not an assignment specification.

If any project instruction, status note, implementation decision,
comment, agent assumption, or source-code behavior conflicts with
`docs/IA_2026_P1.pdf`, stop and surface the conflict before making
changes.

Do not silently reinterpret an academic requirement.

## Source precedence

When sources disagree, use this precedence:

1. `docs/IA_2026_P1.pdf`
   - official university assignment;
   - authoritative source for mandatory requirements.

2. Other project-definition documents under `docs/`
   - interpretation, planning, and implementation documentation;
   - may clarify but may not contradict the official assignment.

3. `docs/STATUS.md`
   - current experimental state;
   - completed milestones;
   - frozen implementation decisions;
   - next planned milestone.

4. `artifacts/split.csv`
   - frozen dataset partition.

5. Source code and generated experimental outputs
   - implementation and experimental evidence.

When there is ambiguity about a requirement, inspect the official PDF
instead of guessing.

## Repository structure

Important locations:

- `docs/`
  Official assignment and project documentation.

- `docs/STATUS.md`
  Current project state and experimental decisions.

- `dataset/`
  Tomato images and reference masks.

- `artifacts/split.csv`
  Frozen training/validation/test partition.

- `src/`
  Reusable implementation.

- `scripts/`
  Executable experiment scripts.

- `results/`
  Generated experiment outputs.

Do not manually edit generated results. Change the implementation and
rerun the corresponding experiment instead.

## Dataset

The dataset contains 62 classification samples:

- 31 ripe tomatoes;
- 31 unripe tomatoes.

One image corresponds to one classification sample.

Pixels are not independent classification samples.

The official assignment requires an approximately 60/20/20
image-level split into:

- training;
- validation;
- test.

The corrected frozen partition in `artifacts/split.csv` contains:

- training: 37 samples (18 ripe, 19 unripe);
- validation: 12 samples (6 ripe, 6 unripe);
- test: 13 samples (7 ripe, 6 unripe).

The original 13 test images were preserved. The former 49 non-test images
were sorted by label/image and subdivided with stratification and seed 42.
`src/protocol.py` verifies the original test-membership fingerprint and
the deterministic corrected partition. `scripts/create_split.py` is
idempotent: rerunning it checks the corrected split without overwriting it.

Do not change any partition membership without explicit human instruction.
Never move an image previously used as test back into development.

Here, development means training plus validation, not a value of `split`.
Fit classifier parameters on training only. Use validation for model
selection and Youden thresholds. Keep the training-fitted model for test;
do not refit on training plus validation after choosing the threshold.

## Data leakage policy

Avoiding data leakage is mandatory.

The test set must never be used to:

- select features;
- select the K-Means channel combination;
- tune segmentation rules;
- select an SFS subset;
- determine the number of PCA components;
- choose a classification threshold;
- choose model hyperparameters;
- choose between alternative methodologies;
- revise a pipeline after observing test performance.

All model-selection and methodological decisions must use development
data only.

The test partition is for final evaluation after decisions have been
frozen.

When cross-validation is used, every learned transformation,
preprocessing operation, feature-selection procedure, and model-selection
operation must be fitted exclusively using the corresponding training
partition.

Do not fit a preprocessing or selection operation once on all
development data and then report cross-validation performance from that
already-fitted transformation.

## Test-set discipline

Historical exposure exists: earlier experiments and the corrected manual
classifier were evaluated on test before SFS and PCA were implemented.
That exposure cannot be undone, but it must not influence any remaining
methodological decision.

Until the manual, SFS, and PCA pipelines are all frozen:

- do not inspect historical or current test predictions, metrics, ROC data,
  figures, or per-image test features;
- do not change feature-selection logic because of test performance;
- do not choose PCA dimensionality from test performance;
- do not choose thresholds from test performance;
- do not run `scripts/extract_segmented_features.py --include-test`;
- do not run `scripts/run_bayes_analysis.py --evaluate-test`.

The active development feature table
`results/features/segmented_features.csv` must contain only training and
validation rows. `scripts/verify_protocol.py` enforces this invariant.

Final test gate: first freeze all three pipelines using only training and
validation. Then generate test features and evaluate every frozen pipeline
without revising any implementation or decision afterward because of test
performance.

Historical outputs are archived under `results/legacy_development_test/`
and are not current results. Do not inspect them to design the remaining
classifiers.

Validation is a selection set: its metrics are not unbiased final estimates.
The current validation images were previously part of exploratory
development; repartitioning does not erase that historical exposure.

## Ground-truth masks

Ground-truth masks may be used for:

- exploratory color analysis;
- evaluation of segmentation using Jaccard;
- development-only diagnostics of segmentation.

Ground-truth masks must not be used to construct classifier input
features.

Classifier features must be extracted exclusively from the region
predicted by the segmentation pipeline.

## Frozen segmentation experiment

K-Means configuration:

- number of clusters: 2;
- initialization: k-means++;
- n_init: 10;
- max_iter: 300;
- tolerance: 1e-4;
- random seed: 42;
- border fraction: 0.05.
- numerical thread count: 1 (via threadpoolctl).

The following RGB channel combinations were recalculated separately on
training and validation:

- R
- G
- B
- RG
- RB
- GB
- RGB

The selected segmentation configuration is:

`RG`

Selection maximizes mean validation Jaccard; ties follow the channel order
above. The corrected means are 0.5476 on training and 0.3808 on validation.
`results/segmentation/selected_configuration.json` records the choice,
parameters and split fingerprint. Downstream scripts load this artifact.

Cluster identification uses minimum border occupancy, with the center
pixel's cluster breaking ties. Diagnostic oracle masks are training-only.

The ground-truth reference mask is not used to select the predicted
tomato cluster at inference time.

The RG configuration is frozen for the remaining classification
experiments.

Do not change it based on test performance.

## Classification feature dataset

The candidate classifier features are mean color measurements calculated
over the foreground predicted by K-Means RG:

- R
- G
- B
- H
- S
- V

The generated feature table is:

`results/features/segmented_features.csv`

Only these six color columns are candidate classifier features.

Never use any of the following as classifier inputs:

- `label`;
- `split`;
- image or mask paths;
- `segmentation_jaccard`;
- foreground-size metadata;
- any value derived from the ground-truth mask.

`segmentation_jaccard` exists only for later analysis of segmentation
quality.

## Analysis-based feature selection

The manually selected feature subset is frozen as:

`R, G, S`

This selection was reconfirmed from training-only exploratory analysis:
absolute class-mean differences G=0.3710, R=0.2156, S=0.1854.

Do not modify this subset based on test performance.

## Bayesian classification

Use Gaussian Naive Bayes for continuous color features.

Positive class:

`ripe`

Negative class:

`unripe`

The Bayesian decision score is:

`log p(x | ripe) - log p(x | unripe)`

Compute the class-conditional Gaussian log densities explicitly using
GaussianNB's training-fitted means and smoothed variances. Do not include
class priors in this score. Use `var_smoothing=1e-9`.

A larger score therefore indicates stronger evidence for the ripe class.

ROC and AUC calculations must explicitly preserve `ripe` as the positive
class.

Do not rely on lexical ordering of string labels for binary ROC/AUC
calculation.

Decision thresholds must be selected using validation only.

The analysis-based classifier uses:

- features R, G, S;
- Gaussian Naive Bayes;
- parameter fitting using the 37 training images only;
- Youden index on the 12 validation scores for threshold selection;
- largest finite threshold on ties, with ripe predicted for score >= threshold;
- no training/validation refit after threshold selection;
- final evaluation using the frozen test partition.

`run_bayes_analysis.py` saves validation outputs and `decision.json` by
default. `--evaluate-test` verifies the saved model/decision before test.
Feature extraction excludes test by default; `--include-test` performs
the already-selected segmentation on test for final evaluation.

## SFS methodology

The SFS experiment is:

`SFS + Gaussian Naive Bayes`

Candidate features:

`R, G, B, H, S, V`

Use the corrected holdout protocol, without mandatory nested CV:

1. define and document the selection metric, stopping and tie rules before
   evaluating candidates;
2. fit each candidate GaussianNB using training only;
3. use validation scores to select additions and the final subset;
4. choose the Youden threshold on validation for the selected model;
5. freeze subset, training-fitted model and threshold;
6. evaluate test without changing the pipeline afterward.

Validation may serve both feature selection and threshold selection under
the assignment, but its performance is selection performance, not an
independent estimate. Test supplies the final comparison.

`scripts/run_sfs.py` implements all six forward stages with candidates in
R,G,B,H,S,V order. Each addition maximizes validation AUC; exact AUC ties
follow that order. The final subset maximizes AUC over the full trajectory;
ties select the smallest subset. There is no early stopping or refit on
training plus validation. Use the existing prior-free score and Youden rule.
Development artifacts are saved in `results/classification/bayes_sfs/`.
Run `python3 scripts/verify_sfs.py` after SFS changes. Keep test hidden until
PCA is implemented and all three pipelines are frozen.

Never use test performance to alter SFS.

The SFS experiment must report the feature-selection trajectory and the
final selected subset so it can be compared with the manual subset
`R, G, S`.

## PCA methodology

When implementing:

`PCA + Gaussian Naive Bayes`

use the candidate color features only.

StandardScaler must be fitted using training data only.

PCA must be fitted using training data only.

During cross-validation, scaling and PCA must be fitted independently
inside each corresponding training fold.

Never fit StandardScaler or PCA on all development data before computing
cross-validation scores.

Choose the number of principal components from training explained variance
and justify it as required by PDF section 3.6. Fit GaussianNB on training
PC scores; use validation for Youden and the frozen pipeline for test.

Never use test to choose PCA dimensionality.

Report at least:

- eigenvalues;
- explained variance ratio;
- cumulative explained variance;
- justification for the selected number of components.

## Fair model comparison

The required classifiers must be compared under equivalent evaluation
conditions.

Do not give one classifier privileged access to validation or test data
that another classifier does not receive.

Use the same split, training-only model fitting, validation-only threshold
selection and final test metrics for all three classifiers. If optional
cross-validation is introduced later, all learned steps must be fitted
inside its training folds; it does not replace the official holdout split.

## Reproducibility

Default random seed:

`42`

Keep random seeds explicit in experiment code.

Prefer deterministic experiments whenever possible.

Keep important experiment parameters visible in source code.

Do not silently introduce randomness.

## Implementation conventions

Prefer:

- Python type hints;
- small reusable functions;
- `pathlib.Path`;
- NumPy for numerical operations;
- pandas for experimental tables;
- scikit-learn for ML algorithms;
- matplotlib for figures;
- explicit input validation;
- explicit experiment constants.

Avoid unnecessary dependencies.

Do not introduce notebooks unless explicitly requested.

Do not overengineer this project.

The implementation must remain sufficiently simple to understand and
explain during the oral evaluation.

## Verification

After modifying Python source code, run at minimum:

    python3 -m compileall src scripts
    git diff --check

For protocol or Bayesian changes, also run `python3 scripts/verify_protocol.py`
after generating the corrected validation artifacts.

Also execute the experiment or script directly affected by the change.

Before declaring a milestone complete, inspect:

    git status --short
    git diff

Do not claim that code or an experiment works without executing the
relevant verification when execution is available.

## Git policy

Do not commit automatically.

Do not push automatically.

Do not rewrite Git history.

Do not use destructive Git commands.

Do not modify unrelated files.

Do not stage changes unless the human explicitly requests it.

The human decides when a milestone is ready to commit.

Commits should represent one coherent project milestone.

## Working procedure

For every substantial implementation task:

1. read the relevant part of `docs/IA_2026_P1.pdf`;
2. read `docs/STATUS.md`;
3. inspect the relevant existing source code;
4. inspect applicable supporting documentation under `docs/`;
5. explain the proposed methodological design before editing;
6. identify data-leakage and reproducibility risks;
7. make the smallest coherent implementation;
8. run the directly relevant experiment;
9. run the required verification;
10. inspect the resulting Git diff;
11. use an appropriate reviewer when useful;
12. report results, assumptions, and limitations.

Prefer methodological correctness, reproducibility, and fidelity to the
official assignment over adding features.
