# RNA Splice-Junction-Based Classification of Drosophila Neuronal Populations

## Research Question

Can RNA splice-junction information from single-cell RNA sequencing distinguish neuronal populations associated with GABAergic and glutamatergic identities in the Drosophila brain?

This project evaluates binary classification of two published annotated neuronal populations: Pm1/Pm2 and TmY14. These labels are not direct measurements of neurotransmitter identity.

## Dataset

- Source: NCBI GEO GSE107451
- Organism: Drosophila melanogaster
- Selected cells: 582
- Pm1/Pm2 cells: 270
- TmY14 cells: 312
- Junction features: 3,399
- Ages represented: 3, 6, and 9 days

Dataset: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107451

## Machine Learning

The project uses RNA junction ratios and event-coverage masks.

- Development: 289 cells from r1 and r2 libraries
- Held-out test: 293 cells from r3 libraries
- Development evaluation: leave-one-library-out cross-validation
- Selected classifier: Logistic Regression
- Regularization parameter: C = 0.01

## Results

| Metric | Held-out test |
|---|---:|
| Accuracy | 74.74% |
| Balanced accuracy | 74.75% |
| F1 score | 74.13% |
| ROC-AUC | 0.814 |
| Correct predictions | 219 / 293 |

## Interpretation

RNA splice-junction measurements contain predictive information for distinguishing the selected neuronal populations.

Development analysis showed that junction coverage contributed more predictive information than junction ratios alone.

Therefore, the results do not establish that alternative splicing regulation independently explains the classification performance.

## Reproducibility

The `scripts/` directory contains the classification and analysis code.

The `results/splicing/` directory contains model-selection results, held-out predictions, performance metrics, and coverage analyses.

Large datasets and trained model binaries are excluded from Git tracking.

The original GEO dataset remains stored separately on Unity HPC.
