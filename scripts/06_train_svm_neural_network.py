from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

ROOT = Path.cwd()
SOURCE = Path(
    "/project/pi_mukulika_ray_umb_edu/"
    "Aging_Dataset_GSE107451/Wild_Datasets_GSE107451/"
    "23_leafletfa/02_splicing_dataset/"
    "GSE107451_w1118_3d_6d_9d_LeafletFA_TRAINING.h5ad"
)

OUT = ROOT / "results/splicing"
OUT.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(
    ROOT / "data/processed/verified_cell_manifest_with_splits.csv"
)

def read_ids(obj):
    if isinstance(obj, h5py.Group):
        categories = obj["categories"].asstr()[()]
        return np.array([
            categories[i] if i >= 0 else ""
            for i in obj["codes"][()]
        ])
    return np.asarray(obj.asstr()[()])

def read_sparse(obj):
    return sparse.csr_matrix(
        (obj["data"][()], obj["indices"][()], obj["indptr"][()]),
        shape=tuple(obj.attrs["shape"])
    )

with h5py.File(SOURCE, "r") as f:
    ids = read_ids(f["obs"]["published_cell_id"])
    assert len(ids) == len(set(ids))
    lookup = {cell: i for i, cell in enumerate(ids)}

    assert manifest["geo_sample_id"].isin(lookup).all()
    rows = [
        lookup[cell] for cell in manifest["geo_sample_id"]
    ]

    ratios = read_sparse(f["layers"]["junc_ratio"])[rows]
    coverage = read_sparse(f["layers"]["psi_mask"])[rows]

assert ratios.shape == coverage.shape == (582, 3399)

X = sparse.hstack(
    [ratios, coverage], format="csr"
).astype(np.float32)

y = manifest["published_population"].eq(
    "Pm1/Pm2"
).astype(int).to_numpy()

dev = manifest["split"].eq("development").to_numpy()
test = manifest["split"].eq("final_test").to_numpy()

assert dev.sum() == 289
assert test.sum() == 293

X_dev = X[dev]
y_dev = y[dev]
groups = manifest.loc[dev, "library"].to_numpy()

assert len(np.unique(groups)) == 6
assert set(groups).isdisjoint(
    set(manifest.loc[test, "library"])
)

def build_model(name):
    if name == "Linear_SVM":
        return make_pipeline(
            StandardScaler(with_mean=False),
            LinearSVC(
                C=0.01,
                class_weight="balanced",
                dual=True,
                max_iter=10000,
                random_state=42
            )
        )

    if name == "Neural_Network":
        return make_pipeline(
            MaxAbsScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=1.0,
                learning_rate_init=0.001,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=42
            )
        )

    raise ValueError(name)

logo = LeaveOneGroupOut()
summary = []
all_folds = []

for name in ["Linear_SVM", "Neural_Network"]:
    predictions = np.zeros(len(y_dev), dtype=int)
    scores = np.zeros(len(y_dev), dtype=float)

    print(f"\n=== {name} ===", flush=True)

    for tr, va in logo.split(X_dev, y_dev, groups):
        model = build_model(name)
        model.fit(X_dev[tr], y_dev[tr])

        predictions[va] = model.predict(X_dev[va])

        if name == "Neural_Network":
            scores[va] = model.predict_proba(X_dev[va])[:, 1]
        else:
            scores[va] = model.decision_function(X_dev[va])

        fold = {
            "model": name,
            "library": groups[va][0],
            "cells": int(len(va)),
            "accuracy": float(
                accuracy_score(y_dev[va], predictions[va])
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_dev[va], predictions[va])
            )
        }

        all_folds.append(fold)
        print(fold, flush=True)

    model_folds = [
        x["balanced_accuracy"]
        for x in all_folds if x["model"] == name
    ]

    result = {
        "model": name,
        "cv_accuracy": float(
            accuracy_score(y_dev, predictions)
        ),
        "cv_balanced_accuracy": float(
            balanced_accuracy_score(y_dev, predictions)
        ),
        "mean_fold_balanced_accuracy": float(
            np.mean(model_folds)
        ),
        "cv_f1": float(f1_score(y_dev, predictions)),
        "cv_roc_auc": float(
            roc_auc_score(y_dev, scores)
        )
    }

    summary.append(result)
    print(json.dumps(result, indent=2), flush=True)

pd.DataFrame(all_folds).to_csv(
    OUT / "svm_neural_network_cv_by_library.csv",
    index=False
)

pd.DataFrame(summary).to_csv(
    OUT / "svm_neural_network_cv_summary.csv",
    index=False
)

print("\n=== FINAL DEVELOPMENT COMPARISON ===", flush=True)

original = pd.read_csv(
    OUT / "development_model_selection.csv"
)

logistic = original[
    (original["model"] == "Logistic_Regression")
    & (np.isclose(original["C"], 0.01))
].iloc[0]

comparison = pd.concat(
    [
        pd.DataFrame([{
            "model": "Logistic_Regression",
            "cv_accuracy": logistic["cv_accuracy"],
            "cv_balanced_accuracy": logistic["cv_balanced_accuracy"],
            "mean_fold_balanced_accuracy":
                logistic["mean_fold_balanced_accuracy"],
            "cv_f1": logistic["cv_f1"],
            "cv_roc_auc": logistic["cv_roc_auc"]
        }]),
        pd.DataFrame(summary)
    ],
    ignore_index=True
)

print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    OUT / "three_model_development_comparison.csv",
    index=False
)

print("\nSaved results to:", OUT, flush=True)
print("Final-test data were not used.", flush=True)
