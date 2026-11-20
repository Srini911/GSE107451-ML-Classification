from pathlib import Path
import json
import h5py
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, confusion_matrix
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
(ROOT / "models").mkdir(exist_ok=True)

manifest = pd.read_csv(
    ROOT / "data/processed/verified_cell_manifest_with_splits.csv"
)
assert len(manifest) == 582
assert manifest["geo_sample_id"].is_unique

def read_ids(obj):
    if isinstance(obj, h5py.Group):
        categories = obj["categories"].asstr()[()]
        return np.array([
            categories[i] if i >= 0 else ""
            for i in obj["codes"][()]
        ])
    return np.asarray(obj.asstr()[()])

def read_sparse(obj):
    shape = tuple(obj.attrs["shape"])
    return sparse.csr_matrix(
        (obj["data"][()], obj["indices"][()], obj["indptr"][()]),
        shape=shape
    )

with h5py.File(SOURCE, "r") as f:
    ids = read_ids(f["obs"]["published_cell_id"])
    assert len(ids) == len(set(ids))

    lookup = {cell: i for i, cell in enumerate(ids)}
    assert manifest["geo_sample_id"].isin(lookup).all()

    rows = np.array([
        lookup[cell] for cell in manifest["geo_sample_id"]
    ])

    ratio = read_sparse(f["layers"]["junc_ratio"])[rows]
    mask = read_sparse(f["layers"]["psi_mask"])[rows]

assert ratio.shape == mask.shape == (582, 3399)
assert np.all(np.isfinite(ratio.data))
assert np.all((ratio.data >= 0) & (ratio.data <= 1))
assert np.all(mask.data == 1)
assert (ratio.multiply(mask) != ratio).nnz == 0

features = sparse.hstack(
    [ratio, mask], format="csr"
).astype(np.float32)

y = manifest["published_population"].eq("Pm1/Pm2").astype(int).to_numpy()
dev = manifest["split"].eq("development").to_numpy()
test = manifest["split"].eq("final_test").to_numpy()

assert dev.sum() == 289
assert test.sum() == 293
assert not np.any(dev & test)

X_dev, X_test = features[dev], features[test]
y_dev, y_test = y[dev], y[test]
groups = manifest.loc[dev, "library"].to_numpy()

assert len(np.unique(groups)) == 6
assert set(groups).isdisjoint(set(manifest.loc[test, "library"]))

logo = LeaveOneGroupOut()
records = []
candidates = []

for name in ["Logistic_Regression", "Linear_SVM"]:
    for C in [0.01, 0.1, 1.0]:
        predictions = np.zeros(len(y_dev), dtype=int)
        scores = np.zeros(len(y_dev), dtype=float)
        fold_balanced = []

        for train_idx, valid_idx in logo.split(X_dev, y_dev, groups):
            if name == "Logistic_Regression":
                estimator = LogisticRegression(
                    C=C, class_weight="balanced",
                    solver="liblinear", max_iter=3000
                )
            else:
                estimator = LinearSVC(
                    C=C, class_weight="balanced",
                    dual=True, max_iter=10000
                )

            pipeline = make_pipeline(
                StandardScaler(with_mean=False),
                estimator
            )

            pipeline.fit(X_dev[train_idx], y_dev[train_idx])

            pred = pipeline.predict(X_dev[valid_idx])
            score = pipeline.decision_function(X_dev[valid_idx])

            predictions[valid_idx] = pred
            scores[valid_idx] = score

            fold_balanced.append(
                balanced_accuracy_score(y_dev[valid_idx], pred)
            )

        result = {
            "model": name,
            "C": C,
            "mean_fold_balanced_accuracy": float(np.mean(fold_balanced)),
            "cv_accuracy": float(accuracy_score(y_dev, predictions)),
            "cv_balanced_accuracy": float(
                balanced_accuracy_score(y_dev, predictions)
            ),
            "cv_f1": float(f1_score(y_dev, predictions)),
            "cv_roc_auc": float(roc_auc_score(y_dev, scores))
        }
        records.append(result)
        candidates.append((result, name, C))
        print(result, flush=True)

cv = pd.DataFrame(records)
cv.to_csv(OUT / "development_model_selection.csv", index=False)

best_result, name, C = max(
    candidates,
    key=lambda x: (
        x[0]["mean_fold_balanced_accuracy"],
        x[0]["cv_accuracy"]
    )
)

if name == "Logistic_Regression":
    estimator = LogisticRegression(
        C=C, class_weight="balanced",
        solver="liblinear", max_iter=3000
    )
else:
    estimator = LinearSVC(
        C=C, class_weight="balanced",
        dual=True, max_iter=10000
    )

model = make_pipeline(StandardScaler(with_mean=False), estimator)
model.fit(X_dev, y_dev)

pred = model.predict(X_test)
score = model.decision_function(X_test)

metrics = {
    "selected_model": name,
    "C": C,
    "test_cells": int(len(y_test)),
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, pred)
    ),
    "test_f1": float(f1_score(y_test, pred)),
    "test_roc_auc": float(roc_auc_score(y_test, score)),
    "confusion_matrix": confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).tolist(),
    "labels": {"0": "TmY14", "1": "Pm1/Pm2"}
}

test_results = manifest.loc[
    test, ["geo_sample_id", "age_day", "sex", "library",
           "published_population"]
].copy()
test_results["predicted_population"] = np.where(
    pred == 1, "Pm1/Pm2", "TmY14"
)
test_results["decision_score"] = score

test_results.to_csv(OUT / "final_test_predictions.csv", index=False)

library_results = []
for library, sub in test_results.groupby("library"):
    true = sub["published_population"].eq("Pm1/Pm2").astype(int)
    predicted = sub["predicted_population"].eq("Pm1/Pm2").astype(int)

    library_results.append({
        "library": library,
        "cells": len(sub),
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy_score(true, predicted)
    })

pd.DataFrame(library_results).to_csv(
    OUT / "final_test_by_library.csv", index=False
)

with open(OUT / "final_test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(model, ROOT / "models/splicing_classifier.joblib")

print("\n=== FINAL TEST RESULTS ===", flush=True)
print(json.dumps(metrics, indent=2), flush=True)
print("\n=== BY LIBRARY ===", flush=True)
print(pd.DataFrame(library_results).to_string(index=False), flush=True)
print("\nResults saved to:", OUT, flush=True)
