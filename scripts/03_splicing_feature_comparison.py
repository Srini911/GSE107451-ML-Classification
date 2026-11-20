from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score
)

ROOT = Path.cwd()
SOURCE = Path(
    "/project/pi_mukulika_ray_umb_edu/"
    "Aging_Dataset_GSE107451/Wild_Datasets_GSE107451/"
    "23_leafletfa/02_splicing_dataset/"
    "GSE107451_w1118_3d_6d_9d_LeafletFA_TRAINING.h5ad"
)

manifest = pd.read_csv(
    ROOT / "data/processed/verified_cell_manifest_with_splits.csv"
)
assert len(manifest) == 582

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
    ratio = read_sparse(f["layers"]["junc_ratio"])[rows]
    mask = read_sparse(f["layers"]["psi_mask"])[rows]

assert ratio.shape == mask.shape == (582, 3399)

dev = manifest["split"].eq("development").to_numpy()
assert dev.sum() == 289

y = manifest.loc[dev, "published_population"].eq(
    "Pm1/Pm2"
).astype(int).to_numpy()
groups = manifest.loc[dev, "library"].to_numpy()

features = {
    "Ratios_only": ratio[dev].astype(np.float32),
    "Coverage_only": mask[dev].astype(np.float32),
    "Ratios_plus_coverage": sparse.hstack(
        [ratio[dev], mask[dev]], format="csr"
    ).astype(np.float32)
}

results = []
logo = LeaveOneGroupOut()

for feature_name, X in features.items():
    preds = np.zeros(len(y), dtype=int)
    scores = np.zeros(len(y), dtype=float)
    fold_balanced = []

    for tr, va in logo.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=0.01,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000
            )
        )

        model.fit(X[tr], y[tr])
        preds[va] = model.predict(X[va])
        scores[va] = model.decision_function(X[va])

        fold_balanced.append(
            balanced_accuracy_score(y[va], preds[va])
        )

    results.append({
        "Features": feature_name,
        "CV_Accuracy": accuracy_score(y, preds),
        "CV_Balanced_Accuracy": balanced_accuracy_score(y, preds),
        "Mean_Fold_Balanced_Accuracy": np.mean(fold_balanced),
        "CV_ROC_AUC": roc_auc_score(y, scores)
    })

df = pd.DataFrame(results)
out = ROOT / "results/splicing"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "feature_comparison_development.csv", index=False)

print("\n=== SPLICING FEATURE COMPARISON ===")
print(df.round(4).to_string(index=False))
print("\nComparison completed using development cells only.")
