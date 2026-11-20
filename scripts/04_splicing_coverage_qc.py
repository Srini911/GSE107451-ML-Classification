from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

ROOT = Path.cwd()
SOURCE = Path(
    "/project/pi_mukulika_ray_umb_edu/"
    "Aging_Dataset_GSE107451/Wild_Datasets_GSE107451/"
    "23_leafletfa/02_splicing_dataset/"
    "GSE107451_w1118_3d_6d_9d_LeafletFA_TRAINING.h5ad"
)

df = pd.read_csv(
    ROOT / "data/processed/verified_cell_manifest_with_splits.csv"
)
assert len(df) == 582
assert df["geo_sample_id"].is_unique

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
    lookup = {value: i for i, value in enumerate(ids)}
    assert df["geo_sample_id"].isin(lookup).all()

    rows = np.array([lookup[x] for x in df["geo_sample_id"]])

    counts = read_sparse(
        f["layers"]["cell_by_junction_matrix"]
    )[rows].tocsr()

    mask = read_sparse(
        f["layers"]["psi_mask"]
    )[rows].tocsr()

assert counts.shape == mask.shape == (582, 3399)

# Keep only development cells.
keep = df["split"].eq("development").to_numpy()
df = df.loc[keep].copy().reset_index(drop=True)
counts = counts[keep].tocsr()
mask = mask[keep].tocsr()

assert len(df) == 289
assert df["library"].nunique() == 6

df["junction_reads"] = np.asarray(
    counts.sum(axis=1)
).ravel()

df["detected_junctions"] = np.diff(counts.indptr)
df["measurable_junctions"] = np.diff(mask.indptr)

df["measurable_fraction"] = (
    df["measurable_junctions"] / counts.shape[1]
)

df["detected_fraction"] = (
    df["detected_junctions"] / counts.shape[1]
)

metrics = [
    "junction_reads",
    "detected_junctions",
    "measurable_junctions",
    "measurable_fraction",
]

print("\n=== OVERALL MEDIANS BY POPULATION ===")
print(
    df.groupby("published_population")[metrics]
      .median()
      .round(4)
      .to_string()
)

print("\n=== MEDIANS WITHIN EACH LIBRARY ===")
print(
    df.groupby(["library", "published_population"])[metrics]
      .median()
      .round(4)
      .to_string()
)

print("\n=== ASSOCIATION WITH JUNCTION READ DEPTH ===")

for name in ["measurable_junctions", "detected_junctions"]:
    rho, pvalue = spearmanr(df["junction_reads"], df[name])
    print(
        f"Overall: reads vs {name}: "
        f"Spearman rho={rho:.4f}, p={pvalue:.4g}"
    )

for library, sub in df.groupby("library"):
    rho, pvalue = spearmanr(
        sub["junction_reads"],
        sub["measurable_junctions"]
    )
    print(
        f"{library}: reads vs measurable junctions: "
        f"rho={rho:.4f}, p={pvalue:.4g}"
    )

print("\n=== CELLS BY POPULATION AND LIBRARY ===")
print(
    pd.crosstab(df["library"], df["published_population"])
    .to_string()
)

out = ROOT / "results/splicing"
out.mkdir(parents=True, exist_ok=True)

df.to_csv(out / "development_coverage_qc.csv", index=False)

print("\nSaved:", out / "development_coverage_qc.csv")
print("Final-test cells were not analyzed.")
