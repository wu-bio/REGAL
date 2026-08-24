#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ad.settings.allow_write_nullable_strings = True


DEFAULT_DATA_ROOT = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_SC_H5AD = DEFAULT_DATA_ROOT / "Wu" / "SC" / "wu_tnbc_reference.h5ad"
DEFAULT_ST_H5AD = DEFAULT_DATA_ROOT / "Wu" / "ST" / "Wu_Breast_CID44971.h5ad"
DEFAULT_OUTDIR = DEFAULT_DATA_ROOT / "Wu" / "ST" / "Wu_Breast_CID44971" / "stereoscope-out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stereoscope deconvolution for an h5ad SC reference and Visium sample."
    )
    parser.add_argument("--sc-h5ad", type=Path, default=DEFAULT_SC_H5AD)
    parser.add_argument("--st-h5ad", type=Path, default=DEFAULT_ST_H5AD)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--analysis-name", default="wu2021_author_parameters")
    parser.add_argument(
        "--label-col",
        default="celltype_minor",
        choices=(
            "celltype_major",
            "celltype_minor",
            "celltype_minor_author",
            "celltype_subset",
        ),
    )
    parser.add_argument("--min-cells-per-type", type=int, default=25)
    parser.add_argument("--max-cells-per-type", type=int, default=500)
    parser.add_argument("--topn-genes", type=int, default=5000)
    parser.add_argument("--sc-epochs", type=int, default=50000)
    parser.add_argument("--st-epochs", type=int, default=50000)
    parser.add_argument("--sc-batch-size", type=int, default=2046)
    parser.add_argument("--st-batch-size", type=int, default=2046)
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request CUDA execution through the Stereoscope CLI.",
    )
    parser.add_argument(
        "--sc-model",
        type=Path,
        default=None,
        help="Optional fitted SC model state used to continue training.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def as_csr(matrix: object) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def top_indices(matrix: sparse.csr_matrix, n: int) -> np.ndarray:
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    n = min(int(n), totals.size)
    if n == totals.size:
        return np.arange(totals.size, dtype=int)
    return np.argpartition(totals, -n)[-n:]


def deterministic_cell_subset(
    labels: pd.Series,
    *,
    min_cells: int,
    max_cells: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    values = labels.astype(str).to_numpy()
    for label in np.unique(values):
        idx = np.flatnonzero(values == label)
        if idx.size < min_cells:
            continue
        if idx.size > max_cells:
            idx = np.sort(rng.choice(idx, size=max_cells, replace=False))
        selected.append(idx)
    if not selected:
        raise ValueError("No reference cell type passed the requested size thresholds.")
    return np.sort(np.concatenate(selected))


def add_author_minor_labels(sc: ad.AnnData) -> None:
    """Reconstruct the minor-tier grouping used in the Wu et al. Stereoscope run."""
    required = {"celltype_major", "celltype_minor"}
    missing = sorted(required.difference(sc.obs.columns))
    if missing:
        raise KeyError(
            "Cannot derive 'celltype_minor_author'; missing SC obs columns: "
            + ", ".join(missing)
        )

    labels = sc.obs["celltype_minor"].astype(str).copy()
    major = sc.obs["celltype_major"].astype(str)
    labels.loc[major == "Cancer Epithelial"] = "Epithelial_cancer"
    labels.loc[major == "Normal Epithelial"] = "Epithelial_normal"
    sc.obs["celltype_minor_author"] = pd.Categorical(labels)


def prepare_inputs(args: argparse.Namespace, tier_dir: Path) -> tuple[Path, Path, dict]:
    log(f"[INFO] Reading SC reference: {args.sc_h5ad}")
    sc = ad.read_h5ad(args.sc_h5ad)
    log(f"[INFO] Reading ST sample: {args.st_h5ad}")
    st = ad.read_h5ad(args.st_h5ad)
    source_sc_shape = list(map(int, sc.shape))
    source_st_shape = list(map(int, st.shape))

    if args.label_col == "celltype_minor_author":
        add_author_minor_labels(sc)
    if args.label_col not in sc.obs:
        raise KeyError(f"Missing SC obs column {args.label_col!r}.")
    if "in_tissue" in st.obs:
        keep_spots = st.obs["in_tissue"].astype(int).to_numpy() == 1
        st = st[keep_spots].copy()

    sc.var_names_make_unique()
    st.var_names_make_unique()
    sc_counts = as_csr(sc.X)
    st_counts = as_csr(st.X)
    nonzero_spots = np.asarray(st_counts.sum(axis=1)).ravel() > 0
    if not nonzero_spots.all():
        st = st[nonzero_spots].copy()
        st_counts = st_counts[nonzero_spots]

    selected_cells = deterministic_cell_subset(
        sc.obs[args.label_col],
        min_cells=args.min_cells_per_type,
        max_cells=args.max_cells_per_type,
        seed=args.seed,
    )
    sc = sc[selected_cells].copy()
    sc_counts = sc_counts[selected_cells]

    common = sc.var_names.intersection(st.var_names, sort=False)
    if common.empty:
        raise ValueError("SC and ST inputs have no common genes.")
    sc_idx = sc.var_names.get_indexer(common)
    st_idx = st.var_names.get_indexer(common)
    sc_common = sc_counts[:, sc_idx].tocsr()
    st_common = st_counts[:, st_idx].tocsr()

    detected_spots = np.asarray((st_common > 0).sum(axis=0)).ravel()
    detected_gene_mask = detected_spots > 10
    common = common[detected_gene_mask]
    sc_common = sc_common[:, detected_gene_mask].tocsr()
    st_common = st_common[:, detected_gene_mask].tocsr()

    sc_top = top_indices(sc_common, args.topn_genes)
    selected_genes = common[sc_top]

    prepared_dir = tier_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    sc_prepared_path = prepared_dir / f"wu_tnbc_reference_{args.label_col}.h5ad"
    st_prepared_path = prepared_dir / f"Wu_Breast_CID44971_{args.label_col}.h5ad"

    sc_prepared = ad.AnnData(
        X=sc_common[:, sc_top].astype(np.int32),
        obs=sc.obs[[args.label_col]].copy(),
        var=pd.DataFrame(index=selected_genes.copy()),
    )
    st_prepared = ad.AnnData(
        X=st_common[:, sc_top].astype(np.int32),
        obs=st.obs.copy(),
        var=pd.DataFrame(index=selected_genes.copy()),
    )
    if "spatial" in st.obsm:
        st_prepared.obsm["spatial"] = np.asarray(st.obsm["spatial"]).copy()
    if "spatial" in st.uns:
        st_prepared.uns["spatial"] = st.uns["spatial"]

    sc_prepared.write_h5ad(sc_prepared_path, compression="gzip")
    st_prepared.write_h5ad(st_prepared_path, compression="gzip")

    cell_counts = sc_prepared.obs[args.label_col].astype(str).value_counts().sort_index()
    prep_summary = {
        "source_sc_h5ad": str(args.sc_h5ad),
        "source_st_h5ad": str(args.st_h5ad),
        "label_col": args.label_col,
        "label_derivation": (
            "collapse Cancer Epithelial and Normal Epithelial to author minor-tier labels"
            if args.label_col == "celltype_minor_author"
            else None
        ),
        "min_cells_per_type": args.min_cells_per_type,
        "max_cells_per_type": args.max_cells_per_type,
        "seed": args.seed,
        "source_sc_shape": source_sc_shape,
        "source_st_shape": source_st_shape,
        "prepared_sc_shape": list(map(int, sc_prepared.shape)),
        "prepared_st_shape": list(map(int, st_prepared.shape)),
        "common_gene_count_after_st_detection_filter": int(common.size),
        "st_min_spots_per_gene_exclusive": 10,
        "prepared_gene_count": int(selected_genes.size),
        "gene_selection": "top expressed genes in subsampled SC reference",
        "topn_genes": int(args.topn_genes),
        "cell_counts": {str(k): int(v) for k, v in cell_counts.items()},
    }
    (prepared_dir / "preparation_summary.json").write_text(
        json.dumps(prep_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(
        "[INFO] Prepared "
        f"{sc_prepared.n_obs} SC cells, {st_prepared.n_obs} spots, "
        f"{selected_genes.size} SC-selected genes."
    )
    return sc_prepared_path, st_prepared_path, prep_summary


def newest_result(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No Stereoscope result matched {pattern!r} under {run_dir}.")
    return matches[-1]


def merge_results(
    args: argparse.Namespace,
    tier_dir: Path,
    st_prepared_path: Path,
    run_dir: Path,
    elapsed_seconds: float,
) -> dict:
    w_path = newest_result(run_dir, "*/W.*.tsv")
    w = pd.read_csv(w_path, sep="\t", index_col=0)
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("Stereoscope proportions contain non-finite values.")
    row_sums = w.sum(axis=1)
    if not np.allclose(row_sums.to_numpy(), 1.0, atol=1e-5):
        raise ValueError("Stereoscope proportion rows do not sum to one.")

    st_prepared = ad.read_h5ad(st_prepared_path, backed="r")
    expected_spots = pd.Index(st_prepared.obs_names.astype(str))
    if not expected_spots.equals(pd.Index(w.index.astype(str))):
        raise ValueError("Stereoscope output spot order does not match the prepared ST input.")

    proportions_path = tier_dir / "stereoscope_proportions.parquet"
    w.to_parquet(proportions_path)

    source_st = ad.read_h5ad(args.st_h5ad)
    aligned = w.reindex(source_st.obs_names.astype(str))
    source_st.obsm[f"stereoscope_{args.label_col}"] = aligned
    source_st.uns[f"stereoscope_{args.label_col}_celltypes"] = list(map(str, w.columns))
    merged_path = tier_dir / "st_with_stereoscope.h5ad"
    source_st.write_h5ad(merged_path, compression="gzip")

    summary = {
        "status": "completed",
        "label_col": args.label_col,
        "n_spots": int(w.shape[0]),
        "n_cell_types": int(w.shape[1]),
        "cell_types": list(map(str, w.columns)),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "nonfinite_values": int((~np.isfinite(w.to_numpy())).sum()),
        "raw_w_path": str(w_path),
        "proportions_path": str(proportions_path),
        "merged_h5ad": str(merged_path),
        "elapsed_seconds": float(elapsed_seconds),
    }
    (tier_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    tier_dir = args.outdir / args.analysis_name / args.label_col
    tier_dir.mkdir(parents=True, exist_ok=True)
    sc_prepared_path, st_prepared_path, prep_summary = prepare_inputs(args, tier_dir)

    if args.prepare_only:
        return

    run_dir = tier_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)
    stereoscope_command = shutil.which("stereoscope") or shutil.which("stereoscope.exe")
    if stereoscope_command is None:
        adjacent = Path(sys.executable).with_name(
            "stereoscope.exe" if sys.platform == "win32" else "stereoscope"
        )
        stereoscope_command = str(adjacent) if adjacent.exists() else None
    if stereoscope_command is None:
        raise FileNotFoundError(
            "Cannot find the Stereoscope CLI in PATH or next to the active Python interpreter."
        )
    command = [
        stereoscope_command,
        "run",
        "--sc_cnt",
        str(sc_prepared_path),
        "--label_colname",
        args.label_col,
        "--st_cnt",
        str(st_prepared_path),
        "--topn_genes",
        str(args.topn_genes),
        "--sc_epochs",
        str(args.sc_epochs),
        "--st_epochs",
        str(args.st_epochs),
        "--keep_barcodes",
        "--silent_mode",
        "--out_dir",
        str(run_dir),
    ]
    if args.sc_batch_size is not None:
        command.extend(["--sc_batch_size", str(args.sc_batch_size)])
    if args.st_batch_size is not None:
        command.extend(["--st_batch_size", str(args.st_batch_size)])
    if args.sc_model is not None:
        command.extend(["--sc_model", str(args.sc_model)])
    if args.gpu:
        command.append("--gpu")

    manifest = {
        "status": "running",
        "command": command,
        "parameters": vars(args)
        | {
            "sc_h5ad": str(args.sc_h5ad),
            "st_h5ad": str(args.st_h5ad),
            "outdir": str(args.outdir),
            "sc_model": str(args.sc_model) if args.sc_model is not None else None,
        },
        "preparation": prep_summary,
    }
    manifest_path = tier_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log("[INFO] Running Stereoscope:")
    log(" ".join(command))
    start = time.time()
    subprocess.run(command, check=True)
    elapsed = time.time() - start
    summary = merge_results(args, tier_dir, st_prepared_path, run_dir, elapsed)
    manifest["status"] = "completed"
    manifest["result"] = summary
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"[DONE] {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
