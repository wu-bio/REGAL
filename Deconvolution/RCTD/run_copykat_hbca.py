#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_SAMPLE_ID = "Breast_BlockA_Section1"
DEFAULT_R_SCRIPT = SCRIPT_DIR / "copykat_hbca.R"
DEFAULT_HBCA_H5AD = Path("/Users/wuyang/Documents/SC-ST data/BREAST/SC/HBCA/hbca_gse113197_ind4_7.h5ad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CopyKAT on ST h5ad using HBCA/GSE113197 as diploid baseline.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cancer-type", default="BREAST")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--st-h5ad", type=Path, default=None)
    parser.add_argument("--hbca-h5ad", type=Path, default=DEFAULT_HBCA_H5AD)
    parser.add_argument("--hbca-ref-dir", type=Path, default=None, help="Optional prebuilt hbca_ref.mtx.gz bundle directory.")
    parser.add_argument("--max-ref-cells-per-individual", type=int, default=200)
    parser.add_argument("--min-ref-counts", type=float, default=500)
    parser.add_argument("--min-ref-genes", type=int, default=300)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--r-script", type=Path, default=DEFAULT_R_SCRIPT)
    parser.add_argument("--r-bin", default="/Users/wuyang/anaconda3/envs/decon_r/bin/Rscript")
    parser.add_argument("--counts-mode", choices=["X", "raw"], default="X")
    parser.add_argument("--max-spots", type=int, default=0, help="Optional random downsample for trial runs; 0 keeps all spots.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-cores", type=int, default=1)
    parser.add_argument("--reuse-selected-ref-cells", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun", action="store_true", help="Ignore stale CopyKAT cache checks and rerun copykat.")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def get_counts_matrix(adata: ad.AnnData, mode: str) -> sparse.csr_matrix:
    matrix = adata.raw.X if mode == "raw" else adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def strip_ensembl_version(names: pd.Index) -> pd.Index:
    return pd.Index(pd.Series(names.astype(str)).str.replace(r"\.\d+$", "", regex=True).values)


def collapse_duplicate_genes(matrix: sparse.csr_matrix, genes: pd.Index) -> tuple[sparse.csr_matrix, pd.Index]:
    genes = strip_ensembl_version(genes)
    if genes.is_unique:
        return matrix, genes
    codes, uniques = pd.factorize(genes.astype(str), sort=False)
    mapper = sparse.csr_matrix(
        (np.ones(len(genes), dtype=np.int8), (np.arange(len(genes), dtype=np.int32), codes.astype(np.int32))),
        shape=(len(genes), len(uniques)),
    )
    return matrix.dot(mapper).tocsr(), pd.Index(uniques.astype(str))


def write_gz_lines(path: Path, values: pd.Index) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for value in values.astype(str):
            handle.write(f"{value}\n")


def write_mtx_bundle(outdir: Path, prefix: str, matrix_genes_by_cells: sparse.csr_matrix, genes: pd.Index, cells: pd.Index) -> dict:
    mtx_path = outdir / f"{prefix}.mtx"
    mtx_gz_path = outdir / f"{prefix}.mtx.gz"
    genes_path = outdir / f"{prefix}.genes.tsv.gz"
    cells_path = outdir / f"{prefix}.barcodes.tsv.gz"

    mmwrite(mtx_path, matrix_genes_by_cells.tocoo())
    with open(mtx_path, "rb") as src, gzip.open(mtx_gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    mtx_path.unlink()

    write_gz_lines(genes_path, genes)
    write_gz_lines(cells_path, cells)
    return {"matrix": str(mtx_gz_path), "genes": str(genes_path), "barcodes": str(cells_path)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_hbca_ref_from_h5ad(args: argparse.Namespace, io_dir: Path) -> dict:
    log(f"[LOAD] HBCA: {args.hbca_h5ad}")
    hbca = ad.read_h5ad(args.hbca_h5ad)
    matrix = hbca.X.tocsr() if sparse.issparse(hbca.X) else sparse.csr_matrix(hbca.X)
    genes = pd.Index(hbca.var_names.astype(str))
    matrix, genes = collapse_duplicate_genes(matrix, genes)

    n_counts = (
        hbca.obs["n_counts"].astype(float).to_numpy()
        if "n_counts" in hbca.obs
        else np.asarray(matrix.sum(axis=1)).ravel()
    )
    n_genes = (
        hbca.obs["n_genes"].astype(int).to_numpy()
        if "n_genes" in hbca.obs
        else np.asarray((matrix > 0).sum(axis=1)).ravel()
    )
    keep = (n_counts >= args.min_ref_counts) & (n_genes >= args.min_ref_genes)
    if "pass_qc" in hbca.obs:
        keep &= hbca.obs["pass_qc"].astype(bool).to_numpy()

    bundle_dir = io_dir / "hbca_ref"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    selected_cells_path = bundle_dir / "selected_hbca_ref_cells.csv"
    selected_meta_path = bundle_dir / "selected_hbca_ref_cells.metadata.json"
    selection_meta = {
        "hbca_h5ad": str(args.hbca_h5ad),
        "hbca_h5ad_sha256": sha256_file(args.hbca_h5ad),
        "seed": int(args.seed),
        "max_ref_cells_per_individual": int(args.max_ref_cells_per_individual),
        "min_ref_counts": float(args.min_ref_counts),
        "min_ref_genes": int(args.min_ref_genes),
    }

    selected: list[int] = []
    if args.reuse_selected_ref_cells and selected_cells_path.exists():
        if not selected_meta_path.exists():
            raise FileNotFoundError(
                f"Missing HBCA ref selection metadata: {selected_meta_path}. "
                "Use --no-reuse-selected-ref-cells to resample."
            )
        old_meta = json.loads(selected_meta_path.read_text(encoding="utf-8"))
        mismatch = {
            key: {"old": old_meta.get(key), "new": value}
            for key, value in selection_meta.items()
            if old_meta.get(key) != value
        }
        if mismatch:
            raise ValueError(
                "Existing HBCA ref selection metadata does not match current parameters. "
                "Use --no-reuse-selected-ref-cells to resample, or use a new --outdir.\n"
                + json.dumps(mismatch, indent=2)
            )
        selected_cells = pd.read_csv(selected_cells_path)["cell"].astype(str)
        selected = hbca.obs_names.astype(str).get_indexer(selected_cells).tolist()
        if any(i < 0 for i in selected):
            missing = selected_cells[np.array(selected) < 0].head(5).tolist()
            raise ValueError(f"Selected HBCA ref cells are missing from h5ad, examples: {missing}")
        log(f"[HBCA] Reusing selected ref cells: {selected_cells_path}")
    else:
        rng = np.random.default_rng(args.seed)
        if "individual" in hbca.obs:
            individuals = hbca.obs["individual"].astype(str).to_numpy()
            for individual in sorted(pd.unique(individuals)):
                idx = np.flatnonzero(keep & (individuals == individual))
                if args.max_ref_cells_per_individual > 0 and len(idx) > args.max_ref_cells_per_individual:
                    idx = rng.choice(idx, size=args.max_ref_cells_per_individual, replace=False)
                selected.extend(idx.tolist())
                log(f"[HBCA] {individual}: selected {len(idx)} ref cells")
        else:
            idx = np.flatnonzero(keep)
            max_cells = args.max_ref_cells_per_individual
            if max_cells > 0 and len(idx) > max_cells:
                idx = rng.choice(idx, size=max_cells, replace=False)
            selected.extend(idx.tolist())
            log(f"[HBCA] selected {len(idx)} ref cells")

    if not selected:
        raise ValueError("No HBCA reference cells passed QC/sampling.")
    selected = sorted(selected)
    ref_matrix = matrix[selected, :].T.tocsr()
    ref_cells = pd.Index(hbca.obs_names[selected].astype(str))
    bundle = write_mtx_bundle(bundle_dir, "hbca_ref", ref_matrix, genes, ref_cells)
    selected_df = pd.DataFrame({"cell": ref_cells})
    if "individual" in hbca.obs:
        selected_df["individual"] = hbca.obs.iloc[selected]["individual"].astype(str).to_numpy()
    selected_df.to_csv(selected_cells_path, index=False)
    selection_meta["n_selected_ref_cells"] = int(len(ref_cells))
    selection_meta["selected_cells_csv"] = str(selected_cells_path)
    selected_meta_path.write_text(json.dumps(selection_meta, indent=2), encoding="utf-8")

    ref_qc = pd.DataFrame(
        {
            "cell": hbca.obs_names.astype(str),
            "selected": False,
            "n_counts": n_counts,
            "n_genes": n_genes,
        }
    )
    if "individual" in hbca.obs:
        ref_qc["individual"] = hbca.obs["individual"].astype(str).to_numpy()
    ref_qc.loc[selected, "selected"] = True
    ref_qc.to_csv(bundle_dir / "hbca_ref_qc.csv.gz", index=False, compression="gzip")
    return {k: Path(v) for k, v in bundle.items()}


def resolve_st_path(args: argparse.Namespace) -> Path:
    if args.st_h5ad is not None:
        return args.st_h5ad
    return args.data_root / args.cancer_type / "ST" / f"{args.sample_id}.h5ad"


def main() -> None:
    args = parse_args()
    st_h5ad = resolve_st_path(args)
    outdir = args.outdir
    if outdir is None:
        outdir = Path("outputs") / "copykat_runs" / args.sample_id / "hbca_gse113197"
    io_dir = outdir / "io"
    io_dir.mkdir(parents=True, exist_ok=True)
    copykat_dir = outdir / "copykat"
    copykat_dir.mkdir(parents=True, exist_ok=True)

    if args.hbca_ref_dir is not None:
        ref_bundle = {
            "matrix": args.hbca_ref_dir / "hbca_ref.mtx.gz",
            "genes": args.hbca_ref_dir / "hbca_ref.genes.tsv.gz",
            "barcodes": args.hbca_ref_dir / "hbca_ref.barcodes.tsv.gz",
        }
        missing = [str(path) for path in ref_bundle.values() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Missing HBCA reference bundle:\n" + "\n".join(missing))
    else:
        ref_bundle = prepare_hbca_ref_from_h5ad(args, io_dir)

    log(f"[LOAD] ST: {st_h5ad}")
    st = ad.read_h5ad(st_h5ad)
    matrix = get_counts_matrix(st, args.counts_mode)
    genes = pd.Index(st.var_names.astype(str))
    matrix, genes = collapse_duplicate_genes(matrix, genes)

    keep = np.asarray(matrix.sum(axis=1)).ravel() > 0
    if "in_tissue" in st.obs:
        keep &= st.obs["in_tissue"].astype(int).to_numpy() == 1
    if args.max_spots and int(keep.sum()) > args.max_spots:
        rng = np.random.default_rng(args.seed)
        idx = np.flatnonzero(keep)
        sampled = rng.choice(idx, size=args.max_spots, replace=False)
        keep = np.zeros(st.n_obs, dtype=bool)
        keep[sampled] = True
    log(f"[FILTER] observed spots: keep {int(keep.sum())}/{st.n_obs}")

    obs_matrix = matrix[keep, :].T.tocsr()
    obs_cells = pd.Index(st.obs_names[keep].astype(str))
    obs_bundle = write_mtx_bundle(io_dir, "obs", obs_matrix, genes, obs_cells)

    params = {
        "sample_name": args.sample_id,
        "id_type": "S",
        "n_cores": args.n_cores,
        "ngene_chr": 1,
        "win_size": 25,
        "ks_cut": 0.2,
        "force_rerun": bool(args.force_rerun),
    }
    params_path = io_dir / "params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    summary = {
        "st_h5ad": str(st_h5ad),
        "outdir": str(outdir),
        "hbca_h5ad": str(args.hbca_h5ad),
        "hbca_ref_dir": str(args.hbca_ref_dir) if args.hbca_ref_dir is not None else None,
        "max_ref_cells_per_individual": args.max_ref_cells_per_individual,
        "reuse_selected_ref_cells": args.reuse_selected_ref_cells,
        "n_observed_spots": int(len(obs_cells)),
        "n_observed_genes": int(len(genes)),
        "obs_bundle": obs_bundle,
        "ref_bundle": {k: str(v) for k, v in ref_bundle.items()},
        "params": params,
    }
    (io_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))

    if args.prepare_only:
        log("[DONE] Prepared CopyKAT IO only.")
        return

    cmd = [
        args.r_bin,
        "--vanilla",
        str(args.r_script),
        str(Path(obs_bundle["matrix"]).resolve()),
        str(Path(obs_bundle["genes"]).resolve()),
        str(Path(obs_bundle["barcodes"]).resolve()),
        str(Path(ref_bundle["matrix"]).resolve()),
        str(Path(ref_bundle["genes"]).resolve()),
        str(Path(ref_bundle["barcodes"]).resolve()),
        str(copykat_dir.resolve()),
        str(params_path.resolve()),
    ]
    log("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    log("[DONE]")


if __name__ == "__main__":
    main()
