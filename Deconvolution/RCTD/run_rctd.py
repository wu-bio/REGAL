#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/regal_mplconfig")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

ad.settings.allow_write_nullable_strings = True


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CANCER_TYPE = "BREAST"
DEFAULT_SAMPLE_ID = "Breast_BlockA_Section1"
DEFAULT_DATA_ROOT = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_R_SCRIPT = SCRIPT_DIR / "rctd.R"
DEFAULT_BREAST_SC_REFERENCE = DEFAULT_DATA_ROOT / "BREAST" / "SC" / "TNBC" / "BREAST_reference.h5ad"
DEFAULT_WU_SC_REFERENCE = DEFAULT_DATA_ROOT / "Wu" / "SC" / "wu_tnbc_reference.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RCTD deconvolution from SC/ST h5ad inputs.")
    parser.add_argument("--cancer-type", default=DEFAULT_CANCER_TYPE)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sc-h5ad", type=Path, default=None)
    parser.add_argument("--st-h5ad", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--r-script", type=Path, default=DEFAULT_R_SCRIPT)
    parser.add_argument("--r-bin", default="Rscript")
    parser.add_argument("--ref-counts-mode", default="X", choices=["X", "raw"])
    parser.add_argument("--st-counts-mode", default="X", choices=["X", "raw"])
    parser.add_argument("--ref-celltype-col", default=None)
    parser.add_argument("--st-coords-source", default="obsm:spatial")
    parser.add_argument("--max-cells-per-type", type=int, default=5000)
    parser.add_argument(
        "--min-cells-per-type",
        type=int,
        default=25,
        help="Drop reference cell types below this size before spacexr (which requires at least 25 cells per type).",
    )
    parser.add_argument("--max-cores", type=int, default=1)
    parser.add_argument("--gene-cutoff", type=float, default=0.000125)
    parser.add_argument("--fc-cutoff", type=float, default=0.5)
    parser.add_argument("--doublets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-st-h5ad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rctd-obsm-key", default="rctd_props")
    parser.add_argument("--strip-ensembl-version", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collapse-duplicate-genes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-st-in-tissue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-st-nonzero-lib", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exclude-st-annotation",
        action="append",
        default=[],
        help="Exact ST annotation value to exclude; may be supplied more than once.",
    )
    parser.add_argument(
        "--st-annotation-col",
        default=None,
        help="ST obs column used by --exclude-st-annotation. If omitted, a standard annotation column is selected.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only export RCTD IO bundles and params; do not call RCTD.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    sc_h5ad = args.sc_h5ad
    if sc_h5ad is None:
        cancer_type = str(args.cancer_type).strip().upper()
        if cancer_type == "BREAST":
            sc_h5ad = DEFAULT_BREAST_SC_REFERENCE
        elif cancer_type == "WU":
            sc_h5ad = DEFAULT_WU_SC_REFERENCE
        else:
            sc_h5ad = args.data_root / args.cancer_type / "SC" / f"{args.cancer_type}_reference.h5ad"

    st_h5ad = args.st_h5ad
    if st_h5ad is None:
        st_h5ad = args.data_root / args.cancer_type / "ST" / f"{args.sample_id}.h5ad"

    outdir = args.outdir
    if outdir is None:
        outdir = args.data_root / args.cancer_type / "ST" / args.sample_id / "rctd-out"

    return Path(sc_h5ad), Path(st_h5ad), Path(outdir)


def get_counts_matrix(adata: ad.AnnData, mode: str, tag: str) -> sparse.csr_matrix:
    if mode == "raw":
        if adata.raw is None:
            raise ValueError(f"[{tag}] counts mode is raw, but adata.raw is None.")
        matrix = adata.raw.X
    else:
        matrix = adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def integer_like_frac(matrix: sparse.csr_matrix, n_sample: int = 200_000, seed: int = 1) -> float:
    data = matrix.data
    if data.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    sample = data[rng.choice(data.size, size=n_sample, replace=False)] if data.size > n_sample else data
    return float(np.mean(np.isclose(sample, np.round(sample))))


def cast_to_int_if_integer_like(matrix: sparse.csr_matrix, tag: str, threshold: float = 0.999) -> sparse.csr_matrix:
    frac = integer_like_frac(matrix)
    if frac < threshold:
        log(f"[WARN] {tag} counts are not integer-like; frac={frac:.4f}, dtype={matrix.dtype}")
        return matrix
    if not np.issubdtype(matrix.dtype, np.integer):
        log(f"[INFO] {tag} counts are integer-like; casting {matrix.dtype} -> int32")
        return matrix.astype(np.int32)
    log(f"[INFO] {tag} counts are integer-like; dtype={matrix.dtype}")
    return matrix.astype(np.int32, copy=False)


def strip_ensembl_version(names: pd.Index) -> pd.Index:
    return pd.Index(pd.Series(names.astype(str)).str.replace(r"\.\d+$", "", regex=True).values)


def collapse_duplicate_genes(matrix: sparse.csr_matrix, genes: pd.Index, tag: str) -> tuple[sparse.csr_matrix, pd.Index]:
    if genes.is_unique:
        return matrix, genes
    codes, uniques = pd.factorize(genes.astype(str), sort=False)
    log(f"[INFO] {tag} collapsing duplicate genes: {len(genes)} -> {len(uniques)}")
    mapper = sparse.csr_matrix(
        (np.ones(len(genes), dtype=np.int8), (np.arange(len(genes), dtype=np.int32), codes.astype(np.int32))),
        shape=(len(genes), len(uniques)),
    )
    return matrix.dot(mapper).tocsr(), pd.Index(uniques.astype(str))


def harmonize_genes(
    ref_matrix: sparse.csr_matrix,
    ref_genes: pd.Index,
    st_matrix: sparse.csr_matrix,
    st_genes: pd.Index,
    *,
    strip_versions: bool,
    collapse_duplicates: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.Index]:
    if strip_versions:
        ref_genes = strip_ensembl_version(ref_genes)
        st_genes = strip_ensembl_version(st_genes)
    if collapse_duplicates:
        ref_matrix, ref_genes = collapse_duplicate_genes(ref_matrix, ref_genes, "SC")
        st_matrix, st_genes = collapse_duplicate_genes(st_matrix, st_genes, "ST")

    genes = ref_genes.intersection(st_genes)
    if genes.empty:
        raise ValueError("No intersecting genes between SC reference and ST sample.")
    log(f"[INFO] Intersect genes: {len(genes)}")

    ref_idx = ref_genes.get_indexer(genes)
    st_idx = st_genes.get_indexer(genes)
    if (ref_idx < 0).any() or (st_idx < 0).any():
        raise ValueError("Internal gene intersection error.")
    return ref_matrix[:, ref_idx].tocsr(), st_matrix[:, st_idx].tocsr(), genes


def filter_st_spots(
    st: ad.AnnData,
    st_matrix: sparse.csr_matrix,
    *,
    filter_in_tissue: bool,
    filter_nonzero_lib: bool,
    annotation_col: Optional[str],
    excluded_annotations: list[str],
) -> tuple[ad.AnnData, sparse.csr_matrix]:
    keep = np.ones(st.n_obs, dtype=bool)
    if filter_in_tissue and "in_tissue" in st.obs:
        keep &= st.obs["in_tissue"].astype(int).to_numpy() == 1
    if filter_nonzero_lib:
        keep &= np.asarray(st_matrix.sum(axis=1)).ravel() > 0
    if excluded_annotations:
        if annotation_col is None:
            annotation_col = next(
                (col for col in ("manual_annotation_fine", "Classification", "manual_annotation") if col in st.obs),
                None,
            )
        if annotation_col is None or annotation_col not in st.obs:
            raise KeyError(
                "Cannot apply --exclude-st-annotation: no usable ST annotation column was found. "
                f"Available obs columns: {list(st.obs.columns)}"
            )
        excluded = {str(value) for value in excluded_annotations}
        annotation = st.obs[annotation_col].astype(str)
        n_excluded = int(annotation.isin(excluded).sum())
        log(f"[INFO] ST annotation filtering on {annotation_col!r}: exclude {sorted(excluded)} ({n_excluded} spots)")
        keep &= ~annotation.isin(excluded).to_numpy()
    if int(keep.sum()) != st.n_obs:
        log(f"[INFO] ST filtering: keep {int(keep.sum())} / {st.n_obs} spots")
        st = st[keep].copy()
        st_matrix = st_matrix[keep, :].tocsr()
    return st, st_matrix


def get_st_coords(st: ad.AnnData, source: str) -> pd.DataFrame:
    source = source.strip()
    if source.lower().startswith("obsm:"):
        key = source.split(":", 1)[1]
        if key not in st.obsm:
            raise KeyError(f"Missing ST obsm[{key!r}]. Available keys: {list(st.obsm.keys())}")
        coords = np.asarray(st.obsm[key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"ST obsm[{key!r}] must be a 2D array with at least two columns.")
        x = coords[:, 0]
        y = coords[:, 1]
    elif source.lower().startswith("obs:"):
        cols = [col.strip() for col in source.split(":", 1)[1].split(",")]
        if len(cols) < 2 or not all(col in st.obs for col in cols[:2]):
            raise KeyError(f"Missing ST obs coordinate columns requested by {source!r}.")
        x = st.obs[cols[0]].to_numpy()
        y = st.obs[cols[1]].to_numpy()
    else:
        raise ValueError(f"Invalid ST coordinate source: {source!r}")
    return pd.DataFrame({"barcode": st.obs_names.astype(str), "x": x, "y": y})


def write_gz_lines(path: Path, lines: pd.Index | np.ndarray | list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for value in lines:
            handle.write(f"{value}\n")


def export_mtx_bundle(
    prefix: str,
    outdir: Path,
    matrix_cells_by_genes: sparse.csr_matrix,
    genes: pd.Index,
    barcodes: pd.Index,
) -> tuple[Path, Path, Path]:
    mtx_path = outdir / f"{prefix}.mtx"
    mtx_gz_path = outdir / f"{prefix}.mtx.gz"
    genes_path = outdir / f"{prefix}.genes.tsv.gz"
    barcodes_path = outdir / f"{prefix}.barcodes.tsv.gz"

    mmwrite(mtx_path, matrix_cells_by_genes.T.tocsr())
    with open(mtx_path, "rb") as src, gzip.open(mtx_gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    mtx_path.unlink()

    write_gz_lines(genes_path, genes.astype(str))
    write_gz_lines(barcodes_path, barcodes.astype(str))
    return mtx_gz_path, genes_path, barcodes_path


def resolve_ref_celltype_col(sc: ad.AnnData, requested_col: Optional[str]) -> str:
    if requested_col:
        col = requested_col
    else:
        col = str(sc.uns.get("reference_label_col", "celltype_major"))
    if col not in sc.obs:
        raise KeyError(f"Reference cell type column {col!r} not found. Available obs columns: {list(sc.obs.columns)}")
    return col


def run_rctd(args: argparse.Namespace) -> None:
    sc_h5ad, st_h5ad, outdir = resolve_paths(args)
    io_dir = outdir / "io"
    rctd_dir = outdir / "rctd"
    io_dir.mkdir(parents=True, exist_ok=True)
    rctd_dir.mkdir(parents=True, exist_ok=True)

    if not args.r_script.exists():
        raise FileNotFoundError(f"Missing R script: {args.r_script}")

    log(f"[LOAD] SC: {sc_h5ad}")
    sc = ad.read_h5ad(sc_h5ad)
    log(f"[LOAD] ST: {st_h5ad}")
    st = ad.read_h5ad(st_h5ad)

    requested_ref_label_col = args.ref_celltype_col
    if requested_ref_label_col is None and str(args.cancer_type).strip().upper() == "WU":
        requested_ref_label_col = "celltype_minor"
    ref_label_col = resolve_ref_celltype_col(sc, requested_ref_label_col)
    log(f"[INFO] Reference label column: {ref_label_col}")

    ref_matrix = cast_to_int_if_integer_like(get_counts_matrix(sc, args.ref_counts_mode, "SC"), "SC")
    st_matrix = cast_to_int_if_integer_like(get_counts_matrix(st, args.st_counts_mode, "ST"), "ST")
    st, st_matrix = filter_st_spots(
        st,
        st_matrix,
        filter_in_tissue=args.filter_st_in_tissue,
        filter_nonzero_lib=args.filter_st_nonzero_lib,
        annotation_col=args.st_annotation_col,
        excluded_annotations=args.exclude_st_annotation,
    )

    ref_matrix, st_matrix, genes = harmonize_genes(
        ref_matrix,
        sc.var_names.copy(),
        st_matrix,
        st.var_names.copy(),
        strip_versions=args.strip_ensembl_version,
        collapse_duplicates=args.collapse_duplicate_genes,
    )
    if ref_matrix.shape[0] != sc.n_obs:
        raise ValueError("SC count matrix row count does not match SC obs.")
    if st_matrix.shape[0] != st.n_obs:
        raise ValueError("ST count matrix row count does not match filtered ST obs.")

    celltypes = sc.obs[ref_label_col].astype(str)
    if celltypes.isna().any() or (celltypes.str.len() == 0).any():
        raise ValueError(f"Reference label column {ref_label_col!r} contains empty values.")
    ref_barcodes_index = sc.obs_names.copy()
    reference_celltype_counts_input = celltypes.value_counts()
    rare_celltypes = reference_celltype_counts_input[
        reference_celltype_counts_input < int(args.min_cells_per_type)
    ]
    if not rare_celltypes.empty:
        log(
            "[INFO] Dropping undersized reference cell types "
            f"(<{int(args.min_cells_per_type)} cells): {rare_celltypes.to_dict()}"
        )
        keep_ref = ~celltypes.isin(rare_celltypes.index).to_numpy()
        ref_matrix = ref_matrix[keep_ref, :].tocsr()
        ref_barcodes_index = ref_barcodes_index[keep_ref]
        celltypes = celltypes.iloc[np.flatnonzero(keep_ref)]
    log(f"[INFO] Reference cell types: {celltypes.nunique()}")

    log("[EXPORT] Writing RCTD IO bundles...")
    ref_mtx, ref_genes, ref_barcodes = export_mtx_bundle(
        "ref", io_dir, ref_matrix, genes, ref_barcodes_index
    )
    st_mtx, st_genes, st_barcodes = export_mtx_bundle("st", io_dir, st_matrix, genes, st.obs_names)

    ref_celltypes = io_dir / "ref.celltypes.tsv.gz"
    write_gz_lines(ref_celltypes, celltypes.to_numpy())

    coords = get_st_coords(st, args.st_coords_source)
    coords = coords.set_index("barcode").loc[st.obs_names.astype(str)].reset_index()
    st_coords = io_dir / "st.coords.tsv.gz"
    with gzip.open(st_coords, "wt", encoding="utf-8") as handle:
        coords.to_csv(handle, sep="\t", index=False)

    params = {
        "quiet": bool(args.quiet),
        "max_cores": int(args.max_cores),
        "max_cells_per_type": int(args.max_cells_per_type),
        "gene_cutoff": float(args.gene_cutoff),
        "fc_cutoff": float(args.fc_cutoff),
        "doublets": bool(args.doublets),
    }
    params_path = io_dir / "params.json"
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "cancer_type": args.cancer_type,
        "sample_id": args.sample_id,
        "sc_h5ad": str(sc_h5ad),
        "st_h5ad": str(st_h5ad),
        "outdir": str(outdir),
        "n_reference_cells_input": int(sc.n_obs),
        "n_reference_cells_used": int(ref_matrix.shape[0]),
        "n_spots": int(st.n_obs),
        "n_genes": int(len(genes)),
        "reference_label_col": ref_label_col,
        "reference_celltype_counts": celltypes.value_counts().to_dict(),
        "dropped_reference_celltypes": rare_celltypes.to_dict(),
        "min_cells_per_type": int(args.min_cells_per_type),
        "st_annotation_col": args.st_annotation_col,
        "excluded_st_annotations": list(args.exclude_st_annotation),
        "params": params,
    }
    (io_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.prepare_only:
        log("[DONE] Prepared RCTD IO bundles only.")
        return

    cmd = [
        args.r_bin,
        "--vanilla",
        str(args.r_script),
        str(ref_mtx),
        str(ref_genes),
        str(ref_barcodes),
        str(ref_celltypes),
        str(st_mtx),
        str(st_genes),
        str(st_barcodes),
        str(st_coords),
        str(rctd_dir),
        str(params_path),
    ]
    log("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    props_path = rctd_dir / "rctd_props.csv.gz"
    if not props_path.exists():
        raise FileNotFoundError(f"Missing RCTD output: {props_path}")
    props = pd.read_csv(props_path, index_col=0)
    props.index = props.index.astype(str)
    keep = st.obs_names.astype(str).isin(props.index)
    if (~keep).any():
        log(f"[WARN] Dropping {int((~keep).sum())} ST spots missing from RCTD output.")
    st = st[keep].copy()
    props = props.loc[st.obs_names.astype(str)]
    log(f"[OK] RCTD proportions: {props.shape[0]} spots x {props.shape[1]} cell types")

    if args.write_st_h5ad:
        st.obsm[args.rctd_obsm_key] = props.to_numpy(dtype=np.float32)
        st.uns["rctd_celltypes"] = props.columns.astype(str).tolist()
        st.uns["rctd_reference_label_col"] = ref_label_col
        st.uns["rctd_source"] = {
            "sc_h5ad": str(sc_h5ad),
            "props_csv": str(props_path),
            "params": params,
        }
        out_h5ad = outdir / "st_with_rctd.h5ad"
        log(f"[SAVE] {out_h5ad}")
        st.write_h5ad(out_h5ad, compression="gzip")

    log("[DONE]")


def main() -> None:
    run_rctd(parse_args())


if __name__ == "__main__":
    main()
