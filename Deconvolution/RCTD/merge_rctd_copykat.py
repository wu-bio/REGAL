#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_SAMPLE_ID = "Breast_BlockA_Section1"
DEFAULT_RCTD_OUTDIR = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/rctd-out"
)
DEFAULT_ST_H5AD = Path("/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad")
EPITHELIAL_COLS = ["Cancer Epithelial", "Normal Epithelial"]
FINAL_EPITHELIAL_COLS = ["Tumor Epithelial", "Normal-like Epithelial"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge RCTD proportions with CopyKAT CNV calls into final deconvolution proportions."
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--rctd-outdir", type=Path, default=DEFAULT_RCTD_OUTDIR)
    parser.add_argument("--st-h5ad", type=Path, default=DEFAULT_ST_H5AD)
    parser.add_argument("--rctd-props", type=Path, default=None)
    parser.add_argument("--copykat-report", type=Path, default=None)
    parser.add_argument("--copykat-col", default="copykat_pred_tirosh")
    parser.add_argument("--epithelial-min-prop", type=float, default=0.5)
    parser.add_argument("--out-h5ad", type=Path, default=None)
    parser.add_argument("--out-props", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    parser.add_argument("--out-summary", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    outdir = args.rctd_outdir
    return {
        "st_h5ad": args.st_h5ad,
        "rctd_props": args.rctd_props or outdir / "rctd" / "rctd_props.csv.gz",
        "copykat_report": args.copykat_report
        or outdir / "copykat_hbca_ref800" / "copykat" / "copykat_pred_report.csv.gz",
        "out_h5ad": args.out_h5ad or outdir / "st_with_rctd_copykat.h5ad",
        "out_props": args.out_props or outdir / "rctd_copykat_final_props.csv.gz",
        "out_report": args.out_report or outdir / "rctd_copykat_final_report.csv.gz",
        "out_summary": args.out_summary or outdir / "rctd_copykat_final_summary.json",
    }


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def dominant_from_props(props: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dominant = props.idxmax(axis=1)
    dominant_prop = props.max(axis=1)
    return dominant, dominant_prop


def build_final_props(
    rctd_props: pd.DataFrame,
    copykat: pd.DataFrame,
    copykat_col: str,
    epithelial_min_prop: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(rctd_props, EPITHELIAL_COLS, "RCTD proportions")
    require_columns(copykat, ["barcode", copykat_col, "copykat_pred_default", "cnv_score", "cnv_cor"], "CopyKAT report")

    copykat = copykat.copy()
    copykat["barcode"] = copykat["barcode"].astype(str)
    copykat = copykat.set_index("barcode", drop=False)

    missing = rctd_props.index.difference(copykat.index)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} RCTD barcodes are missing from CopyKAT report.")

    copykat = copykat.loc[rctd_props.index]
    if not copykat.index.equals(rctd_props.index):
        raise ValueError("Internal barcode alignment error.")

    non_epi_cols = [col for col in rctd_props.columns if col not in EPITHELIAL_COLS]
    final_props = rctd_props[non_epi_cols].copy()
    epithelial_prop = rctd_props[EPITHELIAL_COLS].sum(axis=1)
    final_props["Tumor Epithelial"] = 0.0
    final_props["Normal-like Epithelial"] = 0.0

    pred = copykat[copykat_col].fillna("Unknown").astype(str)
    is_aneuploid = pred.eq("aneuploid")
    is_diploid = pred.eq("diploid")
    is_unknown = ~(is_aneuploid | is_diploid)

    final_props.loc[is_aneuploid, "Tumor Epithelial"] = epithelial_prop.loc[is_aneuploid]
    final_props.loc[is_diploid, "Normal-like Epithelial"] = epithelial_prop.loc[is_diploid]

    # Rare CopyKAT-Unknown spots keep the original RCTD epithelial split so row sums stay intact.
    final_props.loc[is_unknown, "Tumor Epithelial"] = rctd_props.loc[is_unknown, "Cancer Epithelial"]
    final_props.loc[is_unknown, "Normal-like Epithelial"] = rctd_props.loc[is_unknown, "Normal Epithelial"]

    final_props = final_props[non_epi_cols + FINAL_EPITHELIAL_COLS]
    row_sums = final_props.sum(axis=1)
    if not np.allclose(row_sums.to_numpy(), 1.0, atol=1e-4):
        raise ValueError(
            "Final proportions do not sum to 1 within tolerance. "
            f"Observed range: {row_sums.min():.6f}-{row_sums.max():.6f}"
        )

    rctd_dominant, rctd_dominant_prop = dominant_from_props(rctd_props)
    final_dominant, final_dominant_prop = dominant_from_props(final_props)
    epithelial_is_major = epithelial_prop >= epithelial_min_prop

    flags = pd.Series("ok", index=rctd_props.index, dtype="object")
    flags.loc[is_unknown] = "copykat_unknown"
    flags.loc[is_aneuploid & ~epithelial_is_major] = "aneuploid_low_epithelial_prop"
    rctd_cancer_epi_but_diploid = (
        is_diploid & epithelial_is_major & rctd_props["Cancer Epithelial"].gt(rctd_props["Normal Epithelial"])
    )

    report = pd.DataFrame(
        {
            "barcode": rctd_props.index,
            "rctd_dominant": rctd_dominant,
            "rctd_dominant_prop": rctd_dominant_prop,
            "rctd_epithelial_prop": epithelial_prop,
            "rctd_cancer_epithelial_prop": rctd_props["Cancer Epithelial"],
            "rctd_normal_epithelial_prop": rctd_props["Normal Epithelial"],
            "copykat_pred_default": copykat["copykat_pred_default"].astype(str),
            "copykat_pred_tirosh": copykat[copykat_col].fillna("Unknown").astype(str),
            "cnv_score": copykat["cnv_score"],
            "cnv_cor": copykat["cnv_cor"],
            "final_dominant": final_dominant,
            "final_dominant_prop": final_dominant_prop,
            "final_flag": flags,
            "audit_rctd_cancer_epi_but_diploid": rctd_cancer_epi_but_diploid,
        },
        index=rctd_props.index,
    )
    return final_props, report


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    print(f"[LOAD] ST h5ad: {paths['st_h5ad']}", flush=True)
    st = ad.read_h5ad(paths["st_h5ad"])
    print(f"[LOAD] RCTD props: {paths['rctd_props']}", flush=True)
    rctd_props = pd.read_csv(paths["rctd_props"], index_col=0)
    rctd_props.index = rctd_props.index.astype(str)
    print(f"[LOAD] CopyKAT report: {paths['copykat_report']}", flush=True)
    copykat = pd.read_csv(paths["copykat_report"])

    if not st.obs_names.astype(str).equals(rctd_props.index):
        rctd_props = rctd_props.loc[st.obs_names.astype(str)]

    final_props, report = build_final_props(
        rctd_props,
        copykat,
        copykat_col=args.copykat_col,
        epithelial_min_prop=args.epithelial_min_prop,
    )

    print(f"[SAVE] {paths['out_props']}", flush=True)
    final_props.to_csv(paths["out_props"], compression="gzip")
    print(f"[SAVE] {paths['out_report']}", flush=True)
    report.to_csv(paths["out_report"], index=False, compression="gzip")

    st.obsm["rctd_copykat_props"] = final_props.to_numpy(dtype=np.float32)
    st.uns["rctd_copykat_celltypes"] = final_props.columns.astype(str).tolist()
    st.uns["rctd_copykat_method"] = {
        "description": "RCTD epithelial block split by CopyKAT CNV; non-epithelial RCTD proportions retained.",
        "epithelial_input_columns": EPITHELIAL_COLS,
        "copykat_column": args.copykat_col,
        "epithelial_min_prop_for_flags": args.epithelial_min_prop,
    }
    for col in report.columns:
        if col == "barcode":
            continue
        st.obs[col] = report[col].to_numpy()

    print(f"[SAVE] {paths['out_h5ad']}", flush=True)
    st.write_h5ad(paths["out_h5ad"], compression="gzip")

    summary = {
        "sample_id": args.sample_id,
        "n_spots": int(final_props.shape[0]),
        "final_celltypes": final_props.columns.astype(str).tolist(),
        "copykat_counts": report["copykat_pred_tirosh"].value_counts(dropna=False).to_dict(),
        "rctd_dominant_counts": report["rctd_dominant"].value_counts(dropna=False).to_dict(),
        "final_dominant_counts": report["final_dominant"].value_counts(dropna=False).to_dict(),
        "final_flag_counts": report["final_flag"].value_counts(dropna=False).to_dict(),
        "inputs": {key: str(value) for key, value in paths.items() if key.startswith(("st", "rctd", "copykat"))},
        "outputs": {key: str(value) for key, value in paths.items() if key.startswith("out")},
    }
    print(f"[SAVE] {paths['out_summary']}", flush=True)
    paths["out_summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
