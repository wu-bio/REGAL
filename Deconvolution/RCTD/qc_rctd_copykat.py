#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/regal_mplconfig")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


DEFAULT_RCTD_OUTDIR = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/rctd-out"
)
DEFAULT_QC_DIRNAME = "qc_rctd_copykat"
FINAL_PANELS = [
    "Tumor Epithelial",
    "Normal-like Epithelial",
    "CAFs",
    "Myeloid",
    "T-cells",
    "B-cells",
    "Endothelial",
    "PVL",
    "Plasmablasts",
]
COPYKAT_COLORS = {
    "aneuploid": "#C73E1D",
    "diploid": "#2B6CB0",
    "Unknown": "#8A8A8A",
}
FLAG_COLORS = {
    "ok": "#5A8F3D",
    "copykat_unknown": "#8A8A8A",
    "aneuploid_low_epithelial_prop": "#D08C24",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC for final RCTD+CopyKAT deconvolution results.")
    parser.add_argument("--rctd-outdir", type=Path, default=DEFAULT_RCTD_OUTDIR)
    parser.add_argument("--st-h5ad", type=Path, default=None)
    parser.add_argument("--final-props", type=Path, default=None)
    parser.add_argument("--final-report", type=Path, default=None)
    parser.add_argument("--qc-outdir", type=Path, default=None)
    parser.add_argument("--low-umi-threshold", type=float, default=100)
    parser.add_argument("--low-umi-q", type=float, default=None)
    parser.add_argument("--knn-k", type=int, default=8)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    outdir = args.rctd_outdir
    return {
        "st_h5ad": args.st_h5ad or outdir / "st_with_rctd_copykat.h5ad",
        "final_props": args.final_props or outdir / "rctd_copykat_final_props.csv.gz",
        "final_report": args.final_report or outdir / "rctd_copykat_final_report.csv.gz",
        "qc_outdir": args.qc_outdir or outdir / DEFAULT_QC_DIRNAME,
    }


def get_coords(st: ad.AnnData) -> np.ndarray:
    if "spatial" in st.obsm:
        coords = np.asarray(st.obsm["spatial"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]
    for x_col, y_col in [
        ("x", "y"),
        ("X", "Y"),
        ("array_col", "array_row"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ]:
        if x_col in st.obs and y_col in st.obs:
            return np.vstack([st.obs[x_col].astype(float), st.obs[y_col].astype(float)]).T
    raise ValueError("Cannot find spatial coordinates in obsm['spatial'] or common obs columns.")


def libsize_from_x(st: ad.AnnData) -> np.ndarray:
    if sp.issparse(st.X):
        return np.asarray(st.X.sum(axis=1)).ravel().astype(float)
    return np.asarray(st.X.sum(axis=1)).ravel().astype(float)


def entropy(props: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    props = np.clip(props, eps, 1.0)
    return -np.sum(props * np.log(props), axis=1)


def spatial_knn_metrics(coords: np.ndarray, props: np.ndarray, k: int) -> dict[str, float | int]:
    if coords.shape[0] < 3:
        return {
            "spatial_knn_k": int(k),
            "spatial_knn_cosine_mean": None,
            "spatial_knn_cosine_p10": None,
            "spatial_knn_l1diff_mean": None,
            "spatial_knn_l1diff_p90": None,
        }
    nn = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0]))
    nn.fit(coords)
    _, idx = nn.kneighbors(coords)
    neigh = idx[:, 1:]

    norm = np.linalg.norm(props, axis=1, keepdims=True) + 1e-12
    props_norm = props / norm
    cos_vals = []
    l1_vals = []
    for i in range(props.shape[0]):
        ni = neigh[i]
        cos_vals.append((props_norm[i] * props_norm[ni]).sum(axis=1))
        l1_vals.append(np.abs(props[i] - props[ni]).sum(axis=1))
    cos_vals = np.concatenate(cos_vals)
    l1_vals = np.concatenate(l1_vals)
    return {
        "spatial_knn_k": int(k),
        "spatial_knn_cosine_mean": float(np.mean(cos_vals)),
        "spatial_knn_cosine_p10": float(np.quantile(cos_vals, 0.10)),
        "spatial_knn_l1diff_mean": float(np.mean(l1_vals)),
        "spatial_knn_l1diff_p90": float(np.quantile(l1_vals, 0.90)),
    }


def load_inputs(paths: dict[str, Path]) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    st = ad.read_h5ad(paths["st_h5ad"])
    props = pd.read_csv(paths["final_props"], index_col=0)
    props.index = props.index.astype(str)
    report = pd.read_csv(paths["final_report"])
    report["barcode"] = report["barcode"].astype(str)
    report = report.set_index("barcode", drop=False)

    obs_names = st.obs_names.astype(str)
    missing_props = pd.Index(obs_names).difference(props.index)
    missing_report = pd.Index(obs_names).difference(report.index)
    if len(missing_props) or len(missing_report):
        raise ValueError(
            f"Barcode mismatch: missing props={len(missing_props)}, missing report={len(missing_report)}"
        )
    props = props.loc[obs_names]
    report = report.loc[obs_names]
    return st, props, report


def valid_spot_mask(st: ad.AnnData, props: pd.DataFrame, low_umi_threshold: float | None, low_umi_q: float | None):
    p = props.to_numpy(dtype=float)
    lib = libsize_from_x(st)
    nan_row = np.isnan(p).any(axis=1)
    low_umi = np.zeros(st.n_obs, dtype=bool)
    rules = []
    if low_umi_threshold is not None:
        low_umi |= lib < float(low_umi_threshold)
        rules.append(f"libsize<{low_umi_threshold:g}")
    if low_umi_q is not None:
        qthr = float(np.quantile(lib, float(low_umi_q)))
        low_umi |= lib <= qthr
        rules.append(f"libsize<=q{low_umi_q:g}({qthr:.3f})")
    invalid = nan_row | low_umi
    return ~invalid, invalid, nan_row, low_umi, lib, " OR ".join(rules) if rules else None


def plot_hist(props: pd.DataFrame, valid_mask: np.ndarray, out_png: Path) -> None:
    p = props.to_numpy(dtype=float)[valid_mask]
    pmax = p.max(axis=1)
    ent = entropy(p)
    ek = np.exp(ent)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].hist(pmax, bins=50, color="#4C78A8")
    axes[0].set_title("Max final proportion")
    axes[0].set_xlabel("pmax")
    axes[0].set_ylabel("spots")

    axes[1].hist(ent, bins=50, color="#F58518")
    axes[1].set_title("Final proportion entropy")
    axes[1].set_xlabel("entropy")

    axes[2].hist(ek, bins=50, color="#54A24B")
    axes[2].set_title("Effective cell types")
    axes[2].set_xlabel("exp(entropy)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_spatial_props(coords: np.ndarray, props: pd.DataFrame, out_png: Path) -> None:
    panels = [ct for ct in FINAL_PANELS if ct in props.columns]
    if len(panels) < 9:
        remaining = [c for c in props.columns if c not in panels]
        remaining = props[remaining].mean().sort_values(ascending=False).index.tolist()
        panels.extend(remaining[: 9 - len(panels)])
    panels = panels[:9]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10.5))
    means = props.mean()
    for ax, ct in zip(axes.ravel(), panels):
        sca = ax.scatter(coords[:, 0], coords[:, 1], c=props[ct].to_numpy(float), s=6)
        ax.set_title(f"{ct} (mean={means[ct]:.3f})")
        ax.axis("equal")
        ax.axis("off")
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_categorical_spatial(
    ax,
    coords: np.ndarray,
    values: pd.Series,
    color_map: dict[str, str],
    title: str,
) -> None:
    values = values.fillna("Unknown").astype(str)
    for value in sorted(values.unique()):
        idx = values.eq(value).to_numpy()
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=6,
            c=color_map.get(value, "#BBBBBB"),
            label=f"{value} ({idx.sum()})",
        )
    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    ax.legend(frameon=False, fontsize=7, loc="best")


def plot_cnv_epithelial(coords: np.ndarray, props: pd.DataFrame, report: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_categorical_spatial(axes[0, 0], coords, report["copykat_pred_tirosh"], COPYKAT_COLORS, "CopyKAT Tirosh")

    sca = axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=report["cnv_score"].to_numpy(float), s=6)
    axes[0, 1].set_title("CNV score")
    axes[0, 1].axis("equal")
    axes[0, 1].axis("off")
    fig.colorbar(sca, ax=axes[0, 1], fraction=0.046, pad=0.04)

    for ax, ct in zip(axes[1], ["Tumor Epithelial", "Normal-like Epithelial"]):
        scp = ax.scatter(coords[:, 0], coords[:, 1], c=props[ct].to_numpy(float), s=6)
        ax.set_title(ct)
        ax.axis("equal")
        ax.axis("off")
        fig.colorbar(scp, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_audit(report: pd.DataFrame, out_png: Path) -> None:
    copykat_final = pd.crosstab(report["copykat_pred_tirosh"], report["final_dominant"])
    rctd_final = pd.crosstab(report["rctd_dominant"], report["final_dominant"])
    flag_counts = report["final_flag"].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, table, title in [
        (axes[0], copykat_final, "CopyKAT vs final dominant"),
        (axes[1], rctd_final, "RCTD dominant vs final dominant"),
    ]:
        im = ax.imshow(table.to_numpy(), aspect="auto", cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(table.shape[1]))
        ax.set_xticklabels(table.columns, rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(table.shape[0]))
        ax.set_yticklabels(table.index, fontsize=8)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = int(table.iat[i, j])
                if value:
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[2].barh(flag_counts.index.astype(str), flag_counts.to_numpy(), color="#4C78A8")
    axes[2].set_title("Final flags")
    axes[2].set_xlabel("spots")
    axes[2].invert_yaxis()
    for y, value in enumerate(flag_counts.to_numpy()):
        axes[2].text(value, y, f" {value}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def make_summary(
    st: ad.AnnData,
    props: pd.DataFrame,
    report: pd.DataFrame,
    valid_mask: np.ndarray,
    invalid: np.ndarray,
    nan_row: np.ndarray,
    low_umi: np.ndarray,
    lib: np.ndarray,
    low_umi_rule: str | None,
    knn_k: int,
    paths: dict[str, Path],
) -> dict:
    p = props.to_numpy(dtype=float)
    p_valid = p[valid_mask]
    pmax = p_valid.max(axis=1) if p_valid.size else np.array([])
    ent = entropy(p_valid) if p_valid.size else np.array([])
    coords = get_coords(st)
    knn = spatial_knn_metrics(coords[valid_mask], p_valid, knn_k) if p_valid.size else {}

    summary = {
        "n_spots": int(st.n_obs),
        "n_celltypes": int(props.shape[1]),
        "celltypes": props.columns.astype(str).tolist(),
        "props_rowsum_min": float(np.nanmin(p.sum(axis=1))),
        "props_rowsum_max": float(np.nanmax(p.sum(axis=1))),
        "props_any_nan": bool(np.isnan(p).any()),
        "props_any_negative": bool((p < -1e-8).any()),
        "valid_spots_n": int(valid_mask.sum()),
        "invalid_spots_n": int(invalid.sum()),
        "invalid_spots_nanprops_n": int(nan_row.sum()),
        "invalid_spots_lowumi_n": int(low_umi.sum()),
        "low_umi_rule": low_umi_rule,
        "libsize_median": float(np.median(lib)),
        "libsize_p01": float(np.quantile(lib, 0.01)),
        "pmax_median": float(np.median(pmax)) if pmax.size else None,
        "pmax_p10": float(np.quantile(pmax, 0.10)) if pmax.size else None,
        "entropy_mean": float(np.mean(ent)) if ent.size else None,
        "effective_k_mean": float(np.mean(np.exp(ent))) if ent.size else None,
        "copykat_counts": report["copykat_pred_tirosh"].value_counts(dropna=False).to_dict(),
        "rctd_dominant_counts": report["rctd_dominant"].value_counts(dropna=False).to_dict(),
        "final_dominant_counts": report["final_dominant"].value_counts(dropna=False).to_dict(),
        "final_flag_counts": report["final_flag"].value_counts(dropna=False).to_dict(),
        "audit_rctd_cancer_epi_but_diploid_n": int(report["audit_rctd_cancer_epi_but_diploid"].sum())
        if "audit_rctd_cancer_epi_but_diploid" in report
        else None,
        "inputs": {
            "st_h5ad": str(paths["st_h5ad"]),
            "final_props": str(paths["final_props"]),
            "final_report": str(paths["final_report"]),
        },
        "qc_outputs": {
            "qc_report": str(paths["qc_outdir"] / "qc_report.json"),
            "qc_composition_hist": str(paths["qc_outdir"] / "qc_composition_hist.png"),
            "qc_spatial_final_props": str(paths["qc_outdir"] / "qc_spatial_final_props.png"),
            "qc_spatial_cnv_epithelial": str(paths["qc_outdir"] / "qc_spatial_cnv_epithelial.png"),
            "qc_audit_tables": str(paths["qc_outdir"] / "qc_audit_tables.png"),
        },
    }
    summary.update(knn)
    if "annot_type" in st.obs:
        summary["manual_annotation_vs_final_dominant"] = pd.crosstab(
            st.obs["annot_type"].astype(str), report["final_dominant"].astype(str)
        ).to_dict()
        summary["manual_annotation_vs_copykat"] = pd.crosstab(
            st.obs["annot_type"].astype(str), report["copykat_pred_tirosh"].astype(str)
        ).to_dict()
    return summary


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    paths["qc_outdir"].mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {paths['st_h5ad']}", flush=True)
    st, props, report = load_inputs(paths)
    coords = get_coords(st)
    valid_mask, invalid, nan_row, low_umi, lib, low_umi_rule = valid_spot_mask(
        st, props, args.low_umi_threshold, args.low_umi_q
    )

    summary = make_summary(
        st,
        props,
        report,
        valid_mask,
        invalid,
        nan_row,
        low_umi,
        lib,
        low_umi_rule,
        args.knn_k,
        paths,
    )
    report_path = paths["qc_outdir"] / "qc_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plot_hist(props, valid_mask, paths["qc_outdir"] / "qc_composition_hist.png")
    plot_spatial_props(coords, props, paths["qc_outdir"] / "qc_spatial_final_props.png")
    plot_cnv_epithelial(coords, props, report, paths["qc_outdir"] / "qc_spatial_cnv_epithelial.png")
    plot_audit(report, paths["qc_outdir"] / "qc_audit_tables.png")

    print(f"[QC] Done: {paths['qc_outdir']}", flush=True)
    print(f"[QC] final flags: {summary['final_flag_counts']}", flush=True)
    print(f"[QC] final dominant: {summary['final_dominant_counts']}", flush=True)


if __name__ == "__main__":
    main()
