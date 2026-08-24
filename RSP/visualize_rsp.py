from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/regal_mplconfig")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from utils.dataset_registry import get_dataset
from utils.spatial_plotting import (
    ensure_out_dir,
    load_spatial_coords,
    sample_dir_to_h5ad_path,
    save_figure,
    spatial_scatter,
    write_manifest,
)


def _panel_grid(
    n_items: int,
    *,
    panel_width: float,
    panel_height: float,
) -> tuple[plt.Figure, np.ndarray]:
    rows, cols = _adaptive_panel_grid_shape(n_items)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(panel_width * cols, panel_height * rows),
        squeeze=False,
        constrained_layout=True,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.05, h_pad=0.06, wspace=0.04, hspace=0.06)
    return fig, axes


def _adaptive_panel_grid_shape(n_items: int) -> tuple[int, int]:
    n = max(0, int(n_items))
    if n <= 1:
        return 1, 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    while cols <= rows:
        cols += 1
        rows = int(np.ceil(n / cols))
    return rows, cols


def _load_spatial_image_context(h5ad_path: Path, preferred_image: str = "hires") -> tuple[np.ndarray, float] | None:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("anndata is required to load spatial images from h5ad files.") from exc

    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    spatial = adata.uns.get("spatial", {})
    if not spatial:
        return None
    library_id = next(iter(spatial.keys()))
    library = spatial.get(library_id, {})
    images = library.get("images", {})
    if not images:
        return None
    image_key = preferred_image if preferred_image in images else next(iter(images.keys()))
    image = np.asarray(images[image_key])
    scalefactors = library.get("scalefactors", {})
    scale_key = f"tissue_{image_key}_scalef"
    if scale_key in scalefactors:
        scale = float(scalefactors[scale_key])
    elif image_key == "hires" and "tissue_hires_scalef" in scalefactors:
        scale = float(scalefactors["tissue_hires_scalef"])
    elif image_key == "lowres" and "tissue_lowres_scalef" in scalefactors:
        scale = float(scalefactors["tissue_lowres_scalef"])
    else:
        coords = load_spatial_coords(h5ad_path)
        scale = min(image.shape[1] / coords["x"].max(), image.shape[0] / coords["y"].max())
    return image, scale


def _selected_rsps(rsp: pd.DataFrame, limit: int | None) -> list[str]:
    df = rsp.copy()
    if "rsp_score" in df.columns:
        df = df.sort_values("rsp_score", ascending=False)
    ids = df["rsp_id"].astype(str).drop_duplicates().tolist()
    return ids if limit is None else ids[:limit]


def plot_tissue_image(h5ad_path: Path, coords: pd.DataFrame, out_path: Path) -> None:
    image_context = _load_spatial_image_context(h5ad_path, preferred_image="hires")
    if image_context is None:
        return
    image, scale = image_context
    xy = _coords_to_image_space(coords, scale)
    image = _orient_tissue_image(image, xy)
    radius_px = _estimate_spot_radius_px(xy)
    cropped_image, xy = _crop_to_spot_window(image, xy, padding_px=radius_px * 2.5)

    fig, ax = plt.subplots(figsize=(8.0, 8.5), constrained_layout=True)
    ax.imshow(cropped_image, origin="upper")
    ax.scatter(
        xy["x_img"],
        xy["y_img"],
        s=6.0,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.35,
        alpha=0.36,
    )
    ax.set_title("Tissue image with spatial spot centers", loc="left", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, out_path)


def plot_rsp_activation_atlas(activation: pd.DataFrame, coords: pd.DataFrame, rsp_ids: list[str], out_path: Path) -> None:
    fig, axes = _panel_grid(len(rsp_ids), panel_width=4.1, panel_height=3.65)
    coord_index = coords.set_index("spot_id")
    for ax, rsp_id in zip(axes.ravel(), rsp_ids):
        values = pd.Series(0.0, index=coord_index.index)
        sub = activation.loc[activation["rsp_id"].eq(rsp_id), ["spot_id", "activation"]]
        values.loc[sub["spot_id"].astype(str)] = sub["activation"].astype(float).to_numpy()
        handle = spatial_scatter(ax, coords, values.to_numpy(), cmap="magma", point_size=4.6, title=rsp_id)
        if handle is not None:
            fig.colorbar(handle, ax=ax, fraction=0.042, pad=0.015)
    for ax in axes.ravel()[len(rsp_ids) :]:
        ax.axis("off")
    fig.suptitle("RSP activation atlas", x=0.01, ha="left", fontsize=15, fontweight="bold")
    save_figure(fig, out_path)


def _active_activation_for_rsp(activation: pd.DataFrame, rsp_id: str) -> pd.Series:
    sub = activation.loc[activation["rsp_id"].eq(rsp_id), ["spot_id", "activation"]].copy()
    if "activation_active" in activation.columns:
        sub["activation_active"] = activation.loc[activation["rsp_id"].eq(rsp_id), "activation_active"].astype(bool).to_numpy()
    else:
        sub["activation_active"] = False
    sub["activation"] = sub["activation"].astype(float)
    if "activation_active" not in activation.columns:
        sub["activation_active"] = np.isfinite(sub["activation"]) & (sub["activation"] > 0)
    active = sub.loc[np.isfinite(sub["activation"]) & sub["activation_active"], ["spot_id", "activation"]]
    return active.assign(spot_id=active["spot_id"].astype(str)).set_index("spot_id")["activation"]


def _coords_to_image_space(coords: pd.DataFrame, scale: float) -> pd.DataFrame:
    xy = coords[["spot_id", "x", "y"]].copy()
    xy["spot_id"] = xy["spot_id"].astype(str)
    xy["x_img"] = xy["x"].astype(float) * float(scale)
    xy["y_img"] = xy["y"].astype(float) * float(scale)
    return xy.set_index("spot_id")


def _orient_tissue_image(image: np.ndarray, xy: pd.DataFrame | None = None) -> np.ndarray:
    """Orient the H&E image to the scaled spatial-coordinate frame."""
    transposed = _transpose_tissue_image(image)
    if xy is None or xy.empty:
        return transposed

    original_overflow = _spot_bounds_overflow(image, xy)
    transposed_overflow = _spot_bounds_overflow(transposed, xy)
    if original_overflow < transposed_overflow:
        return image
    if transposed_overflow < original_overflow:
        return transposed
    if image.shape[:2] != transposed.shape[:2]:
        return image
    return transposed


def _transpose_tissue_image(image: np.ndarray) -> np.ndarray:
    return np.rot90(np.flipud(image), k=-1)


def _spot_bounds_overflow(image: np.ndarray, xy: pd.DataFrame) -> float:
    height, width = image.shape[:2]
    points = xy[["x_img", "y_img"]].to_numpy(dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return float("inf")
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    return (
        max(0.0, -x_min)
        + max(0.0, x_max - float(width))
        + max(0.0, -y_min)
        + max(0.0, y_max - float(height))
    )


def _estimate_spot_radius_px(xy: pd.DataFrame) -> float:
    points = xy[["x_img", "y_img"]].to_numpy(dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return 8.0
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = distances[:, 1]
    spacing = np.nanmedian(nearest[np.isfinite(nearest) & (nearest > 0)])
    if not np.isfinite(spacing) or spacing <= 0:
        return 8.0
    return max(3.0, spacing * 0.45)


def _crop_to_spot_window(
    image: np.ndarray,
    xy: pd.DataFrame,
    padding_px: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    height, width = image.shape[:2]
    x_min = max(0, int(np.floor(xy["x_img"].min() - padding_px)))
    x_max = min(width, int(np.ceil(xy["x_img"].max() + padding_px)))
    y_min = max(0, int(np.floor(xy["y_img"].min() - padding_px)))
    y_max = min(height, int(np.ceil(xy["y_img"].max() + padding_px)))
    cropped = image[y_min:y_max, x_min:x_max]
    adjusted = xy.copy()
    adjusted["x_img"] = adjusted["x_img"] - x_min
    adjusted["y_img"] = adjusted["y_img"] - y_min
    return cropped, adjusted


def plot_rsp_contour_panels_on_tissue(
    activation: pd.DataFrame,
    coords: pd.DataFrame,
    rsp_ids: list[str],
    h5ad_path: Path,
    out_path: Path,
) -> None:
    image_context = _load_spatial_image_context(h5ad_path, preferred_image="hires")
    if image_context is None:
        return
    image, scale = image_context
    xy = _coords_to_image_space(coords, scale)
    image = _orient_tissue_image(image, xy)
    radius_px = _estimate_spot_radius_px(xy)
    cropped_image, xy = _crop_to_spot_window(image, xy, padding_px=radius_px * 2.5)

    fig, axes = _panel_grid(len(rsp_ids), panel_width=4.1, panel_height=3.75)
    for ax, rsp_id in zip(axes.ravel(), rsp_ids):
        ax.imshow(cropped_image, origin="upper")
        ax.scatter(
            xy["x_img"],
            xy["y_img"],
            s=3.5,
            facecolors="none",
            edgecolors="#333333",
            linewidths=0.25,
            alpha=0.16,
        )
        active_activation = _active_activation_for_rsp(activation, rsp_id)
        active_xy = xy.loc[xy.index.intersection(active_activation.index)].copy()
        if not active_xy.empty:
            active_values = active_activation.loc[active_xy.index].to_numpy(dtype=float)
            ax.scatter(
                active_xy["x_img"],
                active_xy["y_img"],
                s=7.0,
                c=active_values,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                linewidths=0,
                alpha=0.90,
            )
        ax.set_title(rsp_id, loc="left", fontsize=9, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(rsp_ids) :]:
        ax.axis("off")
    fig.suptitle("RSP active spot footprints on cropped tissue image", x=0.01, ha="left", fontsize=15, fontweight="bold")
    save_figure(fig, out_path)


def plot_rsp_top_loadings(loadings: pd.DataFrame, rsp_ids: list[str], out_path: Path, top_n: int = 10) -> None:
    fig, axes = _panel_grid(len(rsp_ids), panel_width=5.0, panel_height=4.15)
    for ax, rsp_id in zip(axes.ravel(), rsp_ids):
        sub = (
            loadings.loc[loadings["rsp_id"].eq(rsp_id)]
            .sort_values("loading", ascending=False)
            .head(top_n)
            .iloc[::-1]
        )
        ax.barh(sub["gene"].astype(str), sub["loading"].astype(float), color="#7b3294")
        ax.set_title(rsp_id, loc="left", fontsize=10, pad=5)
        ax.set_xlabel("loading", fontsize=9, labelpad=3)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    for ax in axes.ravel()[len(rsp_ids) :]:
        ax.axis("off")
    fig.suptitle("Top core loading genes per RSP", x=0.01, ha="left", fontsize=15, fontweight="bold")
    save_figure(fig, out_path)


def plot_rsp_score_summary(rsp: pd.DataFrame, out_path: Path) -> None:
    plot_cols = [c for c in ["rsp_score", "spatial_score", "recurrence_score", "gene_quality_score"] if c in rsp.columns]
    if not plot_cols:
        return
    df = rsp[["rsp_id", *plot_cols]].copy().sort_values(plot_cols[0], ascending=False)
    fig, ax = plt.subplots(figsize=(9.6, 4.5), constrained_layout=True)
    x = np.arange(len(df))
    width = 0.8 / len(plot_cols)
    colors = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]
    for i, col in enumerate(plot_cols):
        ax.bar(x + i * width, df[col].astype(float), width=width, label=col, color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(plot_cols) - 1) / 2)
    ax.set_xticklabels(df["rsp_id"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("score")
    ax.set_title("RSP score summary", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    save_figure(fig, out_path)


def plot_core_score_elbow_curves(
    rsp: pd.DataFrame,
    loadings: pd.DataFrame,
    rsp_ids: list[str],
    out_path: Path,
) -> None:
    if loadings.empty or "core_score" not in loadings.columns:
        return
    fig, axes = _panel_grid(len(rsp_ids), panel_width=5.0, panel_height=3.75)
    rsp_index = rsp.set_index("rsp_id") if "rsp_id" in rsp.columns else pd.DataFrame()
    for panel_index, (ax, rsp_id) in enumerate(zip(axes.ravel(), rsp_ids)):
        sub = loadings.loc[loadings["rsp_id"].astype(str).eq(str(rsp_id))].copy()
        if "core_eligible" in sub.columns:
            sub = sub.loc[sub["core_eligible"].astype(bool)].copy()
        sub["core_score"] = pd.to_numeric(sub["core_score"], errors="coerce")
        sub = sub.loc[np.isfinite(sub["core_score"])].sort_values("core_score", ascending=False)
        scores = sub["core_score"].to_numpy(dtype=float)
        ranks = np.arange(1, scores.size + 1)
        meta = rsp_index.loc[str(rsp_id)] if str(rsp_id) in rsp_index.index else {}
        raw_elbow = _safe_int(_meta_value(meta, "raw_elbow_size"), 0)
        core_count = _safe_int(_meta_value(meta, "core_gene_count"), int(sub.get("is_core_gene", pd.Series(dtype=bool)).sum()))
        eligible_count = _safe_int(_meta_value(meta, "eligible_gene_count"), int(scores.size))
        if scores.size:
            ax.plot(ranks, scores, color="#2b5d7f", linewidth=2.0)
            if 0 < raw_elbow <= scores.size:
                ax.axvline(raw_elbow, color="#888888", linestyle="--", linewidth=1.2, alpha=0.80)
            if 0 < core_count <= scores.size:
                ax.axvline(core_count, color="#d1495b", linewidth=1.4, alpha=0.95)
                ax.scatter([core_count], [scores[core_count - 1]], s=20, color="#d1495b", zorder=5)
        ax.set_title(f"{rsp_id}  core={core_count} / eligible={eligible_count}", fontsize=9, pad=5)
        ax.set_xlabel("eligible gene rank", fontsize=8)
        ax.set_ylabel("core_score", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, color="#d9d9d9", linewidth=0.55, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        if panel_index == 0:
            ax.legend(
                handles=[
                    Line2D([0], [0], color="#2b5d7f", lw=2.0, label="sorted core_score"),
                    Line2D([0], [0], color="#888888", lw=1.2, ls="--", label="curvature suggestion"),
                    Line2D([0], [0], color="#d1495b", lw=1.4, label="final core count"),
                ],
                frameon=False,
                fontsize=7,
                loc="upper right",
            )
    for ax in axes.ravel()[len(rsp_ids) :]:
        ax.axis("off")
    fig.suptitle("RSP core score rank curves", x=0.01, ha="left", fontsize=15, fontweight="bold")
    save_figure(fig, out_path)


def plot_gene_selection_curve(
    gene_filter: pd.DataFrame,
    out_path: Path,
    *,
    adaptive_iqr_multiplier: float = 0.5,
    enable_adaptive_gene_selection: bool = True,
) -> None:
    required = {"gene_evidence_score", "is_kept_for_decomposition"}
    if gene_filter.empty or not required.issubset(gene_filter.columns):
        return
    kept = gene_filter.loc[gene_filter["is_kept_for_decomposition"].astype(bool)].copy()
    kept["gene_evidence_score"] = pd.to_numeric(kept["gene_evidence_score"], errors="coerce")
    kept = kept.loc[np.isfinite(kept["gene_evidence_score"])].sort_values("gene_evidence_score", ascending=False)
    if kept.empty:
        return
    scores = kept["gene_evidence_score"].to_numpy(dtype=float)
    ranks = np.arange(1, scores.size + 1)
    q1, median, q3 = np.quantile(scores, [0.25, 0.50, 0.75])
    multiplier = max(0.0, float(adaptive_iqr_multiplier))
    cutoff = float(np.clip(median + multiplier * (q3 - q1), float(scores.min()), float(scores.max())))
    if "is_selected_for_decomposition" in kept.columns:
        selected_count = int(kept["is_selected_for_decomposition"].astype(bool).sum())
    else:
        selected_count = int(np.count_nonzero(scores >= cutoff))
    selected_count = max(0, min(selected_count, scores.size))

    fig, ax = plt.subplots(figsize=(12.0, 6.0), constrained_layout=True)
    ax.plot(ranks, scores, color="#2b5d7f", linewidth=2.0, label="genes sorted by evidence score")
    if selected_count > 0:
        fill_bottom = cutoff if enable_adaptive_gene_selection else float(scores.min())
        ax.fill_between(ranks[:selected_count], scores[:selected_count], fill_bottom, color="#f4a261", alpha=0.25)
        ax.axvline(selected_count, color="#c43d0b", linestyle="--", linewidth=1.4, label=f"selected genes = {selected_count}")
        arrow_x = max(1, int(selected_count * 0.45))
        arrow_y = float(scores[min(arrow_x - 1, scores.size - 1)])
        ax.annotate(
            "selected high-evidence tail" if enable_adaptive_gene_selection else "all hard-kept genes selected",
            xy=(arrow_x, arrow_y),
            xytext=(max(1, int(scores.size * 0.14)), min(1.0, arrow_y + 0.16)),
            arrowprops={"arrowstyle": "->", "color": "#c43d0b", "lw": 1.3},
            color="#a83205",
            fontsize=10,
        )
    ax.axhline(median, color="#6b7280", linestyle="--", linewidth=1.1, label=f"median = {median:.3f}")
    ax.axhline(q1, color="#9aa4b2", linestyle=":", linewidth=1.1, label=f"Q1 = {q1:.3f}")
    ax.axhline(q3, color="#9aa4b2", linestyle=":", linewidth=1.1, label=f"Q3 = {q3:.3f}")
    if enable_adaptive_gene_selection:
        label = "median + IQR" if np.isclose(multiplier, 1.0) else f"median + {multiplier:g} IQR"
        ax.axhline(cutoff, color="#c43d0b", linewidth=1.8, label=f"cutoff = {label} = {cutoff:.3f}")
        title = "Adaptive RSP Decomposition Gene Selection"
    else:
        title = "RSP Decomposition Gene Selection: Adaptive Step Disabled"
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel("Kept genes sorted by gene_evidence_score")
    ax.set_ylabel("gene_evidence_score")
    ax.set_ylim(bottom=max(0.0, min(float(scores.min()), float(q1)) - 0.03), top=min(1.03, max(1.0, float(scores.max()) + 0.03)))
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save_figure(fig, out_path)


def _safe_int(value, default: int) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _meta_value(meta, key: str):
    if isinstance(meta, pd.Series):
        return meta.get(key, np.nan)
    if isinstance(meta, dict):
        return meta.get(key, np.nan)
    return np.nan


def _adaptive_iqr_multiplier_from_manifest(rsp_out: Path) -> float:
    manifest_path = rsp_out / "manifest.json"
    if not manifest_path.exists():
        return 0.5
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return float(manifest.get("params", {}).get("discovery", {}).get("adaptive_iqr_multiplier", 0.5))
    except Exception:
        return 0.5


def _enable_adaptive_gene_selection_from_manifest(rsp_out: Path) -> bool:
    manifest_path = rsp_out / "manifest.json"
    if not manifest_path.exists():
        return True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return bool(manifest.get("params", {}).get("discovery", {}).get("enable_adaptive_gene_selection", True))
    except Exception:
        return True


def visualize_sample(sample_dir: Path, out_dir: Path | None = None, max_rsps: int | None = None) -> list[Path]:
    rsp_out = sample_dir / "rsp_out"
    rsp_path = rsp_out / "rsp.parquet"
    activation_path = rsp_out / "rsp_activation.parquet"
    loadings_path = rsp_out / "rsp_gene_loadings.parquet"
    core_loadings_path = rsp_out / "rsp_core_gene_loadings.parquet"
    gene_filter_path = rsp_out / "gene_filter_metrics.parquet"
    for path in [rsp_path, activation_path, loadings_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing RSP table: {path}")

    h5ad_path = sample_dir_to_h5ad_path(sample_dir)
    coords = load_spatial_coords(h5ad_path)
    rsp = pd.read_parquet(rsp_path)
    activation = pd.read_parquet(activation_path)
    full_loadings = pd.read_parquet(loadings_path)
    loadings = pd.read_parquet(core_loadings_path if core_loadings_path.exists() else loadings_path)
    rsp_ids = _selected_rsps(rsp, max_rsps)

    final_out = ensure_out_dir(out_dir or sample_dir / "vis" / "rsp")
    outputs = [
        final_out / "fig00_tissue_image.png",
        final_out / "fig01_rsp_activation_atlas.png",
        final_out / "fig02_rsp_top_loading_genes.png",
        final_out / "fig03_rsp_score_summary.png",
        final_out / "fig04_rsp_footprint_contour_panels_on_tissue.png",
        final_out / "fig05_rsp_core_score_elbow_curves.png",
        final_out / "fig06_rsp_decomposition_gene_selection.png",
    ]
    plot_tissue_image(h5ad_path, coords, outputs[0])
    plot_rsp_activation_atlas(activation, coords, rsp_ids, outputs[1])
    plot_rsp_top_loadings(loadings, rsp_ids, outputs[2])
    plot_rsp_score_summary(rsp, outputs[3])
    plot_rsp_contour_panels_on_tissue(activation, coords, rsp_ids, h5ad_path, outputs[4])
    plot_core_score_elbow_curves(rsp, full_loadings, rsp_ids, outputs[5])
    if gene_filter_path.exists():
        gene_filter = pd.read_parquet(gene_filter_path)
        plot_gene_selection_curve(
            gene_filter,
            outputs[6],
            adaptive_iqr_multiplier=_adaptive_iqr_multiplier_from_manifest(rsp_out),
            enable_adaptive_gene_selection=_enable_adaptive_gene_selection_from_manifest(rsp_out),
        )
    outputs = [path for path in outputs if path.exists()]
    write_manifest(final_out / "visualization_manifest.csv", outputs, {"object": "RSP", "sample_dir": str(sample_dir)})
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Regional Specificity Program (RSP) objects.")
    parser.add_argument("--dataset", default="WU", help="Dataset id from utils.dataset_registry.")
    parser.add_argument("--sample", default=None, help="Single sample id. Defaults to all samples in dataset.")
    parser.add_argument("--sample-dir", default=None, help="Explicit sample directory.")
    parser.add_argument("--out-dir", default=None, help="Explicit output directory.")
    parser.add_argument("--max-rsps", type=int, default=None, help="Limit number of RSPs shown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_dir:
        sample_dirs = [Path(args.sample_dir)]
    else:
        dataset = get_dataset(args.dataset)
        sample_ids = [args.sample] if args.sample else dataset.sample_ids
        sample_dirs = [dataset.sample_dir(str(sample_id)) for sample_id in sample_ids]
    for sample_dir in sample_dirs:
        outputs = visualize_sample(sample_dir, Path(args.out_dir) if args.out_dir else None, max_rsps=args.max_rsps)
        print(f"[RSP] {sample_dir.name}: wrote {len(outputs)} figures to {outputs[0].parent}")


if __name__ == "__main__":
    main()
