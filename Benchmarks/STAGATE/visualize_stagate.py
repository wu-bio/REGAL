#!/usr/bin/env python
"""Visualize STAGATE spatial domains and the available manual reference annotation."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "stagate_visualization_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULT_DIR = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/STAGATE"
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize STAGATE domain output.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULT_DIR / "stagate.h5ad")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR / "vis")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--domain-key", default="STAGATE_domain")
    parser.add_argument("--annotation-key", default="manual_annotation")
    parser.add_argument("--point-size", type=float, default=8.0)
    args = parser.parse_args()
    if args.point_size <= 0:
        parser.error("--point-size must be positive.")
    return args


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def natural_order(values: pd.Series) -> list[str]:
    labels = values.astype(str).unique().tolist()
    return sorted(labels, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def categorical_colors(values: pd.Series) -> tuple[pd.Categorical, dict[str, object]]:
    categories = pd.Categorical(values.astype(str), categories=natural_order(values))
    palette = mpl.colormaps["tab20"].resampled(max(len(categories.categories), 1))
    colors = {label: palette(index) for index, label in enumerate(categories.categories)}
    return categories, colors


def plot_category_map(
    axis: mpl.axes.Axes,
    coordinates: np.ndarray,
    values: pd.Series,
    title: str,
    point_size: float,
) -> None:
    categories, color_map = categorical_colors(values)
    for label in categories.categories:
        mask = np.asarray(categories == label)
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=point_size,
            color=color_map[label],
            label=label,
            linewidths=0,
            rasterized=True,
        )
    axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(title, loc="left", fontweight="bold")
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        markerscale=1.8,
        handletextpad=0.25,
        borderaxespad=0,
        fontsize=6,
    )


def plot_overview(
    adata: ad.AnnData,
    coordinates: np.ndarray,
    domain_key: str,
    annotation_key: str,
    point_size: float,
) -> mpl.figure.Figure:
    has_annotation = annotation_key in adata.obs
    ncols = 2 if has_annotation else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.3), squeeze=False)
    if has_annotation:
        plot_category_map(
            axes[0, 0], coordinates, adata.obs[annotation_key], "Manual region annotation", point_size
        )
        domain_axis = axes[0, 1]
    else:
        domain_axis = axes[0, 0]
    plot_category_map(domain_axis, coordinates, adata.obs[domain_key], "STAGATE domains", point_size)
    fig.tight_layout(w_pad=5.5)
    return fig


def plot_domain_sizes(values: pd.Series) -> mpl.figure.Figure:
    counts = values.astype(str).value_counts()
    counts = counts.reindex(natural_order(values))
    fig, axis = plt.subplots(figsize=(max(5.0, 0.32 * len(counts)), 2.7))
    axis.bar(counts.index, counts.values, color="#4C78A8", width=0.75)
    axis.set(xlabel="STAGATE domain", ylabel="Spots")
    axis.set_title("STAGATE domain sizes", loc="left", fontweight="bold")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"STAGATE h5ad does not exist: {args.input}")
    configure_style()
    adata = ad.read_h5ad(args.input)
    if args.spatial_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{args.spatial_key!r}].")
    if args.domain_key not in adata.obs:
        raise KeyError(f"Missing adata.obs[{args.domain_key!r}].")
    coordinates = np.asarray(adata.obsm[args.spatial_key])[:, :2]
    if coordinates.shape != (adata.n_obs, 2):
        raise ValueError(f"Invalid spatial coordinates: {coordinates.shape}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        plot_overview(adata, coordinates, args.domain_key, args.annotation_key, args.point_size),
        args.output_dir / "stagate_spatial_overview",
    )
    save_figure(plot_domain_sizes(adata.obs[args.domain_key]), args.output_dir / "stagate_domain_sizes")
    print(f"Saved STAGATE visualizations to {args.output_dir}.")


if __name__ == "__main__":
    main()
