#!/usr/bin/env python
"""Visualize native continuous STAMP topics and their global gene modules."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "stamp_visualization_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
)
DEFAULT_RESULT_DIR = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/STAMP/K20"
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
    parser = argparse.ArgumentParser(description="Visualize STAMP topic outputs without hard assignment.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--topics", type=Path, default=DEFAULT_RESULT_DIR / "stamp_spot_topics.csv")
    parser.add_argument("--modules", type=Path, default=DEFAULT_RESULT_DIR / "stamp_gene_modules.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR / "vis")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--top-genes", type=int, default=3)
    parser.add_argument("--point-size", type=float, default=7.0)
    args = parser.parse_args()
    if args.top_genes < 1 or args.point_size <= 0:
        parser.error("--top-genes and --point-size must be positive.")
    return args


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_aligned(
    input_path: Path, topics_path: Path, modules_path: Path, spatial_key: str
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame, np.ndarray]:
    adata = ad.read_h5ad(input_path, backed="r")
    topics = pd.read_csv(topics_path, index_col=0)
    modules = pd.read_csv(modules_path, index_col=0)
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{spatial_key!r}].")
    missing = adata.obs_names.difference(topics.index)
    if len(missing):
        raise ValueError(f"STAMP topics are missing {len(missing)} spots.")
    if set(topics.columns) != set(modules.columns):
        raise ValueError("STAMP topic and gene-module names do not match.")
    topics = topics.loc[adata.obs_names, modules.columns]
    coordinates = np.asarray(adata.obsm[spatial_key])[:, :2]
    return adata, topics, modules, coordinates


def top_gene_table(modules: pd.DataFrame, n_top: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for topic in modules.columns:
        ranked = modules[topic].sort_values(ascending=False).head(n_top)
        for rank, (gene, weight) in enumerate(ranked.items(), start=1):
            records.append({"topic": topic, "rank": rank, "gene": gene, "weight": float(weight)})
    return pd.DataFrame(records)


def plot_topic_maps(
    coordinates: np.ndarray,
    topics: pd.DataFrame,
    top_genes: pd.DataFrame,
    point_size: float,
) -> mpl.figure.Figure:
    ncols = min(5, topics.shape[1])
    nrows = int(np.ceil(topics.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.1 * nrows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= topics.shape[1]:
            continue
        topic = topics.columns[index]
        values = topics[topic].to_numpy(dtype=float)
        vmax = max(float(np.nanquantile(values, 0.99)), 1e-12)
        genes = top_genes.loc[top_genes["topic"] == topic, "gene"].astype(str).tolist()
        axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=values, s=point_size, cmap="magma",
            vmin=0, vmax=vmax, linewidths=0, rasterized=True
        )
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_title(f"{topic}\n{', '.join(genes[:2])}", fontsize=6.7, pad=1.5)
    fig.suptitle("STAMP continuous topic proportions (each panel scaled to its 99th percentile)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), pad=0.25)
    return fig


def plot_module_heatmap(modules: pd.DataFrame, top_genes: pd.DataFrame) -> mpl.figure.Figure:
    genes = list(dict.fromkeys(top_genes["gene"].astype(str)))
    matrix = modules.loc[genes].to_numpy(dtype=float)
    limit = max(float(np.nanquantile(np.abs(matrix), 0.98)), 1e-12)
    fig, axis = plt.subplots(figsize=(max(7.0, 0.34 * modules.shape[1]), max(4.5, 0.15 * len(genes))))
    image = axis.imshow(
        matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    )
    axis.set_xticks(np.arange(modules.shape[1]), modules.columns, rotation=90)
    axis.set_yticks(np.arange(len(genes)), genes)
    axis.set(xlabel="STAMP topic", ylabel="Top-weight genes")
    axis.set_title("STAMP global gene-module weights", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.015, label="Module weight")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    for path in (args.input, args.topics, args.modules):
        if not path.is_file():
            raise FileNotFoundError(path)
    configure_style()
    adata, topics, modules, coordinates = load_aligned(
        args.input, args.topics, args.modules, args.spatial_key
    )
    top_genes = top_gene_table(modules, args.top_genes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        plot_topic_maps(coordinates, topics, top_genes, args.point_size),
        args.output_dir / "stamp_topic_maps",
    )
    save_figure(
        plot_module_heatmap(modules, top_genes),
        args.output_dir / "stamp_gene_modules",
    )
    top_genes.to_csv(args.output_dir / "stamp_top_genes.csv", index=False)
    if adata.isbacked:
        adata.file.close()
    print(f"Saved STAMP visualizations for {topics.shape[1]} topics to {args.output_dir}.")


if __name__ == "__main__":
    main()
