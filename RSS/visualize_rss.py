from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/regal_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utils.dataset_registry import get_dataset
from Utils.spatial_plotting import (
    ensure_out_dir,
    load_spatial_coords,
    ranked_grid,
    sample_dir_to_h5ad_path,
    save_figure,
    spatial_scatter,
    write_manifest,
)


def default_rss_visualization_dir(sample_dir: Path) -> Path:
    return sample_dir / "vis" / "rss"


def _load_annotation(h5ad_path: Path, annotation_col: str | None) -> pd.DataFrame | None:
    if not annotation_col:
        return None
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("anndata is required to load spot annotations from h5ad files.") from exc

    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    if annotation_col not in adata.obs.columns:
        return None
    labels = adata.obs[annotation_col].astype("string").fillna("Unannotated")
    labels = labels.replace({"": "Unannotated", "nan": "Unannotated", "None": "Unannotated"})
    return pd.DataFrame(
        {
            "spot_id": adata.obs_names.astype(str),
            "annotation": labels.to_numpy(dtype=str),
        }
    )


def _load_raw_expression(h5ad_path: Path, genes: list[str], layer: str = "X") -> pd.DataFrame:
    try:
        import anndata as ad
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("anndata and scipy are required to plot raw expression maps.") from exc

    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    var_names = pd.Index(adata.var_names.astype(str))
    matched = [gene for gene in genes if gene in var_names]
    if not matched:
        return pd.DataFrame({"spot_id": adata.obs_names.astype(str)})
    gene_positions = [(gene, int(var_names.get_loc(gene))) for gene in matched]
    index_to_gene = {idx: gene for gene, idx in gene_positions}
    read_indices = sorted(index_to_gene)
    matrix = adata.X if layer == "X" else adata.layers[layer]
    sub = matrix[:, read_indices]
    if sp.issparse(sub):
        arr = sub.toarray()
    else:
        arr = np.asarray(sub)
    out = pd.DataFrame(arr, columns=[index_to_gene[idx] for idx in read_indices])
    out = out.loc[:, matched]
    out.insert(0, "spot_id", adata.obs_names.astype(str))
    return out


def _top_rss_genes(rss: pd.DataFrame, n: int) -> list[str]:
    summary = (
        rss.groupby("gene", as_index=False)
        .agg(max_rss=("rss_score", "max"), mean_rss=("rss_score", "mean"), n_spots=("spot_id", "nunique"))
        .sort_values(["max_rss", "mean_rss", "n_spots"], ascending=False)
    )
    return summary.head(n)["gene"].astype(str).tolist()


def plot_raw_vs_rss_atlas(
    rss: pd.DataFrame,
    raw_expr: pd.DataFrame,
    coords: pd.DataFrame,
    genes: list[str],
    out_path: Path,
) -> None:
    n = len(genes)
    fig, axes = plt.subplots(n, 2, figsize=(7.4, 2.6 * n), squeeze=False)
    coord_index = coords.set_index("spot_id")
    raw_indexed = raw_expr.set_index("spot_id")
    for row, gene in enumerate(genes):
        raw_values = pd.Series(0.0, index=coord_index.index)
        if gene in raw_indexed.columns:
            raw_values.loc[raw_indexed.index.astype(str)] = raw_indexed[gene].astype(float).to_numpy()
        raw_plot = np.log1p(raw_values.to_numpy())
        handle = spatial_scatter(
            axes[row, 0],
            coords,
            raw_plot,
            cmap="Blues",
            point_size=5.0,
            title=f"{gene} raw expression",
        )
        if handle is not None:
            fig.colorbar(handle, ax=axes[row, 0], fraction=0.046, pad=0.02)

        rss_values = pd.Series(0.0, index=coord_index.index)
        sub = rss.loc[rss["gene"].eq(gene), ["spot_id", "rss_score"]]
        rss_values.loc[sub["spot_id"].astype(str)] = sub["rss_score"].astype(float).to_numpy()
        handle = spatial_scatter(
            axes[row, 1],
            coords,
            rss_values.to_numpy(),
            cmap="viridis",
            point_size=5.0,
            title=f"{gene} RSS evidence",
        )
        if handle is not None:
            fig.colorbar(handle, ax=axes[row, 1], fraction=0.046, pad=0.02)
    fig.suptitle("Raw expression vs RSS local evidence", x=0.02, ha="left", fontsize=12, fontweight="bold")
    save_figure(fig, out_path)


def plot_annotation_panels(
    annotation: pd.DataFrame | None,
    coords: pd.DataFrame,
    annotation_col: str | None,
    out_path: Path,
) -> None:
    if annotation is None or annotation.empty:
        return
    df = coords.merge(annotation, on="spot_id", how="left")
    df["annotation"] = df["annotation"].fillna("Unannotated").astype(str)
    counts = df["annotation"].value_counts()
    labels = counts.index.tolist()
    rows, cols = ranked_grid(len(labels), max_cols=4)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.25 * rows), squeeze=False)
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(labels), 1)))
    for ax, label, color in zip(axes.ravel(), labels, palette):
        mask = df["annotation"].eq(label)
        ax.scatter(df["x"], df["y"], s=3.0, c="#d0d0d0", linewidths=0, alpha=0.55)
        ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"], s=9.0, c=[color], linewidths=0, alpha=0.98)
        ax.set_title(f"{label} ({int(mask.sum())})", loc="left", fontsize=8, fontweight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(labels) :]:
        ax.axis("off")
    fig.suptitle(f"Spot annotation panels: {annotation_col}", x=0.02, ha="left", fontsize=12, fontweight="bold")
    save_figure(fig, out_path)


def plot_rss_gene_atlas(rss: pd.DataFrame, coords: pd.DataFrame, genes: list[str], out_path: Path) -> None:
    rows, cols = ranked_grid(len(genes), max_cols=4)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    coord_index = coords.set_index("spot_id")
    for ax, gene in zip(axes.ravel(), genes):
        values = pd.Series(0.0, index=coord_index.index)
        sub = rss.loc[rss["gene"].eq(gene), ["spot_id", "rss_score"]]
        values.loc[sub["spot_id"].astype(str)] = sub["rss_score"].astype(float).to_numpy()
        handle = spatial_scatter(ax, coords, values.to_numpy(), cmap="viridis", point_size=5.0, title=gene)
        if handle is not None:
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.ravel()[len(genes) :]:
        ax.axis("off")
    fig.suptitle("RSS evidence maps for top genes", x=0.02, ha="left", fontsize=12, fontweight="bold")
    save_figure(fig, out_path)


def plot_rss_matrix_snapshot(rss: pd.DataFrame, genes: list[str], out_path: Path, max_spots: int = 300) -> None:
    sub = rss.loc[rss["gene"].isin(genes), ["spot_id", "gene", "rss_score"]].copy()
    spot_order = (
        sub.groupby("spot_id")["rss_score"].max().sort_values(ascending=False).head(max_spots).index.astype(str)
    )
    mat = (
        sub[sub["spot_id"].astype(str).isin(spot_order)]
        .pivot_table(index="spot_id", columns="gene", values="rss_score", fill_value=0.0, aggfunc="max")
        .reindex(index=spot_order, columns=genes)
    )
    fig, ax = plt.subplots(figsize=(max(6, len(genes) * 0.35), 5.5))
    im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("RSS matrix snapshot (spots x top genes)", loc="left", fontweight="bold")
    ax.set_xlabel("genes")
    ax.set_ylabel("top RSS spots")
    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=7)
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="RSS")
    save_figure(fig, out_path)


def visualize_sample(
    sample_dir: Path,
    out_dir: Path | None = None,
    top_n: int = 12,
    annotation_col: str | None = None,
) -> list[Path]:
    rss_out = sample_dir / "rss_out"
    rss_path = rss_out / "rss" / "rss_sparse.parquet"
    if not rss_path.exists():
        raise FileNotFoundError(f"Missing RSS sparse table: {rss_path}")
    h5ad_path = sample_dir_to_h5ad_path(sample_dir)
    coords = load_spatial_coords(h5ad_path)
    rss = pd.read_parquet(rss_path)
    rss["spot_id"] = rss["spot_id"].astype(str)
    rss["gene"] = rss["gene"].astype(str)
    genes = _top_rss_genes(rss, top_n)
    raw_expr = _load_raw_expression(h5ad_path, genes[: min(top_n, 6)])
    annotation = _load_annotation(h5ad_path, annotation_col)
    final_out = ensure_out_dir(out_dir or default_rss_visualization_dir(sample_dir))

    outputs = [
        final_out / "fig00_spot_annotation_panels.png",
        final_out / "fig01_raw_expression_vs_rss_evidence.png",
        final_out / "fig02_rss_top_gene_atlas.png",
        final_out / "fig03_rss_matrix_snapshot.png",
    ]
    plot_annotation_panels(annotation, coords, annotation_col, outputs[0])
    plot_raw_vs_rss_atlas(rss, raw_expr, coords, genes[: min(top_n, 6)], outputs[1])
    plot_rss_gene_atlas(rss, coords, genes[: min(top_n, 12)], outputs[2])
    plot_rss_matrix_snapshot(rss, genes[: min(top_n, 20)], outputs[3])
    outputs = [path for path in outputs if path.exists()]
    write_manifest(final_out / "visualization_manifest.csv", outputs, {"object": "RSS", "sample_dir": str(sample_dir)})
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Regional Specificity Score (RSS) objects.")
    parser.add_argument("--dataset", default="WU", help="Dataset id from Utils.dataset_registry.")
    parser.add_argument("--sample", default=None, help="Single sample id. Defaults to all samples in dataset.")
    parser.add_argument("--sample-dir", default=None, help="Explicit sample directory.")
    parser.add_argument("--out-dir", default=None, help="Explicit output directory.")
    parser.add_argument("--top-n", type=int, default=12, help="Number of top RSS genes to visualize.")
    parser.add_argument("--annotation-col", default=None, help="Observation column to show as spot annotations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_dir:
        sample_dirs = [Path(args.sample_dir)]
        annotation_col = args.annotation_col
    else:
        dataset = get_dataset(args.dataset)
        sample_ids = [args.sample] if args.sample else dataset.sample_ids
        sample_dirs = [dataset.sample_dir(str(sample_id)) for sample_id in sample_ids]
        annotation_col = args.annotation_col or dataset.annotation_col
    for sample_dir in sample_dirs:
        outputs = visualize_sample(
            sample_dir,
            Path(args.out_dir) if args.out_dir else None,
            top_n=args.top_n,
            annotation_col=annotation_col,
        )
        print(f"[RSS] {sample_dir.name}: wrote {len(outputs)} figures to {outputs[0].parent}")


if __name__ == "__main__":
    main()
