#!/usr/bin/env python
"""Visualize retained programs from the complete RSS-front-end ablation."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


RUNTIME_CACHE = Path(tempfile.gettempdir()) / "regal_rss_ablation_vis_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
)
DEFAULT_BUNDLE = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/REGAL-RSS/rsp_out"
)
DEFAULT_OUTPUT = DEFAULT_BUNDLE.parent / "vis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_coordinates(path: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(path, backed="r")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    table = pd.DataFrame(coords, index=adata.obs_names.astype(str), columns=["x", "y"])
    if adata.file is not None:
        adata.file.close()
    return table


def top_gene_labels(loadings: pd.DataFrame, n: int = 3) -> dict[str, str]:
    ranked = loadings.sort_values(["rsp_id", "loading_rank"])
    top = ranked.groupby("rsp_id", observed=True, group_keys=False).head(n)
    return top.groupby("rsp_id", observed=True)["gene"].agg(
        lambda values: ", ".join(values.astype(str))
    ).to_dict()


def main() -> None:
    args = parse_args()
    for path in (
        args.input,
        args.bundle / "rsp.parquet",
        args.bundle / "rsp_activation.parquet",
        args.bundle / "rsp_gene_loadings.parquet",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.titlesize": 7,
        }
    )
    coords = load_coordinates(args.input)
    rsp = pd.read_parquet(args.bundle / "rsp.parquet").sort_values(
        ["source_k", "rsp_score"], ascending=[True, False]
    )
    activation = pd.read_parquet(args.bundle / "rsp_activation.parquet")
    loadings = pd.read_parquet(args.bundle / "rsp_gene_loadings.parquet")
    genes = top_gene_labels(loadings)

    n_cols = 5
    n_rows = int(np.ceil(len(rsp) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.05 * n_cols, 2.15 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = mpl.colormaps["magma"]
    for axis, record in zip(axes.ravel(), rsp.to_dict("records")):
        rsp_id = str(record["rsp_id"])
        values = activation.loc[
            activation["rsp_id"].astype(str).eq(rsp_id),
            ["spot_id", "activation", "activation_active"],
        ].copy()
        values["spot_id"] = values["spot_id"].astype(str)
        values = values.set_index("spot_id").reindex(coords.index)
        active = values["activation_active"].fillna(False).to_numpy(dtype=bool)
        score = values["activation"].fillna(0).to_numpy(dtype=float)
        positive = score[active & np.isfinite(score)]
        vmax = float(np.quantile(positive, 0.99)) if positive.size else 1.0

        axis.scatter(coords["x"], coords["y"], s=3.8, c="#e5e7eb", linewidths=0)
        if np.any(active):
            axis.scatter(
                coords.loc[active, "x"],
                coords.loc[active, "y"],
                s=5.0,
                c=np.clip(score[active], 0, vmax),
                cmap=cmap,
                vmin=0,
                vmax=max(vmax, 1e-8),
                linewidths=0,
            )
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        active_fraction = float(record.get("activation_active_fraction", np.mean(active)))
        axis.set_title(
            f"{rsp_id} | K={int(record['source_k'])} | active={active_fraction:.0%}\n"
            f"{genes.get(rsp_id, '')}",
            loc="left",
            pad=2.5,
            fontweight="semibold",
        )

    for axis in axes.ravel()[len(rsp) :]:
        axis.axis("off")
    fig.suptitle(
        "RSPs learned from STAMP-selected HVG expression without RSS",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "no_rss_hvg_rsp_atlas.png"
    svg = args.output_dir / "no_rss_hvg_rsp_atlas.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {svg}")


if __name__ == "__main__":
    main()
