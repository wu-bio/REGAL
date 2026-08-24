from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/regal_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_spatial_coords(h5ad_path: Path, spatial_obsm_key: str = "spatial") -> pd.DataFrame:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("anndata is required to load spatial coordinates from h5ad files.") from exc

    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    if spatial_obsm_key not in adata.obsm:
        raise KeyError(f"Missing obsm[{spatial_obsm_key!r}] in {h5ad_path}")
    coords = np.asarray(adata.obsm[spatial_obsm_key])
    if coords.shape[1] < 2:
        raise ValueError(f"Spatial coordinates must have at least two columns: {h5ad_path}")
    return pd.DataFrame(
        {
            "spot_id": adata.obs_names.astype(str),
            "x": coords[:, 0].astype(float),
            "y": coords[:, 1].astype(float),
        }
    )


def sample_dir_to_h5ad_path(sample_dir: Path) -> Path:
    return sample_dir.parent / f"{sample_dir.name}.h5ad"


def default_visualization_dir(sample_dir: Path, object_name: str) -> Path:
    return sample_dir / "visualization" / object_name


def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ranked_grid(n_items: int, max_cols: int = 4) -> tuple[int, int]:
    if n_items <= 0:
        return 1, 1
    cols = min(max_cols, max(1, math.ceil(math.sqrt(n_items))))
    rows = math.ceil(n_items / cols)
    return rows, cols


def spatial_scatter(
    ax,
    coords: pd.DataFrame,
    values: pd.Series | np.ndarray | None = None,
    *,
    cmap: str = "magma",
    background_color: str = "#dddddd",
    point_size: float = 5.0,
    alpha: float = 1.0,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    ax.scatter(
        coords["x"],
        coords["y"],
        s=point_size,
        c=background_color,
        linewidths=0,
        alpha=0.55,
    )
    handle = None
    if values is not None:
        arr = np.asarray(values, dtype=float)
        mask = np.isfinite(arr)
        if mask.any():
            handle = ax.scatter(
                coords.loc[mask, "x"],
                coords.loc[mask, "y"],
                s=point_size,
                c=arr[mask],
                cmap=cmap,
                linewidths=0,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return handle


def save_figure(fig, path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_manifest(path: Path, outputs: Iterable[Path], metadata: dict) -> None:
    rows = []
    for output in outputs:
        rows.append({"file": str(output), **metadata})
    pd.DataFrame(rows).to_csv(path, index=False)
