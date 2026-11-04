"""Plot manuscript fig06 using paths from the shared YAML configuration."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from typing import Iterable
from typing import Iterable, Sequence
from typing import Sequence
import argparse
from string import Template
from copy import deepcopy
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RSP.visualize_rsp import _coords_to_image_space, _estimate_spot_radius_px, _load_spatial_image_context, _orient_tissue_image
from RSP.visualize_rsp import _crop_to_spot_window
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import sparse
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from Utils.manuscript_figure_style import ANNOTATION_SIZE, AXIS_LABEL_SIZE, COLORBAR_SIZE, EXPORT_DPI, FIGURE_WIDTH_IN, GRID, GRID_LINE_WIDTH, INK, LEGEND_SIZE, OUTLINE_WIDTH, PANEL_TITLE_SIZE, TICK_LABEL_SIZE, marker_expression_cmap, rsp_activity_cmap
from Utils.manuscript_figure_style import EXPORT_DPI, GRID, INK, MID, PAPER, TICK_LABEL_SIZE, add_panel_label as add_shared_panel_label, configure_manuscript_style, rsp_activity_cmap
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import yaml

_CONFIG: dict

_CONFIG_DIR: Path
_ROOTS: dict

RESOURCE_DIR = Path(__file__).resolve().parents[1] / 'resources'
RESOURCE_PATHS = {
    'literature_anchors.yaml': RESOURCE_DIR / 'fig06_7' / 'literature_anchors.yaml',
    'c2.cp.reactome.v2026.1.Hs.symbols.gmt': RESOURCE_DIR / 'shared' / 'c2.cp.reactome.v2026.1.Hs.symbols.gmt',
    'BioPlanet_2019.Enrichr.gmt': RESOURCE_DIR / 'shared' / 'BioPlanet_2019.Enrichr.gmt',
    'h.all.v2026.1.Hs.symbols.gmt': RESOURCE_DIR / 'fig06_7' / 'h.all.v2026.1.Hs.symbols.gmt',
    'scffpe_markers': RESOURCE_DIR / 'shared' / 'scffpe_markers.csv',
}

FIGURE_SELECTION = {'selected_rsp_objects': [{'rsp_id': 'RSP0005',
                           'block': 'dcis1_multilayer',
                           'role': 'dcis1_layer_a',
                           'description': 'A relatively coherent epithelial-associated layer in '
                                          'DCIS#1, representing one component of its multilayer '
                                          'co-expression realization.',
                           'continuity_old_rsp': 'RSP0004',
                           'domain_mode': 'selected',
                           'selected_domains': ['D000028', 'D000029'],
                           'anchor': 'DCIS_1'},
                          {'rsp_id': 'RSP0019',
                           'block': 'dcis1_multilayer',
                           'role': 'dcis1_layer_b',
                           'description': 'Co-localizes with RSP0005 in DCIS#1 but complements it '
                                          'through distinct local spatial placements and '
                                          'morphology.',
                           'continuity_old_rsp': 'RSP0009',
                           'domain_mode': 'selected',
                           'selected_domains': ['D000094', 'D000095', 'D000096'],
                           'anchor': 'DCIS_1'},
                          {'rsp_id': 'RSP0012',
                           'block': 'dcis2_structural_scale',
                           'role': 'concentrated_dcis2_epithelial',
                           'description': 'A concentrated, epithelial-dominant DCIS#2 layer '
                                          'anchored by large DCIS_2 cell structures.',
                           'continuity_old_rsp': None,
                           'domain_mode': 'selected',
                           'selected_domains': ['D000052', 'D000048', 'D000049'],
                           'analysis_scope': 'selected_rsd_with_xenium_background',
                           'anchor': 'DCIS_2'},
                          {'rsp_id': 'RSP0018',
                           'block': 'dcis2_structural_scale',
                           'role': 'myoepithelial_with_discrete_dcis2',
                           'description': 'A broader basal/myoepithelial layer formed around '
                                          'lower-concentration, multiple or small DCIS_2 cell '
                                          'structures.',
                           'continuity_old_rsp': 'RSP0006',
                           'domain_mode': 'selected',
                           'selected_domains': ['D000093', 'D000089', 'D000087'],
                           'analysis_scope': 'selected_rsd_with_xenium_background',
                           'anchor': 'DCIS_2'},
                          {'rsp_id': 'RSP0001',
                           'block': 'invasive_overlap',
                           'role': 'broad_core_invasive',
                           'description': 'A centrally enhanced main layer within the selected '
                                          'invasive tumor landscape.',
                           'continuity_old_rsp': 'RSP0002',
                           'domain_mode': 'selected',
                           'selected_domains': ['D000005'],
                           'anchor': 'Invasive_Tumor'},
                          {'rsp_id': 'RSP0015',
                           'block': 'invasive_overlap',
                           'role': 'cdh2_high_contextual_invasive',
                           'description': 'A peripherally enhanced layer that overlaps RSP0001 '
                                          'within the same invasive tumor landscape.',
                           'continuity_old_rsp': 'RSP0007',
                           'domain_mode': 'selected',
                           'selected_domains': ['D000072'],
                           'anchor': 'Invasive_Tumor'},
                          {'rsp_id': 'RSP0004',
                           'block': 'invasive_overlap',
                           'role': 'nested_local_invasive',
                           'description': 'A locally activated layer nested at selected internal '
                                          'positions of the invasive tumor landscape.',
                           'continuity_old_rsp': 'RSP0008',
                           'domain_mode': 'selected',
                           'selected_domains': ['D000024'],
                           'anchor': 'Invasive_Tumor'},
                          {'rsp_id': 'RSP0000',
                           'block': 'local_carrier_disruption',
                           'role': 'broad_carrier_with_local_depression',
                           'description': 'A broad carrier layer containing a focal low-activity '
                                          'or partial-membership gap at the scale of the '
                                          'triple-receptor island, linking the layer to local '
                                          'pathological modulation.',
                           'continuity_old_rsp': 'RSP0000',
                           'domain_mode': 'selected',
                           'selected_domains': ['D000001'],
                           'cavity_spots': ['AATGAATCGGAAGGCA-1',
                                            'GTACGATTAGTCTACA-1',
                                            'TGCGCTCATAGAACGA-1'],
                           'anchor': 'triple_receptor_island'}]}

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig06'

FIGURE_STEM: Path

H5AD_PATH: Path

MARKERS_PATH: Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'output' / 'figures'

SAMPLE_DIR: Path

SAMPLE_ID: str


XENIUM_MAPPING_PATH: Path

def load_selection():
    selection = deepcopy(FIGURE_SELECTION)
    selection['sample_id'] = option('sample_id')
    selection['analysis_parameters'] = option('analysis_parameters')
    return selection

def path(name: str) -> Path:
    value = _CONFIG['paths'][name]
    return _resolve(value)

def option(name: str):
    return _CONFIG['options'][name]

def _resolve(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Fill in every required path in the configuration.')
    result = Path(Template(value).substitute(_ROOTS)).expanduser()
    return result if result.is_absolute() else _CONFIG_DIR / result

def table(name: str) -> Path:
    return TABLE_DIR / name

def resource(name: str) -> Path:
    return RESOURCE_PATHS[name]


DEFAULT_OBS_COLUMNS = (
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
    "janesick_cluster",
    "janesick_annotation",
    "sample_id",
    "library_id",
    "total_counts",
)

def load_spot_context(
    h5ad_path: str | Path,
    *,
    sample_id: str,
    obs_columns: Iterable[str] = DEFAULT_OBS_COLUMNS,
    spatial_key: str = "spatial",
) -> pd.DataFrame:
    """Read Visium coordinates and selected observation columns."""

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        out = pd.DataFrame({"spot_id": adata.obs_names.astype(str), "sample_id": sample_id})
        if spatial_key not in adata.obsm:
            raise ValueError(f"Missing {spatial_key!r} coordinates in {h5ad_path}")
        spatial = np.asarray(adata.obsm[spatial_key])[:, :2].astype(float)
        out["x_fullres"] = spatial[:, 0]
        out["y_fullres"] = spatial[:, 1]

        scale = _hires_scale(adata)
        out["x_hires"] = out["x_fullres"] * scale
        out["y_hires"] = out["y_fullres"] * scale

        for col in obs_columns:
            if col not in adata.obs or col in out:
                continue
            values = adata.obs[col]
            if pd.api.types.is_numeric_dtype(values):
                out[col] = pd.to_numeric(values, errors="coerce").to_numpy()
            else:
                out[col] = values.astype(str).to_numpy()
    finally:
        adata.file.close()
    return out

def load_normalized_expression(
    h5ad_path: str | Path,
    genes: Iterable[str],
    *,
    target_sum: float = 1.0e4,
    layer: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return log1p library-normalized expression for selected genes."""

    requested = list(dict.fromkeys(str(g) for g in genes if str(g)))
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        selected = [gene for gene in requested if gene in adata.var_names]
        sub = adata[:, selected].to_memory()
        matrix = sub.layers[layer] if layer and layer in sub.layers else sub.X
        values = _dense(matrix).astype(float)
        totals = _row_totals(adata)
        totals = np.maximum(totals.astype(float), 1.0)
        values = np.log1p(values / totals[:, None] * float(target_sum))
        expr = pd.DataFrame(values, index=sub.obs_names.astype(str), columns=selected)
    finally:
        adata.file.close()
    return expr, selected

def _hires_scale(adata: ad.AnnData) -> float:
    registration = adata.uns.get("janesick_registration", {})
    scalefactors = registration.get("visium_scalefactors", {})
    if "tissue_hires_scalef" in scalefactors:
        return float(scalefactors["tissue_hires_scalef"])
    spatial = adata.uns.get("spatial", {})
    if spatial:
        library = next(iter(spatial.values()))
        return float(library.get("scalefactors", {}).get("tissue_hires_scalef", 1.0))
    return 1.0

def _dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
        if sparse.issparse(matrix):
            return matrix.toarray()
    return np.asarray(matrix)

def _row_totals(adata: ad.AnnData) -> np.ndarray:
    if "total_counts" in adata.obs:
        return pd.to_numeric(adata.obs["total_counts"], errors="coerce").fillna(0).to_numpy(float)
    matrix = adata.X
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
        if sparse.issparse(matrix):
            return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    return np.asarray(matrix).sum(axis=1).astype(float)

REGAL_COLORS = {
    "blue": "#3676B8",
    "magenta": "#C65A7A",
    "gold": "#D6A63A",
    "teal": "#2A9D8F",
    "red": "#D9545D",
    "violet": "#665191",
    "warm": "#9E7467",
    "ink": INK,
    "mid": MID,
    "light": GRID,
    "paper": PAPER,
}

def configure_figure_style() -> None:
    """Apply the shared R1-R5 manuscript style."""

    configure_manuscript_style()

def load_spatial_image_context(
    h5ad_path: str | Path,
    *,
    sample_id: str,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Load an oriented H&E image and Visium coordinates in image pixels."""

    h5ad_path = Path(h5ad_path)
    coords = load_spot_context(h5ad_path, sample_id=sample_id)
    coords["spot_id"] = coords["spot_id"].astype(str)
    plot_coords = coords.assign(x=coords["x_fullres"], y=coords["y_fullres"])
    context = _load_spatial_image_context(h5ad_path, preferred_image="hires")
    if context is None:
        raise ValueError(f"Missing spatial image context in {h5ad_path}")
    image, scale = context
    xy = _coords_to_image_space(plot_coords, scale)
    image = _orient_tissue_image(image, xy)
    xy = xy.join(coords.set_index("spot_id").drop(columns=["sample_id"], errors="ignore"), how="left")
    return soften_histology(image), xy, _estimate_spot_radius_px(xy)

def soften_histology(image: np.ndarray, strength: float = 0.30) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype.kind in {"u", "i"}:
        arr = arr.astype(float) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(float)
        if np.nanmax(arr) > 1.0:
            arr = arr / 255.0
    return np.clip(arr * (1.0 - strength) + strength, 0.0, 1.0)

def add_panel_label(
    container: mpl.figure.FigureBase,
    label: str,
    axis: plt.Axes,
    *,
    align_to_canvas_left: bool = False,
) -> mpl.text.Text:
    """Anchor a panel label with the shared physical-offset contract."""

    return add_shared_panel_label(
        container,
        label,
        axis.get_position(),
        align_to_canvas_left=align_to_canvas_left,
    )

def draw_histology(ax: plt.Axes, image: np.ndarray, *, alpha: float = 1.0) -> None:
    ax.imshow(image, origin="upper", alpha=alpha, interpolation="nearest")

def finish_spatial_axis(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    limits: tuple[float, float, float, float] | None = None,
) -> None:
    if limits is None:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
    else:
        x0, x1, y0, y1 = limits
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def bounds_from_points(
    points: pd.DataFrame,
    *,
    x_col: str = "x_img",
    y_col: str = "y_img",
    padding: float = 30.0,
    fixed_width: float | None = None,
    fixed_height: float | None = None,
) -> tuple[float, float, float, float]:
    local = points[[x_col, y_col]].dropna()
    if local.empty:
        raise ValueError("Cannot derive spatial bounds from an empty point table")
    cx = float((local[x_col].min() + local[x_col].max()) / 2.0)
    cy = float((local[y_col].min() + local[y_col].max()) / 2.0)
    width = float(local[x_col].max() - local[x_col].min() + 2.0 * padding)
    height = float(local[y_col].max() - local[y_col].min() + 2.0 * padding)
    width = float(fixed_width) if fixed_width is not None else width
    height = float(fixed_height) if fixed_height is not None else height
    return cx - width / 2.0, cx + width / 2.0, cy - height / 2.0, cy + height / 2.0

def add_scale_bar(
    ax: plt.Axes,
    *,
    px_per_um: float,
    length_um: float,
    color: str = INK,
    linewidth: float = 1.6,
) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    left = min(x0, x1) + width * 0.065
    bottom = max(y0, y1) - height * 0.075
    length_px = float(length_um) * float(px_per_um)
    ax.plot([left, left + length_px], [bottom, bottom], color=color, lw=linewidth, solid_capstyle="butt")
    ax.text(
        left + length_px / 2.0,
        bottom + height * 0.025,
        f"{int(length_um)} µm",
        ha="center",
        va="top",
        fontsize=TICK_LABEL_SIZE,
        color=color,
    )

def estimate_hires_px_per_um(mapping: pd.DataFrame) -> float:
    """Estimate the local affine scale between Xenium microns and H&E pixels."""

    required = ["x_um", "y_um", "x_visium_hires", "y_visium_hires"]
    local = mapping[required].dropna()
    if len(local) > 50000:
        local = local.sample(50000, random_state=13)
    design = np.column_stack([local["x_um"], local["y_um"], np.ones(len(local))])
    target = local[["x_visium_hires", "y_visium_hires"]].to_numpy(float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    transform = coefficients[:2, :].T
    singular_values = np.linalg.svd(transform, compute_uv=False)
    return float(np.mean(singular_values))

def validate_source_columns(table: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def render_row(figure: plt.Figure) -> np.ndarray:
    """Render a row in memory without writing an intermediate file."""
    figure.set_dpi(EXPORT_DPI)
    figure.set_facecolor('white')
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba()).astype(np.float32) / 255.0
    plt.close(figure)
    return image

def stack_row_images(
    images: Iterable[np.ndarray],
    *,
    gap_px: int = 24,
    vertical_padding_px: int = 18,
) -> np.ndarray:
    """Stack row images in memory using the shared R4 composite contract."""

    rows = []
    for image in images:
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected an RGB/RGBA row image, found shape={image.shape}")
        if image.shape[2] == 3:
            image = np.dstack([image, np.ones(image.shape[:2])])
        nonwhite = np.any(image[:, :, :3] < 0.995, axis=2)
        occupied = np.flatnonzero(np.any(nonwhite, axis=1))
        if occupied.size:
            top = max(0, int(occupied[0]) - vertical_padding_px)
            bottom = min(image.shape[0], int(occupied[-1]) + vertical_padding_px + 1)
            image = image[top:bottom]
        rows.append(image)

    widths = {row.shape[1] for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Composite rows must have equal widths, found {sorted(widths)}")
    width = widths.pop()
    total_height = sum(row.shape[0] for row in rows) + gap_px * (len(rows) - 1)
    canvas = np.ones((total_height, width, 4), dtype=float)
    y0 = 0
    for index, row in enumerate(rows):
        height = row.shape[0]
        canvas[y0:y0 + height] = row
        y0 += height + (gap_px if index < len(rows) - 1 else 0)
    return canvas

def save_composite_bundle(
    image: np.ndarray,
    stem: str | Path,
    *,
    dpi: int = EXPORT_DPI,
) -> list[Path]:
    """Save the complete figure as PNG."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, interpolation="nearest")
    ax.axis("off")
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return [path]

REQUIRED_RSP_FILES = {
    "rsp": "rsp.parquet",
    "activation": "rsp_activation.parquet",
    "gene_loadings": "rsp_gene_loadings.parquet",
    "core_gene_loadings": "rsp_core_gene_loadings.parquet",
}

REQUIRED_RSMD_FILES = {
    "domains": "domains.parquet",
    "membership": "domain_spot_membership.parquet",
    "domain_rsp_map": "domain_rsp_map.parquet",
    "metadomains": "metadomains.parquet",
}

SCFFPE_TO_XENIUM = {
    "B Cells": "B_Cells",
    "CD4+ T Cells": "CD4+_T_Cells",
    "CD8+ T Cells": "CD8+_T_Cells",
    "DCIS 1": "DCIS_1",
    "DCIS 2": "DCIS_2",
    "Endothelial": "Endothelial",
    "IRF7+ DCs": "IRF7+_DCs",
    "Invasive Tumor": "Invasive_Tumor",
    "LAMP3+ DCs": "LAMP3+_DCs",
    "Macrophages 1": "Macrophages_1",
    "Macrophages 2": "Macrophages_2",
    "Mast Cells": "Mast_Cells",
    "Myoepi ACTA2+": "Myoepi_ACTA2+",
    "Myoepi KRT15+": "Myoepi_KRT15+",
    "Perivascular-Like": "Perivascular-Like",
    "Prolif Invasive Tumor": "Prolif_Invasive_Tumor",
    "Stromal": "Stromal",
}

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)

def load_rsp_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(sample_dir) / "rsp_out"
    return _load_required_tables(root, REQUIRED_RSP_FILES)

def load_rsmd_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(sample_dir) / "rsmd_out"
    return _load_required_tables(root, REQUIRED_RSMD_FILES)

def load_xenium_mapping(path: str | Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    required = {"visium_barcode", "label", "in_visium_capture_area"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"Xenium mapping is missing columns: {sorted(missing)}")
    keep = mapping["in_visium_capture_area"].astype(bool) & mapping["visium_barcode"].notna()
    return mapping.loc[keep].copy()

def activation_wide(activation: pd.DataFrame, rsp_ids: Iterable[str]) -> pd.DataFrame:
    rsp_ids = list(dict.fromkeys(str(rsp) for rsp in rsp_ids))
    sub = activation.loc[activation["rsp_id"].astype(str).isin(rsp_ids)].copy()
    out = pd.DataFrame(index=sorted(sub["spot_id"].astype(str).unique()))
    out.index.name = "spot_id"
    value_map = {
        "activation": "activation",
        "activation_raw": "activation_raw",
        "activation_active": "active",
        "rank_in_spot": "rank",
    }
    for value_col, prefix in value_map.items():
        wide = sub.pivot(index="spot_id", columns="rsp_id", values=value_col)
        wide.index = wide.index.astype(str)
        for rsp_id in rsp_ids:
            values = wide[rsp_id] if rsp_id in wide else 0.0
            out[f"{prefix}_{rsp_id}"] = pd.Series(values, index=wide.index).reindex(out.index)
    activation_cols = [column for column in out if column.startswith(("activation_", "activation_raw_"))]
    active_cols = [column for column in out if column.startswith("active_")]
    out[activation_cols] = out[activation_cols].fillna(0.0).astype(float)
    out[active_cols] = out[active_cols].fillna(False).astype(bool)
    return out.reset_index()

def selected_rsd_membership(
    selection: dict[str, Any],
    domains: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domain_index = domains.set_index(["rsp_seed_id", "domain_id"], drop=False)
    long_rows: list[pd.DataFrame] = []
    registry_rows: list[dict[str, Any]] = []
    for item in selection["selected_rsp_objects"]:
        rsp_id = str(item["rsp_id"])
        for domain_id in map(str, item.get("selected_domains", [])):
            key = (rsp_id, domain_id)
            if key not in domain_index.index:
                raise ValueError(f"Selected domain does not match current artifacts: {rsp_id}:{domain_id}")
            domain = domain_index.loc[key]
            if isinstance(domain, pd.DataFrame):
                raise ValueError(f"Duplicate domain rows for {rsp_id}:{domain_id}")
            spots = membership.loc[
                membership["domain_key"].astype(str).eq(str(domain["domain_key"])), "spot_id"
            ].astype(str)
            long_rows.append(
                pd.DataFrame(
                    {
                        "spot_id": spots,
                        "rsp_id": rsp_id,
                        "domain_id": domain_id,
                        "domain_key": str(domain["domain_key"]),
                        "block": str(item["block"]),
                        "role": str(item["role"]),
                    }
                )
            )
            registry_rows.append(
                {
                    "rsp_id": rsp_id,
                    "domain_id": domain_id,
                    "domain_key": str(domain["domain_key"]),
                    "rsmd_id": str(domain["rsmd_id"]),
                    "spot_count": int(len(spots)),
                    "qc_pass": bool(domain.get("qc_pass", True)),
                    "geo_centroid_x": _number(domain.get("geo_centroid_x")),
                    "geo_centroid_y": _number(domain.get("geo_centroid_y")),
                    "geo_area_est": _number(domain.get("geo_area_est")),
                    "geo_compactness": _number(domain.get("geo_compactness")),
                    "geo_boundary_ratio": _number(domain.get("geo_boundary_ratio")),
                    "geo_elongation": _number(domain.get("geo_elongation")),
                    "components_count": _number(domain.get("components_count")),
                    "domain_reliability": _number(domain.get("domain_reliability")),
                    "block": str(item["block"]),
                    "role": str(item["role"]),
                    "description": str(item.get("description", "")),
                }
            )
    long = pd.concat(long_rows, ignore_index=True)
    registry = pd.DataFrame(registry_rows)
    wide = (
        long.assign(value=True)
        .pivot_table(index="spot_id", columns="rsp_id", values="value", aggfunc="any", fill_value=False)
        .astype(bool)
    )
    wide.columns = [f"selected_rsd_{column}" for column in wide.columns]
    return long, wide.reset_index(), registry

def load_scffpe_markers(path: str | Path, *, practical_only: bool = True) -> pd.DataFrame:
    markers = pd.read_csv(path)
    if practical_only and "passes_practical_marker_filter" in markers:
        markers = markers.loc[markers["passes_practical_marker_filter"].astype(bool)].copy()
    markers["gene"] = markers["gene"].astype(str)
    markers["cell_label"] = markers["cell_label"].astype(str)
    markers["xenium_label"] = markers["cell_label"].map(SCFFPE_TO_XENIUM).fillna("")
    return markers

def _load_required_tables(root: Path, files: dict[str, str]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required table: {path}")
        tables[key] = pd.read_parquet(path)
    return tables

def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

def xenium_cell_microstructures(
    cells: pd.DataFrame,
    *,
    target_label: str,
    x_col: str = "x_um",
    y_col: str = "y_um",
    eps_um: float | None = None,
    min_samples: int = 5,
    minimum_cells: int = 20,
    prefix: str = "X",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Define connected cell-level microstructures in Xenium coordinates.

    When ``eps_um`` is omitted, the radius is the 75th percentile of the fifth
    nearest-neighbor distance. Small/noise clusters remain explicitly
    unassigned instead of being forced into a nearby structure.
    """

    target = cells.loc[cells["label"].astype(str).eq(target_label)].copy().reset_index(drop=True)
    if target.empty:
        raise ValueError(f"No Xenium cells found for label={target_label!r}")
    coordinates = target[[x_col, y_col]].to_numpy(float)
    neighbor_k = min(6, len(target))
    distances, _ = cKDTree(coordinates).query(coordinates, k=neighbor_k)
    fifth_index = min(5, neighbor_k - 1)
    inferred_eps = float(np.quantile(distances[:, fifth_index], 0.75))
    radius = float(eps_um) if eps_um is not None else inferred_eps
    labels = DBSCAN(eps=radius, min_samples=int(min_samples), n_jobs=-1).fit_predict(coordinates)
    raw_counts = pd.Series(labels[labels >= 0]).value_counts()
    retained = raw_counts.loc[raw_counts.ge(int(minimum_cells))]
    ordered = retained.sort_values(ascending=False).index.tolist()
    width = max(3, len(str(max(1, len(ordered)))))
    id_map = {label: f"{prefix}{rank:0{width}d}" for rank, label in enumerate(ordered, start=1)}
    target["microstructure_id"] = pd.Series(labels).map(id_map).fillna("unassigned")
    target["microstructure_assigned"] = target["microstructure_id"].ne("unassigned")

    rows: list[dict[str, float | int | str]] = []
    for microstructure_id, group in target.loc[target["microstructure_assigned"]].groupby("microstructure_id"):
        xy = group[[x_col, y_col]].to_numpy(float)
        centroid = xy.mean(axis=0)
        compactness = _expanded_hull_compactness(xy, radius=float(np.median(distances[:, 1])) / 2.0)
        rows.append(
            {
                "microstructure_id": microstructure_id,
                "dcis1_cell_count": len(group),
                "centroid_x_um": float(centroid[0]),
                "centroid_y_um": float(centroid[1]),
                "x_span_um": float(np.ptp(xy[:, 0])),
                "y_span_um": float(np.ptp(xy[:, 1])),
                "compactness": compactness,
                "mapped_visium_spot_count": group["visium_barcode"].dropna().astype(str).nunique(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("dcis1_cell_count", ascending=False).reset_index(drop=True)
    parameters = pd.DataFrame(
        [
            {
                "target_label": target_label,
                "n_target_cells": len(target),
                "median_nearest_neighbor_um": float(np.median(distances[:, 1])),
                "fifth_neighbor_q75_um": inferred_eps,
                "dbscan_eps_um": radius,
                "dbscan_min_samples": int(min_samples),
                "minimum_cells": int(minimum_cells),
                "n_retained_microstructures": len(summary),
                "assigned_cell_fraction": float(target["microstructure_assigned"].mean()),
            }
        ]
    )
    return target, summary, parameters

def _expanded_hull_compactness(coordinates: np.ndarray, radius: float) -> float:
    if len(coordinates) < 3:
        return float("nan")
    try:
        hull = ConvexHull(coordinates)
    except Exception:
        return float("nan")
    area = float(hull.volume) + float(hull.area) * radius + np.pi * radius**2
    perimeter = float(hull.area) + 2 * np.pi * radius
    return float(4 * np.pi * area / perimeter**2) if perimeter > 0 else float("nan")

RSP_COLORS = {
    "RSP0005": REGAL_COLORS["blue"],
    "RSP0019": REGAL_COLORS["magenta"],
    "RSP0012": REGAL_COLORS["blue"],
    "RSP0018": REGAL_COLORS["magenta"],
}

DCIS1_RSPS = ("RSP0005", "RSP0019")

DCIS2_RSPS = ("RSP0012", "RSP0018")


RSP_ORDER = (*DCIS1_RSPS, *DCIS2_RSPS)

MARKER_SOURCE_ORDER = ("DCIS 1", "DCIS 2", "Myoepi ACTA2+", "Myoepi KRT15+", "Stromal")

MARKER_SOURCE_DISPLAY = {
    "DCIS 1": "DCIS-1",
    "DCIS 2": "DCIS-2",
}

FIXED_MARKER_SOURCE_COLUMNS = {
    "DCIS 1": (
        ("RSP0005", "SCGB2A2"),
        ("RSP0005", "UGT2B4"),
        ("RSP0005", "AGR2"),
        ("RSP0005", "CEACAM6"),
        ("RSP0005", "PCDH1"),
        ("RSP0005", "ITGB6"),
    ),
    "DCIS 2": (
        ("RSP0012", "CPB1"),
        ("RSP0012", "MUCL1"),
        ("RSP0012", "CD24"),
        ("RSP0012", "KRT19"),
        ("RSP0012", "GREB1"),
        ("RSP0012", "ESR1"),
    ),
    "Myoepi ACTA2+": (
        ("RSP0019", "MYH11"),
        ("RSP0019", "MYLK"),
        ("RSP0019", "KRT14"),
        ("RSP0018", "TP63"),
        ("RSP0018", "COL17A1"),
        ("RSP0018", "COL7A1"),
    ),
    "Myoepi KRT15+": (
        ("RSP0019", "KRT15"),
        ("RSP0019", "KRT17"),
        ("RSP0018", "KRT6B"),
        ("RSP0018", "KRT23"),
        ("RSP0018", "MMP7"),
        ("RSP0018", "DSC2"),
    ),
    "Stromal": (
        ("RSP0018", "FAP"),
        ("RSP0018", "ADAMTS12"),
    ),
}

RSP_PROVENANCE_COLORS = {
    "RSP0005": REGAL_COLORS["blue"],
    "RSP0019": REGAL_COLORS["magenta"],
    "RSP0012": REGAL_COLORS["teal"],
    "RSP0018": REGAL_COLORS["gold"],
}

ACTIVITY_CMAP = rsp_activity_cmap("r4_figure1_activity")

ACTIVITY_NORM = mpl.colors.Normalize(0.0, 1.0)

EXPRESSION_CMAP = marker_expression_cmap("r4_marker_expression")

EXPRESSION_NORM = mpl.colors.Normalize(0.0, 3.0)

ROW_COLORBAR_X = 0.9637

ROW_COLORBAR_WIDTH = 0.00435

DCIS1_TOP_CROP_FRACTION = 0.05

F_LEFT_COLUMN_SHIFT = 0.010

DCIS2_RIGHT_CROP_FRACTION = 0.10

GI_RIGHT_SHIFT = 0.030

BALANCE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "dcis1_balance", ["#C13F70", "#F1F1EE", "#2E73B8"]
)

BALANCE_NORM = mpl.colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

STROMAL_COLOR = "#B57918"

CELL_MARKER_COLORS = {
    "Myoepi ACTA2+": REGAL_COLORS["red"],
    "Myoepi KRT15+": REGAL_COLORS["violet"],
    "Stromal": STROMAL_COLOR,
}

RSD_OUTLINE_HALO_WIDTH = 1.5

RSD_OUTLINE_WIDTH = 0.65

RSD_SMOOTHING_STEPS = 3

RSD_LABEL_OFFSETS = {
    ("RSP0012", "D000048"): (-11, 0),
    ("RSP0012", "D000049"): (-11, 0),
    ("RSP0012", "D000052"): (11, 0),
    ("RSP0018", "D000087"): (-11, 0),
    ("RSP0018", "D000089"): (11, 5),
    ("RSP0018", "D000093"): (11, -5),
}

@dataclass
class Figure1Sources:
    image: np.ndarray
    xy: pd.DataFrame
    spot_radius_px: float
    px_per_um: float
    activation: pd.DataFrame
    rsd_long: pd.DataFrame
    mapping: pd.DataFrame
    dcis1_assignments: pd.DataFrame
    dcis2_assignments: pd.DataFrame
    dcis1_micro: pd.DataFrame
    dcis1_assoc: pd.DataFrame
    dcis2_micro: pd.DataFrame
    dcis2_rsd: pd.DataFrame
    dcis2_entity_de: pd.DataFrame
    dcis2_entity_pathway: pd.DataFrame
    gene_loadings: pd.DataFrame
    scffpe_markers: pd.DataFrame

def load_figure1_sources() -> Figure1Sources:
    selection = load_selection()
    (image, xy, radius) = load_spatial_image_context(H5AD_PATH, sample_id=SAMPLE_ID)
    rsp_tables = load_rsp_tables(SAMPLE_DIR)
    rsmd_tables = load_rsmd_tables(SAMPLE_DIR)
    activation = activation_wide(rsp_tables['activation'], [*DCIS1_RSPS, *DCIS2_RSPS]).set_index('spot_id')
    (rsd_long, _, _) = selected_rsd_membership(selection, rsmd_tables['domains'], rsmd_tables['membership'])
    rsd_long = rsd_long.merge(xy.reset_index()[['spot_id', 'x_img', 'y_img']], on='spot_id', how='left', validate='many_to_one')
    mapping = load_xenium_mapping(XENIUM_MAPPING_PATH)
    px_per_um = estimate_hires_px_per_um(mapping)
    parameters = selection['analysis_parameters']
    (dcis1_assignments, _, _) = xenium_cell_microstructures(mapping, target_label='DCIS_1', eps_um=float(parameters['dcis1_xenium_eps_um']), min_samples=int(parameters['dcis1_xenium_min_samples']), minimum_cells=int(parameters['dcis1_xenium_min_cells']), prefix='DCIS1_X')
    (dcis2_assignments, _, _) = xenium_cell_microstructures(mapping, target_label='DCIS_2', eps_um=float(parameters['dcis2_xenium_eps_um']), min_samples=int(parameters['dcis2_xenium_min_samples']), minimum_cells=int(parameters['dcis2_xenium_min_cells']), prefix='DCIS2_X')
    for assignments in (dcis1_assignments, dcis2_assignments):
        assignments['x_img'] = pd.to_numeric(assignments['x_visium_hires'], errors='coerce')
        assignments['y_img'] = pd.to_numeric(assignments['y_visium_hires'], errors='coerce')
    sources = Figure1Sources(image=image, xy=xy, spot_radius_px=radius, px_per_um=px_per_um, activation=activation, rsd_long=rsd_long, mapping=mapping, dcis1_assignments=dcis1_assignments, dcis2_assignments=dcis2_assignments, dcis1_micro=pd.read_csv(table('dcis1_microstructure_readback.csv')), dcis1_assoc=pd.read_csv(table('dcis1_microstructure_associations.csv')), dcis2_micro=pd.read_csv(table('dcis2_microstructure_readback.csv')), dcis2_rsd=pd.read_csv(table('dcis2_rsd_morphology.csv')), dcis2_entity_de=pd.read_csv(table('dcis2_entity_state_de_readback.csv')), dcis2_entity_pathway=pd.read_csv(table('dcis2_entity_state_pathway_readback.csv')), gene_loadings=rsp_tables['gene_loadings'].copy(), scffpe_markers=load_scffpe_markers(MARKERS_PATH))
    validate_figure1_contract(sources)
    return sources

def validate_figure1_contract(s: Figure1Sources) -> None:
    validate_source_columns(
        s.dcis2_rsd,
        {
            "rsp_id",
            "domain_id",
            "frac_DCIS_2_cells",
            "frac_Myoepi_ACTA2+_cells",
            "frac_Myoepi_KRT15+_cells",
            "frac_Stromal_cells",
            "frac_Invasive_cells",
        },
        name="dcis2_rsd_morphology.csv",
    )
    validate_source_columns(
        s.dcis2_micro,
        {
            "microstructure_id",
            "is_small_structure",
            "small_structure_group",
            "contacted_dcis2_fraction_Myoepi_ACTA2+",
            "contacted_dcis2_fraction_Myoepi_KRT15+",
            "contacted_dcis2_fraction_Stromal",
            "contacted_dcis2_fraction_Invasive",
        },
        name="dcis2_microstructure_readback.csv",
    )
    observed_counts = s.dcis2_micro["small_structure_group"].value_counts().to_dict()
    expected_counts = {"RSP0018-associated": 19, "other-small": 34, "not-in-comparison": 41}
    if observed_counts != expected_counts:
        raise ValueError(f"Unexpected Figure 1H group counts: {observed_counts}")
    if len(s.dcis2_rsd) != 6:
        raise ValueError(f"Expected six DCIS#2 RSDs, found {len(s.dcis2_rsd)}")
    validate_source_columns(
        s.dcis2_entity_de,
        {"gene", "log2fc_case_vs_reference", "q_value", "is_entity_state_signature"},
        name="dcis2_entity_state_de_readback.csv",
    )
    validate_source_columns(
        s.dcis2_entity_pathway,
        {"gene_set_id", "term", "q_value"},
        name="dcis2_entity_state_pathway_readback.csv",
    )

def _activation_values(s: Figure1Sources, rsp_id: str) -> pd.Series:
    return s.activation[f"activation_{rsp_id}"].reindex(s.xy.index).fillna(0.0).astype(float)

def _active_mask(s: Figure1Sources, rsp_id: str) -> pd.Series:
    column = f"active_{rsp_id}"
    if column in s.activation:
        return s.activation[column].reindex(s.xy.index).fillna(False).astype(bool)
    return _activation_values(s, rsp_id).gt(0)

def _row1_image_and_xy(s: Figure1Sources) -> tuple[np.ndarray, pd.DataFrame]:
    image, xy = _crop_to_spot_window(
        s.image,
        s.xy.copy(),
        padding_px=s.spot_radius_px * 2.5,
    )
    raw_like = np.clip((image - 0.30) / 0.70, 0.0, 1.0)
    image = np.clip(raw_like * 0.82 + 0.18, 0.0, 1.0)
    return image, xy

def _draw_full_tissue_activation(
    ax: plt.Axes,
    s: Figure1Sources,
    rsp_id: str,
    image: np.ndarray,
    xy: pd.DataFrame,
) -> None:
    draw_histology(ax, image)
    ax.scatter(
        xy["x_img"], xy["y_img"], s=2.9, facecolors="none",
        edgecolors=REGAL_COLORS["ink"], linewidths=0.17, alpha=0.14,
        rasterized=True, zorder=2,
    )
    values = _activation_values(s, rsp_id)
    active = _active_mask(s, rsp_id) & values.gt(0)
    points = xy.loc[xy.index.intersection(active.index[active])]
    ax.scatter(
        points["x_img"], points["y_img"], c=values.loc[points.index], s=3.0,
        cmap=ACTIVITY_CMAP, norm=ACTIVITY_NORM, linewidths=0, alpha=0.86,
        rasterized=True, zorder=3,
    )
    _draw_selected_rsd_annotations(ax, s, rsp_id, xy, points.index)
    ax.set_title(
        rsp_id,
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, image)

def _draw_selected_rsd_annotations(
    ax: plt.Axes,
    s: Figure1Sources,
    rsp_id: str,
    xy: pd.DataFrame,
    displayed_active_spots: pd.Index,
) -> None:
    if rsp_id not in DCIS2_RSPS:
        return
    selected_domains = s.rsd_long.loc[
        s.rsd_long["rsp_id"].astype(str).eq(rsp_id),
        ["domain_id", "spot_id"],
    ].copy()
    if selected_domains.empty:
        return
    coordinates = xy[["x_img", "y_img"]].to_numpy(dtype=float)
    triangulation = _make_tissue_triangulation(coordinates)
    displayed_active = xy.index.isin(displayed_active_spots)
    domain_order = sorted(
        selected_domains["domain_id"].astype(str).unique(),
        key=lambda value: int(value.removeprefix("D")),
    )
    for domain_id in domain_order:
        domain_spots = selected_domains.loc[
            selected_domains["domain_id"].astype(str).eq(domain_id),
            "spot_id",
        ].astype(str)
        selected = xy.index.isin(domain_spots) & displayed_active
        if int(selected.sum()) < 3:
            continue
        membership = _smooth_rsd_membership(
            triangulation,
            selected.astype(float),
        )
        membership[~displayed_active] = 0.0
        ax.tricontour(
            triangulation,
            membership,
            levels=[0.5],
            colors=["white"],
            linewidths=RSD_OUTLINE_HALO_WIDTH,
            zorder=7,
        )
        ax.tricontour(
            triangulation,
            membership,
            levels=[0.5],
            colors=[INK],
            linewidths=RSD_OUTLINE_WIDTH,
            zorder=8,
        )
        offset = RSD_LABEL_OFFSETS.get((rsp_id, domain_id))
        if offset is None:
            raise ValueError(f"Missing RSD label offset for {rsp_id}:{domain_id}")
        anchor_x, anchor_y = _boundary_label_anchor(
            coordinates,
            selected,
            offset,
        )
        ax.annotate(
            _short_domain_label(domain_id),
            xy=(anchor_x, anchor_y),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right" if offset[0] < 0 else "center",
            va="bottom" if offset[1] > 0 else "top" if offset[1] < 0 else "center",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color=INK,
            clip_on=True,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.25,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": INK,
                "linewidth": 0.45,
                "shrinkA": 1.0,
                "shrinkB": 1.0,
            },
            zorder=9,
        )

def _make_tissue_triangulation(coordinates: np.ndarray) -> mtri.Triangulation:
    triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    nearest = cKDTree(coordinates).query(coordinates, k=2)[0][:, 1]
    spacing = float(np.median(nearest[np.isfinite(nearest)]))
    triangle_points = coordinates[triangulation.triangles]
    edge_lengths = np.stack(
        [
            np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 1], axis=1),
            np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 2], axis=1),
            np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 0], axis=1),
        ],
        axis=1,
    )
    triangulation.set_mask(edge_lengths.max(axis=1) > 2.25 * spacing)
    return triangulation

def _smooth_rsd_membership(
    triangulation: mtri.Triangulation,
    membership: np.ndarray,
) -> np.ndarray:
    triangles = triangulation.get_masked_triangles()
    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    source = np.concatenate([edges[:, 0], edges[:, 1]])
    target = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(source, minlength=len(membership)).astype(float)
    smoothed = membership.astype(float, copy=True)
    for _ in range(RSD_SMOOTHING_STEPS):
        neighbor_sum = np.zeros_like(smoothed)
        np.add.at(neighbor_sum, source, smoothed[target])
        neighbor_mean = np.divide(
            neighbor_sum,
            degree,
            out=smoothed.copy(),
            where=degree > 0,
        )
        smoothed = 0.5 * smoothed + 0.5 * neighbor_mean
    return smoothed

def _boundary_label_anchor(
    coordinates: np.ndarray,
    selected: np.ndarray,
    offset: tuple[float, float],
) -> tuple[float, float]:
    inside = coordinates[selected]
    dx, dy = offset
    if abs(dx) >= abs(dy):
        index = int(np.argmax(inside[:, 0]) if dx > 0 else np.argmin(inside[:, 0]))
    else:
        index = int(np.argmin(inside[:, 1]) if dy > 0 else np.argmax(inside[:, 1]))
    point = inside[index]
    return float(point[0]), float(point[1])

def _short_domain_label(domain_id: str) -> str:
    return f"D{int(str(domain_id).removeprefix('D'))}"

def _draw_pathology_annotation(
    ax: plt.Axes,
    image: np.ndarray,
    xy: pd.DataFrame,
) -> None:
    draw_histology(ax, image)
    colors = {
        "DCIS #1": REGAL_COLORS["teal"],
        "DCIS #2": REGAL_COLORS["gold"],
        "mixed": REGAL_COLORS["violet"],
    }
    display_labels = {
        "DCIS #1": "DCIS#1",
        "DCIS #2": "DCIS#2",
        "mixed": "mixed",
    }
    labels = xy["janesick_annotation"].astype(str) if "janesick_annotation" in xy else pd.Series("", index=xy.index)
    other = xy.loc[~labels.isin(colors)]
    ax.scatter(
        other["x_img"], other["y_img"], s=2.9, facecolors="none", edgecolors="#9DA3A7",
        linewidths=0.20, alpha=0.34, rasterized=True, zorder=3,
    )
    for label, color in colors.items():
        points = xy.loc[labels.eq(label)]
        ax.scatter(
            points["x_img"], points["y_img"], s=3.0, facecolors="none", edgecolors=color,
            linewidths=0.38, alpha=0.96, rasterized=True, zorder=4,
        )
    handles = [
        Line2D([0], [0], marker="s", color=color, markerfacecolor="none", markeredgewidth=0.9,
               markersize=4.6, lw=0, label=display_labels[label])
        for label, color in colors.items()
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3,
              prop={"size": LEGEND_SIZE},
              handlelength=0.8, handletextpad=0.25, columnspacing=0.55)
    ax.set_title(
        "Pathology",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, image)

def build_row1_spatial_entry(s: Figure1Sources) -> plt.Figure:
    image, xy = _row1_image_and_xy(s)
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 2.72), constrained_layout=False)
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.045], wspace=0.065)
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_pathology_annotation(ax_a, image, xy)
    axes = [fig.add_subplot(gs[0, index]) for index in range(1, 5)]
    for ax, rsp_id in zip(axes, RSP_ORDER, strict=True):
        _draw_full_tissue_activation(ax, s, rsp_id, image, xy)
    for ax in (ax_a, *axes):
        ax.set_aspect(0.88, adjustable="box")
    cax = fig.add_subplot(gs[0, 5])
    fig.subplots_adjust(left=0.019, right=0.970, top=0.90, bottom=0.14)
    bbox = cax.get_position()
    cax.remove()
    cax = fig.add_axes([
        ROW_COLORBAR_X,
        bbox.y0 + bbox.height * 0.22,
        ROW_COLORBAR_WIDTH,
        bbox.height * 0.56,
    ])
    mappable = mpl.cm.ScalarMappable(norm=ACTIVITY_NORM, cmap=ACTIVITY_CMAP)
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_ticks([0.0, 1.0])
    colorbar.ax.tick_params(
        labelsize=COLORBAR_SIZE,
        length=2.0,
        width=0.8,
        pad=1.2,
    )
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    colorbar.set_label(
        "RSP activation",
        fontsize=COLORBAR_SIZE,
        labelpad=2.0,
    )
    add_panel_label(fig, "A", ax_a, align_to_canvas_left=True)
    add_panel_label(fig, "B", axes[0])
    return fig

def _fixed_marker_source_columns(s: Figure1Sources) -> list[tuple[str, str, str]]:
    columns = [
        (marker_source, rsp_id, gene)
        for marker_source in MARKER_SOURCE_ORDER
        for rsp_id, gene in FIXED_MARKER_SOURCE_COLUMNS[marker_source]
    ]
    genes = [gene for _, _, gene in columns]
    if len(genes) != len(set(genes)):
        raise ValueError("Fixed Figure 1C marker dictionary contains duplicate genes")

    marker_pairs = set(
        s.scffpe_markers.loc[
            s.scffpe_markers["cell_label"].isin(MARKER_SOURCE_ORDER),
            ["cell_label", "gene"],
        ].itertuples(index=False, name=None)
    )
    loadings = s.gene_loadings.copy()
    for flag in ("is_blacklist_gene", "is_housekeeping_gene", "is_low_info_gene"):
        if flag in loadings:
            loadings = loadings.loc[~loadings[flag].fillna(False).astype(bool)].copy()
    for marker_source, rsp_id, gene in columns:
        if (marker_source, gene) not in marker_pairs:
            raise ValueError(f"{gene} is not an scFFPE marker for {marker_source}")
        loading = loadings.loc[
            loadings["rsp_id"].astype(str).eq(rsp_id)
            & loadings["gene"].astype(str).eq(gene)
        ]
        eligible_column = "in_contribution_70" if marker_source == "Stromal" else "is_core_gene"
        if loading.empty or not loading[eligible_column].fillna(False).astype(bool).any():
            raise ValueError(
                f"{gene} is not an eligible {rsp_id} loading for {marker_source}"
            )
    return columns

def _selected_rsp_spots(s: Figure1Sources) -> dict[str, list[str]]:
    return {
        rsp_id: s.rsd_long.loc[s.rsd_long["rsp_id"].eq(rsp_id), "spot_id"].astype(str).drop_duplicates().tolist()
        for rsp_id in RSP_ORDER
    }

def _draw_micro_violin(ax: plt.Axes, x: float, y: float, values: np.ndarray, *, max_expr: float) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    median = float(np.median(values))
    color = EXPRESSION_CMAP(EXPRESSION_NORM(median))
    if np.nanmax(values) <= 0 or np.unique(np.round(values, 4)).size < 3:
        height = 0.014 + 0.09 * min(median / max_expr, 1.0)
        ax.plot([x - 0.19, x + 0.19], [y, y], color=color, linewidth=0.55, solid_capstyle="round")
        ax.plot([x, x], [y - height, y + height], color=color, linewidth=0.45, solid_capstyle="round")
        return
    grid = np.linspace(0.0, max_expr, 80)
    try:
        density = gaussian_kde(values)(grid)
    except Exception:
        hist, edges = np.histogram(values, bins=20, range=(0.0, max_expr), density=True)
        grid = (edges[:-1] + edges[1:]) / 2.0
        density = hist
    if not np.isfinite(density).any() or float(np.nanmax(density)) <= 0:
        return
    density = density / float(np.nanmax(density)) * 0.26
    y_grid = y + 0.28 - grid / max_expr * 0.56
    ax.fill_betweenx(y_grid, x - density, x + density, facecolor=color, edgecolor="white", linewidth=0.18, alpha=0.96)

def _draw_marker_identity_matrix(
    ax: plt.Axes,
    columns: list[tuple[str, str, str]],
    expression: pd.DataFrame,
    rsp_spots: dict[str, list[str]],
) -> None:
    n_rows, n_cols = len(RSP_ORDER), len(columns)
    ax.set_xlim(-0.55, n_cols - 0.45)
    ax.set_ylim(n_rows - 0.45, -1.08)
    ax.set_yticks(
        range(n_rows),
        RSP_ORDER,
        fontsize=TICK_LABEL_SIZE,
        fontweight="bold",
    )
    for tick, rsp_id in zip(ax.get_yticklabels(), RSP_ORDER, strict=True):
        tick.set_color(RSP_PROVENANCE_COLORS[rsp_id])
    ax.set_xticks(
        range(n_cols),
        [gene for _, _, gene in columns],
        rotation=20,
        ha="right",
        fontsize=ANNOTATION_SIZE,
    )
    for tick in ax.get_xticklabels():
        tick.set_color(REGAL_COLORS["ink"])
        tick.set_fontstyle("italic")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in np.arange(-0.5, n_cols + 0.5, 1.0):
        ax.axvline(x, color=GRID, linewidth=GRID_LINE_WIDTH, zorder=0)
    for y in np.arange(-0.5, n_rows + 0.5, 1.0):
        ax.axhline(y, color=GRID, linewidth=GRID_LINE_WIDTH, zorder=0)
    present = [gene for _, _, gene in columns if gene in expression]
    max_expr = max(float(np.nanpercentile(expression[present].to_numpy(float), 98)), 1.0)
    for row_index, rsp_id in enumerate(RSP_ORDER):
        spots = [spot for spot in rsp_spots[rsp_id] if spot in expression.index]
        for col_index, (_, _, gene) in enumerate(columns):
            values = expression.loc[spots, gene].to_numpy(float) if gene in expression else np.array([])
            _draw_micro_violin(ax, col_index, row_index, values, max_expr=max_expr)

    for col_index, (_, source_rsp, _) in enumerate(columns):
        ax.add_patch(
            Rectangle(
                (col_index - 0.38, -0.61), 0.76, 0.065,
                facecolor=RSP_PROVENANCE_COLORS[source_rsp], edgecolor="none",
                clip_on=False, zorder=5,
            )
        )
    ax.text(-0.62, -0.578, "loading RSP", ha="right", va="center", fontsize=ANNOTATION_SIZE,
            color=REGAL_COLORS["mid"], clip_on=False)

    start = 0
    for marker_source in MARKER_SOURCE_ORDER:
        end = start + len(FIXED_MARKER_SOURCE_COLUMNS[marker_source]) - 1
        y = -0.76
        ax.plot([start - 0.35, end + 0.35], [y, y], color="#595959", linewidth=0.55, clip_on=False)
        ax.plot([start - 0.35, start - 0.35], [y, y + 0.08], color="#595959", linewidth=0.55, clip_on=False)
        ax.plot([end + 0.35, end + 0.35], [y, y + 0.08], color="#595959", linewidth=0.55, clip_on=False)
        ax.text(
            (start + end) / 2,
            y - 0.08,
            MARKER_SOURCE_DISPLAY.get(marker_source, marker_source),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
        )
        start = end + 1

def build_row2_marker_identity(s: Figure1Sources) -> plt.Figure:
    columns = _fixed_marker_source_columns(s)
    genes = [gene for _, _, gene in columns]
    expression, present = load_normalized_expression(H5AD_PATH, genes)
    missing = sorted(set(genes).difference(present))
    if missing:
        raise ValueError(f"Selected marker genes missing from Visium expression: {missing}")
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.05), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.97, 0.018], wspace=0.018)
    ax = fig.add_subplot(gs[0, 0])
    _draw_marker_identity_matrix(ax, columns, expression, _selected_rsp_spots(s))
    cax = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.078, right=0.976, top=0.91, bottom=0.14)
    bbox = cax.get_position()
    cax.remove()
    cax = fig.add_axes([
        ROW_COLORBAR_X,
        bbox.y0 + bbox.height * 0.18,
        ROW_COLORBAR_WIDTH,
        bbox.height * 0.62,
    ])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=EXPRESSION_NORM, cmap=EXPRESSION_CMAP), cax=cax)
    colorbar.set_ticks([0.0, 3.0])
    colorbar.ax.tick_params(
        labelsize=COLORBAR_SIZE,
        length=2.0,
        width=0.8,
        pad=1.2,
    )
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    colorbar.set_label(
        "Median log-expression",
        fontsize=COLORBAR_SIZE,
        labelpad=2.0,
    )
    add_panel_label(fig, "C", ax, align_to_canvas_left=True)
    return fig

def _dcis1_limits(s: Figure1Sources) -> tuple[float, float, float, float]:
    selected = s.rsd_long.loc[s.rsd_long["block"].eq("dcis1_multilayer")]
    x0, x1, y0, y1 = bounds_from_points(selected, padding=24)
    target_height = max(y1 - y0, (x1 - x0) / 1.55)
    center_y = (y0 + y1) / 2.0
    top = center_y - target_height / 2.0 + target_height * DCIS1_TOP_CROP_FRACTION
    bottom = center_y + target_height / 2.0
    return x0, x1, top, bottom

def _muted_histology(image: np.ndarray) -> np.ndarray:
    histology_rgb = np.asarray(image)[..., :3]
    return np.clip(0.80 + 0.20 * histology_rgb, 0.0, 1.0)

def _draw_dcis1_interface_map(
    ax: plt.Axes,
    s: Figure1Sources,
    *,
    xenium_label: str,
    title: str,
    color: str,
    limits: tuple[float, float, float, float],
) -> None:
    draw_histology(ax, _muted_histology(s.image))
    x0, x1, y0, y1 = limits
    context = s.mapping.loc[
        s.mapping["label"].eq(xenium_label)
        & s.mapping["x_visium_hires"].between(x0, x1)
        & s.mapping["y_visium_hires"].between(y0, y1)
    ]
    dense_context = len(context) > 5000
    point_alpha = 0.42 if xenium_label == "Stromal" and dense_context else (
        0.72 if xenium_label == "Stromal" else (0.24 if dense_context else 0.62)
    )
    ax.scatter(
        context["x_visium_hires"], context["y_visium_hires"],
        s=0.24 if dense_context else 0.86, c=color,
        alpha=point_alpha, linewidths=0, rasterized=True, zorder=5,
    )
    finish_spatial_axis(ax, s.image, limits=limits)
    ax.set_title(
        title,
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        color=color,
        pad=4.0,
    )

def _draw_dcis1_balance_map(
    ax: plt.Axes,
    s: Figure1Sources,
    *,
    limits: tuple[float, float, float, float],
) -> None:
    draw_histology(ax, _muted_histology(s.image))
    x0, x1, y0, y1 = limits
    target = s.dcis1_assignments.loc[
        s.dcis1_assignments["microstructure_assigned"].astype(bool)
        & s.dcis1_assignments["x_img"].between(x0, x1)
        & s.dcis1_assignments["y_img"].between(y0, y1)
    ].copy()
    balance = s.dcis1_micro.set_index("microstructure_id")["weighted_mean_dcis1_activation_balance"]
    target["microstructure_balance"] = target["microstructure_id"].astype(str).map(balance)
    target = target.loc[target["microstructure_balance"].notna()]
    ax.scatter(
        target["x_img"], target["y_img"], s=0.72,
        c=target["microstructure_balance"], cmap=BALANCE_CMAP, norm=BALANCE_NORM,
        alpha=0.90, linewidths=0, rasterized=True, zorder=4,
    )
    finish_spatial_axis(ax, s.image, limits=limits)
    ax.set_title(
        "RSP0005 vs RSP0019",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )

def _draw_dcis1_association(axes: Sequence[plt.Axes], s: Figure1Sources) -> None:
    if len(axes) != 6:
        raise ValueError(f"DCIS#1 association grid requires six axes, received {len(axes)}")
    context_specs = [
        (
            "contacted_dcis1_fraction_Myoepi_ACTA2+",
            "frac_Myoepi_ACTA2+",
            "ACTA2+ contact",
            REGAL_COLORS["red"],
        ),
        (
            "contacted_dcis1_fraction_Myoepi_KRT15+",
            "frac_Myoepi_KRT15+",
            "KRT15+ contact",
            REGAL_COLORS["violet"],
        ),
        (
            "contacted_dcis1_fraction_Stromal",
            "frac_Stromal",
            "Stromal contact",
            STROMAL_COLOR,
        ),
    ]
    outcome_specs = [
        ("weighted_mean_activation_RSP0005", "activation_RSP0005", "RSP0005"),
        ("weighted_mean_activation_RSP0019", "activation_RSP0019", "RSP0019"),
    ]
    between = s.dcis1_assoc.loc[s.dcis1_assoc["analysis_level"].eq("between_microstructures")]
    within = s.dcis1_assoc.loc[s.dcis1_assoc["analysis_level"].eq("within_microstructure_centered")]

    for row_index, (between_context, within_context, row_label, row_color) in enumerate(context_specs):
        for col_index, (between_outcome, within_outcome, rsp_id) in enumerate(outcome_specs):
            ax = axes[row_index * 2 + col_index]
            local = s.dcis1_micro[[between_context, between_outcome]].dropna()
            ax.scatter(
                local[between_context], local[between_outcome], s=14.5,
                facecolors=mpl.colors.to_rgba(RSP_COLORS[rsp_id], 0.68),
                edgecolors="white", linewidths=0.32, rasterized=True, zorder=3,
            )
            x_values = local[between_context].to_numpy(float)
            y_values = local[between_outcome].to_numpy(float)
            if len(local) >= 2 and np.ptp(x_values) > 0:
                slope, intercept = np.polyfit(x_values, y_values, 1)
                trend_x = np.array([x_values.min(), x_values.max()])
                ax.plot(
                    trend_x, intercept + slope * trend_x,
                    color=RSP_COLORS[rsp_id], linewidth=0.85,
                    linestyle=(0, (3.0, 2.0)), alpha=0.78, zorder=2,
                )
            between_row = between.loc[
                between["context_variable"].eq(between_context)
                & between["outcome"].eq(between_outcome)
            ].iloc[0]
            within_row = within.loc[
                within["context_variable"].eq(within_context)
                & within["outcome"].eq(within_outcome)
            ].iloc[0]
            between_suffix = ", ns" if float(between_row.q_value) >= 0.05 else ""
            within_suffix = ", ns" if float(within_row.q_value) >= 0.05 else ""
            annotation_y = 0.05 if col_index == 0 else 0.95
            annotation_va = "bottom" if col_index == 0 else "top"
            ax.text(
                0.04, annotation_y,
                f"Between ρ = {between_row.rho:+.2f}{between_suffix}\n"
                f"Within ρ = {within_row.rho:+.2f}{within_suffix}",
                transform=ax.transAxes, ha="left", va=annotation_va, fontsize=ANNOTATION_SIZE,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.45},
            )
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(
                labelsize=TICK_LABEL_SIZE,
                length=2.0,
                width=0.8,
                pad=1.2,
            )
            if col_index == 1:
                ax.set_yticklabels([])
                ax.set_ylabel("")
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0)
            elif row_index == 1:
                ax.set_ylabel("RSP activation", fontsize=AXIS_LABEL_SIZE)
            if row_index < 2:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            else:
                ax.set_xlabel(
                    "Contacted DCIS-1 fraction",
                    fontsize=AXIS_LABEL_SIZE,
                )
            if col_index == 1:
                ax.text(
                    1.10, 0.50, row_label, transform=ax.transAxes,
                    rotation=270, ha="center", va="center", fontsize=ANNOTATION_SIZE,
                    color=row_color, fontweight="bold", clip_on=False,
                )
            if row_index == 0:
                ax.set_title(
                    rsp_id, loc="center", fontsize=PANEL_TITLE_SIZE,
                    color=RSP_COLORS[rsp_id], fontweight="bold", pad=2.0,
                )
            ax.grid(color=GRID, linewidth=GRID_LINE_WIDTH, alpha=0.8)

def build_row3_dcis1_interface(s: Figure1Sources) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 4.90), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, width_ratios=[0.48, 0.52], wspace=0.08)
    spatial = outer[0, 0].subgridspec(2, 2, wspace=0.025, hspace=0.075)
    axes = [
        fig.add_subplot(spatial[0, 0]),
        fig.add_subplot(spatial[0, 1]),
        fig.add_subplot(spatial[1, 0]),
        fig.add_subplot(spatial[1, 1]),
    ]
    limits = _dcis1_limits(s)
    specs = [
        ("Myoepi_ACTA2+", "ACTA2+", REGAL_COLORS["red"]),
        ("Myoepi_KRT15+", "KRT15+", REGAL_COLORS["violet"]),
        ("Stromal", "Stromal", STROMAL_COLOR),
    ]
    for ax, (label, title, color) in zip(axes[:3], specs, strict=True):
        _draw_dcis1_interface_map(ax, s, xenium_label=label, title=title, color=color, limits=limits)
    _draw_dcis1_balance_map(axes[3], s, limits=limits)
    for ax in axes[2:]:
        ax.set_anchor("S")
    add_scale_bar(axes[-1], px_per_um=s.px_per_um, length_um=500)
    right_shell = outer[0, 1].subgridspec(1, 2, width_ratios=[0.94, 0.06], wspace=0.0)
    right = right_shell[0, 0].subgridspec(3, 2, wspace=0.12, hspace=0.16)
    association_axes = [fig.add_subplot(right[row, column]) for row in range(3) for column in range(2)]
    _draw_dcis1_association(association_axes, s)
    fig.subplots_adjust(left=0.025, right=0.988, top=0.92, bottom=0.11)
    cax = axes[3].inset_axes([0.18, -0.075, 0.64, 0.035])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=BALANCE_NORM, cmap=BALANCE_CMAP), cax=cax, orientation="horizontal"
    )
    colorbar.set_ticks([-1.0, 1.0])
    colorbar.set_ticklabels(["RSP0019-high", "RSP0005-high"])
    colorbar.ax.tick_params(
        labelsize=COLORBAR_SIZE,
        length=1.8,
        width=0.8,
        pad=1.0,
    )
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    colorbar.set_label(
        "Relative activation",
        fontsize=COLORBAR_SIZE,
        labelpad=1.0,
    )
    add_panel_label(fig, "D", axes[0], align_to_canvas_left=True)
    add_panel_label(fig, "E", association_axes[0])
    return fig

def _draw_dcis2_region_distribution_maps(axes: Sequence[plt.Axes], s: Figure1Sources) -> None:
    region_spots = set(
        s.rsd_long.loc[s.rsd_long["rsp_id"].isin(DCIS2_RSPS), "spot_id"].astype(str)
    )
    assigned = s.dcis2_assignments.loc[
        s.dcis2_assignments["microstructure_assigned"].astype(bool)
    ].copy()
    region_microstructure_ids = set(
        assigned.loc[
            assigned["visium_barcode"].astype(str).isin(region_spots),
            "microstructure_id",
        ].astype(str)
    )
    region_microstructure_cells = assigned.loc[
        assigned["microstructure_id"].astype(str).isin(region_microstructure_ids)
    ].copy()

    specs = [
        ("Myoepi_ACTA2+", "ACTA2+", CELL_MARKER_COLORS["Myoepi ACTA2+"]),
        ("Myoepi_KRT15+", "KRT15+", CELL_MARKER_COLORS["Myoepi KRT15+"]),
        ("Stromal", "Stromal", CELL_MARKER_COLORS["Stromal"]),
    ]
    roi_context = s.mapping.loc[
        s.mapping["label"].isin([cell_label for cell_label, _, _ in specs])
        & s.mapping["visium_barcode"].astype(str).isin(region_spots)
    ].copy()
    roi_points = pd.concat(
        [
            roi_context[["x_visium_hires", "y_visium_hires"]].rename(
                columns={"x_visium_hires": "x_img", "y_visium_hires": "y_img"}
            ),
            region_microstructure_cells[["x_img", "y_img"]],
        ],
        ignore_index=True,
    )
    x0, x1, y0, y1 = bounds_from_points(roi_points, padding=40)
    target_width = max(x1 - x0, 1.55 * (y1 - y0))
    center_x = (x0 + x1) / 2.0
    limits = (
        center_x - target_width / 2.0,
        center_x + target_width / 2.0 - target_width * DCIS2_RIGHT_CROP_FRACTION,
        y0,
        y1,
    )
    x0, x1, y0, y1 = limits
    context = s.mapping.loc[
        s.mapping["label"].isin([cell_label for cell_label, _, _ in specs])
        & s.mapping["x_visium_hires"].between(x0, x1)
        & s.mapping["y_visium_hires"].between(y0, y1)
    ].copy()

    for ax, (cell_label, title, color) in zip(axes[:3], specs, strict=True):
        draw_histology(ax, _muted_histology(s.image))
        cells = context.loc[context["label"].eq(cell_label)]
        dense_context = len(cells) > 5000
        point_alpha = 0.42 if cell_label == "Stromal" and dense_context else (
            0.72 if cell_label == "Stromal" else (0.24 if dense_context else 0.62)
        )
        ax.scatter(
            cells["x_visium_hires"], cells["y_visium_hires"],
            s=0.24 if dense_context else 0.86, c=color,
            alpha=point_alpha,
            linewidths=0, rasterized=True, zorder=5,
        )
        ax.set_title(
            title,
            loc="center",
            fontsize=PANEL_TITLE_SIZE,
            color=color,
            fontweight="bold",
            pad=4.0,
        )
        finish_spatial_axis(ax, s.image, limits=limits)

    ax = axes[3]
    draw_histology(ax, _muted_histology(s.image))
    ax.scatter(
        region_microstructure_cells["x_img"], region_microstructure_cells["y_img"], s=0.72,
        c=REGAL_COLORS["teal"], alpha=0.88, linewidths=0, rasterized=True, zorder=5,
    )
    ax.set_title(
        "DCIS-2 microstructures",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        color=REGAL_COLORS["teal"],
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, s.image, limits=limits)
    add_scale_bar(ax, px_per_um=s.px_per_um, length_um=500)

def _draw_microenvironment_matrix(ax: plt.Axes, s: Figure1Sources) -> None:
    data = s.dcis2_rsd.copy()
    data = data.sort_values(["rsp_id", "domain_id"]).reset_index(drop=True)
    data["label"] = data.apply(
        lambda row: f"{row['rsp_id']}:{_short_domain_label(row['domain_id'])}",
        axis=1,
    )
    metrics = [
        ("frac_DCIS_2_cells", "DCIS-2"),
        ("frac_Myoepi_ACTA2+_cells", "ACTA2+"),
        ("frac_Myoepi_KRT15+_cells", "KRT15+"),
        ("frac_Stromal_cells", "Stromal"),
        ("frac_Invasive_cells", "Invasive"),
    ]
    values = data[[column for column, _ in metrics]].astype(float)
    scaled = (values - values.min()) / (values.max() - values.min()).replace(0, 1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "microenvironment_heatmap", ["#F3F4F2", "#C8DADB", "#83A8B4", REGAL_COLORS["violet"]]
    )
    x_positions = np.array([0.0, 1.22, 2.22, 3.22, 4.22])
    for row_index, (_, row) in enumerate(data.iterrows()):
        for col_index, (column, _) in enumerate(metrics):
            raw_value = float(row[column])
            percentage = 100.0 * raw_value
            label = f"{percentage:.0f}%" if percentage >= 10 else f"{percentage:.1f}%"
            normalized = float(scaled.iloc[row_index, col_index])
            ax.add_patch(
                mpl.patches.Rectangle(
                    (x_positions[col_index] - 0.45, row_index - 0.43), 0.90, 0.86,
                    facecolor=cmap(normalized), edgecolor="white", linewidth=0.45,
                )
            )
            color = "white" if normalized > 0.62 else REGAL_COLORS["ink"]
            ax.text(
                x_positions[col_index],
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=ANNOTATION_SIZE,
                color=color,
            )

        ax.add_patch(
            mpl.patches.Rectangle(
                (-0.67, row_index - 0.43), 0.10, 0.86,
                facecolor=RSP_COLORS[str(row["rsp_id"])], edgecolor="none",
            )
        )

    ax.text(x_positions[0], -0.78, "Lesion", ha="center", va="center",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=REGAL_COLORS["mid"])
    ax.text(np.mean(x_positions[1:]), -0.78, "Context", ha="center", va="center",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=REGAL_COLORS["mid"])
    ax.axhline(2.5, color="#6F7476", linewidth=0.75, xmin=0.02, xmax=0.99)
    ax.set_xlim(-0.72, 4.72)
    ax.set_ylim(len(data) - 0.5, -0.90)
    ax.set_xticks(
        x_positions,
        [label for _, label in metrics],
        fontsize=ANNOTATION_SIZE,
    )
    ax.set_yticks(
        range(len(data)),
        data["label"],
        fontsize=ANNOTATION_SIZE,
    )
    for tick in ax.get_yticklabels():
        tick.set_color(REGAL_COLORS["ink"])
        tick.set_fontweight("bold")
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

def _draw_small_structure_distributions(ax: plt.Axes, s: Figure1Sources) -> None:
    specs = [
        ("contacted_dcis2_fraction_Myoepi_ACTA2+", "ACTA2+"),
        ("contacted_dcis2_fraction_Myoepi_KRT15+", "KRT15+"),
        ("contacted_dcis2_fraction_Stromal", "Stromal"),
        ("contacted_dcis2_fraction_Invasive", "Invasive"),
    ]
    groups = ["RSP0018-associated", "other-small"]
    colors = [REGAL_COLORS["magenta"], "#AEB5B8"]
    offsets = [-0.16, 0.16]
    for feature_index, (feature, label) in enumerate(specs):
        for group_index, (group, color) in enumerate(zip(groups, colors, strict=True)):
            values = s.dcis2_micro.loc[s.dcis2_micro["small_structure_group"].eq(group), feature].dropna().to_numpy(float)
            phase = np.arange(len(values), dtype=float)
            jitter = np.sin(phase * 1.61803398875) * 0.055
            x = feature_index + offsets[group_index]
            box = ax.boxplot(
                [values], positions=[x], widths=0.22, patch_artist=True,
                showfliers=False, whis=(5, 95), manage_ticks=False,
                boxprops={"facecolor": mpl.colors.to_rgba(color, 0.16), "edgecolor": color, "linewidth": 0.65},
                whiskerprops={"color": color, "linewidth": 0.60},
                capprops={"color": color, "linewidth": 0.60},
                medianprops={"color": REGAL_COLORS["ink"], "linewidth": 1.05},
            )
            for artist in box["boxes"] + box["whiskers"] + box["caps"] + box["medians"]:
                artist.set_zorder(2)
            ax.scatter(np.full(len(values), x) + jitter, values, s=8.2, c=color, alpha=0.74, linewidths=0, rasterized=True)
    ax.set_xlim(-0.48, 3.48)
    ax.set_ylim(-0.04, 1.12)
    ax.set_xticks(
        range(4),
        [label for _, label in specs],
        fontsize=ANNOTATION_SIZE,
    )
    ax.set_ylabel(
        "Contacted DCIS-2 fraction",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=3.5, markerfacecolor=color,
               markeredgewidth=0, label=label)
        for color, label in zip(colors, ["RSP0018-associated (19)", "Other small (34)"], strict=True)
    ]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=2,
        fontsize=LEGEND_SIZE, frameon=False, handlelength=0.7,
        handletextpad=0.15, columnspacing=0.55, borderaxespad=0.0,
    )

def _draw_d000048_volcano(ax: plt.Axes, s: Figure1Sources) -> None:
    data = s.dcis2_entity_de.copy().reset_index(drop=True)
    data["minus_log10_q"] = -np.log10(data["q_value"].clip(lower=1e-300))
    signature = data["is_entity_state_signature"].astype(bool)
    x = data["log2fc_case_vs_reference"].to_numpy(float)
    y = data["minus_log10_q"].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    y_cap = min(max(float(np.nanpercentile(y, 99)), 20.0), 85.0)
    x_cap = min(max(float(np.nanpercentile(np.abs(x[np.isfinite(x)]), 98)), 4.0), 8.0)
    x_plot = np.clip(x, -x_cap, x_cap)
    y_plot = np.clip(y, 0, y_cap)
    colors = np.full(len(data), "#C7CCD0", dtype=object)
    colors[signature.to_numpy() & (x < 0)] = REGAL_COLORS["magenta"]
    colors[signature.to_numpy() & (x > 0)] = RSP_COLORS["RSP0012"]
    ax.scatter(x_plot, y_plot, s=3.4, c=colors, alpha=0.78, linewidths=0, rasterized=True)
    ax.axvline(-0.5, color="#8C8C8C", lw=0.45, linestyle=(0, (2, 2)))
    ax.axvline(0.5, color="#8C8C8C", lw=0.45, linestyle=(0, (2, 2)))
    ax.axhline(-np.log10(0.05), color="#8C8C8C", lw=0.45, linestyle=(0, (2, 2)))
    ax.set_xlim(-x_cap * 1.08, x_cap * 1.08)
    ax.set_ylim(-0.5, y_cap * 1.05)
    label_offsets = {
        "FASN": (8, -14),
        "MT-ND1": (8, -34),
        "IFI27": (-8, -18),
        "COL1A1": (-8, 8),
    }
    missing_labels = sorted(set(label_offsets).difference(data["gene"]))
    if missing_labels:
        raise ValueError(f"Missing fixed volcano labels: {missing_labels}")
    for gene, offset in label_offsets.items():
        row = data.loc[data["gene"].eq(gene)]
        index = int(row.index[0])
        point_x = float(x_plot[index])
        point_y = float(y_plot[index])
        ax.annotate(
            gene,
            xy=(point_x, point_y),
            xytext=offset,
            textcoords="offset points",
            ha="right" if point_x < 0 else "left",
            va="top" if offset[1] < 0 else "bottom",
            fontsize=ANNOTATION_SIZE, fontstyle="italic",
            color=REGAL_COLORS["ink"], clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.25},
            arrowprops={
                "arrowstyle": "-",
                "color": REGAL_COLORS["mid"],
                "linewidth": 0.45,
                "shrinkA": 1.0,
                "shrinkB": 1.0,
            },
        )
    ax.set_xlabel("log2FC", fontsize=AXIS_LABEL_SIZE, labelpad=1.5)
    ax.set_ylabel("−log10 q", fontsize=AXIS_LABEL_SIZE, labelpad=1.5)
    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.grid(color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.text(0.02, 0.98, "D49/D52", transform=ax.transAxes, ha="left", va="top",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=REGAL_COLORS["magenta"])
    ax.text(0.98, 0.98, "D48", transform=ax.transAxes, ha="right", va="top",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=RSP_COLORS["RSP0012"])

def _select_entity_pathways(s: Figure1Sources, direction: str) -> pd.DataFrame:
    desired = {
        "D000048_up": [
            ("MITOCHONDRIAL_TRANSLATION", "Mitochondrial translation"),
            ("ESTROGEN_RESPONSE_EARLY", "Estrogen response"),
            ("RESPIRATORY_ELECTRON_TRANSPORT", "Respiratory transport"),
            ("Biosynthesis of unsaturated fatty acids", "Unsaturated FA synthesis"),
        ],
        "D000048_down": [
            ("EPITHELIAL_MESENCHYMAL_TRANSITION", "EMT"),
            ("EXTRACELLULAR_MATRIX_ORGANIZATION", "ECM organization"),
            ("INTERFERON_GAMMA_RESPONSE", "Interferon-γ response"),
            ("CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM", "Cytokine signaling"),
        ],
    }[direction]
    local = s.dcis2_entity_pathway.loc[s.dcis2_entity_pathway["gene_set_id"].eq(direction)].copy()
    rows = []
    for pattern, label in desired:
        match = local.loc[local["term"].astype(str).str.contains(pattern, regex=False)].sort_values("q_value")
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["display_term"] = label
        rows.append(row)
    return pd.DataFrame(rows)

def _draw_entity_pathway_mirror(ax: plt.Axes, s: Figure1Sources) -> None:
    up = _select_entity_pathways(s, "D000048_up").copy()
    down = _select_entity_pathways(s, "D000048_down").copy()
    up["score"] = -np.log10(up["q_value"].clip(lower=1e-300))
    down["score"] = -np.log10(down["q_value"].clip(lower=1e-300))
    up = up.sort_values("score", ascending=True).reset_index(drop=True)
    down = down.sort_values("score", ascending=True).reset_index(drop=True)
    gap = 0.50
    y_down = np.arange(len(down), dtype=float)
    y_up = np.arange(len(up), dtype=float) + len(down) + gap
    ax.barh(y_down, -down["score"], height=0.58, color=REGAL_COLORS["magenta"], alpha=0.86)
    ax.barh(y_up, up["score"], height=0.58, color=RSP_COLORS["RSP0012"], alpha=0.86)
    ax.axvline(0, color="#4B4F50", lw=0.55)
    for y_value, label in zip(y_down, down["display_term"], strict=True):
        ax.text(0.20, y_value, label, ha="left", va="center", fontsize=ANNOTATION_SIZE, color=INK)
    for y_value, label in zip(y_up, up["display_term"], strict=True):
        ax.text(-0.20, y_value, label, ha="right", va="center", fontsize=ANNOTATION_SIZE, color=INK)
    max_score = max(float(up["score"].max()), float(down["score"].max()), 1.0)
    limit = np.ceil(max_score / 5.0) * 5.0
    ax.set_xlim(-limit * 1.04, limit * 1.04)
    ax.set_ylim(-0.65, len(up) + len(down) + gap - 0.35)
    ax.set_yticks([])
    ticks = np.linspace(-limit, limit, 5)
    ax.set_xticks(
        ticks,
        [f"{abs(value):.0f}" for value in ticks],
        fontsize=TICK_LABEL_SIZE,
    )
    ax.set_xlabel("−log10 q", fontsize=AXIS_LABEL_SIZE, labelpad=16.0)
    ax.grid(axis="x", color=GRID, lw=GRID_LINE_WIDTH)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.02, -0.14, "D49/D52 higher", transform=ax.transAxes, ha="left", va="top",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=REGAL_COLORS["magenta"])
    ax.text(0.98, -0.14, "D48 higher", transform=ax.transAxes, ha="right", va="top",
            fontsize=ANNOTATION_SIZE, fontweight="bold", color=RSP_COLORS["RSP0012"])

def build_row4_dcis2_organization(s: Figure1Sources) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 5.25), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, width_ratios=[0.44, 0.56], wspace=0.08)

    left = outer[0, 0].subgridspec(2, 2, wspace=-0.10, hspace=0.12)
    axes_f = [fig.add_subplot(left[row, col]) for row in range(2) for col in range(2)]
    _draw_dcis2_region_distribution_maps(axes_f, s)

    right = outer[0, 1].subgridspec(
        2,
        2,
        width_ratios=[0.95, 1.05],
        wspace=0.34,
        hspace=0.26,
    )
    ax_g = fig.add_subplot(right[0, 0])
    _draw_microenvironment_matrix(ax_g, s)
    ax_h = fig.add_subplot(right[0, 1])
    _draw_small_structure_distributions(ax_h, s)
    ax_i = fig.add_subplot(right[1, 0])
    _draw_d000048_volcano(ax_i, s)
    ax_j = fig.add_subplot(right[1, 1])
    _draw_entity_pathway_mirror(ax_j, s)

    fig.subplots_adjust(left=0.025, right=0.988, top=0.935, bottom=0.115)
    for ax in axes_f[2:]:
        ax.set_anchor("S")
    for ax in (axes_f[0], axes_f[2]):
        position = ax.get_position()
        ax.set_position([
            position.x0 - F_LEFT_COLUMN_SHIFT,
            position.y0,
            position.width,
            position.height,
        ])
    for ax in (ax_g, ax_i):
        position = ax.get_position()
        ax.set_position([
            position.x0 + GI_RIGHT_SHIFT,
            position.y0,
            position.width,
            position.height,
        ])
    for ax in (ax_i, ax_j):
        position = ax.get_position()
        ax.set_position([
            position.x0,
            position.y0,
            position.width,
            position.height + 0.018,
        ])
    add_panel_label(fig, "F", axes_f[0], align_to_canvas_left=True)
    add_panel_label(fig, "G", ax_g)
    add_panel_label(fig, "H", ax_h)
    add_panel_label(fig, "I", ax_i)
    add_panel_label(fig, "J", ax_j)
    return fig


def export_figure1() -> list[Path]:
    configure_figure_style()
    sources = load_figure1_sources()
    builders = (build_row1_spatial_entry, build_row2_marker_identity, build_row3_dcis1_interface, build_row4_dcis2_organization,)
    rows = (render_row(builder(sources)) for builder in builders)
    return save_composite_bundle(stack_row_images(rows), FIGURE_STEM)

def draw_figure() -> None:
    for output in export_figure1():
        print(output)

def configure():
    global FIGURE_STEM, H5AD_PATH, MARKERS_PATH, SAMPLE_DIR, SAMPLE_ID, XENIUM_MAPPING_PATH
    SAMPLE_ID = option('sample_id')
    SAMPLE_DIR = path('sample_dir')
    H5AD_PATH = path('h5ad')
    XENIUM_MAPPING_PATH = path('xenium_mapping')
    FIGURE_STEM = OUTPUT_DIR / 'fig06'
    MARKERS_PATH = resource('scffpe_markers')

def main() -> None:
    global _CONFIG, _CONFIG_DIR, _ROOTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_file = args.config.expanduser().resolve()
    _CONFIG_DIR = config_file.parent
    with config_file.open() as handle:
        document = yaml.safe_load(handle)
    _ROOTS = {'HOME': str(Path.home())}
    for name, value in document['roots'].items():
        _ROOTS[name] = Template(value).substitute(_ROOTS)
    _CONFIG = document['fig06']
    configure()
    result = draw_figure()
    if result is not None:
        for output in result:
            print(output)

if __name__ == '__main__':
    main()
