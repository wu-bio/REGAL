"""Plot manuscript fig07 using paths from the shared YAML configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from typing import Iterable
from typing import Iterable, Sequence
from typing import Sequence
import argparse
from string import Template
from copy import deepcopy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RSP.visualize_rsp import _coords_to_image_space, _estimate_spot_radius_px, _load_spatial_image_context, _orient_tissue_image
from RSP.visualize_rsp import _crop_to_spot_window
from matplotlib.lines import Line2D
from scipy import sparse
from scipy import stats
from Utils.manuscript_figure_style import ANNOTATION_SIZE, AXIS_LABEL_SIZE, COLORBAR_SIZE, EXPORT_DPI, FIGURE_WIDTH_IN, GRID, GRID_LINE_WIDTH, INK, LEGEND_SIZE, OUTLINE_WIDTH, PANEL_TITLE_SIZE, TICK_LABEL_SIZE
from Utils.manuscript_figure_style import EXPORT_DPI, GRID, INK, MID, PAPER, TICK_LABEL_SIZE, add_panel_label as add_shared_panel_label, configure_manuscript_style, rsp_activity_cmap
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
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

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig07'

FIGURE_STEM: Path

H5AD_PATH: Path

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

RSP_ACTIVITY_CMAP = rsp_activity_cmap("r4_activity")

RSP_ACTIVITY_NORM = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

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

def loading_basis_pair(loadings: pd.DataFrame, rsp_a: str, rsp_b: str) -> pd.DataFrame:
    columns = [
        "gene",
        "loading",
        "loading_rank",
        "loading_fraction",
        "in_contribution_70",
        "is_core_gene",
    ]
    a = loadings.loc[loadings["rsp_id"].astype(str).eq(rsp_a), columns].rename(
        columns={column: f"{rsp_a}_{column}" for column in columns if column != "gene"}
    )
    b = loadings.loc[loadings["rsp_id"].astype(str).eq(rsp_b), columns].rename(
        columns={column: f"{rsp_b}_{column}" for column in columns if column != "gene"}
    )
    out = a.merge(b, on="gene", how="outer")
    for rsp_id in (rsp_a, rsp_b):
        for column in ("loading", "loading_fraction"):
            out[f"{rsp_id}_{column}"] = out[f"{rsp_id}_{column}"].fillna(0.0)
        for column in ("in_contribution_70", "is_core_gene"):
            out[f"{rsp_id}_{column}"] = out[f"{rsp_id}_{column}"].fillna(False).astype(bool)
    out["loading_difference"] = out[f"{rsp_a}_loading"] - out[f"{rsp_b}_loading"]
    out["shared_core"] = out[f"{rsp_a}_is_core_gene"] & out[f"{rsp_b}_is_core_gene"]
    out["shared_contribution70"] = out[f"{rsp_a}_in_contribution_70"] & out[f"{rsp_b}_in_contribution_70"]
    out["loading_class"] = np.select(
        [out["shared_core"], out["loading_difference"] > 0, out["loading_difference"] < 0],
        ["shared_core", f"{rsp_a}_weighted", f"{rsp_b}_weighted"],
        default="balanced_or_zero",
    )
    return out.sort_values("loading_difference", ascending=False).reset_index(drop=True)

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


QUANTITATIVE_COLUMN_LEFTS = (0.458, 0.749)

INVASIVE_RSPS = ("RSP0001", "RSP0015", "RSP0004")

CARRIER_RSP = "RSP0000"

ROW1_RSP_ORDER = (*INVASIVE_RSPS, CARRIER_RSP)

INVASIVE_RSP_COLORS = {
    "RSP0001": "#4C5680",
    "RSP0015": REGAL_COLORS["magenta"],
    "RSP0004": "#B77A42",
}

INVASIVE_CONTEXT_COLORS = {
    "invasive": "#8E7065",
    "stromal": "#8A7B45",
    "macrophages": "#6C5591",
    "endothelial": "#3F7A73",
}

CONTEXT_FEATURES = {
    "fraction_cells_contact_Stromal": ("Stromal", INVASIVE_CONTEXT_COLORS["stromal"]),
    "fraction_cells_contact_Endothelial": ("Endothelial", INVASIVE_CONTEXT_COLORS["endothelial"]),
    "fraction_cells_contact_Macrophages_1": ("Macrophages 1", INVASIVE_CONTEXT_COLORS["macrophages"]),
    "fraction_cells_contact_Macrophages_2": ("Macrophages 2", INVASIVE_CONTEXT_COLORS["macrophages"]),
}

HIGHLIGHT_CONTEXT = {
    "RSP0015": "fraction_cells_contact_Stromal",
    "RSP0004": "fraction_cells_contact_Endothelial",
}

LOADING_LABEL_GENES = {
    "RSP0015": ("EXO1", "FOXM1", "GATA3"),
    "RSP0004": ("MKI67", "CCNB1", "ITGB6"),
}

LOADING_LABEL_OFFSETS = {
    "RSP0015": {
        "EXO1": (-7, 7),
        "FOXM1": (-7, 10),
        "GATA3": (7, 7),
    },
    "RSP0004": {
        "MKI67": (-7, 7),
        "CCNB1": (-7, 20),
        "ITGB6": (7, -10),
    },
}

CARRIER_GROUP_ORDER = ("cavity_core", "depressed_rim", "outer_carrier", "distant_carrier")

CARRIER_GROUP_LABELS = {
    "cavity_core": "Cavity",
    "depressed_rim": "Rim",
    "outer_carrier": "Outer\nring",
    "distant_carrier": "Other\nactive",
}

CARRIER_GROUP_COLORS = {
    "cavity_core": REGAL_COLORS["red"],
    "depressed_rim": "#C9877B",
    "outer_carrier": "#6B93B6",
    "distant_carrier": REGAL_COLORS["mid"],
}

VOLCANO_LABEL_OFFSETS = {
    "G0S2": (-8, 9),
    "ERBB2": (7, 9),
    "ESR1": (-7, 10),
    "PGR": (7, 8),
}

FOCAL_PATHWAY_LABELS = {
    "focal_region_up": [
        ("DISEASES_OF_SIGNAL_TRANSDUCTION_BY_GROWTH_FACTOR_RECEPTORS", "Growth-factor receptor signaling"),
        ("CELL_JUNCTION_ORGANIZATION", "Cell-junction organization"),
        ("CELL_CELL_COMMUNICATION", "Cell–cell communication"),
        ("REGULATION_OF_CDH1_FUNCTION", "Regulation of CDH1 function"),
    ],
    "reference_region_up": [
        ("PEPTIDE_LIGAND_BINDING_RECEPTORS", "Peptide-ligand receptors"),
        ("CHEMOKINE_RECEPTORS_BIND_CHEMOKINES", "Chemokine–receptor binding"),
        ("ANTIMICROBIAL_PEPTIDES", "Antimicrobial peptides"),
        ("GPCR_LIGAND_BINDING", "GPCR ligand binding"),
    ],
}

@dataclass
class Figure2Sources:
    image: np.ndarray
    xy: pd.DataFrame
    spot_radius_px: float
    px_per_um: float
    activation: pd.DataFrame
    rsd_long: pd.DataFrame
    mapping: pd.DataFrame
    gene_loadings: pd.DataFrame
    invasive_context: pd.DataFrame
    invasive_continuous_spots: pd.DataFrame
    carrier_spatial: pd.DataFrame
    carrier_de: pd.DataFrame
    carrier_pathway: pd.DataFrame

def load_figure2_sources() -> Figure2Sources:
    selection = load_selection()
    (image, xy, radius) = load_spatial_image_context(H5AD_PATH, sample_id=SAMPLE_ID)
    mapping = load_xenium_mapping(XENIUM_MAPPING_PATH)
    rsp_tables = load_rsp_tables(SAMPLE_DIR)
    rsmd_tables = load_rsmd_tables(SAMPLE_DIR)
    activation = activation_wide(rsp_tables['activation'], ROW1_RSP_ORDER).set_index('spot_id')
    (rsd_long, _, _) = selected_rsd_membership(selection, rsmd_tables['domains'], rsmd_tables['membership'])
    sources = Figure2Sources(image=image, xy=xy, spot_radius_px=radius, px_per_um=estimate_hires_px_per_um(mapping), activation=activation, rsd_long=rsd_long, mapping=mapping, gene_loadings=rsp_tables['gene_loadings'].copy(), invasive_context=pd.read_csv(table('invasive_context_readback.csv')), invasive_continuous_spots=pd.read_csv(table('invasive_continuous_context_spots.csv')), carrier_spatial=pd.read_csv(table('carrier_disruption_spatial_readback.csv')), carrier_de=pd.read_csv(table('carrier_disruption_de_readback.csv')), carrier_pathway=pd.read_csv(table('carrier_disruption_pathway_readback.csv')))
    validate_figure2_contract(sources)
    return sources

def validate_figure2_contract(s: Figure2Sources) -> None:
    required_activation = {
        *(f"activation_{rsp_id}" for rsp_id in ROW1_RSP_ORDER),
        *(f"active_{rsp_id}" for rsp_id in ROW1_RSP_ORDER),
    }
    missing_activation = sorted(required_activation.difference(s.activation.columns))
    if missing_activation:
        raise ValueError(f"Figure 2 activation table is missing columns: {missing_activation}")
    for rsp_id in ROW1_RSP_ORDER:
        values = pd.to_numeric(s.activation[f"activation_{rsp_id}"], errors="coerce").dropna()
        if values.empty or values.min() < 0.0 or values.max() > 1.0:
            raise ValueError(f"{rsp_id} activation is outside the shared 0-1 display contract")

    validate_source_columns(
        s.rsd_long,
        {"spot_id", "rsp_id", "domain_id", "block"},
        name="selected RSD membership",
    )
    validate_source_columns(
        s.mapping,
        {"visium_barcode", "label", "x_visium_hires", "y_visium_hires"},
        name="Xenium-to-Visium mapping",
    )
    validate_source_columns(
        s.gene_loadings,
        {"rsp_id", "gene", "loading", "loading_rank"},
        name="rsp_gene_loadings.parquet",
    )
    validate_source_columns(
        s.invasive_context,
        {
            "readback_type", "target_rsp", "reference_rsp", "feature",
            "standardized_beta", "ci_low", "ci_high", "q_value",
        },
        name="invasive_context_readback.csv",
    )
    validate_source_columns(
        s.invasive_continuous_spots,
        {
            "target_rsp", "reference_rsp", "feature", "adjusted_context", "adjusted_signal",
        },
        name="invasive_continuous_context_spots.csv",
    )
    context_rows = s.invasive_context.loc[
        s.invasive_context["readback_type"].astype(str).eq("continuous_context_association")
    ]
    expected_pairs = {(target, feature) for target in ("RSP0015", "RSP0004") for feature in CONTEXT_FEATURES}
    observed_pairs = set(zip(context_rows["target_rsp"].astype(str), context_rows["feature"].astype(str)))
    if observed_pairs != expected_pairs:
        raise ValueError("Figure 2F/G continuous-context rows do not match the 2 x 4 contract")
    validate_source_columns(
        s.carrier_spatial,
        {
            "readback_type", "spot_id", "carrier_group", "activation_RSP0000",
            "geneset_RSP0000_core_or_contribution70",
        },
        name="carrier_disruption_spatial_readback.csv",
    )
    validate_source_columns(
        s.carrier_de,
        {"gene", "log2fc_case_vs_reference", "q_value"},
        name="carrier_disruption_de_readback.csv",
    )

    spot_rows = s.carrier_spatial.loc[
        s.carrier_spatial["readback_type"].astype(str).eq("spot_gradient")
    ].drop_duplicates("spot_id")
    focal_counts = spot_rows["carrier_group"].value_counts().to_dict()
    if focal_counts.get("cavity_core") != 3 or focal_counts.get("depressed_rim") != 9:
        raise ValueError(f"Unexpected focal-disruption spot counts: {focal_counts}")
    if set(CARRIER_GROUP_ORDER).difference(focal_counts):
        raise ValueError(f"Missing focal-disruption carrier groups: {focal_counts}")
    validate_source_columns(
        s.carrier_pathway,
        {
            "gene_set_id", "term", "database", "q_value",
        },
        name="carrier_disruption_pathway_readback.csv",
    )
    if set(s.carrier_pathway["gene_set_id"].astype(str)) != set(FOCAL_PATHWAY_LABELS):
        raise ValueError("Figure 2J pathway table must contain focal_region_up and reference_region_up")
    if not s.carrier_pathway["database"].astype(str).eq("Reactome").all():
        raise ValueError("Figure 2J pathway table must use Reactome only")

def _activation_values(s: Figure2Sources, rsp_id: str) -> pd.Series:
    return s.activation[f"activation_{rsp_id}"].reindex(s.xy.index).fillna(0.0).astype(float)

def _active_mask(s: Figure2Sources, rsp_id: str) -> pd.Series:
    return s.activation[f"active_{rsp_id}"].reindex(s.xy.index).fillna(False).astype(bool)

def _row1_image_and_xy(s: Figure2Sources) -> tuple[np.ndarray, pd.DataFrame]:
    image, xy = _crop_to_spot_window(
        s.image,
        s.xy.copy(),
        padding_px=s.spot_radius_px * 2.5,
    )
    raw_like = np.clip((image - 0.30) / 0.70, 0.0, 1.0)
    return np.clip(raw_like * 0.82 + 0.18, 0.0, 1.0), xy

def _invasive_territory_spots(s: Figure2Sources) -> set[str]:
    return set(
        s.rsd_long.loc[
            s.rsd_long["rsp_id"].astype(str).isin(("RSP0001", "RSP0015")),
            "spot_id",
        ].astype(str)
    )

def _invasive_context_spots(s: Figure2Sources) -> set[str]:
    return set(
        s.rsd_long.loc[
            s.rsd_long["rsp_id"].astype(str).isin(INVASIVE_RSPS),
            "spot_id",
        ].astype(str)
    )

def _focal_disruption_spots(s: Figure2Sources) -> set[str]:
    spot_rows = s.carrier_spatial.loc[
        s.carrier_spatial["readback_type"].astype(str).eq("spot_gradient")
        & s.carrier_spatial["carrier_group"].astype(str).isin(("cavity_core", "depressed_rim"))
    ]
    return set(spot_rows["spot_id"].astype(str))

def _draw_orientation_map(
    ax: plt.Axes,
    s: Figure2Sources,
    image: np.ndarray,
    xy: pd.DataFrame,
) -> None:
    draw_histology(ax, image)
    invasive_ids = _invasive_territory_spots(s)
    focal_ids = _focal_disruption_spots(s)
    invasive = xy.loc[xy.index.intersection(invasive_ids)]
    focal = xy.loc[xy.index.intersection(focal_ids)]
    highlighted = invasive_ids | focal_ids
    other = xy.loc[~xy.index.astype(str).isin(highlighted)]

    ax.scatter(
        other["x_img"], other["y_img"], s=2.9, facecolors="none", edgecolors="#9DA3A7",
        linewidths=0.20, alpha=0.34, rasterized=True, zorder=3,
    )
    ax.scatter(
        invasive["x_img"], invasive["y_img"], s=3.0, facecolors="none",
        edgecolors=REGAL_COLORS["blue"], linewidths=0.38, alpha=0.96,
        rasterized=True, zorder=4,
    )
    ax.scatter(
        focal["x_img"], focal["y_img"], s=5.0, facecolors=REGAL_COLORS["red"],
        edgecolors="white", linewidths=0.24, alpha=0.98, rasterized=True, zorder=5,
    )
    handles = [
        Line2D(
            [0], [0], marker="s", color=REGAL_COLORS["blue"], markerfacecolor="none",
            markeredgewidth=0.9, markersize=4.6, lw=0, label="Invasive territory",
        ),
        Line2D(
            [0], [0], marker="o", color=REGAL_COLORS["red"], markerfacecolor=REGAL_COLORS["red"],
            markeredgewidth=0.0, markersize=4.2, lw=0, label="Focal disruption",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        prop={"size": LEGEND_SIZE},
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.7,
    )
    ax.set_title(
        "Orientation",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, image)

def _draw_full_tissue_activation(
    ax: plt.Axes,
    s: Figure2Sources,
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
        cmap=RSP_ACTIVITY_CMAP, norm=RSP_ACTIVITY_NORM, linewidths=0, alpha=0.86,
        rasterized=True, zorder=3,
    )
    ax.set_title(
        rsp_id,
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, image)

def build_row1_spatial_entry(s: Figure2Sources) -> plt.Figure:
    image, xy = _row1_image_and_xy(s)
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.05), constrained_layout=False)
    grid = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.04], wspace=0.012)
    ax_a = fig.add_subplot(grid[0, 0])
    _draw_orientation_map(ax_a, s, image, xy)

    activation_axes = [fig.add_subplot(grid[0, index]) for index in range(1, 5)]
    for ax, rsp_id in zip(activation_axes, ROW1_RSP_ORDER, strict=True):
        _draw_full_tissue_activation(ax, s, rsp_id, image, xy)

    colorbar_axis = fig.add_subplot(grid[0, 5])
    fig.subplots_adjust(left=0.018, right=0.985, top=0.91, bottom=0.079)
    bbox = colorbar_axis.get_position()
    colorbar_axis.remove()
    colorbar_axis = fig.add_axes([
        bbox.x0 + bbox.width * 0.18 - 0.006,
        bbox.y0 + bbox.height * 0.22,
        bbox.width * 0.58,
        bbox.height * 0.56,
    ])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=RSP_ACTIVITY_NORM, cmap=RSP_ACTIVITY_CMAP),
        cax=colorbar_axis,
    )
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
    add_panel_label(fig, "B", activation_axes[0])
    return fig

def _invasive_context_limits(s: Figure2Sources) -> tuple[float, float, float, float]:
    territory = s.xy.loc[s.xy.index.intersection(_invasive_context_spots(s)), ["x_img", "y_img"]]
    return bounds_from_points(
        territory,
        padding=s.spot_radius_px * 2.2,
        fixed_width=max(float(territory["x_img"].max() - territory["x_img"].min()) + 4.4 * s.spot_radius_px, 1.38 * (float(territory["y_img"].max() - territory["y_img"].min()) + 4.4 * s.spot_radius_px)),
    )

def _draw_invasive_context_maps(axes: Sequence[plt.Axes], s: Figure2Sources) -> None:
    limits = _invasive_context_limits(s)
    context_spots = _invasive_context_spots(s)
    mapping = s.mapping.loc[
        s.mapping["visium_barcode"].astype(str).isin(context_spots)
    ].copy()
    image = np.clip(np.asarray(s.image, dtype=float) * 0.20 + 0.80, 0.0, 1.0)
    specs = [
        (
            ("Invasive_Tumor", "Prolif_Invasive_Tumor"),
            "Invasive tumor",
            INVASIVE_CONTEXT_COLORS["invasive"],
        ),
        (("Stromal",), "Stromal", INVASIVE_CONTEXT_COLORS["stromal"]),
        (
            ("Macrophages_1", "Macrophages_2"),
            "Macrophages",
            INVASIVE_CONTEXT_COLORS["macrophages"],
        ),
        (("Endothelial",), "Endothelial", INVASIVE_CONTEXT_COLORS["endothelial"]),
    ]
    for ax, (labels, cell_name, color) in zip(axes, specs, strict=True):
        cells = mapping.loc[mapping["label"].astype(str).isin(labels)]
        draw_histology(ax, image)
        ax.scatter(
            mapping["x_visium_hires"], mapping["y_visium_hires"],
            s=0.12, c="#9FA3A3", alpha=0.055, linewidths=0, rasterized=True, zorder=2,
        )
        ax.scatter(
            cells["x_visium_hires"], cells["y_visium_hires"],
            s=0.62, c=color, alpha=0.76, linewidths=0, rasterized=True, zorder=4,
        )
        if cell_name == "Invasive tumor":
            proliferative = mapping.loc[mapping["label"].astype(str).eq("Prolif_Invasive_Tumor")]
            ax.scatter(
                proliferative["x_visium_hires"], proliferative["y_visium_hires"],
                s=0.72, c=REGAL_COLORS["red"], alpha=0.84, linewidths=0,
                rasterized=True, zorder=5,
            )
        title = "Invasive / proliferative" if cell_name == "Invasive tumor" else cell_name
        ax.set_title(
            title,
            loc="center",
            fontsize=PANEL_TITLE_SIZE,
            color=color,
            fontweight="bold",
            pad=4.0,
        )
        finish_spatial_axis(ax, s.image, limits=limits)
    add_scale_bar(axes[-1], px_per_um=s.px_per_um, length_um=250)

def _continuous_context_data(s: Figure2Sources) -> pd.DataFrame:
    data = s.invasive_context.loc[
        s.invasive_context["readback_type"].astype(str).eq("continuous_context_association")
        & s.invasive_context["target_rsp"].astype(str).isin(("RSP0015", "RSP0004"))
        & s.invasive_context["feature"].astype(str).isin(CONTEXT_FEATURES),
    ].copy()
    for column in ("standardized_beta", "ci_low", "ci_high", "q_value"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    expected = len(CONTEXT_FEATURES) * 2
    if len(data) != expected:
        raise ValueError(f"Expected {expected} continuous-context rows for Figure 2F/G, found {len(data)}")
    return data

def _loading_pair_data(s: Figure2Sources, target_rsp: str) -> pd.DataFrame:
    data = loading_basis_pair(s.gene_loadings, "RSP0001", target_rsp)
    keep = (
        data["RSP0001_is_core_gene"]
        | data[f"{target_rsp}_is_core_gene"]
        | data["RSP0001_in_contribution_70"]
        | data[f"{target_rsp}_in_contribution_70"]
    )
    return data.loc[keep].copy()

def _draw_loading_pair(ax: plt.Axes, s: Figure2Sources, target_rsp: str) -> None:
    data = _loading_pair_data(s, target_rsp)
    reference_loading = pd.to_numeric(data["RSP0001_loading"], errors="coerce").fillna(0.0)
    target_loading = pd.to_numeric(data[f"{target_rsp}_loading"], errors="coerce").fillna(0.0)
    shared = data["shared_core"] | data["shared_contribution70"]
    reference_weighted = ~shared & reference_loading.gt(target_loading)
    target_weighted = ~shared & target_loading.gt(reference_loading)
    ax.scatter(reference_loading, target_loading, s=7, c="#D9DCDD", alpha=0.55,
               linewidths=0, rasterized=True, zorder=1)
    ax.scatter(reference_loading.loc[shared], target_loading.loc[shared], s=11,
               c=REGAL_COLORS["mid"], alpha=0.72, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(reference_loading.loc[reference_weighted], target_loading.loc[reference_weighted],
               s=12, c=INVASIVE_RSP_COLORS["RSP0001"], alpha=0.78,
               linewidths=0, rasterized=True, zorder=3)
    ax.scatter(reference_loading.loc[target_weighted], target_loading.loc[target_weighted],
               s=12, c=INVASIVE_RSP_COLORS[target_rsp], alpha=0.82,
               linewidths=0, rasterized=True, zorder=4)
    limit = max(float(reference_loading.max()), float(target_loading.max())) * 1.08
    ax.plot([0.0, limit], [0.0, limit], color="#AEB3B5", lw=0.6, linestyle=(0, (2, 2)), zorder=0)
    indexed = data.set_index(data["gene"].astype(str), drop=False)
    label_genes = LOADING_LABEL_GENES[target_rsp]
    missing_labels = sorted(set(label_genes).difference(indexed.index))
    if missing_labels:
        raise ValueError(f"Missing fixed loading labels for {target_rsp}: {missing_labels}")
    labels = indexed.loc[list(label_genes)]
    for row in labels.itertuples(index=False):
        x_value = float(getattr(row, "RSP0001_loading"))
        y_value = float(getattr(row, f"{target_rsp}_loading"))
        offset = LOADING_LABEL_OFFSETS[target_rsp][str(row.gene)]
        ax.annotate(
            str(row.gene),
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right",
            va="top" if offset[1] < 0 else "bottom",
            fontsize=ANNOTATION_SIZE + 0.5,
            fontstyle="italic",
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.2},
            arrowprops={
                "arrowstyle": "-",
                "color": REGAL_COLORS["mid"],
                "linewidth": 0.40,
                "shrinkA": 1.0,
                "shrinkB": 1.0,
            },
        )
    ax.set_xlim(-0.02 * limit, limit)
    ax.set_ylim(-0.02 * limit, limit)
    ax.set_aspect("auto")
    ax.set_xlabel("RSP0001 loading", fontsize=AXIS_LABEL_SIZE, labelpad=1.5)
    ax.set_ylabel(f"{target_rsp} loading", fontsize=AXIS_LABEL_SIZE, labelpad=1.5)
    ax.tick_params(
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.grid(color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.set_title(
        f"RSP0001 vs {target_rsp}",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        color=INK,
        fontweight="bold",
        pad=4.0,
    )

def _draw_combined_context_coefficients(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_limits: tuple[float, float],
) -> None:
    features = list(CONTEXT_FEATURES)
    y_positions = np.arange(len(features))[::-1]
    ax.axvline(0.0, color="#AEB3B5", lw=0.65, linestyle=(0, (2, 2)), zorder=0)
    handles = []
    for target_rsp, offset in (("RSP0015", 0.10), ("RSP0004", -0.10)):
        target = summary.loc[
            summary["target_rsp"].astype(str).eq(target_rsp)
        ].set_index("feature")
        rows = target.loc[features]
        positions = y_positions + offset
        beta = rows["standardized_beta"].to_numpy(float)
        low = rows["ci_low"].to_numpy(float)
        high = rows["ci_high"].to_numpy(float)
        color = INVASIVE_RSP_COLORS[target_rsp]
        ax.hlines(positions, low, high, color=color, lw=1.0, zorder=2)
        ax.scatter(
            beta, positions, s=18, c=color,
            edgecolors="white", linewidths=0.45, zorder=3,
        )
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=4.2,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.45, label=target_rsp,
            )
        )
    ax.set_yticks(y_positions, [CONTEXT_FEATURES[feature][0] for feature in features])
    ax.set_xlim(*x_limits)
    ax.set_ylim(-0.65, len(features) - 0.35)
    ax.set_xlabel(
        "Independent association (β)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=1.5,
    )
    ax.tick_params(
        axis="x",
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.tick_params(
        axis="y",
        labelsize=ANNOTATION_SIZE,
        length=0,
        pad=1.5,
    )
    ax.grid(axis="x", color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=LEGEND_SIZE,
        frameon=False,
        handlelength=0.7,
        handletextpad=0.20,
        columnspacing=0.70,
        borderaxespad=0.0,
    )

def _draw_adjusted_context_facets(
    axes: Sequence[plt.Axes],
    s: Figure2Sources,
    summary: pd.DataFrame,
) -> None:
    series = []
    for target_rsp in ("RSP0015", "RSP0004"):
        feature = HIGHLIGHT_CONTEXT[target_rsp]
        data = s.invasive_continuous_spots.loc[
            s.invasive_continuous_spots["target_rsp"].astype(str).eq(target_rsp)
            & s.invasive_continuous_spots["feature"].astype(str).eq(feature)
        ].copy()
        x = pd.to_numeric(data["adjusted_context"], errors="coerce")
        y = pd.to_numeric(data["adjusted_signal"], errors="coerce")
        valid = x.notna() & y.notna()
        x, y = x.loc[valid], y.loc[valid]
        row = summary.loc[
            summary["target_rsp"].astype(str).eq(target_rsp)
            & summary["feature"].astype(str).eq(feature)
        ].iloc[0]
        series.append((target_rsp, feature, x, y, row))

    all_x = pd.concat([item[2] for item in series], ignore_index=True)
    all_y = pd.concat([item[3] for item in series], ignore_index=True)
    x_padding = 0.05 * max(float(all_x.max() - all_x.min()), 1.0)
    y_padding = 0.05 * max(float(all_y.max() - all_y.min()), 1.0)
    x_limits = (float(all_x.min() - x_padding), float(all_x.max() + x_padding))
    y_limits = (float(all_y.min() - y_padding), float(all_y.max() + y_padding))

    for index, (ax, item) in enumerate(zip(axes, series, strict=True)):
        target_rsp, feature, x, y, row = item
        color = INVASIVE_RSP_COLORS[target_rsp]
        ax.scatter(
            x, y, s=3.4, c=color, alpha=0.14,
            linewidths=0, rasterized=True, zorder=1,
        )
        line_x = np.linspace(float(x.quantile(0.01)), float(x.quantile(0.99)), 100)
        ax.plot(
            line_x,
            float(row.standardized_beta) * line_x,
            color=color,
            lw=1.2,
            zorder=3,
        )
        ax.text(
            0.04,
            0.96,
            f"β={float(row.standardized_beta):.2f}\nq={float(row.q_value):.2g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_SIZE,
            color=color,
            fontweight="bold",
        )
        ax.axhline(0.0, color="#D3D6D7", lw=0.45, zorder=0)
        ax.axvline(0.0, color="#D3D6D7", lw=0.45, zorder=0)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.text(
            0.96,
            0.06,
            CONTEXT_FEATURES[feature][0],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=AXIS_LABEL_SIZE,
            color=CONTEXT_FEATURES[feature][1],
            fontweight="bold",
            zorder=6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 0.8,
            },
        )
        if index == 0:
            ax.set_ylabel(
                "Adjusted program signal",
                fontsize=AXIS_LABEL_SIZE,
                labelpad=1.5,
            )
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(
            labelsize=TICK_LABEL_SIZE,
            length=1.8,
            width=0.8,
        )
        ax.set_title(
            target_rsp,
            loc="center",
            fontsize=PANEL_TITLE_SIZE,
            color=color,
            fontweight="bold",
            pad=4.0,
        )

def build_row2_invasive_organization(s: Figure2Sources) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 4.75), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, width_ratios=[0.39, 0.61], wspace=0.18)
    spatial_grid = outer[0, 0].subgridspec(2, 2, wspace=0.05, hspace=0.26)
    spatial_axes = [
        fig.add_subplot(spatial_grid[row, column]) for row in range(2) for column in range(2)
    ]
    _draw_invasive_context_maps(spatial_axes, s)

    right = outer[0, 1].subgridspec(
        2,
        2,
        wspace=0.28,
    )
    loading_axes = [fig.add_subplot(right[0, column]) for column in range(2)]
    _draw_loading_pair(loading_axes[0], s, "RSP0015")
    _draw_loading_pair(loading_axes[1], s, "RSP0004")
    context_summary = _continuous_context_data(s)
    coefficient_min = float(context_summary["ci_low"].min())
    coefficient_max = float(context_summary["ci_high"].max())
    coefficient_padding = 0.08 * (coefficient_max - coefficient_min)
    coefficient_limits = (coefficient_min - coefficient_padding, coefficient_max + coefficient_padding)
    coefficient_axis = fig.add_subplot(right[1, 0])
    adjusted_grid = right[1, 1].subgridspec(1, 2, wspace=0.22)
    adjusted_axes = [
        fig.add_subplot(adjusted_grid[0, 0]),
        fig.add_subplot(adjusted_grid[0, 1]),
    ]
    _draw_combined_context_coefficients(
        coefficient_axis,
        context_summary,
        coefficient_limits,
    )
    _draw_adjusted_context_facets(adjusted_axes, s, context_summary)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.94, bottom=0.085)
    fig.canvas.draw()
    upper_y0 = min(axis.get_position().y0 for axis in spatial_axes[:2])
    upper_y1 = max(axis.get_position().y1 for axis in spatial_axes[:2])
    for axis in loading_axes:
        bounds = axis.get_position()
        axis.set_position([bounds.x0, upper_y0, bounds.width, upper_y1 - upper_y0])
    lower_y0 = min(axis.get_position().y0 for axis in spatial_axes[2:])
    lower_y1 = max(axis.get_position().y1 for axis in spatial_axes[2:])
    for axis in [coefficient_axis, *adjusted_axes]:
        bounds = axis.get_position()
        axis.set_position([bounds.x0, lower_y0, bounds.width, lower_y1 - lower_y0])
    for axis in (loading_axes[0], coefficient_axis):
        bounds = axis.get_position()
        axis.set_position([
            QUANTITATIVE_COLUMN_LEFTS[0],
            bounds.y0,
            bounds.width,
            bounds.height,
        ])
    adjusted_left = adjusted_axes[0].get_position()
    adjusted_right = adjusted_axes[-1].get_position()
    figure_height_pt = fig.bbox.height * 72.0 / fig.dpi
    fig.text(
        (adjusted_left.x0 + adjusted_right.x1) / 2.0,
        lower_y0 - 17.0 / figure_height_pt,
        "Adjusted context",
        ha="center",
        va="top",
        fontsize=AXIS_LABEL_SIZE,
        color=INK,
    )
    add_panel_label(fig, "C", spatial_axes[0], align_to_canvas_left=True)
    add_panel_label(fig, "D", loading_axes[0])
    add_panel_label(fig, "E", loading_axes[1])
    add_panel_label(fig, "F", coefficient_axis)
    add_panel_label(fig, "G", adjusted_axes[0])
    return fig

def _carrier_gradient_spots(s: Figure2Sources) -> pd.DataFrame:
    return (
        s.carrier_spatial.loc[
            s.carrier_spatial["readback_type"].astype(str).eq("spot_gradient")
            & s.carrier_spatial["carrier_group"].astype(str).isin(CARRIER_GROUP_ORDER)
        ]
        .drop_duplicates("spot_id")
        .copy()
    )

def _draw_carrier_gradient(
    ax: plt.Axes,
    spots: pd.DataFrame,
    feature: str,
    y_label: str,
    *,
    seed: int,
) -> None:
    positions = np.arange(len(CARRIER_GROUP_ORDER), dtype=float)
    values_by_group = [
        pd.to_numeric(
            spots.loc[spots["carrier_group"].astype(str).eq(group), feature], errors="coerce"
        ).dropna().to_numpy(float)
        for group in CARRIER_GROUP_ORDER
    ]
    boxes = ax.boxplot(
        values_by_group,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": REGAL_COLORS["ink"], "linewidth": 0.8},
        whiskerprops={"color": REGAL_COLORS["mid"], "linewidth": 0.6},
        capprops={"color": REGAL_COLORS["mid"], "linewidth": 0.6},
    )
    for patch, group in zip(boxes["boxes"], CARRIER_GROUP_ORDER, strict=True):
        color = CARRIER_GROUP_COLORS[group]
        patch.set_facecolor(mpl.colors.to_rgba(color, 0.26))
        patch.set_edgecolor(color)
        patch.set_linewidth(0.75)

    rng = np.random.default_rng(seed)
    medians = []
    for position, group, values in zip(positions, CARRIER_GROUP_ORDER, values_by_group, strict=True):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        is_distant = group == "distant_carrier"
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=1.4 if is_distant else 6.0,
            c=CARRIER_GROUP_COLORS[group],
            alpha=0.12 if is_distant else 0.52,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        medians.append(float(np.median(values)))
    ax.plot(positions, medians, color=REGAL_COLORS["ink"], lw=0.8, zorder=4)
    ax.scatter(positions, medians, s=12, c=REGAL_COLORS["ink"], edgecolors="white", linewidths=0.4, zorder=5)

    counts = spots["carrier_group"].astype(str).value_counts()
    ax.set_xticks(
        positions,
        [f"{CARRIER_GROUP_LABELS[group]}\nn={int(counts[group])}" for group in CARRIER_GROUP_ORDER],
    )
    ax.set_xlim(-0.55, len(CARRIER_GROUP_ORDER) - 0.45)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.tick_params(
        axis="x",
        labelsize=ANNOTATION_SIZE,
        length=0,
        pad=2.0,
    )
    ax.tick_params(
        axis="y",
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.grid(axis="y", color=GRID, linewidth=GRID_LINE_WIDTH)

def _draw_focal_volcano(ax: plt.Axes, s: Figure2Sources) -> None:
    data = s.carrier_de.copy()
    data["log2fc"] = pd.to_numeric(data["log2fc_case_vs_reference"], errors="coerce")
    q_values = pd.to_numeric(data["q_value"], errors="coerce").clip(lower=np.finfo(float).tiny)
    data["minus_log10_q"] = -np.log10(q_values)
    data = data.dropna(subset=["log2fc", "minus_log10_q"]).copy()
    anchor_extent = data.loc[
        data["is_reported_structure_anchor"].astype(bool), "log2fc"
    ].abs().max()
    x_cap = min(
        max(float(data["log2fc"].abs().quantile(0.995)), float(anchor_extent) + 0.45, 4.0),
        7.0,
    )
    y_cap = min(max(float(data["minus_log10_q"].quantile(0.995)), 6.0), 10.0)
    data["x_plot"] = data["log2fc"].clip(-x_cap, x_cap)
    data["y_plot"] = data["minus_log10_q"].clip(0.0, y_cap)

    signature = data["q_value"].le(0.05) & data["log2fc"].abs().ge(0.5)
    colors = np.full(len(data), "#C7CCD0", dtype=object)
    colors[signature.to_numpy() & data["log2fc"].lt(0).to_numpy()] = REGAL_COLORS["blue"]
    colors[signature.to_numpy() & data["log2fc"].gt(0).to_numpy()] = REGAL_COLORS["red"]
    ax.scatter(
        data["x_plot"],
        data["y_plot"],
        s=3.4,
        c=colors,
        alpha=0.78,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.axvline(-0.5, color="#8C8C8C", lw=0.45, linestyle=(0, (2, 2)), zorder=0)
    ax.axvline(0.5, color="#8C8C8C", lw=0.45, linestyle=(0, (2, 2)), zorder=0)
    ax.axhline(-np.log10(0.05), color="#8C8C8C", lw=0.45, linestyle=(0, (2, 2)), zorder=0)

    indexed = data.set_index(data["gene"].astype(str), drop=False)
    missing_labels = sorted(set(VOLCANO_LABEL_OFFSETS).difference(indexed.index))
    if missing_labels:
        raise ValueError(f"Missing fixed volcano labels: {missing_labels}")
    for gene, offset in VOLCANO_LABEL_OFFSETS.items():
        row = indexed.loc[gene]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        point_x = float(row["x_plot"])
        point_y = float(row["y_plot"])
        ax.annotate(
            gene,
            (point_x, point_y),
            xytext=offset,
            textcoords="offset points",
            ha="right" if point_x < 0 or offset[0] < 0 else "left",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontstyle="italic",
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.25},
            arrowprops={
                "arrowstyle": "-",
                "color": REGAL_COLORS["mid"],
                "linewidth": 0.45,
                "shrinkA": 1.0,
                "shrinkB": 1.0,
            },
            clip_on=True,
        )

    ax.set_xlim(-x_cap * 1.04, x_cap * 1.04)
    ax.set_ylim(-0.15, y_cap * 1.04)
    ax.set_xlabel("log2FC", fontsize=AXIS_LABEL_SIZE, labelpad=1.5)
    ax.set_ylabel("−log10 q", fontsize=AXIS_LABEL_SIZE, labelpad=1.5)
    ax.tick_params(
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.grid(color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.text(
        0.02, 0.98, "Reference region", transform=ax.transAxes, ha="left", va="top",
        fontsize=ANNOTATION_SIZE, color=REGAL_COLORS["blue"], fontweight="bold",
    )
    ax.text(
        0.98, 0.98, "Focal region", transform=ax.transAxes, ha="right", va="top",
        fontsize=ANNOTATION_SIZE, color=REGAL_COLORS["red"], fontweight="bold",
    )

def _select_focal_pathways(s: Figure2Sources, direction: str) -> pd.DataFrame:
    local = s.carrier_pathway.loc[
        s.carrier_pathway["gene_set_id"].astype(str).eq(direction)
        & pd.to_numeric(s.carrier_pathway["q_value"], errors="coerce").le(0.05)
    ].copy()
    rows = []
    for pattern, label in FOCAL_PATHWAY_LABELS[direction]:
        match = local.loc[local["term"].astype(str).str.contains(pattern, regex=False)].sort_values("q_value")
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["display_term"] = label
        rows.append(row)
    return pd.DataFrame(rows)

def _draw_focal_pathway_mirror(ax: plt.Axes, s: Figure2Sources) -> None:
    up = _select_focal_pathways(s, "focal_region_up").copy()
    down = _select_focal_pathways(s, "reference_region_up").copy()
    if len(up) != 4 or len(down) != 4:
        raise ValueError(f"Figure 2J requires four pathways per direction, found up={len(up)}, down={len(down)}")
    up["score"] = -np.log10(pd.to_numeric(up["q_value"], errors="coerce").clip(lower=1e-300))
    down["score"] = -np.log10(pd.to_numeric(down["q_value"], errors="coerce").clip(lower=1e-300))
    up = up.sort_values("score", ascending=True).reset_index(drop=True)
    down = down.sort_values("score", ascending=True).reset_index(drop=True)

    gap = 0.50
    y_down = np.arange(len(down), dtype=float)
    y_up = np.arange(len(up), dtype=float) + len(down) + gap
    ax.barh(y_down, -down["score"], height=0.58, color=REGAL_COLORS["blue"], alpha=0.86)
    ax.barh(y_up, up["score"], height=0.58, color=REGAL_COLORS["red"], alpha=0.86)
    ax.axvline(0, color="#4B4F50", linewidth=0.55)
    for y_value, label in zip(y_down, down["display_term"], strict=True):
        ax.text(
            0.16, y_value, label, ha="left", va="center",
            fontsize=ANNOTATION_SIZE, color=INK,
        )
    for y_value, label in zip(y_up, up["display_term"], strict=True):
        ax.text(
            -0.16, y_value, label, ha="right", va="center",
            fontsize=ANNOTATION_SIZE, color=INK,
        )

    max_score = max(float(up["score"].max()), float(down["score"].max()), 1.0)
    limit = np.ceil(max_score / 2.0) * 2.0
    ax.set_xlim(-limit * 1.04, limit * 1.04)
    ax.set_ylim(-0.65, len(up) + len(down) + gap - 0.35)
    ticks = np.linspace(-limit, limit, 5)
    ax.set_xticks(ticks, [f"{abs(value):.0f}" for value in ticks])
    ax.set_yticks([])
    ax.set_xlabel("−log10 q", fontsize=AXIS_LABEL_SIZE, labelpad=17.0)
    ax.tick_params(
        axis="x",
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.grid(axis="x", color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.02, -0.115, "Reference region", transform=ax.transAxes, ha="left", va="top",
        fontsize=ANNOTATION_SIZE, fontweight="bold", color=REGAL_COLORS["blue"],
        clip_on=False,
    )
    ax.text(
        0.98, -0.115, "Focal region", transform=ax.transAxes, ha="right", va="top",
        fontsize=ANNOTATION_SIZE, fontweight="bold", color=REGAL_COLORS["red"],
        clip_on=False,
    )

def build_row3_focal_disruption(s: Figure2Sources) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.35), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.44, 0.29, 0.27],
        wspace=0.22,
    )
    gradient_grid = outer[0, 0].subgridspec(1, 2, wspace=0.26)
    gradient_axes = [fig.add_subplot(gradient_grid[0, column]) for column in range(2)]
    spots = _carrier_gradient_spots(s)
    _draw_carrier_gradient(
        gradient_axes[0], spots, "activation_RSP0000", "RSP0000 activation", seed=31,
    )
    _draw_carrier_gradient(
        gradient_axes[1], spots, "geneset_RSP0000_core_or_contribution70",
        "Loading-derived score", seed=37,
    )
    volcano_axis = fig.add_subplot(outer[0, 1])
    _draw_focal_volcano(volcano_axis, s)
    pathway_axis = fig.add_subplot(outer[0, 2])
    _draw_focal_pathway_mirror(pathway_axis, s)

    fig.subplots_adjust(left=0.040, right=0.985, top=0.90, bottom=0.185)
    fig.canvas.draw()
    for axis, target_left in zip(
        (volcano_axis, pathway_axis),
        QUANTITATIVE_COLUMN_LEFTS,
        strict=True,
    ):
        bounds = axis.get_position()
        axis.set_position([target_left, bounds.y0, bounds.width, bounds.height])
    add_panel_label(fig, "H", gradient_axes[0], align_to_canvas_left=True)
    add_panel_label(fig, "I", volcano_axis)
    add_panel_label(fig, "J", pathway_axis)
    return fig


def export_figure2() -> list[Path]:
    configure_figure_style()
    sources = load_figure2_sources()
    builders = (build_row1_spatial_entry, build_row2_invasive_organization, build_row3_focal_disruption,)
    rows = (render_row(builder(sources)) for builder in builders)
    return save_composite_bundle(stack_row_images(rows), FIGURE_STEM)

def draw_figure() -> None:
    for output in export_figure2():
        print(output)

def configure():
    global FIGURE_STEM, H5AD_PATH, SAMPLE_DIR, SAMPLE_ID, XENIUM_MAPPING_PATH
    SAMPLE_ID = option('sample_id')
    SAMPLE_DIR = path('sample_dir')
    H5AD_PATH = path('h5ad')
    XENIUM_MAPPING_PATH = path('xenium_mapping')
    FIGURE_STEM = OUTPUT_DIR / 'fig07'

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
    _CONFIG = document['fig07']
    configure()
    result = draw_figure()
    if result is not None:
        for output in result:
            print(output)

if __name__ == '__main__':
    main()
