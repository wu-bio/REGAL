"""Plot manuscript fig05 using paths from the shared YAML configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
import argparse
from string import Template
from copy import deepcopy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RSP.visualize_rsp import _active_activation_for_rsp, _coords_to_image_space, _crop_to_spot_window, _estimate_spot_radius_px, _load_spatial_image_context, _orient_tissue_image
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from Utils.manuscript_figure_style import ANNOTATION_SIZE, AXIS_LABEL_SIZE, COLORBAR_SIZE, EXPORT_DPI, FIGURE_WIDTH_IN, GRID, GRID_LINE_WIDTH, INK, MID, OUTLINE_WIDTH, PANEL_TITLE_SIZE, TICK_LABEL_SIZE, add_panel_label, configure_manuscript_style, contribution_heatmap_cmap, marker_expression_cmap, rsp_activity_cmap
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
    'GO_Biological_Process_2023.gmt': RESOURCE_DIR / 'fig05' / 'GO_Biological_Process_2023.gmt',
    'deepst_deg_table': RESOURCE_DIR / 'fig05' / 'Supplementary Table 1.csv',
}

FIGURE_SELECTION = {'sample': {'sample_id': 'Breast_BlockA_Section1'},
 'selected_rsp_objects': [{'rsp_id': 'RSP0003',
                           'landscape_relations': ['within_annotation_activity_layers',
                                                   'tumor_structure_activity_backbone'],
                           'narrative_role': 'IDC_2 broad tumor activity layer and IDC comparator '
                                             'backbone',
                           'used_in_panels': ['idc2_activity_layers', 'dcis_lcis_vs_idc_backbone'],
                           'selection_note': 'Manual representative for the broad IDC_2-associated '
                                             'tumor activity backbone.',
                           'rsds': ['RSP0003:D000011']},
                          {'rsp_id': 'RSP0004',
                           'landscape_relations': ['within_annotation_activity_layers'],
                           'narrative_role': 'IDC_2 local activity sublayer',
                           'used_in_panels': ['idc2_activity_layers'],
                           'selection_note': 'Manual representative for a local activity layer '
                                             'inside the IDC_2 annotated region.',
                           'rsds': ['RSP0004:D000014']},
                          {'rsp_id': 'RSP0008',
                           'landscape_relations': ['within_annotation_activity_layers'],
                           'narrative_role': 'IDC_2 overlay activity layer',
                           'used_in_panels': ['idc2_activity_layers'],
                           'selection_note': 'Manual representative for an additional '
                                             'IDC_2-associated overlay activity layer.',
                           'rsds': ['RSP0008:D000029']},
                          {'rsp_id': 'RSP0005',
                           'landscape_relations': ['tumor_structure_activity_backbone',
                                                   'immune_like_overlay'],
                           'narrative_role': 'DCIS/LCIS-associated tumor activity backbone',
                           'used_in_panels': ['dcis_lcis_vs_idc_backbone', 'immune_like_overlay'],
                           'selection_note': 'Manual representative for DCIS/LCIS-associated tumor '
                                             'RSDs and their overlap with immune-like activity.',
                           'rsds': ['RSP0005:D000015']},
                          {'rsp_id': 'RSP0018',
                           'landscape_relations': ['immune_like_overlay'],
                           'narrative_role': 'Immune-like activity layer overlaying '
                                             'tumor-associated spatial entities',
                           'used_in_panels': ['immune_like_overlay'],
                           'selection_note': 'Manual representative with lymphoid/immune '
                                             'loading-gene support, used to analyze local overlay '
                                             'on RSP0005-backed entities.',
                           'rsds': ['RSP0018:D000089']},
                          {'rsp_id': 'RSP0009',
                           'landscape_relations': ['discrete_rsp_instantiation',
                                                   'entity_level_realized_contribution_variation'],
                           'narrative_role': 'Recurrent activity layer instantiated as multiple '
                                             'discrete RSD entities',
                           'used_in_panels': ['rsp0009_discrete_entities',
                                              'entity_realized_contribution_summary'],
                           'selection_note': 'Manual representative for entity-level comparison of '
                                             'realized contribution from one shared source RSP '
                                             'loading-gene backbone.',
                           'rsds': ['RSP0009:D000030',
                                    'RSP0009:D000031',
                                    'RSP0009:D000032',
                                    'RSP0009:D000033']}]}

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig05'

PANEL_DIR = Path(__file__).resolve().parents[1] / 'output' / 'figures'


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
    "manual_annotation",
    "manual_annotation_fine",
    "rctd_dominant",
    "rctd_dominant_prop",
    "rctd_epithelial_prop",
    "rctd_cancer_epithelial_prop",
    "rctd_normal_epithelial_prop",
    "copykat_pred_default",
    "copykat_pred_tirosh",
    "cnv_score",
    "cnv_cor",
    "final_dominant",
    "final_dominant_prop",
    "final_flag",
)

def load_spot_context(
    h5ad_path: str | Path,
    *,
    sample_id: str,
    obs_columns: Iterable[str] = DEFAULT_OBS_COLUMNS,
    spatial_key: str = "spatial",
) -> pd.DataFrame:
    """Load coordinates and selected obs context columns from an h5ad file."""

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        out = pd.DataFrame({"spot_id": adata.obs_names.astype(str), "sample_id": sample_id})
        if spatial_key in adata.obsm:
            spatial = np.asarray(adata.obsm[spatial_key])[:, :2]
            out["x"] = spatial[:, 0].astype(float)
            out["y"] = spatial[:, 1].astype(float)
        elif {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(adata.obs.columns):
            out["x"] = pd.to_numeric(adata.obs["pxl_col_in_fullres"], errors="coerce").to_numpy(float)
            out["y"] = pd.to_numeric(adata.obs["pxl_row_in_fullres"], errors="coerce").to_numpy(float)
        else:
            raise ValueError(f"Missing spatial coordinates in {h5ad_path}")

        for col in obs_columns:
            if col not in adata.obs:
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
    genes: Iterable[str] | None = None,
    *,
    target_sum: float = 1.0e4,
    layer: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return log1p library-normalized expression for selected genes or all genes."""

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        requested = None if genes is None else list(dict.fromkeys(str(g) for g in genes if str(g)))
        if requested is None:
            selected = adata.var_names.astype(str).tolist()
        else:
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

def _dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)

def _row_totals(adata: ad.AnnData) -> np.ndarray:
    if "total_counts" in adata.obs:
        return pd.to_numeric(adata.obs["total_counts"], errors="coerce").fillna(0).to_numpy(dtype=float)
    matrix = adata.X
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    if hasattr(matrix, "to_memory"):
        in_memory = matrix.to_memory()
        if sparse.issparse(in_memory):
            return np.asarray(in_memory.sum(axis=1)).ravel().astype(float)
        return np.asarray(in_memory).sum(axis=1).astype(float)
    return np.asarray(matrix).sum(axis=1).astype(float)

REQUIRED_RSP_FILES = {
    "rsp": "rsp.parquet",
    "activation": "rsp_activation.parquet",
    "gene_loadings": "rsp_gene_loadings.parquet",
    "core_gene_loadings": "rsp_core_gene_loadings.parquet",
}

REQUIRED_RSMD_FILES = {
    "domains": "domains.parquet",
    "membership": "domain_spot_membership.parquet",
    "metadomains": "metadomains.parquet",
}

def load_selection():
    selection = deepcopy(FIGURE_SELECTION)
    selection['sample'].update(h5ad_path=str(path('h5ad')), sample_dir=str(path('sample_dir')))
    selection['verification_references'] = {'deepst_deg_table': {'path': str(resource('deepst_deg_table'))}}
    return selection

def rsp_out_dir(sample_dir: str | Path) -> Path:
    path = Path(sample_dir)
    return path if path.name == "rsp_out" else path / "rsp_out"

def rsmd_out_dir(sample_dir: str | Path) -> Path:
    path = Path(sample_dir)
    return path if path.name == "rsmd_out" else path / "rsmd_out"

def load_rsp_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = rsp_out_dir(sample_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_RSP_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing RSP table: {path}")
        tables[key] = pd.read_parquet(path)
    return tables

def load_rsmd_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = rsmd_out_dir(sample_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_RSMD_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing RSMD table: {path}")
        tables[key] = pd.read_parquet(path)
    return tables

def parse_rsd_ref(rsd_ref: str) -> tuple[str, str]:
    parts = str(rsd_ref).split(":")
    if len(parts) != 2 or not parts[0].startswith("RSP") or not parts[1].startswith("D"):
        raise ValueError(f"Invalid RSD reference {rsd_ref!r}; expected RSPxxxx:Dxxxxx")
    return parts[0], parts[1]

def domain_spots(membership: pd.DataFrame, domain_key: str) -> list[str]:
    return membership.loc[membership["domain_key"].astype(str).eq(str(domain_key)), "spot_id"].astype(str).tolist()



ROW_FIGURE_HEIGHTS = {
    "row1": 2.25,
    "row2": 3.35,
    "row3": 5.35,
    "row4": 3.45,
}

RIGHT_COLORBAR_X = 0.964

RIGHT_COLORBAR_WIDTH = 0.004

DISPLAY_RSP_IDS = (
    "RSP0003",
    "RSP0004",
    "RSP0008",
    "RSP0005",
    "RSP0018",
    "RSP0009",
)

REGAL_COLORS = {
    "blue": "#3676B8",
    "red": "#D94C5C",
    "teal": "#2A9D8F",
    "warm": "#9E7467",
}

@dataclass(frozen=True)
class Row1Sources:
    image_coordinates: pd.DataFrame
    tissue_image: np.ndarray
    activation: pd.DataFrame
    rsd_spots: dict[str, list[str]]

@dataclass(frozen=True)
class Row2Sources:
    selected_genes: dict[str, list[str]]
    expression: pd.DataFrame
    rsp_spots: dict[str, list[str]]
    context: pd.DataFrame

@dataclass(frozen=True)
class Row3Sources:
    image_coordinates: pd.DataFrame
    expression: pd.DataFrame
    rsd_spots: dict[str, list[str]]
    idc2_de: pd.DataFrame
    backbone_de: pd.DataFrame
    contribution: pd.DataFrame
    go: pd.DataFrame

@dataclass(frozen=True)
class Row4Sources:
    go: pd.DataFrame
    readback: pd.DataFrame
    statistics: pd.DataFrame

@dataclass(frozen=True)
class R3FigureSources:
    row1: Row1Sources
    row2: Row2Sources
    row3: Row3Sources
    row4: Row4Sources

def load_figure_sources() -> R3FigureSources:
    selection = load_selection()
    sample = selection['sample']
    h5ad_path = path('h5ad')
    sample_dir = path('sample_dir')
    coordinates = load_spot_context(h5ad_path, sample_id=sample['sample_id'])
    coordinates['spot_id'] = coordinates['spot_id'].astype(str)
    image_context = _load_spatial_image_context(h5ad_path, preferred_image='hires')
    if image_context is None:
        raise ValueError(f'Missing spatial image context in {h5ad_path}')
    (tissue_image, scale) = image_context
    image_coordinates = _coords_to_image_space(coordinates, scale)
    tissue_image = _orient_tissue_image(tissue_image, image_coordinates)
    rsp_tables = load_rsp_tables(sample_dir)
    rsmd_tables = load_rsmd_tables(sample_dir)
    row1_radius = _estimate_spot_radius_px(image_coordinates)
    (row1_image, row1_xy) = _crop_to_spot_window(tissue_image, image_coordinates, padding_px=row1_radius * 2.5)
    row1 = Row1Sources(image_coordinates=row1_xy, tissue_image=row1_image, activation=rsp_tables['activation'].copy(), rsd_spots=ROW1__selected_rsd_spots(selection, rsmd_tables['domains'], rsmd_tables['membership']))
    selected_genes = ROW2__validated_representative_genes(rsp_tables['gene_loadings'])
    requested_genes = [gene for genes in selected_genes.values() for gene in genes]
    (row2_expression, present_genes) = load_normalized_expression(h5ad_path, requested_genes)
    selected_genes = {rsp_id: [gene for gene in genes if gene in present_genes] for (rsp_id, genes) in selected_genes.items()}
    rsp_spots = ROW2__selected_rsp_spots(selection, rsmd_tables['domains'], rsmd_tables['membership'])
    row2 = Row2Sources(selected_genes=selected_genes, expression=row2_expression, rsp_spots=rsp_spots, context=ROW2__rsp_context_rows())
    rsd_spots = ROW3__rsd_spot_sets(rsmd_tables['domains'], rsmd_tables['membership'], ROW3_MARKER_REGION_REFS)
    (row3_expression, present_markers) = load_normalized_expression(h5ad_path, ROW3_MARKER_GENES)
    missing_markers = sorted(set(ROW3_MARKER_GENES) - set(present_markers))
    if missing_markers:
        raise ValueError(f'Marker genes missing from h5ad: {missing_markers}')
    row3 = Row3Sources(image_coordinates=image_coordinates, expression=row3_expression, rsd_spots=rsd_spots, idc2_de=pd.read_csv(table('idc2_region_a_vs_b_de_readback.csv')), backbone_de=pd.read_csv(table('dcis_lcis_vs_idc_de_readback.csv')), contribution=pd.read_csv(table('rsp0009_entity_realized_contribution.csv')), go=pd.read_csv(table('loading_gene_go_annotation.csv')))
    row4 = Row4Sources(go=row3.go, readback=pd.read_csv(table('dcis_lcis_overlay_spot_readback.csv')), statistics=pd.read_csv(table('dcis_lcis_overlay_statistics.csv')))
    return R3FigureSources(row1=row1, row2=row2, row3=row3, row4=row4)

ROW1_PANEL_TITLE_KWARGS = {
    'loc': 'center',
    'fontsize': PANEL_TITLE_SIZE,
    'fontweight': 'bold',
    'pad': 4.0,
}

ROW1_ACTIVITY_CMAP = rsp_activity_cmap('r3_activity')

ROW1_ACTIVITY_NORM = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

ROW1_RSD_OUTLINE_COLOR = INK

ROW1_RSD_OUTLINE_HALO_WIDTH = 1.5

ROW1_RSD_OUTLINE_WIDTH = 0.65

ROW1_RSD_SMOOTHING_STEPS = 3

def ROW1__selected_rsd_spots(
    selection: dict,
    domains: pd.DataFrame,
    membership: pd.DataFrame,
) -> dict[str, list[str]]:
    domain_index = domains.set_index(['rsp_seed_id', 'domain_id'], drop=False)
    selected: dict[str, list[str]] = {}
    for item in selection['selected_rsp_objects']:
        rsp_id = str(item['rsp_id'])
        for rsd_ref in item.get('rsds', []):
            ref_rsp, domain_id = parse_rsd_ref(str(rsd_ref))
            if ref_rsp != rsp_id or (rsp_id, domain_id) not in domain_index.index:
                raise ValueError(f'Invalid selected RSD for row 1: {rsd_ref}')
            domain_key = str(domain_index.loc[(rsp_id, domain_id), 'domain_key'])
            selected[str(rsd_ref)] = domain_spots(membership, domain_key)
    return selected

def _populate_row1(
    container: mpl.figure.FigureBase,
    sources: Row1Sources,
) -> None:
    image = ROW1__soften_tissue_image(sources.tissue_image)
    grid = container.add_gridspec(
        1,
        6,
        width_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.060,
        left=0.022,
        right=0.956,
        top=0.89,
        bottom=0.05,
    )
    axes = [
        container.add_subplot(grid[0, column])
        for column in range(6)
    ]
    for (ax, rsp_id) in zip(axes, DISPLAY_RSP_IDS, strict=True):
        ROW1__draw_rsp_activation(
            ax,
            sources.image_coordinates,
            image,
            sources.activation,
            rsp_id,
        )
        ROW1__draw_rsd_annotations(
            ax,
            sources.image_coordinates,
            sources.activation,
            sources.rsd_spots,
            rsp_id,
        )
    cax = container.add_axes(
        [RIGHT_COLORBAR_X, 0.240, RIGHT_COLORBAR_WIDTH, 0.460]
    )
    ROW1__draw_activity_colorbar(container, cax)
    add_panel_label(
        container,
        'A',
        axes[0].get_position(),
        align_to_canvas_left=True,
    )


def ROW1__draw_rsp_activation(ax: plt.Axes, xy: pd.DataFrame, image: np.ndarray, activation: pd.DataFrame, rsp_id: str) -> None:
    ax.imshow(image, origin='upper')
    ax.scatter(xy['x_img'], xy['y_img'], s=2.9, facecolors='none', edgecolors=INK, linewidths=0.17, alpha=0.14, rasterized=True)
    active_activation = _active_activation_for_rsp(activation, rsp_id)
    active_xy = xy.loc[xy.index.intersection(active_activation.index)].copy()
    if not active_xy.empty:
        active_values = active_activation.loc[active_xy.index].to_numpy(dtype=float)
        ax.scatter(active_xy['x_img'], active_xy['y_img'], s=2.9, c=active_values, cmap=ROW1_ACTIVITY_CMAP, norm=ROW1_ACTIVITY_NORM, linewidths=0, alpha=0.84, rasterized=True)
    ax.set_title(rsp_id, **ROW1_PANEL_TITLE_KWARGS)
    ROW1__finish_image_ax(ax, image)

def ROW1__draw_rsd_annotations(
    ax: plt.Axes,
    xy: pd.DataFrame,
    activation: pd.DataFrame,
    rsd_spots: dict[str, list[str]],
    rsp_id: str,
) -> None:
    coordinates = xy[['x_img', 'y_img']].to_numpy(dtype=float)
    triangulation = ROW1__make_tissue_triangulation(coordinates)
    displayed_active = xy.index.isin(
        _active_activation_for_rsp(activation, rsp_id).index
    )
    for rsd_ref, spots in rsd_spots.items():
        ref_rsp, domain_id = parse_rsd_ref(rsd_ref)
        if ref_rsp != rsp_id:
            continue
        selected = xy.index.isin(spots) & displayed_active
        if int(selected.sum()) < 3:
            continue
        membership = ROW1__smooth_rsd_membership(
            triangulation,
            selected.astype(float),
        )
        membership[~displayed_active] = 0.0
        ax.tricontour(
            triangulation,
            membership,
            levels=[0.5],
            colors=['white'],
            linewidths=ROW1_RSD_OUTLINE_HALO_WIDTH,
            zorder=7,
        )
        ax.tricontour(
            triangulation,
            membership,
            levels=[0.5],
            colors=[ROW1_RSD_OUTLINE_COLOR],
            linewidths=ROW1_RSD_OUTLINE_WIDTH,
            zorder=8,
        )
        label_x, label_y = ROW1__interior_label_position(coordinates, selected)
        ax.text(
            label_x,
            label_y,
            f'D{int(domain_id[1:])}',
            ha='center',
            va='center',
            fontsize=ANNOTATION_SIZE,
            fontweight='bold',
            color=INK,
            bbox={
                'facecolor': 'white',
                'edgecolor': 'none',
                'alpha': 0.78,
                'pad': 0.45,
            },
            zorder=9,
        )

def ROW1__make_tissue_triangulation(coordinates: np.ndarray) -> mtri.Triangulation:
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

def ROW1__smooth_rsd_membership(
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
    for _ in range(ROW1_RSD_SMOOTHING_STEPS):
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

def ROW1__interior_label_position(
    coordinates: np.ndarray,
    selected: np.ndarray,
) -> tuple[float, float]:
    inside = coordinates[selected]
    outside = coordinates[~selected]
    if outside.size == 0:
        center = np.median(inside, axis=0)
        return float(center[0]), float(center[1])
    nonmember_clearance = cKDTree(outside).query(inside, k=1)[0]
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    tissue_edge_clearance = np.minimum.reduce(
        [
            inside[:, 0] - x_min,
            x_max - inside[:, 0],
            inside[:, 1] - y_min,
            y_max - inside[:, 1],
        ]
    )
    clearance = np.minimum(nonmember_clearance, tissue_edge_clearance)
    point = inside[int(np.argmax(clearance))]
    return float(point[0]), float(point[1])

def ROW1__draw_activity_colorbar(
    container: mpl.figure.FigureBase,
    cax: plt.Axes,
) -> None:
    mappable = mpl.cm.ScalarMappable(norm=ROW1_ACTIVITY_NORM, cmap=ROW1_ACTIVITY_CMAP)
    colorbar = container.colorbar(mappable, cax=cax)
    colorbar.set_ticks([0.0, 1.0])
    colorbar.ax.set_yticklabels(['0', '1'])
    colorbar.ax.tick_params(
        labelsize=COLORBAR_SIZE,
        length=2.0,
        width=0.8,
        pad=1.5,
    )
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    colorbar.set_label(
        'RSP activation',
        fontsize=COLORBAR_SIZE,
        labelpad=3.0,
    )

def ROW1__soften_tissue_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype.kind in {'u', 'i'}:
        arr_float = arr.astype(float) / np.iinfo(arr.dtype).max
    else:
        arr_float = arr.astype(float)
        if np.nanmax(arr_float) > 1.0:
            arr_float = arr_float / 255.0
    softened = arr_float * 0.74 + 0.26
    return np.clip(softened, 0.0, 1.0)

def ROW1__finish_image_ax(ax: plt.Axes, image: np.ndarray) -> None:
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

ROW2_GENES_PER_RSP = 4

ROW2_REPRESENTATIVE_GENES = {
    'RSP0003': ['SLITRK6', 'VTCN1', 'CYP4B1', 'CEACAM6'],
    'RSP0004': ['PTGES', 'NAT1', 'ABCC11', 'ABCC12'],
    'RSP0008': ['SHISA2', 'MAB21L4', 'PDZK1IP1', 'GINS2'],
    'RSP0005': ['KLHDC7B', 'FCGR3B', 'SCUBE3', 'EXOC2'],
    'RSP0018': ['TRBC2', 'CD52', 'CCL5', 'PTPRC'],
    'RSP0009': ['KRT37', 'KAT6B', 'GPC3', 'HK2'],
}

ROW2_MANUAL_LABELS = {'RSP0003': 'Invasive', 'RSP0004': 'Invasive', 'RSP0008': 'Invasive', 'RSP0005': 'Tumor', 'RSP0018': 'Surrounding', 'RSP0009': 'Tumor'}

ROW2_EXPRESSION_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    'r3_expression',
    ['#F2F3EE', '#D9E3D5', '#B3CDB5', '#7EAA88', '#4F7E62'],
)

ROW2_EXPRESSION_NORM = mpl.colors.Normalize(vmin=0.0, vmax=1.5)

def ROW2__validated_representative_genes(gene_loadings: pd.DataFrame) -> dict[str, list[str]]:
    selected = {
        rsp_id: list(ROW2_REPRESENTATIVE_GENES[rsp_id])
        for rsp_id in DISPLAY_RSP_IDS
    }
    for rsp_id in DISPLAY_RSP_IDS:
        genes = selected[rsp_id]
        if len(genes) != ROW2_GENES_PER_RSP:
            raise ValueError(
                f'{rsp_id} requires {ROW2_GENES_PER_RSP} representative genes'
            )
        sub = gene_loadings.loc[
            gene_loadings['rsp_id'].astype(str).eq(rsp_id)
            & gene_loadings['gene'].astype(str).isin(genes)
        ].copy()
        valid_genes = set(
            sub.loc[
                pd.to_numeric(sub['loading'], errors='coerce').fillna(0).gt(0),
                'gene',
            ].astype(str)
        )
        invalid_genes = [gene for gene in genes if gene not in valid_genes]
        if invalid_genes:
            raise ValueError(
                f'{rsp_id} representative genes are not positive loading genes: '
                f'{invalid_genes}'
            )
    return selected

def ROW2__selected_rsp_spots(selection: dict, domains: pd.DataFrame, membership: pd.DataFrame) -> dict[str, list[str]]:
    domain_index = domains.set_index(['rsp_seed_id', 'domain_id'], drop=False)
    out: dict[str, list[str]] = {}
    for item in selection['selected_rsp_objects']:
        rsp_id = str(item['rsp_id'])
        if rsp_id not in DISPLAY_RSP_IDS:
            continue
        spots: list[str] = []
        for rsd_ref in item.get('rsds', []):
            (ref_rsp, domain_id) = parse_rsd_ref(str(rsd_ref))
            if ref_rsp != rsp_id or (rsp_id, domain_id) not in domain_index.index:
                continue
            domain_key = str(domain_index.loc[(rsp_id, domain_id), 'domain_key'])
            spots.extend(domain_spots(membership, domain_key))
        out[rsp_id] = list(dict.fromkeys(spots))
    return out

def ROW2__rsp_context_rows() -> pd.DataFrame:
    rows = [
        {'rsp_id': rsp_id, 'manual': ROW2_MANUAL_LABELS[rsp_id]}
        for rsp_id in DISPLAY_RSP_IDS
    ]
    return pd.DataFrame(rows).set_index('rsp_id')

def _populate_row2(
    container: mpl.figure.FigureBase,
    sources: Row2Sources,
) -> None:
    columns = [
        (rsp_id, gene)
        for rsp_id in DISPLAY_RSP_IDS
        for gene in sources.selected_genes.get(rsp_id, [])
    ]
    grid = container.add_gridspec(
        1,
        2,
        width_ratios=[0.935, 0.065],
        wspace=0.015,
        left=0.075,
        right=0.960,
        top=0.93,
        bottom=0.11,
    )
    axis = container.add_subplot(grid[0, 0])
    context_axis = container.add_subplot(grid[0, 1], sharey=axis)
    colorbar_axis = container.add_axes(
        [RIGHT_COLORBAR_X, 0.325, RIGHT_COLORBAR_WIDTH, 0.430]
    )
    ROW2__draw_micro_violin_matrix(
        axis,
        columns,
        sources.expression,
        sources.rsp_spots,
    )
    ROW2__draw_context_strips(context_axis, sources.context)
    ROW2__draw_expression_colorbar(container, colorbar_axis)
    add_panel_label(
        container,
        'B',
        axis.get_position(),
        align_to_canvas_left=True,
    )


def ROW2__draw_micro_violin_matrix(ax: plt.Axes, columns: list[tuple[str, str]], expression: pd.DataFrame, rsp_spots: dict[str, list[str]]) -> None:
    n_rows = len(DISPLAY_RSP_IDS)
    n_cols = len(columns)
    ax.set_xlim(-0.55, n_cols - 0.45)
    ax.set_ylim(n_rows - 0.45, -0.80)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        DISPLAY_RSP_IDS,
        fontsize=TICK_LABEL_SIZE,
        fontweight='bold',
    )
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [gene for (_, gene) in columns],
        rotation=20,
        ha='right',
        va='top',
        fontsize=ANNOTATION_SIZE,
        fontstyle='italic',
    )
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in np.arange(-0.5, n_cols + 0.5, 1.0):
        ax.axvline(x, color='#EFEFEF', linewidth=0.35, zorder=0)
    for y in np.arange(-0.5, n_rows + 0.5, 1.0):
        ax.axhline(y, color='#EFEFEF', linewidth=0.35, zorder=0)
    max_expr = ROW2__robust_expression_max(expression, columns)
    for (row_index, rsp_id) in enumerate(DISPLAY_RSP_IDS):
        spots = [spot for spot in rsp_spots.get(rsp_id, []) if spot in expression.index]
        for (col_index, (_, gene)) in enumerate(columns):
            values = expression.loc[spots, gene].to_numpy(dtype=float) if spots and gene in expression.columns else np.array([])
            ROW2__draw_micro_violin(
                ax,
                col_index,
                row_index,
                values,
                max_expr=max_expr,
            )
    ROW2__draw_gene_group_brackets(ax, columns)

def ROW2__draw_micro_violin(
    ax: plt.Axes,
    x: float,
    y: float,
    values: np.ndarray,
    *,
    max_expr: float,
) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    median = float(np.median(values))
    color = ROW2_EXPRESSION_CMAP(ROW2_EXPRESSION_NORM(median))
    if np.nanmax(values) <= 0 or np.unique(np.round(values, 4)).size < 3:
        height = 0.018 + 0.11 * min(median / max_expr, 1.0)
        ax.plot([x - 0.23, x + 0.23], [y, y], color=color, linewidth=0.55, alpha=0.95, solid_capstyle='round')
        ax.plot([x, x], [y - height, y + height], color=color, linewidth=0.45, alpha=0.95, solid_capstyle='round')
        return
    grid = np.linspace(0.0, max_expr, 80)
    try:
        density = gaussian_kde(values)(grid)
    except Exception:
        (hist, edges) = np.histogram(values, bins=20, range=(0.0, max_expr), density=True)
        grid = (edges[:-1] + edges[1:]) / 2.0
        density = hist
    if not np.isfinite(density).any() or float(np.nanmax(density)) <= 0:
        return
    density = density / float(np.nanmax(density)) * 0.32
    y_grid = y + 0.34 - grid / max_expr * 0.68
    ax.fill_betweenx(y_grid, x - density, x + density, facecolor=color, edgecolor='white', linewidth=0.18, alpha=1.0, zorder=3)

def ROW2__draw_gene_group_brackets(ax: plt.Axes, columns: list[tuple[str, str]]) -> None:
    start = 0
    while start < len(columns):
        rsp_id = columns[start][0]
        end = start
        while end + 1 < len(columns) and columns[end + 1][0] == rsp_id:
            end += 1
        y = -0.68
        ax.plot([start - 0.35, end + 0.35], [y, y], color='#595959', linewidth=0.55, clip_on=False)
        ax.plot([start - 0.35, start - 0.35], [y, y + 0.08], color='#595959', linewidth=0.55, clip_on=False)
        ax.plot([end + 0.35, end + 0.35], [y, y + 0.08], color='#595959', linewidth=0.55, clip_on=False)
        ax.text(
            (start + end) / 2,
            y,
            rsp_id,
            ha='center',
            va='center',
            fontsize=ANNOTATION_SIZE,
            fontweight='bold',
            clip_on=False,
            bbox={
                'facecolor': 'white',
                'edgecolor': 'none',
                'pad': 0.65,
            },
            zorder=2,
        )
        start = end + 1

def ROW2__draw_context_strips(ax: plt.Axes, context: pd.DataFrame) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(len(DISPLAY_RSP_IDS) - 0.45, -0.80)
    ax.set_xticks([])
    ax.set_yticks(range(len(DISPLAY_RSP_IDS)))
    ax.tick_params(axis='both', length=0)
    plt.setp(ax.get_yticklabels(), visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (y, rsp_id) in enumerate(DISPLAY_RSP_IDS):
        value = str(context.loc[rsp_id, 'manual']) if rsp_id in context.index else ''
        ax.text(
            0.0,
            y,
            value,
            ha='left',
            va='center',
            fontsize=ANNOTATION_SIZE,
            color=INK,
        )
    ax.text(
        0.0,
        -0.72,
        'Manual',
        ha='left',
        va='center',
        fontsize=TICK_LABEL_SIZE,
        fontweight='bold',
    )

def ROW2__draw_expression_colorbar(
    container: mpl.figure.FigureBase,
    cax: plt.Axes,
) -> None:
    mappable = mpl.cm.ScalarMappable(
        norm=ROW2_EXPRESSION_NORM,
        cmap=ROW2_EXPRESSION_CMAP,
    )
    colorbar = container.colorbar(mappable, cax=cax)
    colorbar.set_ticks([0.0, 1.5])
    colorbar.ax.tick_params(
        labelsize=COLORBAR_SIZE,
        length=2.0,
        width=0.8,
        pad=1.5,
    )
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    colorbar.set_label(
        'log-normalized expression',
        fontsize=COLORBAR_SIZE,
        labelpad=3.0,
    )

def ROW2__robust_expression_max(expression: pd.DataFrame, columns: list[tuple[str, str]]) -> float:
    genes = [gene for (_, gene) in columns if gene in expression.columns]
    if not genes:
        return 1.0
    value = float(np.nanpercentile(expression[genes].to_numpy(dtype=float), 98))
    return max(value, 1.0)

ROW3_MARKER_REGION_REFS = ['RSP0003:D000011', 'RSP0005:D000015', 'RSP0009:D000030']

ROW3_MARKER_GENES = ['ABCC11', 'ABCC12', 'SHISA2', 'SLITRK6', 'FCGR3B', 'SCUBE3', 'KRT37', 'KAT6B', 'GPC3']

ROW3_VOLCANO_LABEL_GENES = {'IDC_2 region A vs B': ['SFRP2', 'DCN', 'ABCC12'], 'RSP0005 vs RSP0003': ['CRISP3', 'CPB1']}

ROW3_RSP0009_MATRIX_GENES = ['GPC3', 'HK2', 'CITED2', 'BNIP3', 'KRT37', 'PCED1B', 'WFDC2', 'SOX9']

ROW3_RSD_SHORT_LABELS = {'RSP0009:D000030': 'D30', 'RSP0009:D000031': 'D31', 'RSP0009:D000032': 'D32', 'RSP0009:D000033': 'D33'}

ROW3_MARKER_CMAP = marker_expression_cmap('r3_marker')

ROW3_CONTRIBUTION_CMAP = contribution_heatmap_cmap('r3_contribution')

ROW3_VOLCANO_COLORS = {'stable': GRID, 'left': REGAL_COLORS['blue'], 'right': REGAL_COLORS['red']}

ROW3_GO_COLOR = REGAL_COLORS['warm']

ROW3_RSP0009_GO_TERMS = [
    'epithelial cell differentiation',
    'intermediate filament organization',
    'epithelium development',
    'fatty acid catabolic process',
    'apoptotic mitochondrial changes',
    'negative regulation of programmed cell death',
]

def ROW3__rsd_spot_sets(domains: pd.DataFrame, membership: pd.DataFrame, rsd_refs: list[str]) -> dict[str, list[str]]:
    domain_index = domains.set_index(['rsp_seed_id', 'domain_id'], drop=False)
    out = {}
    for rsd_ref in rsd_refs:
        (rsp_id, domain_id) = parse_rsd_ref(rsd_ref)
        domain_key = str(domain_index.loc[(rsp_id, domain_id), 'domain_key'])
        out[rsd_ref] = domain_spots(membership, domain_key)
    return out

def _populate_row3(
    container: mpl.figure.FigureBase,
    sources: Row3Sources,
) -> None:
    outer = container.add_gridspec(
        1,
        2,
        width_ratios=[0.49, 0.51],
        wspace=0.090,
        left=0.015,
        right=0.985,
        top=0.95,
        bottom=0.07,
    )
    d_grid = outer[0, 0].subgridspec(
        3,
        3,
        wspace=0.012,
        hspace=0.21,
    )
    d_axes = [
        container.add_subplot(d_grid[row, column])
        for row in range(3)
        for column in range(3)
    ]
    ROW3__draw_marker_maps(
        d_axes,
        sources.image_coordinates,
        sources.expression,
        sources.rsd_spots,
    )

    right = outer[0, 1].subgridspec(
        2,
        1,
        height_ratios=[0.47, 0.53],
        hspace=0.40,
    )
    e_grid = right[0].subgridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.18,
    )
    e_axes = [
        container.add_subplot(e_grid[0, 0]),
        container.add_subplot(e_grid[0, 1]),
    ]
    ROW3__draw_volcano(
        e_axes[0],
        sources.idc2_de,
        'IDC_2 region A vs B',
        left_label='RSP0008-only',
        right_label='RSP0004-only',
    )
    ROW3__draw_volcano(
        e_axes[1],
        sources.backbone_de,
        'RSP0005 vs RSP0003',
        left_label='RSP0003',
        right_label='RSP0005',
    )

    f_grid = right[1].subgridspec(
        1,
        3,
        width_ratios=[1.0, 0.74, 1.0],
        wspace=0,
    )
    f_matrix_axis = container.add_subplot(f_grid[0, 0])
    f_go_axis = container.add_subplot(f_grid[0, 2])
    ROW3__draw_rsp0009_contribution_matrix(
        f_matrix_axis,
        sources.contribution,
    )
    ROW3__draw_rsp0009_go_terms(f_go_axis, sources.go)

    add_panel_label(
        container,
        'C',
        d_axes[0].get_position(),
        align_to_canvas_left=True,
    )
    add_panel_label(
        container,
        'D',
        e_axes[0].get_position(),
    )
    add_panel_label(
        container,
        'E',
        f_matrix_axis.get_position(),
    )


def ROW3__draw_marker_maps(axes: list[plt.Axes], xy: pd.DataFrame, expression: pd.DataFrame, rsd_spots: dict[str, list[str]]) -> None:
    region_spots = {rsd_ref: [spot for spot in rsd_spots[rsd_ref] if spot in xy.index and spot in expression.index] for rsd_ref in ROW3_MARKER_REGION_REFS}
    selected_spots = sorted({spot for spots in region_spots.values() for spot in spots})
    cropped_xy = ROW3__crop_xy_for_spots(xy, selected_spots)
    for (ax, gene) in zip(axes, ROW3_MARKER_GENES):
        ROW3__draw_deepst_style_marker_map(ax, cropped_xy, expression, region_spots, gene)

def ROW3__draw_deepst_style_marker_map(ax: plt.Axes, xy: pd.DataFrame, expression: pd.DataFrame, region_spots: dict[str, list[str]], gene: str) -> None:
    plot_spots = [spot for spots in region_spots.values() for spot in spots if spot in xy.index]
    values = expression.loc[plot_spots, gene].to_numpy(dtype=float)
    vmax = ROW3__marker_vmax(values)
    ax.scatter(xy.loc[plot_spots, 'x_img'], xy.loc[plot_spots, 'y_img'], s=3.9, c='#E8E8E8', linewidths=0, alpha=0.7, rasterized=True)
    if plot_spots:
        ax.scatter(xy.loc[plot_spots, 'x_img'], xy.loc[plot_spots, 'y_img'], s=3.9, c=values, cmap=ROW3_MARKER_CMAP, norm=mpl.colors.Normalize(vmin=0.0, vmax=vmax), linewidths=0, alpha=0.96, rasterized=True)
    ax.set_title(
        gene,
        loc='center',
        fontsize=PANEL_TITLE_SIZE,
        fontweight='bold',
        fontstyle='italic',
        pad=3.0,
    )
    ROW3__add_short_colorbar(ax, vmax)
    ROW3__finish_xy_ax(ax, xy)

def ROW3__marker_vmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 3.0
    vmax = float(np.nanpercentile(finite, 98.0))
    return max(1.5, min(5.0, np.ceil(vmax * 2.0) / 2.0))

def ROW3__add_short_colorbar(ax: plt.Axes, vmax: float) -> None:
    cax = ax.inset_axes([1.015, 0.25, 0.032, 0.48])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=vmax), cmap=ROW3_MARKER_CMAP)
    cb = plt.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(['0', f'{vmax:g}'])
    cb.ax.tick_params(
        labelsize=ANNOTATION_SIZE,
        length=1.4,
        width=0.6,
        pad=0.5,
    )

def ROW3__draw_volcano(ax: plt.Axes, de: pd.DataFrame, title: str, *, left_label: str, right_label: str) -> None:
    df = de.copy()
    x = pd.to_numeric(df['logfoldchanges'], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(df['minus_log10_padj'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
    y_cap = min(max(float(np.nanpercentile(y, 99)), 20.0), 85.0)
    y_plot = np.clip(y, 0, y_cap)
    passes = df['passes_de_threshold'].fillna(False).astype(bool).to_numpy()
    colors = np.full(len(df), ROW3_VOLCANO_COLORS['stable'], dtype=object)
    colors[passes & (x < 0)] = ROW3_VOLCANO_COLORS['left']
    colors[passes & (x > 0)] = ROW3_VOLCANO_COLORS['right']
    x_cap = min(max(float(np.nanpercentile(np.abs(x[np.isfinite(x)]), 98)), 4.0), 8.0)
    x_plot = np.clip(x, -x_cap, x_cap)
    ax.scatter(x_plot, y_plot, s=3.4, c=colors, linewidths=0, alpha=0.78, rasterized=True)
    ax.axvline(-2, color=MID, linewidth=GRID_LINE_WIDTH, linestyle=(0, (2, 2)))
    ax.axvline(2, color=MID, linewidth=GRID_LINE_WIDTH, linestyle=(0, (2, 2)))
    ax.axhline(-np.log10(0.05), color=MID, linewidth=GRID_LINE_WIDTH, linestyle=(0, (2, 2)))
    ax.set_xlim(-x_cap * 1.08, x_cap * 1.08)
    ax.set_ylim(-0.5, y_cap * 1.05)
    ROW3__label_volcano_genes(ax, df, x_plot, y_plot, title)
    ax.set_xlabel('log2FC', fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.set_ylabel('-log10 adjusted P', fontsize=AXIS_LABEL_SIZE, labelpad=1.0)
    ax.tick_params(
        axis='both',
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.grid(color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.text(
        0.02,
        0.98,
        left_label,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=ANNOTATION_SIZE,
        fontweight='bold',
        color=ROW3_VOLCANO_COLORS['left'],
    )
    ax.text(
        0.98,
        0.98,
        right_label,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=ANNOTATION_SIZE,
        fontweight='bold',
        color=ROW3_VOLCANO_COLORS['right'],
    )

def ROW3__label_volcano_genes(ax: plt.Axes, df: pd.DataFrame, x: np.ndarray, y: np.ndarray, title: str) -> None:
    genes = ROW3_VOLCANO_LABEL_GENES.get(title, [])
    index_by_gene = {gene: i for (i, gene) in enumerate(df['gene'].astype(str))}
    offsets = [1.0, 1.0]
    for (label_index, gene) in enumerate(genes):
        i = index_by_gene.get(gene)
        if i is None:
            continue
        ha = 'right' if x[i] < 0 else 'left'
        dx = -0.1 if x[i] < 0 else 0.1
        y_pos = min(float(y[i] + offsets[label_index % len(offsets)]), float(ax.get_ylim()[1] * 0.82))
        ax.text(
            x[i] + dx,
            y_pos,
            gene,
            fontsize=ANNOTATION_SIZE,
            fontstyle='italic',
            ha=ha,
            va='bottom',
            color=INK,
            clip_on=True,
            bbox={
                'facecolor': 'white',
                'edgecolor': 'none',
                'alpha': 0.75,
                'pad': 0.25,
            },
        )

def ROW3__draw_rsp0009_contribution_matrix(ax: plt.Axes, contribution: pd.DataFrame) -> None:
    genes = [gene for gene in ROW3_RSP0009_MATRIX_GENES if gene in set(contribution['gene'].astype(str))]
    regions = [ref for ref in ROW3_RSD_SHORT_LABELS if ref in set(contribution['region'].astype(str))]
    mat = contribution.loc[contribution['gene'].astype(str).isin(genes) & contribution['region'].astype(str).isin(regions)].pivot_table(index='gene', columns='region', values='realized_contribution', aggfunc='mean').reindex(index=genes, columns=regions)
    values = mat.to_numpy(dtype=float)
    row_max = np.nanmax(values, axis=1, keepdims=True)
    scaled = np.divide(values, row_max, out=np.zeros_like(values), where=np.isfinite(row_max) & (row_max > 0))
    im = ax.imshow(scaled, aspect='auto', cmap=ROW3_CONTRIBUTION_CMAP, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(
        genes,
        fontsize=ANNOTATION_SIZE,
        fontstyle='italic',
    )
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(
        [ROW3_RSD_SHORT_LABELS[r] for r in regions],
        rotation=0,
        ha='center',
        fontsize=ANNOTATION_SIZE,
    )
    ax.tick_params(length=0)
    for i in range(len(genes) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.55)
    for j in range(len(regions) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.15, 1.03, 0.40, 0.05])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([])
    cb.ax.text(
        0.04,
        0.5,
        'low',
        transform=cb.ax.transAxes,
        ha='left',
        va='center',
        fontsize=ANNOTATION_SIZE,
        color='white',
    )
    cb.ax.text(
        0.96,
        0.5,
        'high',
        transform=cb.ax.transAxes,
        ha='right',
        va='center',
        fontsize=ANNOTATION_SIZE,
        color='white',
    )
    cb.ax.xaxis.set_label_position('top')
    cb.set_label(
        'Relative contribution',
        fontsize=ANNOTATION_SIZE,
        labelpad=2.0,
    )

def ROW3__draw_rsp0009_go_terms(ax: plt.Axes, go: pd.DataFrame) -> None:
    sub = go.loc[go['rsp_id'].astype(str).eq('RSP0009')].copy()
    sub['key'] = sub['go_name_normalized'].astype(str).str.lower()
    rows = []
    for term in ROW3_RSP0009_GO_TERMS:
        hit = sub.loc[sub['key'].eq(term.lower())].sort_values('p_adj').head(1)
        if not hit.empty:
            rows.append(hit.iloc[0])
    if len(rows) < 5:
        fallback = sub.sort_values('p_adj').head(6)
        rows = list(pd.DataFrame(rows).to_dict(orient='records')) if rows else []
        seen = {str(row['go_name_normalized']).lower() for row in rows}
        for rec in fallback.to_dict(orient='records'):
            if str(rec['go_name_normalized']).lower() not in seen:
                rows.append(rec)
            if len(rows) == 6:
                break
    selected = pd.DataFrame(rows)
    selected['score'] = -np.log10(pd.to_numeric(selected['p_adj'], errors='coerce').clip(lower=1e-300))
    selected['label'] = selected['go_name_normalized'].map(ROW3__go_label)
    selected = selected.sort_values('score', ascending=True)
    y = np.arange(len(selected))
    ax.barh(y, selected['score'], color=ROW3_GO_COLOR, height=0.58, edgecolor='none', alpha=0.92)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.5, len(selected) - 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(
        selected['label'],
        fontsize=ANNOTATION_SIZE,
    )
    ax.set_xlabel(
        '-log10 adjusted P-value',
        fontsize=AXIS_LABEL_SIZE,
        labelpad=2.0,
    )
    ax.tick_params(
        axis='x',
        labelsize=TICK_LABEL_SIZE,
        length=2,
        width=0.8,
    )
    ax.tick_params(
        axis='y',
        labelsize=ANNOTATION_SIZE,
        length=2,
        width=0.8,
        pad=1.8,
    )
    ax.grid(axis='x', color=GRID, linewidth=GRID_LINE_WIDTH)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(0.45)

def ROW3__go_label(term: str) -> str:
    replacements = {
        'epithelial cell differentiation': 'epithelial differentiation',
        'intermediate filament organization': 'intermediate filament',
        'epithelium development': 'epithelium development',
        'fatty acid catabolic process': 'fatty acid catabolism',
        'apoptotic mitochondrial changes': 'mitochondrial apoptosis',
        'negative regulation of programmed cell death': 'cell-death regulation',
    }
    label = replacements.get(str(term), str(term))
    return label[:34] + '...' if len(label) > 37 else label

def ROW3__crop_xy_for_spots(xy: pd.DataFrame, spots: list[str]) -> pd.DataFrame:
    sub = xy.loc[xy.index.intersection(spots)]
    radius_px = _estimate_spot_radius_px(xy)
    pad = radius_px * 5.0
    x_min = float(sub['x_img'].min() - pad)
    x_max = float(sub['x_img'].max() + pad)
    y_min = float(sub['y_img'].min() - pad)
    y_max = float(sub['y_img'].max() + pad)
    in_crop = xy.loc[xy['x_img'].between(x_min, x_max) & xy['y_img'].between(y_min, y_max)].copy()
    in_crop['x_img'] = in_crop['x_img'] - x_min
    in_crop['y_img'] = in_crop['y_img'] - y_min
    in_crop.attrs['crop_width'] = x_max - x_min
    in_crop.attrs['crop_height'] = y_max - y_min
    return in_crop

def ROW3__finish_xy_ax(ax: plt.Axes, xy: pd.DataFrame) -> None:
    ax.set_xlim(0, float(xy.attrs['crop_width']))
    ax.set_ylim(float(xy.attrs['crop_height']), 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

ROW4_STATE_ORDER = ['RSP0005-only', 'RSP0005+RSP0018', 'RSP0018-only']

ROW4_STATE_COLORS = {'RSP0005-only': '#D98C80', 'RSP0005+RSP0018': '#7B6FA8', 'RSP0018-only': REGAL_COLORS['teal']}

ROW4_FUNCTION_COLORS = {'RSP0018': REGAL_COLORS['red'], 'RSP0005': REGAL_COLORS['blue']}

ROW4_RSP0018_TERMS = ['positive regulation of cytokine production', 'antigen receptor-mediated signaling pathway', 'inflammatory response', 't cell activation', 'cytokine-mediated signaling pathway', 'myeloid leukocyte differentiation']

ROW4_RSP0005_TERMS = ['negative regulation of apoptotic process', 'fatty acid oxidation', 'fatty acid catabolic process', 'negative regulation of epithelial cell proliferation', 'regulation of dna metabolic process', 'establishment or maintenance of epithelial cell apical/basal polarity']

def _populate_row4(
    container: mpl.figure.FigureBase,
    sources: Row4Sources,
) -> None:
    grid = container.add_gridspec(
        1,
        2,
        width_ratios=[0.36, 0.64],
        wspace=0.16,
        left=0.04,
        right=0.985,
        top=0.94,
        bottom=0.12,
    )
    g_ax = container.add_subplot(grid[0, 0])
    ROW4__draw_mirrored_go(g_ax, sources.go)
    g_bounds = g_ax.get_position()
    g_ax.set_position(
        [
            g_bounds.x0 - 0.012,
            g_bounds.y0,
            g_bounds.width,
            g_bounds.height,
        ]
    )
    h_grid = grid[0, 1].subgridspec(1, 3, wspace=0.22)
    h1 = container.add_subplot(h_grid[0, 0])
    h2 = container.add_subplot(h_grid[0, 1])
    h3 = container.add_subplot(h_grid[0, 2])
    ROW4__draw_state_violin(
        h1,
        sources.readback,
        'RSP0005 activation',
        show_ylabels=True,
    )
    ROW4__draw_state_violin(
        h2,
        sources.readback,
        'RSP0018 activation',
        show_ylabels=False,
    )
    ROW4__draw_state_violin(
        h3,
        sources.readback,
        'immune loading score',
        show_ylabels=False,
    )
    ROW4__annotate_immune_comparison(h3, sources.statistics)
    add_panel_label(
        container,
        'F',
        g_ax.get_position(),
        align_to_canvas_left=True,
    )
    add_panel_label(
        container,
        'G',
        h1.get_position(),
    )


def ROW4__draw_mirrored_go(ax: plt.Axes, go: pd.DataFrame) -> None:
    rsp18 = ROW4__selected_go_terms(go, 'RSP0018', ROW4_RSP0018_TERMS)
    rsp5 = ROW4__selected_go_terms(go, 'RSP0005', ROW4_RSP0005_TERMS)
    gap = 0.42
    y18 = np.arange(len(rsp18))[::-1] + len(rsp5) + gap
    y5 = np.arange(len(rsp5))[::-1]
    ax.barh(y18, rsp18['score'], height=0.58, color=ROW4_FUNCTION_COLORS['RSP0018'], edgecolor='none')
    ax.barh(y5, -rsp5['score'], height=0.58, color=ROW4_FUNCTION_COLORS['RSP0005'], edgecolor='none')
    ax.axvline(0, color=INK, linewidth=GRID_LINE_WIDTH)
    for (y, label) in zip(y18, rsp18['label']):
        ax.text(
            -0.15,
            y,
            label,
            ha='right',
            va='center',
            fontsize=ANNOTATION_SIZE,
            color=INK,
        )
    for (y, label) in zip(y5, rsp5['label']):
        ax.text(
            0.15,
            y,
            label,
            ha='left',
            va='center',
            fontsize=ANNOTATION_SIZE,
            color=INK,
        )
    max_score = max(float(rsp18['score'].max()), float(rsp5['score'].max()), 1.0)
    limit = max(4.0, np.ceil(max_score))
    ax.set_xlim(-limit * 1.06, limit * 1.1)
    ax.set_ylim(-0.8, len(rsp18) + len(rsp5) + gap - 0.2)
    ax.set_yticks([])
    ax.set_xlabel(
        '-log10 adjusted P-value',
        fontsize=AXIS_LABEL_SIZE,
        labelpad=2.0,
    )
    ax.tick_params(
        axis='x',
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.grid(axis='x', color=GRID, linewidth=GRID_LINE_WIDTH)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.02,
        0.94,
        'RSP0018',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=TICK_LABEL_SIZE,
        fontweight='bold',
        color=ROW4_FUNCTION_COLORS['RSP0018'],
    )
    ax.text(
        0.02,
        0.08,
        'RSP0005',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=TICK_LABEL_SIZE,
        fontweight='bold',
        color=ROW4_FUNCTION_COLORS['RSP0005'],
    )

def ROW4__selected_go_terms(go: pd.DataFrame, rsp_id: str, terms: list[str]) -> pd.DataFrame:
    sub = go.loc[go['rsp_id'].astype(str).eq(rsp_id)].copy()
    sub['key'] = sub['go_name_normalized'].astype(str).str.lower()
    chosen = []
    for term in terms:
        row = sub.loc[sub['key'].eq(term.lower())].sort_values('p_adj').head(1)
        if row.empty:
            continue
        chosen.append(row.iloc[0])
    out = pd.DataFrame(chosen)
    if out.empty:
        raise ValueError(f'No GO terms selected for {rsp_id}')
    out['score'] = -np.log10(pd.to_numeric(out['p_adj'], errors='coerce').clip(lower=1e-300))
    out['score'] = out['score'].clip(upper=12)
    out['label'] = out['go_name_normalized'].map(ROW4__term_label)
    return out

def ROW4__term_label(term: str) -> str:
    replacements = {'positive regulation of cytokine production': 'cytokine production', 'antigen receptor-mediated signaling pathway': 'antigen receptor signaling', 'cellular response to cytokine stimulus': 'response to cytokine', 'cytokine-mediated signaling pathway': 'cytokine signaling', 'myeloid leukocyte differentiation': 'myeloid differentiation', 'antiviral innate immune response': 'antiviral innate response', 'negative regulation of apoptotic process': 'anti-apoptotic regulation', 'negative regulation of epithelial cell proliferation': 'epithelial proliferation control', 'regulation of dna metabolic process': 'DNA metabolic regulation', 'establishment or maintenance of epithelial cell apical/basal polarity': 'epithelial polarity'}
    label = replacements.get(str(term), str(term))
    return label[:36] + '...' if len(label) > 39 else label

def ROW4__draw_state_violin(ax: plt.Axes, data: pd.DataFrame, value_col: str, *, show_ylabels: bool) -> None:
    values = [data.loc[data['state'].eq(state), value_col].dropna().to_numpy(float) for state in ROW4_STATE_ORDER]
    positions = np.arange(len(ROW4_STATE_ORDER))
    parts = ax.violinplot(values, positions=positions, vert=False, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for (body, state) in zip(parts['bodies'], ROW4_STATE_ORDER):
        body.set_facecolor(ROW4_STATE_COLORS[state])
        body.set_edgecolor('none')
        body.set_alpha(0.74)
    for (pos, arr, state) in zip(positions, values, ROW4_STATE_ORDER):
        if len(arr) == 0:
            continue
        (q1, med, q3) = np.nanpercentile(arr, [25, 50, 75])
        ax.plot([q1, q3], [pos, pos], color=INK, linewidth=1.2, solid_capstyle='round')
        ax.scatter([med], [pos], s=10, color=INK, zorder=4, linewidths=0)
        jitter = ROW4__deterministic_jitter(len(arr), scale=0.11)
        sample = ROW4__deterministic_sample(arr, n=160)
        ax.scatter(sample, np.full(len(sample), pos) + jitter[:len(sample)], s=2.4, color=INK, alpha=0.18, linewidths=0, rasterized=True)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        ROW4_STATE_ORDER if show_ylabels else [],
        fontsize=ANNOTATION_SIZE,
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        value_col,
        fontsize=AXIS_LABEL_SIZE,
        labelpad=2.0,
    )
    ax.tick_params(
        axis='x',
        labelsize=TICK_LABEL_SIZE,
        length=2.0,
        width=0.8,
    )
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, linewidth=GRID_LINE_WIDTH)
    for spine in ax.spines.values():
        spine.set_visible(False)

def ROW4__annotate_immune_comparison(ax: plt.Axes, statistics: pd.DataFrame) -> None:
    if len(statistics) != 1:
        raise ValueError('Expected exactly one prespecified row-4 statistical comparison.')
    row = statistics.iloc[0]
    if str(row['comparison']) != 'RSP0005+RSP0018 vs RSP0005-only':
        raise ValueError(f"Unexpected row-4 comparison: {row['comparison']}")
    p_value = float(row['p_value'])
    exponent = int(np.floor(np.log10(p_value)))
    coefficient = p_value / (10 ** exponent)
    ax.text(
        0.5,
        0.985,
        rf'$P={coefficient:.2f}\times10^{{{exponent}}}$',
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=ANNOTATION_SIZE,
        color=INK,
    )

def ROW4__deterministic_sample(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return np.sort(arr)[idx]

def ROW4__deterministic_jitter(n: int, scale: float) -> np.ndarray:
    if n == 0:
        return np.array([])
    phase = np.arange(n, dtype=float)
    return np.sin(phase * 1.61803398875) * scale

def _save_figure_bundle(
    figure: plt.Figure,
    stem: Path,
) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ('.png',):
        path = stem.with_suffix(suffix)
        figure.savefig(
            path,
            dpi=EXPORT_DPI,
            facecolor='white',
        )
        written.append(path)
    plt.close(figure)
    return written

def build_composite_figure(
    sources: R3FigureSources,
) -> plt.Figure:
    row_keys = ('row1', 'row2', 'row3', 'row4')
    row_heights = [ROW_FIGURE_HEIGHTS[key] for key in row_keys]
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_IN, sum(row_heights))
    )
    subfigures = figure.subfigures(
        len(row_keys),
        1,
        height_ratios=row_heights,
        hspace=0,
    )
    populators = (
        (_populate_row1, sources.row1),
        (_populate_row2, sources.row2),
        (_populate_row3, sources.row3),
        (_populate_row4, sources.row4),
    )
    for subfigure, (populate, row_sources) in zip(
        subfigures,
        populators,
        strict=True,
    ):
        populate(subfigure, row_sources)
    return figure

def export_tumor_activity_landscape_figure() -> list[Path]:
    configure_manuscript_style()
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    sources = load_figure_sources()
    return _save_figure_bundle(build_composite_figure(sources), PANEL_DIR / 'fig05')

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
    _CONFIG = document['fig05']
    result = export_tumor_activity_landscape_figure()
    if result is not None:
        for output in result:
            print(output)

if __name__ == '__main__':
    main()
