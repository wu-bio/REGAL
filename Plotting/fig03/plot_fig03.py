"""Plot manuscript fig03 using paths from the shared YAML configuration."""
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

from RSP.visualize_rsp import _crop_to_spot_window, _estimate_spot_radius_px, _orient_tissue_image
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from Utils.manuscript_figure_style import ANNOTATION_SIZE, AXIS_LABEL_SIZE, COLORBAR_SIZE, DATA_LINE_WIDTH, FIGURE_WIDTH_IN, GRID, GRID_LINE_WIDTH, INK, LEGEND_SIZE, METHOD_COLORS, MID, OUTLINE_WIDTH, PANEL_TITLE_SIZE, REFERENCE_LINE_WIDTH, TICK_LABEL_SIZE, add_panel_label, configure_manuscript_style
from Utils.manuscript_figure_style import EXPORT_DPI, marker_expression_cmap, rsp_activity_cmap
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

_CONFIG: dict

_CONFIG_DIR: Path
_ROOTS: dict

FIGURE_SELECTION = {'sample': {'sample_id': 'V1_Breast_Cancer_Block_A_Section_1'},
 'object_views': {'pathology_annotation': {'obs_key': 'manual_annotation',
                                           'categories': [{'value': 'Invasive',
                                                           'display_label': 'Inv.'},
                                                          {'value': 'Tumor',
                                                           'display_label': 'Tumor'},
                                                          {'value': 'Surrounding tumor',
                                                           'display_label': 'Surr.'},
                                                          {'value': 'Healthy',
                                                           'display_label': 'Healthy'}]},
                  'sparkx_gene': 'CXCL14',
                  'sparkx_reason': 'Independently significant SPARK-X gene with broad detection '
                                   'and a clearly structured continuous expression field; not '
                                   'selected from RSP loadings.',
                  'highlighted_domain': '5',
                  'highlighted_domain_reason': 'Compact lower-right STAGATE domain with multiple '
                                               'RSPs above the mean activation threshold; selected '
                                               'as the clearest spatially recognizable example for '
                                               'linking panels C and D.',
                  'highlighted_rsp': 'RSP0000',
                  'highlighted_rsp_reason': 'RSP with above-threshold activation across multiple '
                                            'STAGATE domains and a clear vertical trace in the '
                                            'domain-by-RSP matrix.',
                  'high_mean_activation_threshold': 0.2},
 'program_fields': {'matching_rule': '0.7_positive_spatial_spearman_plus_0.3_positive_gene_spearman',
                    'activities': [{'order': 2,
                                    'family': 'SIRLNT-CRAT',
                                    'display_label': 'IDC-4',
                                    'regal': 'RSP0000',
                                    'nnmf': 'signature_22',
                                    'stamp': 'Topic1',
                                    'role': 'IDC-4-associated regional field',
                                    'reason': 'representative IDC-4-associated activity with '
                                              'coherent spatial correspondence across methods'},
                                   {'order': 1,
                                    'family': 'SLITRK6-VTCN1',
                                    'display_label': 'IDC-2',
                                    'regal': 'RSP0003',
                                    'nnmf': 'signature_10',
                                    'stamp': 'Topic19',
                                    'role': 'IDC-2-associated regional field',
                                    'reason': 'localized IDC-2-associated epithelial organization '
                                              'recovered across methods'},
                                   {'order': 3,
                                    'family': 'KLHDC7B-FCGR3B',
                                    'display_label': 'DCIS',
                                    'regal': 'RSP0005',
                                    'nnmf': 'signature_23',
                                    'stamp': 'Topic14',
                                    'role': 'DCIS-associated regional field',
                                    'reason': 'focal DCIS-associated activity with a consistent '
                                              'regional footprint across methods'},
                                   {'order': 4,
                                    'family': 'KRT14-KRT17',
                                    'display_label': 'Basal',
                                    'regal': 'RSP0016',
                                    'nnmf': 'signature_20',
                                    'stamp': 'Topic11',
                                    'role': 'basal epithelial regional field',
                                    'reason': 'fine multi-focal basal structure with coherent '
                                              'high-activity correspondence across methods'},
                                   {'order': 5,
                                    'family': 'TRBC2-CD52',
                                    'display_label': 'Immune',
                                    'regal': 'RSP0018',
                                    'nnmf': 'signature_19',
                                    'stamp': 'Topic10',
                                    'role': 'immune-associated regional field',
                                    'reason': 'recurrent immune activity across separated tissue '
                                              'positions'},
                                   {'order': 6,
                                    'family': 'CCDC80-CAVIN1',
                                    'display_label': 'Stromal',
                                    'regal': 'RSP0017',
                                    'nnmf': 'signature_1',
                                    'stamp': 'Topic20',
                                    'role': 'stromal-associated regional field',
                                    'reason': 'complementary stromal organization with strong '
                                              'native matches in both neighboring methods'}]},
 'rsp_stability': {'k_grid': [4, 6, 8, 10, 12, 14, 16, 18],
                   'seeds_per_k': 3,
                   'examples': [{'role': 'broad-scale family',
                                 'rsp_id': 'RSP0019',
                                 'color': '#D88A3D',
                                 'candidate_ids': ['RSP_k4_seed0_anchored_c3',
                                                   'RSP_k4_seed1_anchored_c3',
                                                   'RSP_k6_seed0_anchored_c5'],
                                 'reason': 'lowest centered local-K window with complete '
                                           'three-seed support'},
                                {'role': 'fine-scale family',
                                 'rsp_id': 'RSP0014',
                                 'color': '#6F4C9B',
                                 'candidate_ids': ['RSP_k16_seed0_anchored_c11',
                                                   'RSP_k16_seed1_anchored_c11',
                                                   'RSP_k18_seed0_anchored_c13'],
                                 'reason': 'highest centered local-K window with complete '
                                           'three-seed support'}]}}

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig03'

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


def activity_cmap() -> mpl.colors.Colormap:
    return rsp_activity_cmap("r1_activity")

def gene_cmap() -> mpl.colors.Colormap:
    return marker_expression_cmap("r1_gene")

def soften_histology(image: np.ndarray, amount: float = 0.20) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    return np.clip(array * (1.0 - amount) + amount, 0.0, 1.0)

def scale_field(values: Iterable[float]) -> np.ndarray:
    array = np.clip(np.asarray(list(values), dtype=float), 0, None)
    upper = float(np.quantile(array, 0.99))
    if not np.isfinite(upper) or upper <= 0:
        return np.zeros_like(array)
    return np.clip(array / upper, 0.0, 1.0)

def finish_spatial_axis(axis: plt.Axes) -> None:
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

def draw_continuous_spatial_field(
    axis: plt.Axes,
    coordinates: np.ndarray,
    values: Iterable[float],
    *,
    point_size: float = 2.4,
) -> None:
    scaled = scale_field(values)
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=scaled,
        cmap=activity_cmap(),
        vmin=0,
        vmax=1,
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    axis.invert_yaxis()
    finish_spatial_axis(axis)

def make_triangulation(coordinates: np.ndarray) -> mtri.Triangulation:
    triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    nearest = cKDTree(coordinates).query(coordinates, k=2)[0][:, 1]
    spacing = float(np.median(nearest))
    points = coordinates[triangulation.triangles]
    lengths = np.stack(
        (
            np.linalg.norm(points[:, 0] - points[:, 1], axis=1),
            np.linalg.norm(points[:, 1] - points[:, 2], axis=1),
            np.linalg.norm(points[:, 2] - points[:, 0], axis=1),
        ),
        axis=1,
    )
    triangulation.set_mask(lengths.max(axis=1) > 2.25 * spacing)
    return triangulation

def save_figure_bundle(
    figure: plt.Figure,
    stem: str | Path,
) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in (".png",):
        path = stem.with_suffix(suffix)
        figure.savefig(
            path,
            dpi=EXPORT_DPI,
            facecolor="white",
        )
        paths.append(path)
    plt.close(figure)
    return paths

@dataclass(frozen=True)
class SpatialContext:
    """Minimal spatial and histology context shared by the R1 panels."""

    spot_ids: pd.Index
    obs: pd.DataFrame
    coordinates: np.ndarray
    image: np.ndarray
    image_coordinates: np.ndarray

def _as_float_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype.kind in {"u", "i"}:
        array = array.astype(float) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(float)
        if np.nanmax(array) > 1.0:
            array = array / 255.0
    return np.clip(array, 0.0, 1.0)

def load_spatial_context(h5ad_path: str | Path) -> SpatialContext:
    """Load only the fields required to keep all R1 spatial maps aligned."""

    h5ad_path = Path(h5ad_path)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if "spatial" not in adata.obsm:
            raise KeyError(f"Missing adata.obsm['spatial'] in {h5ad_path}")
        spatial_entries = adata.uns.get("spatial", {})
        if not spatial_entries:
            raise KeyError(f"Missing adata.uns['spatial'] in {h5ad_path}")
        library_id = next(iter(spatial_entries))
        entry = spatial_entries[library_id]
        image = _as_float_image(np.asarray(entry["images"]["hires"]))
        image_scale = float(entry["scalefactors"]["tissue_hires_scalef"])
        coordinates = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
        spot_ids = pd.Index(adata.obs_names.astype(str), name="spot_id")
        obs = adata.obs.copy()
        obs.index = spot_ids
    finally:
        if adata.file is not None:
            adata.file.close()
    return SpatialContext(
        spot_ids=spot_ids,
        obs=obs,
        coordinates=coordinates,
        image=image,
        image_coordinates=coordinates * image_scale,
    )

def load_log_normalized_expression(
    h5ad_path: str | Path,
    genes: Iterable[str],
) -> pd.DataFrame:
    """Return library-normalized log1p expression for a requested gene subset."""

    requested = list(dict.fromkeys(str(gene) for gene in genes))
    adata = ad.read_h5ad(h5ad_path)
    available = [gene for gene in requested if gene in adata.var_names]
    if not available:
        return pd.DataFrame(index=adata.obs_names.astype(str))
    library_size = np.asarray(adata.X.sum(axis=1)).reshape(-1).astype(float)
    matrix = adata[:, available].X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    normalized = np.log1p(matrix / np.maximum(library_size[:, None], 1.0) * 1e4)
    return pd.DataFrame(
        normalized,
        index=pd.Index(adata.obs_names.astype(str), name="spot_id"),
        columns=available,
    )

@dataclass(frozen=True)
class ArtifactPaths:
    h5ad: Path
    sparkx: Path
    stagate: Path
    nmf: Path
    nnmf: Path
    stamp: Path
    regal: Path
    regal_rss_ablation: Path

def load_selection():
    selection = deepcopy(FIGURE_SELECTION)
    return selection

def artifact_paths(selection: dict[str, Any]) -> ArtifactPaths:
    paths = ArtifactPaths(h5ad=path('h5ad'), sparkx=path('sparkx'), stagate=path('stagate'), nmf=path('nmf'), nnmf=path('nnmf'), stamp=path('stamp'), regal=path('regal'), regal_rss_ablation=path('regal_rss_ablation'))
    required = (paths.h5ad, paths.sparkx, paths.stagate, paths.nmf, paths.nnmf, paths.stamp, paths.regal, paths.regal_rss_ablation)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing R1 artifacts: {missing}')
    return paths

def _matrix_csv(path: Path, index_name: str) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = pd.Index(matrix.index.astype(str), name=index_name)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

def load_program_outputs(
    paths: ArtifactPaths,
    spot_ids: pd.Index,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load native loading matrices and continuous fields without rescaling them."""

    regal_loadings_long = pd.read_parquet(paths.regal / "rsp_gene_loadings.parquet")
    regal_loadings = regal_loadings_long.pivot(
        index="gene", columns="rsp_id", values="loading"
    ).fillna(0.0)
    regal_fields_long = pd.read_parquet(paths.regal / "rsp_activation.parquet")
    regal_fields = regal_fields_long.pivot(
        index="spot_id", columns="rsp_id", values="activation"
    )
    outputs = {
        "NMF": {
            "loadings": _matrix_csv(paths.nmf / "nmf_gene_loadings.csv", "gene"),
            "fields": _matrix_csv(paths.nmf / "nmf_spot_weights.csv", "spot_id"),
        },
        "NNMF": {
            "loadings": _matrix_csv(paths.nnmf / "nnmf_signatures.csv", "gene"),
            "fields": _matrix_csv(paths.nnmf / "nnmf_weights.csv", "spot_id"),
        },
        "STAMP": {
            "loadings": _matrix_csv(paths.stamp / "stamp_gene_modules.csv", "gene"),
            "fields": _matrix_csv(paths.stamp / "stamp_spot_topics.csv", "spot_id"),
        },
        "REGAL": {
            "loadings": regal_loadings.apply(pd.to_numeric, errors="coerce").fillna(0.0),
            "fields": regal_fields.apply(pd.to_numeric, errors="coerce").fillna(0.0),
        },
    }
    spot_ids = pd.Index(spot_ids.astype(str), name="spot_id")
    for model, objects in outputs.items():
        loadings = objects["loadings"]
        fields = objects["fields"]
        fields.index = pd.Index(fields.index.astype(str), name="spot_id")
        fields.columns = fields.columns.astype(str)
        missing_spots = spot_ids.difference(fields.index)
        if len(missing_spots):
            raise ValueError(f"{model} is missing {len(missing_spots)} shared spots")
        if set(loadings.columns) != set(fields.columns):
            raise ValueError(f"{model} loading and field component axes do not match")
        objects["fields"] = fields.loc[spot_ids, loadings.columns]
    return outputs

def load_stagate_domains(paths: ArtifactPaths, spot_ids: pd.Index) -> pd.Series:
    adata = ad.read_h5ad(paths.stagate, backed="r")
    try:
        if "STAGATE_domain" not in adata.obs:
            raise KeyError("STAGATE output lacks obs['STAGATE_domain']")
        domains = adata.obs["STAGATE_domain"].astype(str).copy()
        domains.index = domains.index.astype(str)
    finally:
        if adata.file is not None:
            adata.file.close()
    missing = pd.Index(spot_ids.astype(str)).difference(domains.index)
    if len(missing):
        raise ValueError(f"STAGATE is missing {len(missing)} shared spots")
    return domains.loc[pd.Index(spot_ids.astype(str))].rename("STAGATE_domain")

def validate_source_table(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return table



ROW_FIGURE_WIDTH = FIGURE_WIDTH_IN

ROW_FIGURE_HEIGHTS = {
    "row1": 3.23,
    "row2": 5.35,
    "row3": 3.13,
    "row4": 3.35,
}

PATHOLOGY_COLORS = {
    "Invasive": "#D94C5C",
    "Tumor": "#3676B8",
    "Surrounding tumor": "#2A9D8F",
    "Healthy": "#E5B642",
}

@dataclass
class R1FigureSources:
    selection: dict[str, Any]
    spatial: SpatialContext
    domains: pd.Series
    outputs: dict[str, dict[str, pd.DataFrame]]
    gene_expression: pd.Series
    spatial_gene: pd.DataFrame
    domain_rsp: pd.DataFrame
    program_structure: pd.DataFrame
    loading_profiles: pd.DataFrame
    rss_ablation: pd.DataFrame
    family_support: pd.DataFrame
    family_coherence: pd.DataFrame
    family_members: pd.DataFrame
    family_fields: pd.DataFrame

def load_figure_sources() -> R1FigureSources:
    selection = load_selection()
    paths = artifact_paths(selection)
    spatial = load_spatial_context(paths.h5ad)
    domains = load_stagate_domains(paths, spatial.spot_ids)
    outputs = load_program_outputs(paths, spatial.spot_ids)
    gene = str(selection['object_views']['sparkx_gene'])
    gene_expression = load_log_normalized_expression(paths.h5ad, [gene])[gene]
    tables = {'spatial_gene': validate_source_table(table('spatial_gene_evidence.csv'), {'gene', 'adjusted_pvalue', 'significant_gene_count'}), 'domain_rsp': validate_source_table(table('domain_rsp_association.csv'), {'domain', 'rsp_id', 'mean_activation'}), 'program_structure': validate_source_table(table('program_structure_metrics.csv'), {'model', 'component', 'program_score_field_spearman'}), 'loading_profiles': validate_source_table(table('positive_loading_profiles.csv'), {'model', 'gene_rank', 'curve_median'}), 'rss_ablation': validate_source_table(table('rss_ablation_metrics.csv'), {'record_type', 'metric', 'model'}), 'family_support': validate_source_table(table('rsp_family_support.csv'), {'rsp_id', 'k', 'seed_count', 'rsp_order', 'k_order'}), 'family_coherence': validate_source_table(table('rsp_family_coherence.csv'), {'primary_rsp_id', 'spatial_field_correlation', 'active_field_jaccard', 'gene_loading_correlation'}), 'family_members': validate_source_table(table('rsp_family_members.csv'), {'primary_rsp_id', 'candidate_id', 'selected_for_panel'}), 'family_fields': validate_source_table(table('rsp_family_spatial_fields.csv'), {'example_rsp', 'map_id', 'map_order', 'spot_id', 'activation'})}
    return R1FigureSources(selection=selection, spatial=spatial, domains=domains, outputs=outputs, gene_expression=gene_expression.loc[spatial.spot_ids], **tables)

def _row1_image_background(
    axis: plt.Axes,
    sources: R1FigureSources,
    *,
    soften_amount: float,
) -> np.ndarray:
    xy = pd.DataFrame(
        {
            "x_img": sources.spatial.image_coordinates[:, 0],
            "y_img": sources.spatial.image_coordinates[:, 1],
        },
        index=sources.spatial.spot_ids,
    )
    image = _orient_tissue_image(sources.spatial.image, xy)
    radius_px = _estimate_spot_radius_px(xy)
    image, xy = _crop_to_spot_window(image, xy, padding_px=radius_px * 2.5)
    image = soften_histology(image, soften_amount)
    coordinates = xy.loc[sources.spatial.spot_ids, ["x_img", "y_img"]].to_numpy(dtype=float)
    axis.imshow(image, origin="upper", interpolation="nearest")
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    finish_spatial_axis(axis)
    return coordinates

def _row1_frame(axis: plt.Axes) -> None:
    axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axis.transAxes,
            fill=False,
            edgecolor="#DDDDE0",
            linewidth=0.6,
            zorder=8,
            clip_on=False,
        )
    )

def _row1_pathology(axis: plt.Axes, sources: R1FigureSources) -> None:
    coordinates = _row1_image_background(axis, sources, soften_amount=0.26)
    annotation_config = sources.selection["object_views"]["pathology_annotation"]
    annotation = sources.spatial.obs[str(annotation_config["obs_key"])].astype(str)
    categories = annotation_config["categories"]
    labels = [str(item["value"]) for item in categories]
    missing = sorted(set(labels).difference(annotation.unique()))
    if missing:
        raise ValueError(f"Frozen pathology annotations are unavailable: {missing}")
    handles = []
    for item in categories:
        label = str(item["value"])
        selected = annotation.eq(label).to_numpy()
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=3.0,
            facecolors="none",
            edgecolors=PATHOLOGY_COLORS[label],
            linewidths=0.38,
            alpha=0.94,
            rasterized=True,
            zorder=4,
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                color=PATHOLOGY_COLORS[label],
                markerfacecolor="none",
                markeredgewidth=OUTLINE_WIDTH,
                markersize=5.2,
                linewidth=0,
                label=str(item["display_label"]),
            )
        )
    axis.set_title(
        "Manual annotation",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=2.5,
    )
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.040),
        ncol=4,
        prop={"size": LEGEND_SIZE},
        handlelength=0.75,
        columnspacing=0.54,
        handletextpad=0.3,
        borderpad=0,
        borderaxespad=0,
    )

def _row1_gene(axis: plt.Axes, sources: R1FigureSources) -> None:
    gene = str(sources.selection["object_views"]["sparkx_gene"])
    values = sources.gene_expression.to_numpy(dtype=float)
    upper = float(np.quantile(values, 0.99))
    coordinates = _row1_image_background(axis, sources, soften_amount=0.72)
    collection = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values,
        cmap=gene_cmap(),
        vmin=0,
        vmax=upper,
        s=3.0,
        linewidths=0,
        alpha=0.94,
        rasterized=True,
        zorder=4,
    )
    _row1_frame(axis)
    axis.set_title(
        rf"$\mathbfit{{{gene}}}$ (SPARK-X)",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=2.5,
    )
    gene_row = sources.spatial_gene.iloc[0]
    pvalue = float(gene_row["adjusted_pvalue"])
    pvalue_text = "FDR < 1e-300" if pvalue == 0 else f"FDR={pvalue:.1e}"
    axis.text(
        0.5,
        -0.040,
        f"{int(gene_row['significant_gene_count']):,} SVGs  ·  {pvalue_text}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color=INK,
    )
    color_axis = axis.inset_axes([0.925, 0.28, 0.020, 0.44])
    colorbar = axis.figure.colorbar(collection, cax=color_axis)
    colorbar.set_ticks([0, upper])
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.tick_params(labelsize=COLORBAR_SIZE, length=2.2, width=0.7, pad=1)
    colorbar.outline.set_linewidth(0.6)

def _row1_domains(axis: plt.Axes, sources: R1FigureSources) -> None:
    values = sources.domains.astype(int).to_numpy()
    colors = mpl.colormaps["tab20"](np.linspace(0, 1, values.max() + 1))
    colors[:, :3] = 0.52 * colors[:, :3] + 0.48
    coordinates = _row1_image_background(axis, sources, soften_amount=0.82)
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=colors[values],
        s=3.0,
        linewidths=0,
        rasterized=True,
        zorder=4,
    )
    highlighted = str(sources.selection["object_views"]["highlighted_domain"])
    triangulation = make_triangulation(coordinates)
    selected = sources.domains.eq(highlighted).to_numpy(dtype=float)
    axis.tricontour(
        triangulation,
        selected,
        levels=[0.5],
        colors=[INK],
        linewidths=REFERENCE_LINE_WIDTH,
        zorder=5,
    )
    center = np.median(coordinates[selected.astype(bool)], axis=0)
    axis.text(
        center[0],
        center[1],
        highlighted,
        ha="center",
        va="center",
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        color=INK,
    )
    _row1_frame(axis)
    axis.set_title(
        "STAGATE domains",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=2.5,
    )
    axis.text(
        0.5,
        -0.040,
        f"{values.max() + 1} mutually exclusive domains",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color=INK,
    )

def _row1_heatmap(axis: plt.Axes, sources: R1FigureSources) -> None:
    table = sources.domain_rsp.copy()
    table["domain"] = table["domain"].astype(float).astype(int).astype(str)
    row_order = table.sort_values("domain_order")["domain"].drop_duplicates().astype(str).tolist()
    column_order = table.sort_values("rsp_order")["rsp_id"].drop_duplicates().astype(str).tolist()
    matrix = table.pivot(index="domain", columns="rsp_id", values="mean_activation").loc[row_order, column_order]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "domain_rsp",
        ["#FBFAFC", "#DDD8EA", "#A99AC7", "#6F4C9B", "#3F245F"],
    )
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=float(matrix.to_numpy().max()))
    axis.set_box_aspect(0.52)
    axis.set_xticks(
        np.arange(matrix.shape[1]),
        [value.replace("RSP00", "") for value in column_order],
        rotation=90,
        fontsize=TICK_LABEL_SIZE,
    )
    axis.set_yticks(np.arange(matrix.shape[0]), row_order, fontsize=TICK_LABEL_SIZE)
    axis.tick_params(length=2.4, width=0.7, pad=1)
    axis.set_xlabel(
        "Regional specificity program (RSP)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=1,
    )
    axis.set_ylabel("STAGATE domain", fontsize=AXIS_LABEL_SIZE, labelpad=0.5)
    highlighted_domain = str(sources.selection["object_views"]["highlighted_domain"])
    highlighted_rsp = str(sources.selection["object_views"]["highlighted_rsp"])
    domain_position = row_order.index(highlighted_domain)
    rsp_position = column_order.index(highlighted_rsp)
    axis.add_patch(Rectangle((-0.5, domain_position - 0.5), matrix.shape[1], 1, fill=False, edgecolor="#D6A63A", linewidth=OUTLINE_WIDTH, clip_on=False))
    axis.add_patch(Rectangle((rsp_position - 0.5, -0.5), 1, matrix.shape[0], fill=False, edgecolor="#3676B8", linewidth=OUTLINE_WIDTH, clip_on=False))
    axis.legend(
        handles=[
            Line2D([], [], color="#D6A63A", linewidth=REFERENCE_LINE_WIDTH, label=f"Domain {highlighted_domain}: multiple RSPs"),
            Line2D([], [], color="#3676B8", linewidth=REFERENCE_LINE_WIDTH, label=f"{highlighted_rsp}: multiple domains"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=2,
        fontsize=LEGEND_SIZE,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.036, pad=0.018)
    colorbar.ax.tick_params(labelsize=COLORBAR_SIZE, length=2.2, width=0.7)
    colorbar.set_label("Mean activation", fontsize=COLORBAR_SIZE, labelpad=1)

def _populate_row1(container: mpl.figure.FigureBase, sources: R1FigureSources) -> None:
    grid = container.add_gridspec(
        1,
        7,
        width_ratios=[1.0, 0.08, 1.0, 0.08, 1.0, 0.18, 1.80],
        wspace=0,
        left=0.020,
        right=0.965,
        top=0.925,
        bottom=0.166,
    )
    axes = [container.add_subplot(grid[0, index]) for index in (0, 2, 4, 6)]
    _row1_pathology(axes[0], sources)
    _row1_gene(axes[1], sources)
    _row1_domains(axes[2], sources)
    _row1_heatmap(axes[3], sources)
    for label, axis in zip("ABCD", axes):
        add_panel_label(
            container,
            label,
            axis.get_position(),
            align_to_canvas_left=label == "A",
        )


def _populate_row2(container: mpl.figure.FigureBase, sources: R1FigureSources) -> None:
    activities = sorted(sources.selection["program_fields"]["activities"], key=lambda item: item["order"])
    methods = ("NNMF", "STAMP", "REGAL")
    column_count = len(activities)
    grid = container.add_gridspec(
        3,
        column_count + 1,
        width_ratios=[0.14, *([1] * column_count)],
        wspace=0.035,
        hspace=0.055,
        left=0.018,
        right=0.968,
        top=0.94,
        bottom=0.02,
    )
    axes = np.empty((3, column_count), dtype=object)
    for row, model in enumerate(methods):
        label_axis = container.add_subplot(grid[row, 0])
        label_axis.set_axis_off()
        label_axis.text(
            1.30,
            0.5,
            model,
            ha="right",
            va="center",
            fontsize=AXIS_LABEL_SIZE,
            fontweight="bold",
            color=INK,
            clip_on=False,
        )
    for column, activity in enumerate(activities):
        rsp_id = str(activity["regal"])
        for row, model in enumerate(methods):
            axis = container.add_subplot(grid[row, column + 1])
            axes[row, column] = axis
            component = rsp_id if model == "REGAL" else str(activity[model.lower()])
            values = sources.outputs[model]["fields"][component]
            draw_continuous_spatial_field(axis, sources.spatial.coordinates, values, point_size=2.8)
            _row1_frame(axis)
        axes[0, column].text(
            0.5,
            1.025,
            str(activity["display_label"]),
            transform=axes[0, column].transAxes,
            ha="center",
            va="bottom",
            fontsize=PANEL_TITLE_SIZE,
            fontweight="bold",
            color=INK,
        )
    add_panel_label(
        container,
        "E",
        grid[:, :].get_position(container),
        align_to_canvas_left=True,
    )
    cax = container.add_axes([0.969, 0.25, 0.007, 0.50])
    colorbar = container.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=activity_cmap()),
        cax=cax,
    )
    colorbar.set_ticks([0, 1], labels=["0", "1"])
    colorbar.ax.tick_params(labelsize=COLORBAR_SIZE, length=2.2, width=0.7, pad=1.5)
    colorbar.set_label("Scaled activity", fontsize=COLORBAR_SIZE, labelpad=2.0)
    colorbar.outline.set_linewidth(0.6)


def _row3_loading_profiles(axis: plt.Axes, table: pd.DataFrame) -> None:
    curves = table
    for model in ("NMF", "NNMF", "STAMP", "REGAL"):
        data = curves.loc[curves["model"].eq(model)].sort_values("gene_rank")
        color = METHOD_COLORS[model]
        axis.fill_between(
            data["gene_rank"],
            data["curve_q25"],
            data["curve_q75"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(
            data["gene_rank"],
            data["curve_median"],
            color=color,
            linewidth=1.5 if model == "NMF" else DATA_LINE_WIDTH,
            linestyle="--" if model == "NMF" else "-",
            label=model,
        )
    for target in (0.5, 0.7, 0.9):
        axis.axhline(
            target,
            color=GRID,
            linewidth=GRID_LINE_WIDTH,
            linestyle="--",
            zorder=0,
        )
    axis.set_xscale("log")
    axis.set_xlim(left=1)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Gene rank by loading (log scale)")
    axis.set_ylabel("Cumulative positive loading")
    axis.legend(
        loc="lower right",
        ncol=1,
        fontsize=LEGEND_SIZE,
        labelspacing=0.25,
    )

def _row3_field_concordance(axis: plt.Axes, table: pd.DataFrame) -> None:
    components = table.copy()
    models = ("NMF", "NNMF", "STAMP", "REGAL")
    values = [components.loc[components["model"].eq(model), "program_score_field_spearman"].dropna().to_numpy(float) for model in models]
    boxes = axis.boxplot(
        values,
        positions=np.arange(4),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": REFERENCE_LINE_WIDTH},
        whiskerprops={"linewidth": OUTLINE_WIDTH},
        capprops={"linewidth": OUTLINE_WIDTH},
    )
    rng = np.random.default_rng(2026)
    for index, (model, data) in enumerate(zip(models, values)):
        color = METHOD_COLORS[model]
        boxes["boxes"][index].set(
            facecolor=mpl.colors.to_rgba(color, 0.16),
            edgecolor=color,
            linewidth=OUTLINE_WIDTH,
        )
        boxes["medians"][index].set(color=color)
        for line in boxes["whiskers"][2 * index : 2 * index + 2] + boxes["caps"][2 * index : 2 * index + 2]:
            line.set(color=color)
        axis.scatter(np.full(len(data), index) + rng.uniform(-0.11, 0.11, len(data)), data, s=6.5, color=color, alpha=0.46, linewidths=0, zorder=3)
    axis.axhline(0, color=GRID, linewidth=GRID_LINE_WIDTH, linestyle="--", zorder=0)
    axis.set_xticks(
        np.arange(4),
        [f"{model}\nn={len(data)}" for model, data in zip(models, values)],
        fontsize=TICK_LABEL_SIZE,
    )
    axis.set_ylabel("Spearman ρ")

def _row3_joint_structure(axis: plt.Axes, table: pd.DataFrame) -> None:
    components = table
    offsets = {"NMF": (-6, 8), "NNMF": (-7, -9), "STAMP": (-7, 8), "REGAL": (7, -9)}
    alignments = {"NMF": "right", "NNMF": "right", "STAMP": "right", "REGAL": "left"}
    for model in ("NMF", "NNMF", "STAMP", "REGAL"):
        data = components.loc[components["model"].eq(model), ["moran_i", "local_contrast_z"]].dropna()
        x = data["moran_i"].to_numpy(float)
        y = data["local_contrast_z"].to_numpy(float)
        color = METHOD_COLORS[model]
        axis.scatter(x, y, s=14, color=color, alpha=0.46, linewidths=0)
        x25, xm, x75 = np.quantile(x, [0.25, 0.5, 0.75])
        y25, ym, y75 = np.quantile(y, [0.25, 0.5, 0.75])
        axis.errorbar(xm, ym, xerr=np.array([[xm - x25], [x75 - xm]]), yerr=np.array([[ym - y25], [y75 - ym]]), fmt="o", markersize=5.3, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.65, ecolor=color, elinewidth=1.0, capsize=2.0, zorder=4)
        axis.annotate(
            model,
            (xm, ym),
            xytext=offsets[model],
            textcoords="offset points",
            ha=alignments[model],
            va="center",
            color=color,
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
        )
    axis.axhline(0, color=GRID, linewidth=GRID_LINE_WIDTH, linestyle="--", zorder=0)
    axis.set_xlabel("Spatial continuity (Moran’s I)")
    axis.set_ylabel("Local regional contrast")

def _row3_ablation(axis: plt.Axes, table: pd.DataFrame) -> None:
    metrics = ("top50_loading_fraction", "local_contrast_z", "program_score_field_spearman")
    labels = (
        "Loading\nconcentration",
        "Local regional\ncontrast",
        "Field–expression\nconcordance",
    )
    positions = np.arange(3, dtype=float)
    width = 0.34
    rng = np.random.default_rng(2026)
    for metric_index, metric in enumerate(metrics):
        for model_index, model in enumerate(("RSS input", "HVG expression")):
            raw = table.loc[table["record_type"].eq("program") & table["metric"].eq(metric) & table["model"].eq(model), "relative_value"].dropna().to_numpy(float)
            summary = table.loc[table["record_type"].eq("summary") & table["metric"].eq(metric) & table["model"].eq(model)].iloc[0]
            position = positions[metric_index] + (-width / 2 if model_index == 0 else width / 2)
            color = METHOD_COLORS[model]
            median = float(summary["relative_median"])
            q25 = float(summary["relative_q25"])
            q75 = float(summary["relative_q75"])
            axis.bar(
                position,
                median,
                width=width * 0.88,
                color=mpl.colors.to_rgba(color, 0.72),
                edgecolor=color,
                linewidth=OUTLINE_WIDTH,
                zorder=1,
            )
            axis.errorbar(
                position,
                median,
                yerr=np.array([[median - q25], [q75 - median]]),
                fmt="none",
                ecolor=color,
                elinewidth=OUTLINE_WIDTH,
                capsize=2.0,
                capthick=OUTLINE_WIDTH,
                zorder=3,
            )
            axis.scatter(np.full(len(raw), position) + rng.uniform(-width * 0.28, width * 0.28, len(raw)), raw, s=5.5, color=color, alpha=0.38, linewidths=0, zorder=4)
            if model == "HVG expression":
                axis.text(
                    position,
                    max(median, q75) + 0.06,
                    f"{median:.2f}×",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_SIZE,
                    color=MID,
                )
    axis.axhline(
        1.0,
        color=METHOD_COLORS["RSS input"],
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
        alpha=0.7,
    )
    axis.axhline(0.0, color=GRID, linewidth=GRID_LINE_WIDTH)
    axis.set_xticks(positions, labels, fontsize=TICK_LABEL_SIZE)
    axis.tick_params(axis="x", pad=1.0)
    axis.set_ylim(0.0, 1.72)
    axis.set_ylabel("Relative to RSS-input median")
    axis.grid(axis="y", color=GRID, linewidth=GRID_LINE_WIDTH, zorder=0)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=METHOD_COLORS["RSS input"], label="RSS input (n=21)"),
            Line2D([], [], marker="o", linestyle="", color=METHOD_COLORS["HVG expression"], label="HVG expression (n=17)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=LEGEND_SIZE,
        ncol=2,
        handletextpad=0.2,
        columnspacing=0.5,
        borderaxespad=0,
    )

def _populate_row3(container: mpl.figure.FigureBase, sources: R1FigureSources) -> None:
    grid = container.add_gridspec(
        1,
        4,
        width_ratios=[1.30, 1.08, 1.42, 1.40],
        wspace=0.23,
        left=0.045,
        right=0.985,
        top=0.894,
        bottom=0.209,
    )
    axes = [container.add_subplot(grid[0, index]) for index in range(4)]
    _row3_loading_profiles(axes[0], sources.loading_profiles)
    _row3_field_concordance(axes[1], sources.program_structure)
    _row3_joint_structure(axes[2], sources.program_structure)
    _row3_ablation(axes[3], sources.rss_ablation)
    for axis in axes:
        axis.tick_params(labelsize=TICK_LABEL_SIZE)
        axis.xaxis.label.set_size(AXIS_LABEL_SIZE)
        axis.yaxis.label.set_size(AXIS_LABEL_SIZE)
    for label, axis in zip("FGHI", axes):
        add_panel_label(
            container,
            label,
            axis.get_position(),
            align_to_canvas_left=label == "F",
        )


def _row4_support(axis: plt.Axes, sources: R1FigureSources) -> None:
    table = sources.family_support
    rsp_order = table.sort_values("rsp_order")["rsp_id"].drop_duplicates().astype(str).tolist()
    k_order = table.sort_values("k_order")["k"].drop_duplicates().astype(int).tolist()
    matrix = table.pivot(index="rsp_id", columns="k", values="seed_fraction").loc[rsp_order, k_order].T
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seed_support", ["#F7F6F8", "#D7CFE2", "#A48DBD", "#6F4C9B"])
    image = axis.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    axis.set_xlabel("Retained RSP", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("Factorization rank K", fontsize=AXIS_LABEL_SIZE, labelpad=1)
    rsp_labels = [value.replace("RSP00", "") for value in rsp_order]
    axis.set_xticks(
        np.arange(len(rsp_order)),
        rsp_labels,
        rotation=90,
        fontsize=TICK_LABEL_SIZE,
    )
    axis.set_yticks(np.arange(len(k_order)), k_order, fontsize=TICK_LABEL_SIZE)
    axis.set_xticks(np.arange(-0.5, len(rsp_order), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(k_order), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=GRID_LINE_WIDTH)
    axis.tick_params(which="minor", bottom=False, left=False)
    example_colors = {
        str(item["rsp_id"]).replace("RSP00", ""): item["color"]
        for item in sources.selection["rsp_stability"]["examples"]
    }
    for tick in axis.get_xticklabels():
        if tick.get_text() in example_colors:
            tick.set_color(example_colors[tick.get_text()])
            tick.set_fontweight("bold")
    color_axis = axis.inset_axes([0.70, 0.83, 0.24, 0.035])
    colorbar = axis.figure.colorbar(image, cax=color_axis, orientation="horizontal", ticks=[0, 1 / 3, 2 / 3, 1])
    colorbar.ax.set_xticklabels(["0/3", "1/3", "2/3", "3/3"])
    colorbar.ax.tick_params(axis="x", labelsize=COLORBAR_SIZE, length=1.8, pad=1)
    colorbar.ax.set_title("Seed support", fontsize=COLORBAR_SIZE, pad=2)

def _row4_coherence(axis: plt.Axes, sources: R1FigureSources) -> None:
    specs = (
        ("spatial_field_correlation", "Activation\nfield ρ"),
        ("active_field_jaccard", "Activation\nfield Jaccard"),
        ("gene_loading_correlation", "Gene-loading\nprofile ρ"),
    )
    values = [sources.family_coherence[column].to_numpy(float) for column, _ in specs]
    axis.boxplot(
        values,
        positions=np.arange(3),
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": METHOD_COLORS["REGAL"], "linewidth": REFERENCE_LINE_WIDTH},
        boxprops={
            "facecolor": "#EEE9F3",
            "edgecolor": METHOD_COLORS["REGAL"],
            "linewidth": OUTLINE_WIDTH,
        },
        whiskerprops={"color": METHOD_COLORS["REGAL"], "linewidth": OUTLINE_WIDTH},
        capprops={"color": METHOD_COLORS["REGAL"], "linewidth": OUTLINE_WIDTH},
    )
    rng = np.random.default_rng(2026)
    example_colors = {str(item["rsp_id"]): item["color"] for item in sources.selection["rsp_stability"]["examples"]}
    for index, (column, _) in enumerate(specs):
        axis.scatter(np.full(len(sources.family_coherence), index) + rng.uniform(-0.14, 0.14, len(sources.family_coherence)), sources.family_coherence[column], s=6.5, color="#9A8AAA", alpha=0.50, linewidths=0, zorder=3)
        for rsp_id, color in example_colors.items():
            value = float(sources.family_coherence.loc[sources.family_coherence["primary_rsp_id"].eq(rsp_id), column].iloc[0])
            axis.scatter(index, value, marker="D", s=18, color=color, edgecolor="white", linewidth=0.4, zorder=5)
    axis.set_xticks(
        np.arange(3),
        [label for _, label in specs],
        fontsize=TICK_LABEL_SIZE,
    )
    axis.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    axis.set_ylim(0.4, 1.01)
    axis.set_ylabel("Stability-family coherence", fontsize=AXIS_LABEL_SIZE)
    axis.grid(axis="y", color=GRID, linewidth=GRID_LINE_WIDTH)

def _row4_example(
    container: mpl.figure.FigureBase,
    spec: mpl.gridspec.SubplotSpec,
    sources: R1FigureSources,
    example: dict[str, Any],
    panel_label: str,
) -> None:
    rsp_id = str(example["rsp_id"])
    grid = spec.subgridspec(2, 2, wspace=0.10, hspace=0.08)
    axes = [container.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    fields = sources.family_fields.loc[sources.family_fields["example_rsp"].eq(rsp_id)].copy()
    members = sources.family_members.loc[sources.family_members["primary_rsp_id"].eq(rsp_id)].copy()
    for axis, map_order in zip(axes, range(4)):
        map_rows = fields.loc[fields["map_order"].eq(map_order)].set_index("spot_id").reindex(sources.spatial.spot_ids)
        if map_rows["activation"].isna().any():
            raise ValueError(f"Example map {rsp_id}/{map_order} is incomplete")
        map_id = str(map_rows["map_id"].dropna().iloc[0])
        draw_continuous_spatial_field(axis, sources.spatial.coordinates, map_rows["activation"], point_size=2.3)
        _row1_frame(axis)
        member = members.loc[members["candidate_id"].astype(str).eq(map_id)].iloc[0]
        if map_order < 3:
            seed_offset = int(member.get("source_seed_offset", 0))
            run_label = f"K={int(member['source_k'])}\nseed {seed_offset + 1}"
        else:
            run_label = f"Retained\n{rsp_id}"
        axis.text(
            0.97,
            0.94,
            run_label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=ANNOTATION_SIZE,
            color=INK,
            linespacing=1.05,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )
    bounds = spec.get_position(container)
    container.text(
        (bounds.x0 + bounds.x1) / 2,
        bounds.y1 + 0.005,
        rsp_id,
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        color=example["color"],
    )
    add_panel_label(container, panel_label, axes[0].get_position())

def _populate_row4(container: mpl.figure.FigureBase, sources: R1FigureSources) -> None:
    grid = container.add_gridspec(
        1,
        7,
        width_ratios=[1.90, 0.48, 1.60, 0.18, 2.10, 0.06, 2.10],
        wspace=0.0,
        left=0.035,
        right=0.985,
        top=0.902,
        bottom=0.161,
    )
    axis_a = container.add_subplot(grid[0, 0])
    axis_b = container.add_subplot(grid[0, 2])
    _row4_support(axis_a, sources)
    _row4_coherence(axis_b, sources)
    add_panel_label(
        container,
        "J",
        axis_a.get_position(),
        align_to_canvas_left=True,
    )
    add_panel_label(container, "K", axis_b.get_position())
    examples = sources.selection["rsp_stability"]["examples"]
    _row4_example(container, grid[0, 4], sources, examples[0], "L")
    _row4_example(container, grid[0, 6], sources, examples[1], "M")


def export_regional_activity_figure(
    sources: R1FigureSources | None = None,
) -> plt.Figure:
    sources = sources or load_figure_sources()
    row_keys = ("row1", "row2", "row3", "row4")
    row_heights = [ROW_FIGURE_HEIGHTS[key] for key in row_keys]
    figure = plt.figure(figsize=(ROW_FIGURE_WIDTH, sum(row_heights)))
    subfigures = figure.subfigures(
        len(row_keys),
        1,
        height_ratios=row_heights,
        hspace=0,
    )
    for subfigure, populate in zip(
        subfigures,
        (_populate_row1, _populate_row2, _populate_row3, _populate_row4),
        strict=True,
    ):
        populate(subfigure, sources)
    return figure

def draw_figure() -> list[Path]:
    configure_manuscript_style()
    sources = load_figure_sources()
    return save_figure_bundle(export_regional_activity_figure(sources), PANEL_DIR / 'fig03')

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
    _CONFIG = document['fig03']
    result = draw_figure()
    if result is not None:
        for output in result:
            print(output)

if __name__ == '__main__':
    main()
