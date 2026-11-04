"""Plot manuscript fig04 using paths from the shared YAML configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
from string import Template
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle
from scipy import sparse
from Utils.manuscript_figure_style import ANNOTATION_SIZE, COLORBAR_SIZE, DATA_LINE_WIDTH, EXPORT_DPI, FIGURE_WIDTH_IN, GRID, GRID_LINE_WIDTH, INK, MID, OUTLINE_WIDTH, PANEL_TITLE_SIZE, add_panel_label, configure_manuscript_style, marker_expression_cmap, rsp_activity_cmap
import anndata as ad
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_CONFIG: dict

_CONFIG_DIR: Path
_ROOTS: dict

RESOURCE_DIR = Path(__file__).resolve().parents[1] / 'resources'
RESOURCE_PATHS = {
    'Slide-seqV2.png': RESOURCE_DIR / 'fig04' / 'Slide-seqV2.png',
    'Stereo-seq.png': RESOURCE_DIR / 'fig04' / 'Stereo-seq.png',
}

LIVER_REFERENCE = {'markers': {'portal': ['ASS1', 'SDS', 'GLS2', 'CPS1', 'HAL', 'G6PC'],
             'central': ['CYP1A2', 'CYP2E1', 'CYP3A4', 'GLUL']},
 'marker_notes': {'portal': 'SpNeigh reports ASS1, SDS, and GLS2 as periportal examples.',
                  'central': 'SpNeigh reports CYP1A1, CYP3A4, and CYP2E1 as pericentral examples; '
                             'CYP1A1 is not present in this MERFISH panel, so CYP1A2 is retained '
                             'as a panel-available CYP marker.',
                  'midlobular': 'Hep_2 is used as the middle axis point; canonical midlobular '
                                'markers HAMP, HAMP2, and CCND1 are not present in the 317-gene '
                                'panel.'},
 'representative_selection': {'per_sample': 2, 'min_abs_spearman': 0.3},
 'selected_representatives': [{'sample_id': 'AM042',
                               'representatives': [{'rsp_id': 'RSP0006',
                                                    'evidence_context': 'central',
                                                    'selection_note': 'manual central-axis '
                                                                      'representative with CYP1A2, '
                                                                      'CYP3A4, and CYP2E1 support'},
                                                   {'rsp_id': 'RSP0016',
                                                    'evidence_context': 'portal',
                                                    'selection_note': 'manual portal-axis '
                                                                      'representative with SDS, '
                                                                      'GLS2, and G6PC support'}]}],
 'claim_boundary': 'canonical zonation proxy evidence'}

MOB_REFERENCE = {'samples': {'Puck_200127_15': {'platform': 'Slide-seqV2',
                                'role': 'high-resolution MOB laminar reference sample',
                                'note': 'used for visual-marker representative RSP selection'},
             'Stereo_seq': {'platform': 'Stereo-seq',
                            'role': 'cross-platform MOB laminar evidence sample',
                            'note': 'used conservatively; AOB representative is not selected'}},
 'layers': {'ONL': {'canonical_primary': ['Omp', 'Kctd12'],
                    'canonical_secondary': ['Gnal', 'Cnga2', 'Adcy3'],
                    'st_contrast': ['Nrsn1']},
            'GL': {'canonical_primary': ['Th', 'Calb2', 'Cck'],
                   'canonical_secondary': ['Slc6a3', 'Scgn', 'Ddc', 'Calb1', 'Reln'],
                   'st_contrast': ['Slc6a11']},
            'EPL': {'canonical_primary': ['Cck', 'Cdhr1', 'Slc6a11'],
                    'canonical_secondary': ['Gad1', 'Gad2', 'Pvalb', 'Slc17a7'],
                    'st_contrast': ['Doc2g']},
            'MCL': {'canonical_primary': ['Gabra1', 'Cdhr1', 'Slc17a7', 'Doc2g'],
                    'canonical_secondary': ['Tbr1', 'Tbx21'],
                    'st_contrast': ['Nefh', 'Shisa3']},
            'IPL': {'canonical_primary': ['Gad1', 'Gad2', 'Pcp4', 'Penk'],
                    'canonical_secondary': ['Slc17a7'],
                    'st_contrast': []},
            'GCL': {'canonical_primary': ['Pcp4', 'Nrgn', 'Ppp3ca', 'Penk'],
                    'canonical_secondary': ['Calb2', 'Dlx1', 'Dlx2', 'Gad1', 'Gad2', 'Scgn'],
                    'st_contrast': []},
            'RMS': {'canonical_primary': ['Mbp', 'Sox11'],
                    'canonical_secondary': ['Mki67', 'Top2a', 'Dcx', 'Stmn2', 'Tubb3'],
                    'st_contrast': []},
            'AOB': {'canonical_primary': ['Tbx21', 'Reln'],
                    'canonical_secondary': ['Gad1', 'Gad2', 'Gnao1', 'Gnai2'],
                    'st_contrast': ['Fxyd6', 'Atp2b4']}},
 'selected_representatives': [{'sample_id': 'Puck_200127_15',
                               'representatives': [{'rsp_id': 'RSP0000',
                                                    'evidence_context': 'ONL',
                                                    'interpretation_context': 'ONL',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Omp and '
                                                                      'Kctd12 support'},
                                                   {'rsp_id': 'RSP0020',
                                                    'evidence_context': 'GL',
                                                    'interpretation_context': 'GL-like',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Calb2, '
                                                                      'Cck, and Th support'},
                                                   {'rsp_id': 'RSP0017',
                                                    'evidence_context': 'EPL',
                                                    'interpretation_context': 'EPL-like Cck/Cdhr1 '
                                                                              'axis',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Cck and '
                                                                      'Slc17a7 support'},
                                                   {'rsp_id': 'RSP0019',
                                                    'evidence_context': 'MCL',
                                                    'interpretation_context': 'MCL/EPL transition',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with '
                                                                      'Slc17a7, Cdhr1, and Gabra1 '
                                                                      'support'},
                                                   {'rsp_id': 'RSP0007',
                                                    'evidence_context': 'GCL',
                                                    'interpretation_context': 'GCL',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Pcp4, '
                                                                      'Nrgn, and Ppp3ca support'},
                                                   {'rsp_id': 'RSP0010',
                                                    'evidence_context': 'RMS',
                                                    'interpretation_context': 'RMS/neuroblast-like',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Sox11, '
                                                                      'Dcx, Stmn2, and Tubb3 '
                                                                      'support'},
                                                   {'rsp_id': 'RSP0009',
                                                    'evidence_context': 'AOB',
                                                    'interpretation_context': 'AOB-like',
                                                    'selection_note': 'localized representative '
                                                                      'with Atp2b4 and Gnao1 '
                                                                      'support'}]},
                              {'sample_id': 'Stereo_seq',
                               'representatives': [{'rsp_id': 'RSP0001',
                                                    'evidence_context': 'ONL',
                                                    'interpretation_context': 'ONL',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Kctd12 '
                                                                      'and ONL-associated genes'},
                                                   {'rsp_id': 'RSP0009',
                                                    'evidence_context': 'GL',
                                                    'interpretation_context': 'GL-like',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Calb2, '
                                                                      'Th, and Slc6a11 support'},
                                                   {'rsp_id': 'RSP0007',
                                                    'evidence_context': 'EPL',
                                                    'interpretation_context': 'EPL-like Cck/Cdhr1 '
                                                                              'axis',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Cck and '
                                                                      'Slc17a7 support'},
                                                   {'rsp_id': 'RSP0003',
                                                    'evidence_context': 'MCL',
                                                    'interpretation_context': 'MCL/EPL transition',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Gabra1, '
                                                                      'Slc17a7, and Cdhr1 support'},
                                                   {'rsp_id': 'RSP0004',
                                                    'evidence_context': 'GCL',
                                                    'interpretation_context': 'GCL',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Ppp3ca '
                                                                      'and Nrgn support'},
                                                   {'rsp_id': 'RSP0017',
                                                    'evidence_context': 'RMS',
                                                    'interpretation_context': 'RMS/MBP-like',
                                                    'selection_note': 'visual-marker '
                                                                      'representative with Mbp and '
                                                                      'neuroblast-associated '
                                                                      'support'}]}],
 'claim_boundary': 'representative laminar marker-axis evidence'}

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig04'

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


def load_obs_coordinates(
    h5ad_path: str | Path,
    *,
    sample_id: str,
    spatial_key: str = "spatial",
    obs_columns: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if spatial_key not in adata.obsm:
            raise ValueError(f"Missing obsm[{spatial_key!r}] in {h5ad_path}")
        spatial = np.asarray(adata.obsm[spatial_key])[:, :2]
        out = pd.DataFrame(
            {
                "spot_id": adata.obs_names.astype(str),
                "sample_id": sample_id,
                "x": spatial[:, 0].astype(float),
                "y": spatial[:, 1].astype(float),
            }
        )
        for col in obs_columns:
            if col in adata.obs:
                out[col] = adata.obs[col].astype(str).to_numpy()
    finally:
        adata.file.close()
    return out

def load_selected_gene_expression(
    h5ad_path: str | Path,
    genes: list[str] | tuple[str, ...],
    *,
    sample_id: str,
    layer: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    requested = list(dict.fromkeys(str(g) for g in genes if str(g)))
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        present = [gene for gene in requested if gene in adata.var_names]
        if not present:
            out = pd.DataFrame({"spot_id": adata.obs_names.astype(str), "sample_id": sample_id})
            return out
        sub = adata[:, present].to_memory()
        matrix = sub.layers[layer] if layer and layer in sub.layers else sub.X
        values = _dense(matrix).astype(float)
        if normalize:
            totals = _row_totals(adata)
            totals = np.maximum(totals.astype(float), 1.0)
            values = np.log1p(values / totals[:, None] * 1.0e4)
        out = pd.DataFrame(values, columns=present)
        out.insert(0, "sample_id", sample_id)
        out.insert(0, "spot_id", sub.obs_names.astype(str))
    finally:
        adata.file.close()
    return out

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
    return np.asarray(matrix).sum(axis=1).astype(float)

REQUIRED_RSP_FILES = {
    "rsp": "rsp.parquet",
    "activation": "rsp_activation.parquet",
    "gene_loadings": "rsp_gene_loadings.parquet",
    "core_gene_loadings": "rsp_core_gene_loadings.parquet",
    "qc": "rsp_qc.parquet",
    "family_members": "rsp_family_members.parquet",
}

@dataclass(frozen=True)
class SampleConfig:
    dataset: str
    sample_id: str
    platform: str
    h5ad_path: Path
    sample_dir: Path
    include: bool = True
    exclusion_reason: str = ""

def resolve_rsp_out(sample_dir: str | Path) -> Path:
    path = Path(sample_dir)
    if path.name == "rsp_out":
        return path
    return path / "rsp_out"

def load_rsp_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    rsp_out = resolve_rsp_out(sample_dir)
    if not rsp_out.exists():
        raise FileNotFoundError(f"Missing rsp_out directory: {rsp_out}")

    tables: dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_RSP_FILES.items():
        path = rsp_out / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing RSP table: {path}")
        tables[key] = pd.read_parquet(path)
    return tables

def activation_matrix(rsp_activation: pd.DataFrame) -> pd.DataFrame:
    required = {"spot_id", "rsp_id", "activation"}
    missing = required - set(rsp_activation.columns)
    if missing:
        raise ValueError(f"rsp_activation is missing columns: {sorted(missing)}")
    mat = rsp_activation.pivot_table(index="spot_id", columns="rsp_id", values="activation", aggfunc="first")
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    return mat.fillna(0.0).astype(float)

CENTRAL = "#D9545D"

PORTAL = "#3676B8"

PORTAL_LIGHT = "#6B9BC8"

CENTRAL_LIGHT = "#E17A72"

AXIS_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "r2_portal_central",
    [PORTAL, "#F7F7F7", CENTRAL],
)



ROW_FIGURE_HEIGHTS = {
    "row1": 2.55,
    "row2": 3.45,
    "row3": 5.95,
}

MOB_LAYERS = ("ONL", "GL", "EPL", "MCL", "GCL", "RMS")

MOB_MARKERS = {
    ("Puck_200127_15", "ONL"): "Omp",
    ("Stereo_seq", "ONL"): "Kctd12",
    ("Puck_200127_15", "GL"): "Calb2",
    ("Stereo_seq", "GL"): "Calb2",
    ("Puck_200127_15", "EPL"): "Cck",
    ("Stereo_seq", "EPL"): "Cck",
    ("Puck_200127_15", "MCL"): "Cdhr1",
    ("Stereo_seq", "MCL"): "Gabra1",
    ("Puck_200127_15", "GCL"): "Pcp4",
    ("Stereo_seq", "GCL"): "Ppp3ca",
    ("Puck_200127_15", "RMS"): "Sox11",
    ("Stereo_seq", "RMS"): "Mbp",
}

@dataclass
class LiverFigureSources:
    sample_id: str
    coordinates: pd.DataFrame
    activation: pd.DataFrame
    expression: pd.DataFrame
    central_rsp: str
    portal_rsp: str

@dataclass(frozen=True)
class BrainPanelSpec:
    marker: str
    anatomical_label: str
    rsp_id: str
    loading_genes: str

@dataclass
class BrainFigureSources:
    coordinates: pd.DataFrame
    activation: pd.DataFrame
    expression: pd.DataFrame
    panels: tuple[BrainPanelSpec, ...]

@dataclass
class MobSampleSources:
    sample_id: str
    coordinates: pd.DataFrame
    activation: pd.DataFrame
    expression: pd.DataFrame
    rsp_by_layer: dict[str, str]
    marker_by_layer: dict[str, str]
    guide_image: np.ndarray

@dataclass
class R2FigureSources:
    liver: LiverFigureSources
    brain: BrainFigureSources
    mob: tuple[MobSampleSources, ...]

def load_figure_sources() -> R2FigureSources:
    return R2FigureSources(
        liver=_load_liver_sources(),
        brain=_load_brain_sources(),
        mob=_load_mob_sources(),
    )

def _load_liver_sources() -> LiverFigureSources:
    label_column = option('liver_label_column')
    proxy_scores = {str(key): float(value) for key, value in option('liver_proxy_scores').items()}
    representatives = pd.read_csv(table('liver_representative_rsps.csv'))
    sample_id = _selected_liver_sample(representatives)
    sample = SampleConfig('liver', sample_id, 'MERFISH', path('liver_h5ad'), path('liver_sample_dir'))
    coordinates = load_obs_coordinates(sample.h5ad_path, sample_id=sample_id, obs_columns=[label_column])
    coordinates['zonation_proxy'] = coordinates[label_column].map(proxy_scores)
    activation = activation_matrix(load_rsp_tables(sample.sample_dir)['activation'])
    representatives = representatives.loc[representatives['sample_id'].astype(str).eq(sample_id)]
    markers = ['SDS', 'GLS2', 'CYP2E1', 'CYP1A2']
    expression = load_selected_gene_expression(sample.h5ad_path, markers, sample_id=sample_id, normalize=True)
    return LiverFigureSources(sample_id=sample_id, coordinates=coordinates, activation=activation, expression=expression, central_rsp=_representative_rsp(representatives, 'central'), portal_rsp=_representative_rsp(representatives, 'portal'))

def _load_brain_sources() -> BrainFigureSources:
    sample = SampleConfig('brain', 'MouseBrain_Xenium', 'Xenium', path('brain_h5ad'), path('brain_sample_dir'))
    coordinates = load_obs_coordinates(sample.h5ad_path, sample_id=sample.sample_id)
    activation = activation_matrix(load_rsp_tables(sample.sample_dir)['activation'])
    raw_specs = (('Tmem132d', 'Cortex', 'RSP0013', 'Slc17a7, Satb2, Tmem132d'), ('Cplx3', 'L6b / interface', 'RSP0020', 'Cplx3, Ccn2'), ('Gjc3', 'White matter', 'RSP0019', 'Gjc3, Sox10, Opalin'), ('Cpne6', 'Hippocampus', 'RSP0000', 'Cpne6, Pip5k1b, Bhlhe22'), ('Necab2', 'Thalamus', 'RSP0006', 'Slc17a6, Necab2, Calb2'), ('Dcn', 'Meningeal / boundary', 'RSP0023', 'Dcn, Col1a1, Aldh1a2'))
    expression = load_selected_gene_expression(sample.h5ad_path, [spec[0] for spec in raw_specs], sample_id=sample.sample_id, normalize=True)
    panels: list[BrainPanelSpec] = []
    for (marker, anatomical_label, rsp_id, loading_genes) in raw_specs:
        if rsp_id not in activation.columns:
            raise ValueError(f'Missing selected Brain RSP: {rsp_id}')
        panels.append(BrainPanelSpec(marker=marker, anatomical_label=anatomical_label, rsp_id=rsp_id, loading_genes=loading_genes))
    return BrainFigureSources(coordinates=coordinates, activation=activation, expression=expression, panels=tuple(panels))

def _load_mob_sources() -> tuple[MobSampleSources, ...]:
    representatives = pd.read_csv(table('mob_representative_rsps.csv'))
    configs = (SampleConfig('mob', 'Puck_200127_15', 'Slide-seqV2', path('puck_h5ad'), path('puck_sample_dir')), SampleConfig('mob', 'Stereo_seq', 'Stereo-seq', path('stereo_h5ad'), path('stereo_sample_dir')))
    guide_paths = {'Puck_200127_15': resource('Slide-seqV2.png'), 'Stereo_seq': resource('Stereo-seq.png')}
    sources: list[MobSampleSources] = []
    for sample in configs:
        coordinates = load_obs_coordinates(sample.h5ad_path, sample_id=sample.sample_id)
        if sample.sample_id == 'Stereo_seq':
            coordinates = _rotate_stereo_seq_coords(coordinates)
        activation = activation_matrix(load_rsp_tables(sample.sample_dir)['activation'])
        marker_by_layer = {layer: MOB_MARKERS[sample.sample_id, layer] for layer in MOB_LAYERS}
        expression = load_selected_gene_expression(sample.h5ad_path, list(marker_by_layer.values()), sample_id=sample.sample_id, normalize=True)
        rsp_by_layer: dict[str, str] = {}
        for layer in MOB_LAYERS:
            match = representatives.loc[representatives['sample_id'].astype(str).eq(sample.sample_id) & representatives['evidence_context'].astype(str).eq(layer)]
            if not match.empty:
                rsp_by_layer[layer] = str(match.iloc[0]['rsp_id'])
        sources.append(MobSampleSources(sample_id=sample.sample_id, coordinates=coordinates, activation=activation, expression=expression, rsp_by_layer=rsp_by_layer, marker_by_layer=marker_by_layer, guide_image=mpimg.imread(guide_paths[sample.sample_id])))
    return tuple(sources)

def _populate_row1(
    container: mpl.figure.FigureBase,
    sources: R2FigureSources,
) -> None:
    liver = sources.liver
    grid = container.add_gridspec(
        1,
        9,
        width_ratios=[
            1.32,
            0.07,
            1.32,
            0.07,
            1.32,
            0.22,
            1.0,
            0.40,
            1.0,
        ],
        wspace=0,
        left=0.025,
        right=0.985,
        top=0.890,
        bottom=0.050,
    )
    panel_columns = (0, 2, 4, 6, 8)
    axes = [
        container.add_subplot(grid[0, column])
        for column in panel_columns
    ]
    point_size = _point_size(liver.coordinates)

    _spatial_map(
        axes[0],
        liver.coordinates,
        liver.coordinates["zonation_proxy"],
        cmap=AXIS_CMAP,
        size=point_size,
        scale_within_panel=False,
        vmin=-1,
        vmax=1,
    )
    for axis, rsp_id in (
        (axes[1], liver.central_rsp),
        (axes[2], liver.portal_rsp),
    ):
        _spatial_map(
            axis,
            liver.coordinates,
            _activation_for(
                liver.coordinates,
                liver.activation,
                rsp_id,
            ),
            cmap=rsp_activity_cmap("r2_liver_rsp"),
            size=point_size,
        )
    _draw_liver_marker_gradients(
        axes[3],
        liver.coordinates,
        liver.expression,
        ["SDS", "GLS2", "CYP2E1", "CYP1A2"],
    )
    _draw_liver_rsp_activation_trends(
        axes[4],
        liver.coordinates,
        liver.activation,
        liver.portal_rsp,
        liver.central_rsp,
    )
    for axis in axes[3:]:
        axis.set_box_aspect(0.80)
    for column, title in zip(
        panel_columns,
        (
            "Zonation proxy",
            f"Central RSP ({liver.central_rsp})",
            f"Portal RSP ({liver.portal_rsp})",
            "Canonical markers",
            f"{liver.sample_id} RSP trends",
        ),
        strict=True,
    ):
        _add_container_title(
            container,
            grid[0, column].get_position(container),
            title,
            y=0.900,
        )

    add_panel_label(
        container,
        "A",
        grid[0, 0].get_position(container),
        align_to_canvas_left=True,
    )
    add_panel_label(
        container,
        "B",
        grid[0, 2:5].get_position(container),
    )
    add_panel_label(
        container,
        "C",
        grid[0, 6:9].get_position(container),
    )


def _populate_row2(
    container: mpl.figure.FigureBase,
    sources: R2FigureSources,
) -> None:
    brain = sources.brain
    n_panels = len(brain.panels)
    grid = container.add_gridspec(
        2,
        n_panels + 1,
        width_ratios=[1] * n_panels + [0.035],
        wspace=0.10,
        hspace=0.31,
        left=0.025,
        right=0.970,
        top=0.900,
        bottom=0.020,
    )
    axes = np.empty((2, n_panels), dtype=object)
    marker_map = marker_expression_cmap("r2_brain_marker")
    activity_map = rsp_activity_cmap("r2_brain_rsp")
    point_size = _point_size(brain.coordinates)

    for column, panel in enumerate(brain.panels):
        marker_axis = container.add_subplot(grid[0, column])
        rsp_axis = container.add_subplot(grid[1, column])
        axes[0, column] = marker_axis
        axes[1, column] = rsp_axis
        _spatial_map(
            marker_axis,
            brain.coordinates,
            _gene_values(
                brain.coordinates,
                brain.expression,
                panel.marker,
            ),
            cmap=marker_map,
            size=point_size,
        )
        _add_stacked_title(
            marker_axis,
            panel.anatomical_label,
            panel.marker,
            secondary_italic=True,
        )
        _spatial_map(
            rsp_axis,
            brain.coordinates,
            _activation_for(
                brain.coordinates,
                brain.activation,
                panel.rsp_id,
            ),
            cmap=activity_map,
            size=point_size,
        )
        _add_stacked_title(
            rsp_axis,
            panel.rsp_id,
            panel.loading_genes,
            secondary_italic=True,
        )

    marker_cax = container.add_subplot(grid[0, n_panels])
    rsp_cax = container.add_subplot(grid[1, n_panels])
    _unit_colorbar(
        container,
        marker_cax,
        marker_map,
        "Scaled marker expression",
    )
    _unit_colorbar(
        container,
        rsp_cax,
        activity_map,
        "Scaled RSP activation",
    )
    add_panel_label(
        container,
        "D",
        axes[0, 0].get_position(),
        align_to_canvas_left=True,
    )
    add_panel_label(
        container,
        "E",
        axes[1, 0].get_position(),
        align_to_canvas_left=True,
    )


def _populate_row3(
    container: mpl.figure.FigureBase,
    sources: R2FigureSources,
) -> None:
    grid = container.add_gridspec(
        5,
        7,
        height_ratios=[1, 1, 0.18, 1, 1],
        width_ratios=[1.30, 1, 1, 1, 1, 1, 1],
        hspace=0.16,
        wspace=0.035,
        left=0.020,
        right=0.990,
        top=0.940,
        bottom=0.020,
    )
    activity_map = rsp_activity_cmap("r2_mob_rsp")
    marker_map = marker_expression_cmap("r2_mob_marker")

    for sample_index, sample in enumerate(sources.mob):
        rsp_row = 0 if sample_index == 0 else 3
        marker_row = rsp_row + 1
        guide_axis = container.add_subplot(
            grid[rsp_row : marker_row + 1, 0]
        )
        _draw_reference_image(
            guide_axis,
            sample.guide_image,
        )
        add_panel_label(
            container,
            "F" if sample_index == 0 else "G",
            guide_axis.get_position(),
            align_to_canvas_left=True,
        )

        first_rsp_axis: plt.Axes | None = None
        first_marker_axis: plt.Axes | None = None
        for column, layer in enumerate(MOB_LAYERS, start=1):
            rsp_axis = container.add_subplot(grid[rsp_row, column])
            marker_axis = container.add_subplot(grid[marker_row, column])
            if first_rsp_axis is None:
                first_rsp_axis = rsp_axis
            if first_marker_axis is None:
                first_marker_axis = marker_axis
            marker = sample.marker_by_layer[layer]
            rsp_id = sample.rsp_by_layer.get(layer)
            if rsp_id is None:
                _blank_spatial_axis(rsp_axis)
                _blank_spatial_axis(marker_axis)
                _add_inline_title(rsp_axis, layer, "not selected")
                _add_inline_title(
                    marker_axis,
                    layer,
                    marker,
                    secondary_italic=True,
                )
                continue
            _spatial_map(
                rsp_axis,
                sample.coordinates,
                _activation_for(
                    sample.coordinates,
                    sample.activation,
                    rsp_id,
                ),
                cmap=activity_map,
                size=_point_size(sample.coordinates) * 1.25,
                margin=0.025,
            )
            _spatial_map(
                marker_axis,
                sample.coordinates,
                _gene_values(
                    sample.coordinates,
                    sample.expression,
                    marker,
                ),
                cmap=marker_map,
                size=_point_size(sample.coordinates) * 1.25,
                margin=0.025,
            )
            _add_inline_title(rsp_axis, layer, rsp_id)
            _add_inline_title(
                marker_axis,
                layer,
                marker,
                secondary_italic=True,
            )

        if first_rsp_axis is None or first_marker_axis is None:
            raise RuntimeError("MOB row did not create spatial axes")
        add_panel_label(
            container,
            "H" if sample_index == 0 else "J",
            first_rsp_axis.get_position(),
        )
        add_panel_label(
            container,
            "I" if sample_index == 0 else "K",
            first_marker_axis.get_position(),
        )


def export_canonical_regional_programs_figure(
    sources: R2FigureSources | None = None,
) -> plt.Figure:
    sources = sources or load_figure_sources()
    row_keys = ("row1", "row2", "row3")
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
    for subfigure, populate in zip(
        subfigures,
        (_populate_row1, _populate_row2, _populate_row3),
        strict=True,
    ):
        populate(subfigure, sources)
    return figure

def export_manuscript_spatial_figures() -> list[Path]:
    configure_manuscript_style()
    sources = load_figure_sources()
    return _save_figure_bundle(
        export_canonical_regional_programs_figure(sources),
        PANEL_DIR / 'fig04',
    )

def _activation_for(
    coordinates: pd.DataFrame,
    activation: pd.DataFrame,
    rsp_id: str,
) -> pd.Series:
    if rsp_id not in activation.columns:
        return pd.Series(np.nan, index=coordinates.index)
    return coordinates["spot_id"].map(activation[rsp_id])

def _gene_values(
    coordinates: pd.DataFrame,
    expression: pd.DataFrame,
    gene: str,
) -> pd.Series:
    if gene not in expression.columns:
        return pd.Series(np.nan, index=coordinates.index)
    return coordinates["spot_id"].map(
        expression.set_index("spot_id")[gene]
    )

def _rotate_stereo_seq_coords(
    coordinates: pd.DataFrame,
) -> pd.DataFrame:
    rotated = coordinates.copy()
    x = rotated["x"].astype(float)
    y = rotated["y"].astype(float)
    x0 = x - x.mean()
    y0 = y - y.mean()
    rotated["x"] = -y0
    rotated["y"] = -x0
    return rotated

def _selected_liver_sample(representatives: pd.DataFrame) -> str:
    score = (
        representatives.assign(
            abs_rho=pd.to_numeric(
                representatives["spearman_rho_zonation"],
                errors="coerce",
            ).abs()
        )
        .groupby("sample_id", as_index=False)["abs_rho"]
        .mean()
        .sort_values(
            ["abs_rho", "sample_id"],
            ascending=[False, True],
        )
    )
    return str(score.iloc[0]["sample_id"])

def _representative_rsp(
    representatives: pd.DataFrame,
    context: str,
) -> str:
    subset = representatives.loc[
        representatives["evidence_context"].astype(str).eq(context)
    ]
    return str(subset.iloc[0]["rsp_id"]) if not subset.empty else ""

def _draw_liver_marker_gradients(
    axis: plt.Axes,
    coordinates: pd.DataFrame,
    expression: pd.DataFrame,
    markers: list[str],
) -> None:
    merged = coordinates[["spot_id", "zonation_proxy"]].merge(
        expression,
        on="spot_id",
        how="left",
    )
    marker_colors = {
        "SDS": PORTAL,
        "GLS2": PORTAL_LIGHT,
        "CYP2E1": CENTRAL,
        "CYP1A2": CENTRAL_LIGHT,
    }
    label_positions = {
        "SDS": (-0.70, 0.12),
        "GLS2": (-0.55, -0.12),
        "CYP2E1": (-0.82, 0.18),
        "CYP1A2": (-0.45, -0.18),
    }
    for marker in markers:
        if marker not in merged.columns:
            continue
        summary = (
            merged.groupby("zonation_proxy", as_index=False)[marker]
            .mean()
            .sort_values("zonation_proxy")
        )
        x = summary["zonation_proxy"].to_numpy(dtype=float)
        y = summary[marker].to_numpy(dtype=float)
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=DATA_LINE_WIDTH,
            markersize=4.2,
            color=marker_colors[marker],
        )
        label_x, label_offset = label_positions[marker]
        label_y = float(np.interp(label_x, x, y)) + label_offset
        axis.text(
            label_x,
            label_y,
            marker,
            color=marker_colors[marker],
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            fontstyle="italic",
            ha="left",
            va="center",
            clip_on=False,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.15,
            },
        )
    axis.set_xlim(-1.08, 1.08)
    axis.set_xticks([-1, 0, 1], ["portal", "mid", "central"])
    axis.set_xlabel("Portal-central proxy")
    axis.set_ylabel("Mean normalized expression")
    axis.grid(axis="y", color=GRID, linewidth=GRID_LINE_WIDTH)

def _draw_liver_rsp_activation_trends(
    axis: plt.Axes,
    coordinates: pd.DataFrame,
    activation: pd.DataFrame,
    portal_rsp: str,
    central_rsp: str,
) -> None:
    trend_specs = (
        (portal_rsp, "portal", PORTAL),
        (central_rsp, "central", CENTRAL),
    )
    label_offsets = {"portal": -0.008, "central": 0.012}
    for rsp_id, label, color in trend_specs:
        values = _scale01(
            _activation_for(coordinates, activation, rsp_id)
        )
        merged = coordinates[["zonation_proxy"]].copy()
        merged["activation"] = values.to_numpy(dtype=float)
        summary = (
            merged.groupby("zonation_proxy", as_index=False)["activation"]
            .mean()
            .sort_values("zonation_proxy")
        )
        x = (
            summary["zonation_proxy"].to_numpy(dtype=float) + 1.0
        ) / 2.0
        y = summary["activation"].to_numpy(dtype=float)
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=DATA_LINE_WIDTH,
            markersize=4.2,
            color=color,
        )
        axis.text(
            0.97,
            y[-1] + label_offsets[label],
            f"{rsp_id} · {label}",
            color=color,
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            ha="right",
            va="center",
            clip_on=False,
        )
    axis.set_xlim(-0.05, 1.05)
    axis.set_xticks([0, 0.5, 1], ["portal", "mid", "central"])
    axis.set_xlabel("Portal-central proxy")
    axis.set_ylabel("Mean scaled activation")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color=GRID, linewidth=GRID_LINE_WIDTH)

def _scale01(values: pd.Series) -> pd.Series:
    array = pd.Series(values).astype(float)
    finite = np.isfinite(array)
    if not finite.any():
        return array
    lower = float(array[finite].min())
    upper = float(array[finite].max())
    if upper <= lower:
        return pd.Series(np.zeros(len(array)), index=array.index)
    return (array - lower) / (upper - lower)

def _scale_for_display(values: np.ndarray) -> np.ndarray:
    scaled = np.full(values.shape, np.nan, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return scaled
    lower = float(np.nanpercentile(finite, 1))
    upper = float(np.nanpercentile(finite, 99))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return scaled
    if upper <= lower:
        lower = float(np.nanmin(finite))
        upper = float(np.nanmax(finite))
    if upper <= lower:
        scaled[np.isfinite(values)] = 0.0
        return scaled
    scaled[np.isfinite(values)] = np.clip(
        (values[np.isfinite(values)] - lower) / (upper - lower),
        0.0,
        1.0,
    )
    return scaled

def _spatial_map(
    axis: plt.Axes,
    coordinates: pd.DataFrame,
    values: pd.Series | np.ndarray,
    *,
    cmap: str | mpl.colors.Colormap,
    size: float,
    scale_within_panel: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    margin: float = 0.0,
) -> mpl.collections.PathCollection:
    raw_values = pd.Series(values).astype(float).to_numpy()
    display_values = (
        _scale_for_display(raw_values)
        if scale_within_panel
        else raw_values
    )
    if scale_within_panel:
        vmin, vmax = 0.0, 1.0
    finite = np.isfinite(display_values)
    order = np.argsort(
        np.nan_to_num(display_values, nan=-np.inf)
    )
    axis.scatter(
        coordinates["x"].iloc[order],
        coordinates["y"].iloc[order],
        c=GRID,
        s=size,
        linewidths=0,
        rasterized=True,
    )
    collection = axis.scatter(
        coordinates["x"].iloc[order][finite[order]],
        coordinates["y"].iloc[order][finite[order]],
        c=display_values[order][finite[order]],
        s=size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    axis.set_aspect("equal", adjustable="box")
    if margin > 0:
        axis.margins(x=margin, y=margin)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.invert_yaxis()
    for spine in axis.spines.values():
        spine.set_visible(False)
    return collection

def _blank_spatial_axis(axis: plt.Axes) -> None:
    axis.text(
        0.5,
        0.5,
        "not selected",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=ANNOTATION_SIZE,
        color=MID,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

def _add_container_title(
    container: mpl.figure.FigureBase,
    bounds: mpl.transforms.Bbox,
    title: str,
    *,
    y: float,
) -> None:
    container.text(
        (bounds.x0 + bounds.x1) / 2,
        y,
        title,
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        color=INK,
    )

def _add_stacked_title(
    axis: plt.Axes,
    primary: str,
    secondary: str,
    *,
    secondary_italic: bool = False,
) -> None:
    children = [
        TextArea(
            primary,
            textprops={
                "fontsize": PANEL_TITLE_SIZE,
                "fontweight": "bold",
                "color": INK,
                "ha": "center",
            },
        )
    ]
    if secondary:
        children.append(
            TextArea(
                secondary,
                textprops={
                    "fontsize": ANNOTATION_SIZE,
                    "fontstyle": (
                        "italic" if secondary_italic else "normal"
                    ),
                    "color": INK,
                    "ha": "center",
                },
            )
        )
    title_box = VPacker(
        children=children,
        align="center",
        pad=0,
        sep=1.0,
    )
    annotation = AnnotationBbox(
        title_box,
        (0.5, 0.990),
        xycoords=axis.transAxes,
        box_alignment=(0.5, 0.0),
        frameon=False,
        pad=0,
    )
    annotation.set_clip_on(False)
    axis.add_artist(annotation)

def _add_inline_title(
    axis: plt.Axes,
    primary: str,
    secondary: str,
    *,
    secondary_italic: bool = False,
) -> None:
    title_box = HPacker(
        children=[
            TextArea(
                f"{primary} ·",
                textprops={
                    "fontsize": PANEL_TITLE_SIZE,
                    "fontweight": "bold",
                    "color": INK,
                },
            ),
            TextArea(
                secondary,
                textprops={
                    "fontsize": PANEL_TITLE_SIZE,
                    "fontweight": "bold",
                    "fontstyle": (
                        "italic" if secondary_italic else "normal"
                    ),
                    "color": INK,
                },
            ),
        ],
        align="center",
        pad=0,
        sep=2.0,
    )
    annotation = AnnotationBbox(
        title_box,
        (0.5, 0.975),
        xycoords=axis.transAxes,
        box_alignment=(0.5, 0.0),
        frameon=False,
        pad=0,
    )
    annotation.set_clip_on(False)
    axis.add_artist(annotation)

def _unit_colorbar(
    container: mpl.figure.FigureBase,
    axis: plt.Axes,
    cmap: mpl.colors.Colormap,
    label: str,
) -> None:
    colorbar = container.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(0, 1),
            cmap=cmap,
        ),
        cax=axis,
    )
    colorbar.set_ticks([0, 1], labels=["0", "1"])
    colorbar.ax.tick_params(
        labelsize=COLORBAR_SIZE,
        length=2.2,
        width=0.7,
        pad=1.5,
    )
    colorbar.set_label(
        label,
        fontsize=COLORBAR_SIZE,
        labelpad=2.0,
    )
    colorbar.outline.set_linewidth(0.6)

def _draw_reference_image(
    axis: plt.Axes,
    image: np.ndarray,
) -> None:
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axis.transAxes,
            fill=False,
            edgecolor=GRID,
            linewidth=OUTLINE_WIDTH,
            clip_on=False,
        )
    )

def _point_size(coordinates: pd.DataFrame) -> float:
    count = coordinates.shape[0]
    if count > 30000:
        return 0.65
    if count > 18000:
        return 0.95
    return 1.25

def _save_figure_bundle(
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
    _CONFIG = document['fig04']
    for output in export_manuscript_spatial_figures():
        print(output)

if __name__ == '__main__':
    main()
