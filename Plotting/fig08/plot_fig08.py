"""Plot manuscript fig08 using paths from the shared YAML configuration."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import argparse
from string import Template
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import spearmanr
from Utils.manuscript_figure_style import ANNOTATION_SIZE, AXIS_LABEL_SIZE, COLORBAR_SIZE, EXPORT_DPI, FIGURE_WIDTH_IN, GRID, GRID_LINE_WIDTH, INK, LEGEND_SIZE, OUTLINE_WIDTH, PANEL_TITLE_SIZE, TICK_LABEL_SIZE
from Utils.manuscript_figure_style import EXPORT_DPI, GRID, INK, MID, PAPER, add_panel_label as add_shared_panel_label, configure_manuscript_style
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
    'hypothesis_anchors.yaml': RESOURCE_DIR / 'fig08' / 'hypothesis_anchors.yaml',
    'c2.cp.reactome.v2026.1.Hs.symbols.gmt': RESOURCE_DIR / 'shared' / 'c2.cp.reactome.v2026.1.Hs.symbols.gmt',
    'BioPlanet_2019.Enrichr.gmt': RESOURCE_DIR / 'shared' / 'BioPlanet_2019.Enrichr.gmt',
}

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig08'

ARTIFACT_DIR: Path

FIGURE_STEM: Path

H5AD_PATH: Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'output' / 'figures'

SAMPLE_ID: str

STEREOSCOPE_PATH: Path

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


REGAL_COLORS = {
    "ink": INK,
    "mid": MID,
    "carrier": "#2A9D8F",
    "immune": "#3676B8",
    "overlap": "#C65A7A",
    "light": GRID,
    "paper": PAPER,
}

RSP_ACTIVITY_NORM = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

def configure_figure_style() -> None:
    """Apply the shared R1-R5 manuscript style."""

    configure_manuscript_style()

def load_spatial_image_context(
    h5ad_path: str | Path,
    *,
    sample_id: str,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Load the Visium hires image and spot coordinates in image pixels."""

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        spatial = adata.uns.get("spatial", {})
        if sample_id not in spatial:
            raise ValueError(f"Missing spatial entry {sample_id!r} in {h5ad_path}")
        entry = spatial[sample_id]
        image = np.asarray(entry["images"]["hires"])
        scale = float(entry["scalefactors"]["tissue_hires_scalef"])
        diameter = float(entry["scalefactors"]["spot_diameter_fullres"]) * scale
        coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2] * scale
        index = pd.Index(adata.obs_names.astype(str), name="spot_id")
    finally:
        if adata.file is not None:
            adata.file.close()

    xy = pd.DataFrame(coords, index=index, columns=["x_img", "y_img"])
    return soften_histology(image), xy, diameter / 2.0

def soften_histology(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype.kind in {"u", "i"}:
        arr = arr.astype(float) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(float)
        if np.nanmax(arr) > 1.0:
            arr = arr / 255.0
    return np.clip(arr * 0.70 + 0.30, 0.0, 1.0)

def draw_histology(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    alpha: float = 1.0,
) -> None:
    ax.imshow(image, origin="upper", interpolation="nearest", alpha=alpha)

def finish_spatial_axis(
    ax: plt.Axes,
    xy: pd.DataFrame,
    *,
    spot_radius_px: float,
    anchor: str = "N",
) -> None:
    padding = max(spot_radius_px * 2.2, 4.0)
    ax.set_xlim(float(xy["x_img"].min()) - padding, float(xy["x_img"].max()) + padding)
    ax.set_ylim(float(xy["y_img"].max()) + padding, float(xy["y_img"].min()) - padding)
    ax.set_aspect("equal")
    ax.set_anchor(anchor)
    ax.set_axis_off()

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
    """Stack row images in memory while trimming only vertical white margins."""

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
) -> list[Path]:
    """Save the complete figure as PNG."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    fig = plt.figure(
        figsize=(width / EXPORT_DPI, height / EXPORT_DPI),
        dpi=EXPORT_DPI,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, interpolation="nearest")
    ax.axis("off")
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, dpi=EXPORT_DPI)
    plt.close(fig)
    return [png_path]


CARRIER_RSP = "RSP0026"

IMMUNE_RSP = "RSP0024"

CORE_CELLTYPES = (
    ("CAFs MSC iCAF-like", "iCAF"),
    ("CAFs myCAF-like", "myCAF"),
    ("T cells CD4+", "CD4 T"),
    ("T cells CD8+", "CD8 T"),
)

REGIONAL_STATE_ORDER = (
    "neither",
    "iCAF_RSP_only",
    "immune_RSP_only",
    "RSP_overlap",
)

REGIONAL_STATE_LABELS = {
    "neither": "Neither",
    "iCAF_RSP_only": "RSP0026 only",
    "immune_RSP_only": "RSP0024 only",
    "RSP_overlap": "Overlap",
}

REGIONAL_STATE_COLORS = {
    "neither": "#C8CED0",
    "iCAF_RSP_only": "#00A896",
    "immune_RSP_only": "#277DA1",
    "RSP_overlap": "#D83F87",
}

PATHOLOGY_ORDER = (
    "Stroma",
    "DCIS",
    "Invasive cancer + lymphocytes",
    "Lymphocytes",
)

PATHOLOGY_LABELS = {
    "Stroma": "Stroma",
    "DCIS": "DCIS",
    "Invasive cancer + lymphocytes": "Invasive + lymphocytes",
    "Lymphocytes": "Lymphocytes",
}

PATHOLOGY_COLORS = {
    "Stroma": "#A6761D",
    "DCIS": "#666666",
    "Invasive cancer + lymphocytes": "#E17C42",
    "Lymphocytes": "#7561A8",
}

ABUNDANCE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "r5_abundance",
    ["#FFF7BC", "#FEC44F", "#FE9929", "#D7301F", "#7F0000"],
)

SELECTION_EVIDENCE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "r5_selection_evidence",
    ["#F5F5F2", "#B8C0C2", "#49545A"],
)

CARRIER_ACTIVITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "r5_carrier_activity",
    ["#F1F8F5", "#A8D9CD", "#43AF99", "#126B5D"],
)

IMMUNE_ACTIVITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "r5_immune_activity",
    ["#F2F6FB", "#A8C8E6", "#4F88C4", "#245786"],
)

INTERFACE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "r5_interface",
    ["#F7F5F0", "#E7C1CF", "#C65A7A", "#704F6C"],
)

ROBUSTNESS_COLOR = "#4C566A"

RSP0026_PATHWAYS = (
    "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION",
    "REACTOME_ECM_PROTEOGLYCANS",
    "REACTOME_INTEGRIN_CELL_SURFACE_INTERACTIONS",
    "REACTOME_REGULATION_OF_INSULIN_LIKE_GROWTH_FACTOR_IGF_TRANSPORT_AND_UPTAKE_BY_INSULIN_LIKE_GROWTH_FACTOR_BINDING_PROTEINS_IGFBPS",
)

RSP0024_PATHWAYS = (
    "REACTOME_ADAPTIVE_IMMUNE_SYSTEM",
    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM",
    "REACTOME_TCR_SIGNALING",
    "REACTOME_IMMUNOREGULATORY_INTERACTIONS_BETWEEN_A_LYMPHOID_AND_A_NON_LYMPHOID_CELL",
    "REACTOME_CHEMOKINE_RECEPTORS_BIND_CHEMOKINES",
    "REACTOME_CELL_SURFACE_INTERACTIONS_AT_THE_VASCULAR_WALL",
)

PATHWAY_LABELS = {
    "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION": "ECM organization",
    "REACTOME_ECM_PROTEOGLYCANS": "ECM proteoglycans",
    "REACTOME_INTEGRIN_CELL_SURFACE_INTERACTIONS": "integrin surface interactions",
    "REACTOME_REGULATION_OF_INSULIN_LIKE_GROWTH_FACTOR_IGF_TRANSPORT_AND_UPTAKE_BY_INSULIN_LIKE_GROWTH_FACTOR_BINDING_PROTEINS_IGFBPS": "IGF transport by IGFBPs",
    "REACTOME_ADAPTIVE_IMMUNE_SYSTEM": "adaptive immune system",
    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM": "cytokine signaling",
    "REACTOME_TCR_SIGNALING": "TCR signaling",
    "REACTOME_IMMUNOREGULATORY_INTERACTIONS_BETWEEN_A_LYMPHOID_AND_A_NON_LYMPHOID_CELL": "lymphoid–stromal interactions",
    "REACTOME_CHEMOKINE_RECEPTORS_BIND_CHEMOKINES": "chemokine-receptor binding",
    "REACTOME_CELL_SURFACE_INTERACTIONS_AT_THE_VASCULAR_WALL": "vascular-wall interactions",
}

@dataclass
class R5FigureSources:
    image: np.ndarray
    xy: pd.DataFrame
    spot_radius_px: float
    full_spots: pd.Index
    activation: pd.DataFrame
    stereoscope: pd.DataFrame
    retrieval: pd.DataFrame
    spatial_concordance: pd.DataFrame
    enrichment: pd.DataFrame
    regional_pathology: pd.DataFrame
    regional_celltypes: pd.DataFrame
    regional_context: pd.DataFrame
    interface_lr: pd.DataFrame

def _rsp_spatial_concordance(
    activation: pd.DataFrame,
    stereoscope: pd.DataFrame,
    domain_spots: pd.Index,
) -> pd.DataFrame:
    spots = domain_spots.intersection(activation.index).intersection(stereoscope.index)
    icaf = stereoscope.loc[spots, "CAFs MSC iCAF-like"].astype(float)
    tcell = stereoscope.loc[spots, ["T cells CD4+", "T cells CD8+"]].sum(axis=1)
    rows = []
    for rsp_id in activation.columns.astype(str):
        values = activation.loc[spots, rsp_id].astype(float)
        variable = values.nunique(dropna=True) > 1
        rows.append(
            {
                "rsp_id": rsp_id,
                "n_spots": len(spots),
                "icaf_spearman_rho": float(spearmanr(values, icaf).statistic) if variable else np.nan,
                "tcell_spearman_rho": float(spearmanr(values, tcell).statistic) if variable else np.nan,
            }
        )
    return pd.DataFrame(rows)

def load_r5_figure_sources() -> R5FigureSources:
    (image, xy, radius) = load_spatial_image_context(H5AD_PATH, sample_id=SAMPLE_ID)
    stereoscope = pd.read_parquet(STEREOSCOPE_PATH)
    stereoscope.index = stereoscope.index.astype(str)
    activation_long = pd.read_parquet(ARTIFACT_DIR / 'rsp_out/rsp_activation.parquet')
    activation_long['spot_id'] = activation_long['spot_id'].astype(str)
    activation_long['rsp_id'] = activation_long['rsp_id'].astype(str)
    activation = activation_long.pivot(index='spot_id', columns='rsp_id', values='activation')
    full_spots = xy.index.intersection(stereoscope.index).intersection(activation.index)
    regional_context = pd.read_csv(table('regional_interface_spots.csv')).set_index('spot_id')
    regional_context.index = regional_context.index.astype(str)
    spatial_concordance = _rsp_spatial_concordance(activation, stereoscope, regional_context.index)
    sources = R5FigureSources(image=image, xy=xy, spot_radius_px=radius, full_spots=full_spots, activation=activation.loc[full_spots, [CARRIER_RSP, IMMUNE_RSP]].copy(), stereoscope=stereoscope.loc[full_spots].copy(), retrieval=pd.read_csv(table('rsp_hypothesis_retrieval.csv')), spatial_concordance=spatial_concordance, enrichment=pd.read_csv(table('regional_program_enrichment.csv')), regional_pathology=pd.read_csv(table('regional_pathology_readback.csv')), regional_celltypes=pd.read_csv(table('regional_celltype_readback.csv')), regional_context=regional_context, interface_lr=pd.read_csv(table('interface_lr_readback.csv')))
    validate_r5_figure_contract(sources)
    return sources

def validate_r5_figure_contract(s: R5FigureSources) -> None:
    validate_source_columns(
        s.retrieval,
        {
            "query",
            "rsp_id",
            "odds_ratio",
            "fdr_bh",
        },
        name="rsp_hypothesis_retrieval.csv",
    )
    if s.retrieval["rsp_id"].nunique() != 27 or s.retrieval["query"].nunique() != 4:
        raise ValueError("R5 retrieval must contain 27 RSPs x 4 state queries")

    validate_source_columns(
        s.spatial_concordance,
        {"rsp_id", "n_spots", "icaf_spearman_rho", "tcell_spearman_rho"},
        name="descriptive RSP spatial concordance",
    )
    if set(s.spatial_concordance["n_spots"].astype(int)) != {805}:
        raise ValueError("RSP spatial concordance must use the fixed 805-spot analysis domain")

    validate_source_columns(
        s.enrichment,
        {"rsp_id", "term", "odds_ratio", "fdr_bh"},
        name="regional_program_enrichment.csv",
    )
    selected_terms = set(RSP0026_PATHWAYS).union(RSP0024_PATHWAYS)
    enrichment_terms = set(s.enrichment["term"].astype(str))
    missing_terms = sorted(selected_terms.difference(enrichment_terms))
    if missing_terms:
        raise ValueError(f"Missing selected R5 pathway terms: {missing_terms}")
    selected_pathways = pd.concat(
        [
            s.enrichment.loc[
                s.enrichment["rsp_id"].astype(str).eq(CARRIER_RSP)
                & s.enrichment["term"].astype(str).isin(RSP0026_PATHWAYS)
            ],
            s.enrichment.loc[
                s.enrichment["rsp_id"].astype(str).eq(IMMUNE_RSP)
                & s.enrichment["term"].astype(str).isin(RSP0024_PATHWAYS)
            ],
        ],
        ignore_index=True,
    )
    if len(selected_pathways) != len(selected_terms) or not selected_pathways["fdr_bh"].lt(0.05).all():
        raise ValueError("All displayed R5 pathways must have FDR < 0.05")

    validate_source_columns(
        s.regional_context,
        {
            "regional_context",
            "regal_interface_score",
            "regal_top10",
        },
        name="regional_interface_spots.csv",
    )
    if len(s.regional_context) != 805:
        raise ValueError(f"Expected 805 pre-specified regional-context spots, found {len(s.regional_context)}")
    counts = s.regional_context["regional_context"].value_counts().to_dict()
    expected = {"neither": 449, "iCAF_RSP_only": 191, "immune_RSP_only": 134, "RSP_overlap": 31}
    if counts != expected:
        raise ValueError(f"Unexpected R5 regional-state counts: {counts}")

    validate_source_columns(
        s.interface_lr,
        {
            "candidate_axis",
            "regal_selected_detected_fraction",
            "regal_remainder_detected_fraction",
            "regal_fold_change",
            "regal_fdr_bh",
            "leaveout_fold_change",
            "leaveout_fdr_bh",
        },
        name="interface_lr_readback.csv",
    )
    if len(s.interface_lr) != 12:
        raise ValueError("R5 molecular-interface readback must contain 12 candidate axes")

    missing_cells = [cell for cell, _ in CORE_CELLTYPES if cell not in s.stereoscope]
    if missing_cells:
        raise ValueError(f"Missing core Stereoscope cell types: {missing_cells}")
    validate_source_columns(
        s.regional_celltypes,
        {"regional_context", "celltype", "n_spots", "mean"},
        name="regional_celltype_readback.csv",
    )
    validate_source_columns(
        s.regional_pathology,
        {"regional_context", "Classification", "n_spots", "fraction_within_context"},
        name="regional_pathology_readback.csv",
    )
    displayed_pathology = s.regional_pathology.loc[
        s.regional_pathology["regional_context"].astype(str).isin(
            ["iCAF_RSP_only", "immune_RSP_only", "RSP_overlap"]
        )
        & s.regional_pathology["Classification"].astype(str).isin(PATHOLOGY_ORDER)
    ]
    pathology_totals = displayed_pathology.groupby("regional_context")[
        "fraction_within_context"
    ].sum()
    if set(pathology_totals.index.astype(str)) != {
        "iCAF_RSP_only",
        "immune_RSP_only",
        "RSP_overlap",
    } or not np.allclose(pathology_totals.to_numpy(float), 1.0):
        raise ValueError("Displayed ST pathology fractions must sum to one in each regional state")
    if len(s.full_spots) != 1160:
        raise ValueError(f"Expected 1160 common display spots, found {len(s.full_spots)}")
    if s.activation.min().min() < 0 or s.activation.max().max() > 1:
        raise ValueError("RSP activation values are outside the shared 0-1 display contract")

def _style_quant_axis(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_LABEL_SIZE, length=3.0, width=0.8)

def _align_quantitative_axes_to_spatial_box(
    spatial_ax: plt.Axes,
    *quantitative_axes: plt.Axes,
) -> None:
    """Match quantitative axes to the active top and bottom of an equal-aspect map."""

    spatial_ax.apply_aspect()
    spatial_position = spatial_ax.get_position()
    for ax in quantitative_axes:
        position = ax.get_position()
        ax.set_position(
            [position.x0, spatial_position.y0, position.width, spatial_position.height],
            which="both",
        )

def _format_q(value: float) -> str:
    if value >= 0.05:
        return "ns"
    if value < 1e-4:
        return f"q={value:.1e}"
    return f"q={value:.3f}"

def _draw_caf_selection(ax: plt.Axes, s: R5FigureSources) -> None:
    icaf = s.retrieval.loc[
        s.retrieval["query"].astype(str).eq("icaf"),
        ["rsp_id", "odds_ratio"],
    ].rename(columns={"odds_ratio": "query_odds_ratio"})
    broad = s.retrieval.loc[
        s.retrieval["query"].astype(str).eq("broad_caf"),
        ["rsp_id", "fdr_bh"],
    ].rename(columns={"fdr_bh": "color_fdr"})
    data = (
        icaf.merge(broad, on="rsp_id")
        .merge(s.spatial_concordance, on="rsp_id")
        .dropna(subset=["icaf_spearman_rho"])
    )
    color = -np.log10(data["color_fdr"].clip(lower=1e-30))
    vmax = float(np.ceil(color.max()))
    scatter = ax.scatter(
        data["query_odds_ratio"],
        data["icaf_spearman_rho"],
        s=34,
        c=color,
        cmap=SELECTION_EVIDENCE_CMAP,
        vmin=0,
        vmax=vmax,
        edgecolors="#AAB1B3",
        linewidths=0.55,
        alpha=0.92,
    )
    selected = data.set_index("rsp_id").loc[CARRIER_RSP]
    ax.scatter(
        [selected["query_odds_ratio"]],
        [selected["icaf_spearman_rho"]],
        s=76,
        facecolors="none",
        edgecolors=REGAL_COLORS["carrier"],
        linewidths=1.45,
        zorder=4,
    )
    ax.axhline(0, color="#8D9698", lw=0.65, linestyle=(0, (3, 2)))
    ax.annotate(
        CARRIER_RSP,
        (float(selected["query_odds_ratio"]), float(selected["icaf_spearman_rho"])),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        color=REGAL_COLORS["carrier"],
    )
    ax.set_xlim(0, float(data["query_odds_ratio"].max()) * 1.08)
    ax.set_ylim(float(data["icaf_spearman_rho"].min()) - 0.06, 0.58)
    ax.set_xlabel("iCAF enrichment OR", fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.set_ylabel("iCAF spatial correlation (ρ)", fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.set_title(
        "iCAF query",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    _style_quant_axis(ax)
    cax = inset_axes(ax, width="3%", height="46%", loc="lower right", borderpad=0.9)
    colorbar = ax.figure.colorbar(scatter, cax=cax)
    colorbar.set_ticks([0, vmax])
    colorbar.ax.tick_params(labelsize=COLORBAR_SIZE, length=2.0, width=0.8)
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    colorbar.set_label("Broad-CAF\n−log10 FDR", fontsize=COLORBAR_SIZE, labelpad=2.0)

def _draw_tcell_selection(ax: plt.Axes, s: R5FigureSources) -> None:
    data = (
        s.retrieval.loc[
            s.retrieval["query"].astype(str).eq("t_cell"),
            ["rsp_id", "odds_ratio", "fdr_bh"],
        ]
        .merge(s.spatial_concordance, on="rsp_id")
        .dropna(subset=["tcell_spearman_rho"])
    )
    color = -np.log10(data["fdr_bh"].clip(lower=1e-30))
    vmax = float(np.ceil(color.max()))
    scatter = ax.scatter(
        data["odds_ratio"],
        data["tcell_spearman_rho"],
        s=34,
        c=color,
        cmap=SELECTION_EVIDENCE_CMAP,
        vmin=0,
        vmax=vmax,
        edgecolors="#AAB1B3",
        linewidths=0.55,
        alpha=0.92,
    )
    selected = data.set_index("rsp_id").loc[IMMUNE_RSP]
    ax.scatter(
        [selected["odds_ratio"]],
        [selected["tcell_spearman_rho"]],
        s=76,
        facecolors="none",
        edgecolors=REGAL_COLORS["immune"],
        linewidths=1.45,
        zorder=4,
    )
    ax.axhline(0, color="#8D9698", lw=0.65, linestyle=(0, (3, 2)))
    ax.annotate(
        IMMUNE_RSP,
        (float(selected["odds_ratio"]), float(selected["tcell_spearman_rho"])),
        xytext=(-46, -2),
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        color=REGAL_COLORS["immune"],
    )
    ax.set_xlabel("T-cell enrichment OR", fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.set_ylabel("CD4+CD8 T spatial correlation (ρ)", fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.set_xlim(0, float(data["odds_ratio"].max()) * 1.08)
    ax.set_ylim(float(data["tcell_spearman_rho"].min()) - 0.06, 0.70)
    ax.set_title(
        "T-cell query",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    _style_quant_axis(ax)
    cax = inset_axes(ax, width="3%", height="46%", loc="lower right", borderpad=0.9)
    colorbar = ax.figure.colorbar(scatter, cax=cax)
    colorbar.set_ticks([0, vmax])
    colorbar.ax.tick_params(labelsize=COLORBAR_SIZE, length=2.0, width=0.8)
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    colorbar.set_label("T-cell\n−log10 FDR", fontsize=COLORBAR_SIZE, labelpad=2.0)

def _selected_pathway_table(
    s: R5FigureSources,
    rsp_id: str,
    terms: tuple[str, ...],
) -> pd.DataFrame:
    data = s.enrichment.loc[
        s.enrichment["rsp_id"].astype(str).eq(rsp_id)
        & s.enrichment["term"].astype(str).isin(terms)
    ].copy()
    data["term"] = pd.Categorical(data["term"], categories=terms, ordered=True)
    data = data.sort_values("term")
    data["score"] = data["odds_ratio"].astype(float)
    data["label"] = data["term"].astype(str).map(PATHWAY_LABELS)
    return data

def _draw_mirrored_pathways(ax: plt.Axes, s: R5FigureSources) -> None:
    immune = _selected_pathway_table(s, IMMUNE_RSP, RSP0024_PATHWAYS)
    carrier = _selected_pathway_table(s, CARRIER_RSP, RSP0026_PATHWAYS)
    gap = 0.48
    y_carrier = np.arange(len(carrier))[::-1]
    y_immune = np.arange(len(immune))[::-1] + len(carrier) + gap
    ax.barh(
        y_immune,
        immune["score"],
        height=0.58,
        color=REGAL_COLORS["immune"],
        alpha=0.94,
    )
    ax.barh(
        y_carrier,
        -carrier["score"],
        height=0.58,
        color=REGAL_COLORS["carrier"],
        alpha=0.94,
    )
    ax.axvline(0, color="#4E5658", linewidth=0.7)
    for y, label in zip(y_immune, immune["label"], strict=True):
        ax.text(
            -0.18,
            y,
            label,
            ha="right",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=INK,
        )
    for y, label in zip(y_carrier, carrier["label"], strict=True):
        ax.text(
            0.18,
            y,
            label,
            ha="left",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=INK,
        )
    ax.set_xlim(-14, 14)
    ax.set_ylim(-1.10, len(immune) + len(carrier) + gap + 0.20)
    ax.set_yticks([])
    ticks = np.array([-10, -5, 0, 5, 10], dtype=float)
    ax.set_xticks(ticks, [f"{abs(int(value))}" for value in ticks])
    ax.set_xlabel("Enrichment OR", fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.set_title(
        "Selected pathways",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    ax.text(
        -13.55,
        float(y_immune.max()) + 0.72,
        "RSP0024",
        ha="left",
        va="center",
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        color=REGAL_COLORS["immune"],
    )
    ax.text(
        -13.55,
        float(y_carrier.min()) - 0.70,
        "RSP0026",
        ha="left",
        va="center",
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        color=REGAL_COLORS["carrier"],
    )
    _style_quant_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.grid(axis="y", visible=False)

def _draw_base_spots(ax: plt.Axes, s: R5FigureSources) -> pd.DataFrame:
    xy = s.xy.loc[s.xy.index.intersection(s.full_spots)]
    ax.scatter(
        xy["x_img"],
        xy["y_img"],
        s=2.8,
        facecolors="none",
        edgecolors="#7E8789",
        linewidths=0.18,
        alpha=0.18,
        rasterized=True,
        zorder=2,
    )
    return xy

def _draw_abundance_map(
    ax: plt.Axes,
    s: R5FigureSources,
    *,
    celltype: str,
    title: str,
) -> None:
    draw_histology(ax, s.image, alpha=0.28)
    xy = _draw_base_spots(ax, s)
    values = s.stereoscope.loc[xy.index, celltype].astype(float)
    positive = values.gt(0)
    vmax = max(float(values.quantile(0.99)), 1e-6)
    norm = mpl.colors.PowerNorm(gamma=0.38, vmin=0, vmax=vmax)
    points = xy.loc[positive]
    scatter = ax.scatter(
        points["x_img"],
        points["y_img"],
        c=values.loc[points.index],
        s=6.8,
        cmap=ABUNDANCE_CMAP,
        norm=norm,
        linewidths=0,
        alpha=0.96,
        rasterized=True,
        zorder=3,
    )
    ax.set_title(
        title,
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, xy, spot_radius_px=s.spot_radius_px, anchor="NW")
    cax = ax.inset_axes([0.86, 0.06, 0.025, 0.34])
    colorbar = ax.figure.colorbar(scatter, cax=cax)
    colorbar.set_ticks([0, vmax])
    colorbar.ax.set_yticklabels(["0", f"{vmax:.2f}"])
    colorbar.ax.tick_params(labelsize=COLORBAR_SIZE, length=2.0, width=0.8, pad=1.0)
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)

def _draw_rsp_map(
    ax: plt.Axes,
    s: R5FigureSources,
    *,
    rsp_id: str,
) -> None:
    draw_histology(ax, s.image)
    xy = _draw_base_spots(ax, s)
    values = s.activation.loc[xy.index, rsp_id].astype(float)
    positive = values.gt(0)
    points = xy.loc[positive]
    is_carrier = rsp_id == CARRIER_RSP
    activity_cmap = CARRIER_ACTIVITY_CMAP if is_carrier else IMMUNE_ACTIVITY_CMAP
    ax.scatter(
        points["x_img"],
        points["y_img"],
        c=values.loc[points.index],
        s=6.8,
        cmap=activity_cmap,
        norm=RSP_ACTIVITY_NORM,
        linewidths=0,
        alpha=0.97,
        rasterized=True,
        zorder=3,
    )
    title_color = REGAL_COLORS["carrier"] if is_carrier else REGAL_COLORS["immune"]
    ax.set_title(
        rsp_id,
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        color=title_color,
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, xy, spot_radius_px=s.spot_radius_px)
    cax = ax.inset_axes([0.90, 0.06, 0.025, 0.34])
    colorbar = ax.figure.colorbar(
        mpl.cm.ScalarMappable(norm=RSP_ACTIVITY_NORM, cmap=activity_cmap),
        cax=cax,
    )
    colorbar.set_ticks([0, 1])
    colorbar.ax.tick_params(labelsize=COLORBAR_SIZE, length=2.0, width=0.8, pad=1.0)
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)

def _draw_regional_state_map(ax: plt.Axes, s: R5FigureSources) -> None:
    draw_histology(ax, s.image)
    full_xy = _draw_base_spots(ax, s)
    for state in REGIONAL_STATE_ORDER:
        spots = s.regional_context.index[
            s.regional_context["regional_context"].astype(str).eq(state)
        ]
        points = s.xy.loc[s.xy.index.intersection(spots)]
        ax.scatter(
            points["x_img"],
            points["y_img"],
            s=6.2,
            c=REGIONAL_STATE_COLORS[state],
            linewidths=0,
            alpha=0.92 if state != "neither" else 0.56,
            rasterized=True,
            zorder=3,
        )
    ax.set_title(
        "Regional states",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, full_xy, spot_radius_px=s.spot_radius_px, anchor="NW")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.0,
            markerfacecolor=REGIONAL_STATE_COLORS[state],
            markeredgewidth=0,
            label=REGIONAL_STATE_LABELS[state],
        )
        for state in REGIONAL_STATE_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.02),
        ncol=2,
        fontsize=LEGEND_SIZE,
        handletextpad=0.25,
        columnspacing=0.75,
    )

def _celltype_contrast_q(s: R5FigureSources, celltype: str) -> float:
    rows = s.regional_celltypes.loc[
        s.regional_celltypes["celltype"].astype(str).eq(celltype)
        & s.regional_celltypes["comparison"].astype(str).eq("RSP_overlap > iCAF_RSP_only")
    ]
    if rows.empty:
        raise ValueError(f"Missing overlap contrast for {celltype}")
    return float(rows.iloc[0]["fdr_bh"])

def _draw_core_abundance_contrast(
    ax: plt.Axes,
    s: R5FigureSources,
    *,
    celltype: str,
    title: str,
    seed: int,
    show_ylabel: bool,
) -> None:
    groups = ("iCAF_RSP_only", "RSP_overlap")
    colors = (REGAL_COLORS["carrier"], REGAL_COLORS["overlap"])
    values = []
    rng = np.random.default_rng(seed)
    for position, (group, color) in enumerate(zip(groups, colors, strict=True)):
        spots = s.regional_context.index[
            s.regional_context["regional_context"].astype(str).eq(group)
        ]
        array = s.stereoscope.loc[spots, celltype].to_numpy(float)
        values.append(array)
        jitter = rng.uniform(-0.13, 0.13, len(array))
        ax.scatter(
            np.full(len(array), position) + jitter,
            array,
            s=3.3,
            c=color,
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    boxes = ax.boxplot(
        values,
        positions=[0, 1],
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": REGAL_COLORS["ink"], "lw": 0.85},
        whiskerprops={"color": REGAL_COLORS["mid"], "lw": 0.55},
        capprops={"color": REGAL_COLORS["mid"], "lw": 0.55},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.24)
    q_value = _celltype_contrast_q(s, celltype)
    ax.text(
        0.98,
        0.95,
        _format_q(q_value),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color=INK,
    )
    ax.set_xticks(
        [0, 1],
        ["RSP0026 only\nn=191", "Overlap\nn=31"],
        fontsize=TICK_LABEL_SIZE,
    )
    ax.set_title(
        title,
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    if show_ylabel:
        ax.set_ylabel("Cell proportion", fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    else:
        ax.set_ylabel("")
    _style_quant_axis(ax)

def _draw_pathology_composition(ax: plt.Axes, s: R5FigureSources) -> None:
    contexts = ("iCAF_RSP_only", "immune_RSP_only", "RSP_overlap")
    x = np.arange(len(contexts), dtype=float)
    bottom = np.zeros(len(contexts))
    for pathology in PATHOLOGY_ORDER:
        fractions = []
        for context in contexts:
            rows = s.regional_pathology.loc[
                s.regional_pathology["regional_context"].astype(str).eq(context)
                & s.regional_pathology["Classification"].astype(str).eq(pathology)
            ]
            fractions.append(
                float(rows.iloc[0]["fraction_within_context"]) if not rows.empty else 0.0
            )
        fractions_array = np.asarray(fractions)
        ax.bar(
            x,
            fractions_array,
            bottom=bottom,
            width=0.58,
            color=PATHOLOGY_COLORS[pathology],
            label=PATHOLOGY_LABELS[pathology],
        )
        bottom += fractions_array
    counts = s.regional_context["regional_context"].astype(str).value_counts()
    ax.set_xticks(
        x,
        [
            f"RSP0026 only\nn={counts.get('iCAF_RSP_only', 0)}",
            f"RSP0024 only\nn={counts.get('immune_RSP_only', 0)}",
            f"Overlap\nn={counts.get('RSP_overlap', 0)}",
        ],
        fontsize=TICK_LABEL_SIZE,
    )
    ax.set_xlim(-0.45, len(contexts) - 0.55)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_ylabel("Composition (%)", fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.set_title(
        "Pathology composition",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.10),
        ncol=2,
        fontsize=LEGEND_SIZE,
        handlelength=0.75,
        handletextpad=0.20,
        columnspacing=0.90,
        frameon=False,
    )
    _style_quant_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=GRID_LINE_WIDTH)
    ax.grid(axis="x", visible=False)

def _draw_continuous_interface(ax: plt.Axes, s: R5FigureSources) -> None:
    draw_histology(ax, s.image)
    full_xy = _draw_base_spots(ax, s)
    domain_xy = s.xy.loc[s.xy.index.intersection(s.regional_context.index)]
    ax.scatter(
        domain_xy["x_img"],
        domain_xy["y_img"],
        s=3.2,
        c="#BBC2C4",
        linewidths=0,
        alpha=0.42,
        rasterized=True,
        zorder=3,
    )
    scores = s.regional_context["regal_interface_score"].astype(float)
    positive = scores.gt(0)
    points = s.xy.loc[s.xy.index.intersection(scores.index[positive])]
    vmax = max(float(scores.quantile(0.99)), 1e-6)
    norm = mpl.colors.PowerNorm(gamma=0.55, vmin=0, vmax=vmax)
    scatter = ax.scatter(
        points["x_img"],
        points["y_img"],
        c=scores.loc[points.index],
        s=6.8,
        cmap=INTERFACE_CMAP,
        norm=norm,
        linewidths=0,
        alpha=0.94,
        rasterized=True,
        zorder=4,
    )
    top_spots = s.regional_context.index[s.regional_context["regal_top10"].astype(bool)]
    top_xy = s.xy.loc[s.xy.index.intersection(top_spots)]
    ax.scatter(
        top_xy["x_img"],
        top_xy["y_img"],
        s=10.5,
        facecolors="none",
        edgecolors=REGAL_COLORS["overlap"],
        linewidths=0.65,
        alpha=0.95,
        rasterized=True,
        zorder=5,
    )
    ax.set_title(
        "Continuous interface score",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    finish_spatial_axis(ax, full_xy, spot_radius_px=s.spot_radius_px, anchor="NW")
    cax = ax.inset_axes([1.05, 0.08, 0.035, 0.34])
    colorbar = ax.figure.colorbar(scatter, cax=cax)
    colorbar.set_ticks([0, vmax])
    colorbar.ax.set_yticklabels(["0", f"{vmax:.2f}"])
    colorbar.ax.tick_params(labelsize=COLORBAR_SIZE, length=2.0, width=0.8)
    colorbar.outline.set_linewidth(OUTLINE_WIDTH)
    colorbar.set_label("Interface score", fontsize=COLORBAR_SIZE, labelpad=2.0)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=4.0,
                markerfacecolor="none",
                markeredgecolor=REGAL_COLORS["overlap"],
                label="Top 10% interface (n=81)",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=LEGEND_SIZE,
    )

def _lr_axis_order(s: R5FigureSources) -> list[str]:
    data = s.interface_lr.loc[s.interface_lr["regal_fdr_bh"].lt(0.05)].copy()
    if data.empty:
        raise ValueError("No primary candidate axes pass FDR < 0.05")
    return data.sort_values(["regal_fdr_bh", "regal_fold_change"])["candidate_axis"].astype(str).tolist()

def _draw_lr_detection(ax: plt.Axes, s: R5FigureSources, order: list[str]) -> None:
    data = s.interface_lr.set_index("candidate_axis").loc[order]
    y = np.arange(len(data))
    selected = data["regal_selected_detected_fraction"].to_numpy(float)
    remainder = data["regal_remainder_detected_fraction"].to_numpy(float)
    for yy, left, right in zip(y, remainder, selected, strict=True):
        ax.plot([left, right], [yy, yy], color="#ADB5B7", lw=0.7, zorder=1)
    ax.scatter(remainder, y, s=17, c="#AEB6B8", label="Background", zorder=2)
    ax.scatter(selected, y, s=19, c=REGAL_COLORS["overlap"], label="Top interface", zorder=3)
    ax.set_yticks(y, order, fontsize=TICK_LABEL_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Detected fraction", fontsize=AXIS_LABEL_SIZE, labelpad=2.0)
    ax.set_title(
        "Co-detection coverage",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        ncol=1,
        fontsize=LEGEND_SIZE,
        handletextpad=0.25,
        columnspacing=0.8,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.86,
    )
    _style_quant_axis(ax)

def _draw_lr_effects(
    ax: plt.Axes,
    s: R5FigureSources,
    order: list[str],
) -> None:
    data = s.interface_lr.set_index("candidate_axis").loc[order]
    y = np.arange(len(data))
    primary = data["regal_fold_change"].fillna(0).to_numpy(float)
    leaveout = data["leaveout_fold_change"].fillna(0).to_numpy(float)
    for yy, first, second in zip(y, primary, leaveout, strict=True):
        ax.plot([first, second], [yy, yy], color="#ADB5B7", lw=0.7, zorder=1)
    leaveout_sig = data["leaveout_fdr_bh"].lt(0.05).to_numpy()
    ax.scatter(
        primary,
        y,
        s=24,
        facecolors=np.full(len(primary), REGAL_COLORS["overlap"]),
        edgecolors=REGAL_COLORS["overlap"],
        linewidths=0.85,
        label="Primary interface",
        zorder=3,
    )
    ax.scatter(
        leaveout,
        y,
        s=24,
        facecolors=np.where(leaveout_sig, ROBUSTNESS_COLOR, "white"),
        edgecolors=ROBUSTNESS_COLOR,
        linewidths=0.85,
        label="Leave-out",
        zorder=3,
    )
    ax.axvline(1.0, color="#8D9698", lw=0.7, linestyle=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(-0.15, max(7.2, float(np.nanmax([primary, leaveout])) * 1.05))
    ax.set_xlabel(
        "Interface/background fold change",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=2.0,
    )
    ax.set_title(
        "Interface enrichment",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=4.0,
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        ncol=1,
        fontsize=LEGEND_SIZE,
        handletextpad=0.25,
        columnspacing=0.8,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.86,
    )
    _style_quant_axis(ax)

def build_row1_hypothesis_retrieval(s: R5FigureSources) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.35), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 0.40, 1.0, 0.08, 2.0],
        wspace=0.0,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[0, 4])
    _draw_caf_selection(ax_a, s)
    _draw_tcell_selection(ax_b, s)
    _draw_mirrored_pathways(ax_c, s)
    fig.subplots_adjust(left=0.042, right=0.985, top=0.82, bottom=0.18)
    position_c = ax_c.get_position()
    ax_c.set_position(
        [
            position_c.x0 + 0.035,
            position_c.y0,
            position_c.width - 0.035,
            position_c.height,
        ]
    )
    add_panel_label(fig, "A", ax_a, align_to_canvas_left=True)
    add_panel_label(fig, "B", ax_b)
    add_panel_label(fig, "C", ax_c)
    return fig

def build_row2_spatial_translation(s: R5FigureSources) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.15), constrained_layout=False)
    grid = fig.add_gridspec(1, 6, wspace=0.045)
    axes = [fig.add_subplot(grid[0, index]) for index in range(6)]
    for ax, (celltype, label) in zip(axes[:4], CORE_CELLTYPES, strict=True):
        _draw_abundance_map(ax, s, celltype=celltype, title=f"{label} abundance")
    _draw_rsp_map(axes[4], s, rsp_id=CARRIER_RSP)
    _draw_rsp_map(axes[5], s, rsp_id=IMMUNE_RSP)
    fig.subplots_adjust(left=0.028, right=0.990, top=0.82, bottom=0.025)
    add_panel_label(fig, "D", axes[0], align_to_canvas_left=True)
    add_panel_label(fig, "E", axes[4])
    return fig

def build_row3_regional_transition(s: R5FigureSources) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.90), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.18, 3.04, 1.78], wspace=0.24)
    ax_f = fig.add_subplot(grid[0, 0])
    _draw_regional_state_map(ax_f, s)
    middle = grid[0, 1].subgridspec(2, 2, wspace=0.24, hspace=0.72)
    contrast_axes = [
        fig.add_subplot(middle[row, col]) for row in range(2) for col in range(2)
    ]
    for index, (ax, (celltype, label), seed) in enumerate(zip(
        contrast_axes,
        CORE_CELLTYPES,
        (17, 23, 31, 37),
        strict=True,
    )):
        _draw_core_abundance_contrast(
            ax,
            s,
            celltype=celltype,
            title=label,
            seed=seed,
            show_ylabel=index in (0, 2),
        )
    ax_h = fig.add_subplot(grid[0, 2])
    _draw_pathology_composition(ax_h, s)
    fig.subplots_adjust(left=0.028, right=0.985, top=0.85, bottom=0.20)
    add_panel_label(fig, "F", ax_f, align_to_canvas_left=True)
    add_panel_label(fig, "G", contrast_axes[0])
    add_panel_label(fig, "H", ax_h)
    return fig

def build_row4_molecular_interface(s: R5FigureSources) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.55), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.55, 2.20, 2.25], wspace=0.34)
    ax_i = fig.add_subplot(grid[0, 0])
    _draw_continuous_interface(ax_i, s)
    order = _lr_axis_order(s)
    ax_j = fig.add_subplot(grid[0, 1])
    ax_k = fig.add_subplot(grid[0, 2])
    _draw_lr_detection(ax_j, s, order)
    _draw_lr_effects(ax_k, s, order)
    fig.subplots_adjust(left=0.028, right=0.985, top=0.84, bottom=0.16)
    position_j = ax_j.get_position()
    ax_j.set_position(
        [
            position_j.x0 + 0.012,
            position_j.y0,
            position_j.width,
            position_j.height,
        ]
    )
    _align_quantitative_axes_to_spatial_box(ax_i, ax_j, ax_k)
    add_panel_label(fig, "I", ax_i, align_to_canvas_left=True)
    add_panel_label(fig, "J", ax_j)
    add_panel_label(fig, "K", ax_k)
    return fig

def export_scrna_spatial_bridge_figure() -> list[Path]:
    configure_figure_style()
    sources = load_r5_figure_sources()
    builders = (build_row1_hypothesis_retrieval, build_row2_spatial_translation, build_row3_regional_transition, build_row4_molecular_interface,)
    rows = (render_row(builder(sources)) for builder in builders)
    return save_composite_bundle(stack_row_images(rows), FIGURE_STEM)

def draw_figure() -> None:
    for output in export_scrna_spatial_bridge_figure():
        print(output)

def configure():
    global ARTIFACT_DIR, FIGURE_STEM, H5AD_PATH, SAMPLE_ID, STEREOSCOPE_PATH
    FIGURE_STEM = OUTPUT_DIR / 'fig08'
    H5AD_PATH = path('h5ad')
    ARTIFACT_DIR = path('sample_dir')
    STEREOSCOPE_PATH = path('stereoscope')
    SAMPLE_ID = option('sample_id')

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
    _CONFIG = document['fig08']
    configure()
    result = draw_figure()
    if result is not None:
        for output in result:
            print(output)

if __name__ == '__main__':
    main()
