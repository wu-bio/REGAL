from __future__ import annotations

"""Shared visual constants for REGAL manuscript figures."""

from typing import Any

import matplotlib as mpl


FIGURE_WIDTH_IN = 13.8
EXPORT_DPI = 600

FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
PANEL_LABEL_SIZE = 15.0
PANEL_LABEL_X_PAD_PT = 12.0
PANEL_LABEL_Y_PAD_PT = 18.0
PANEL_LABEL_CANVAS_LEFT_PT = 8.0
PANEL_TITLE_SIZE = 13.0
AXIS_LABEL_SIZE = 11.0
TICK_LABEL_SIZE = 10.0
LEGEND_SIZE = 10.0
COLORBAR_SIZE = 10.0
ANNOTATION_SIZE = 9.5

SPINE_WIDTH = 1.2
DATA_LINE_WIDTH = 1.8
REFERENCE_LINE_WIDTH = 1.2
GRID_LINE_WIDTH = 0.6
OUTLINE_WIDTH = 1.0

INK = "#252525"
MID = "#7C8389"
GRID = "#D8DCDD"
PAPER = "#F7F5F0"

METHOD_COLORS = {
    "NMF": "#D8DCDD",
    "NNMF": "#738190",
    "STAMP": "#D6A63A",
    "REGAL": "#665191",
    "RSS input": "#665191",
    "HVG expression": "#B8D5D8",
}

RSP_ACTIVITY_COLORS = [
    "#F4F1EB",
    "#B8D5D8",
    "#5A9CB0",
    "#3676B8",
    "#665191",
]

MARKER_EXPRESSION_COLORS = [
    "#F3F8F8",
    "#B8DDD8",
    "#70AAA4",
    "#155E63",
]

DISTRIBUTION_EXPRESSION_COLORS = [
    "#E8E0DC",
    "#CDBBB6",
    "#B29590",
    "#92706F",
    "#704F55",
]

CONTRIBUTION_HEATMAP_COLORS = [
    "#2166AC",
    "#92C5DE",
    "#F7F7F7",
    "#F4A582",
    "#B2182B",
]


def configure_manuscript_style() -> None:
    """Apply the shared REGAL manuscript style to matplotlib."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": TICK_LABEL_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.labelweight": "normal",
            "axes.titleweight": "bold",
            "axes.titlelocation": "center",
            "axes.linewidth": SPINE_WIDTH,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": LEGEND_SIZE,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rsp_activity_cmap(name: str = "regal_rsp_activity") -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, RSP_ACTIVITY_COLORS)


def marker_expression_cmap(name: str = "regal_marker_expression") -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, MARKER_EXPRESSION_COLORS)


def distribution_expression_cmap(
    name: str = "regal_distribution_expression",
) -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        name,
        DISTRIBUTION_EXPRESSION_COLORS,
    )


def contribution_heatmap_cmap(
    name: str = "regal_contribution_heatmap",
) -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        name,
        CONTRIBUTION_HEATMAP_COLORS,
    )


def add_panel_label(
    container: Any,
    label: str,
    bounds: Any,
    *,
    x_pad_pt: float = PANEL_LABEL_X_PAD_PT,
    y_pad_pt: float = PANEL_LABEL_Y_PAD_PT,
    align_to_canvas_left: bool = False,
) -> mpl.text.Text:
    """Place a panel label using fixed physical panel or canvas offsets."""

    container_width_pt = container.bbox.width * 72.0 / container.dpi
    container_height_pt = container.bbox.height * 72.0 / container.dpi
    if container_width_pt <= 0 or container_height_pt <= 0:
        raise ValueError("Panel-label container must have positive dimensions")
    x_position = (
        PANEL_LABEL_CANVAS_LEFT_PT / container_width_pt
        if align_to_canvas_left
        else bounds.x0 - x_pad_pt / container_width_pt
    )
    return container.text(
        x_position,
        bounds.y1 + y_pad_pt / container_height_pt,
        label,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
