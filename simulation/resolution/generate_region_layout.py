"""Generate the fixed REGAL spatial simulation design without external inputs.

The vector contours were traced once from regional annotations and embedded here.
Use generate_layout() for the base partition and build_supports() for overlapping
region/activity masks. Neither function reads files, plots or generates expression.
Run this file directly only when arrays, design notes and the overview PNG are needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.path import Path as PolygonPath
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, label

RegionGrid = NDArray[np.int16]
Coordinates = NDArray[np.float64]
SupportGrid = NDArray[np.bool_]

DEFAULT_NSIDE = 128
SPATIAL_EXTENT = 128.0
NESTED_PARENT_ID = 13
NESTED_REGION_ID = 16
REGION_IDS = tuple(range(1, NESTED_REGION_ID + 1))


class OverlapSpec(TypedDict):
    """A local source-to-target extension in fixed design-space coordinates."""

    source: int
    target: int
    center: tuple[int, int]
    radii: tuple[int, int]
    depth: int


ACTIVITY_GROUPS: dict[str, tuple[int, ...]] = {
    "A1": (5, 10, 11, 16),
    "A2": (4, 12, 13),
    "A3": (1,),
    "A4": (2,),
    "A5": (3,),
    "A6": (6,),
    "A7": (7,),
    "A8": (8,),
    "A9": (9,),
    "A10": (14,),
    "A11": (15,),
}

# The source gains support; the target retains its original activity.
OVERLAP_SPECS: list[OverlapSpec] = [
    {"source": 14, "target": 1, "center": (39, 20), "radii": (14, 14), "depth": 7},
    {"source": 14, "target": 2, "center": (64, 20), "radii": (22, 17), "depth": 8},
    {"source": 14, "target": 3, "center": (92, 16), "radii": (14, 15), "depth": 7},
    {"source": 15, "target": 2, "center": (82, 53), "radii": (16, 15), "depth": 7},
    {"source": 15, "target": 10, "center": (92, 68), "radii": (14, 18), "depth": 6},
]

ACTIVITY_COLORS = (
    "#387BAD", "#DB9863", "#E8BB8E", "#47745C", "#859BC5", "#87BC7F", "#E38586", "#A342D2",
    "#ADB957", "#A87972", "#83C5D2",
)
PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
}

# Geometry provenance only; original annotation names are not activity truth.
DESIGN_NOTES = {
    "source_geometry": "Contours traced from existing BREAST regional annotations, simplified and fixed in Python.",
    "source_sha256": "6c9e23b461057754f5deee32a1af00919a7a91dcf16415702e15d65b32f470fb",
    "source_unchanged": True,
    "grouping": {
        "R01": ["IDC_5"],
        "R02": ["DCIS/LCIS_3"],
        "R03": ["IDC_4"],
        "R04": ["Tumor_edge_1"],
        "R05": ["DCIS/LCIS_5"],
        "R06": ["IDC_6"],
        "R07": ["IDC_7"],
        "R08": ["Healthy_1"],
        "R09": ["IDC_2"],
        "R10": ["DCIS/LCIS_4"],
        "R11": ["DCIS/LCIS_1"],
        "R12": ["Tumor_edge_3", "IDC_1"],
        "R13": ["Tumor_edge_4"],
        "R14": ["Tumor_edge_2", "IDC_3"],
        "R15": ["Healthy_2", "Tumor_edge_5"],
        "R16": ["DCIS/LCIS_2"],
    },
    "removed_scattered_label": (
        "Tumor_edge_6 (182 original positions); these positions are assigned by neighboring retained regions."
    ),
    "smoothing": (
        "Nearest retained label on normalized square, Gaussian sigma 0.875 design units; "
        "contours simplified at tolerance 0.35 units."
    ),
    "runtime_dependency": "No raw input, stored label arrays, images or external geometry files required.",
    "n_rings": 16,
    "n_control_points": 598,
    "nested_region": {"R16": {"parent": "R13", "source_positions": 28, "source": "DCIS/LCIS_2"}},
}

# Closed contour rings in the fixed design space; nested rings use even-odd filling.
REGION_CONTOURS = [
    # R01
    [
        [
            [0.125, 0.0], [37.0, 0.125], [37.0, 6.625], [36.25, 9.375],
            [37.0, 12.625], [37.0, 16.375], [38.75, 19.875], [39.0, 22.625],
            [42.75, 29.875], [42.25, 32.875], [40.75, 34.875], [39.5, 37.875],
            [39.5, 42.625], [41.25, 46.625], [41.5, 48.875], [43.5, 52.875],
            [43.25, 54.125], [41.125, 55.0], [38.125, 58.25], [37.5, 61.125],
            [35.75, 64.375], [34.375, 65.75], [28.125, 67.5], [23.875, 65.5],
            [18.625, 59.75], [16.25, 58.125], [15.75, 56.125], [14.5, 54.875],
            [14.25, 53.125], [12.75, 51.375], [12.5, 49.375], [11.0, 46.125],
            [11.0, 40.375], [9.25, 36.375], [7.375, 35.0], [0.0, 34.875],
            [0.125, 0.0],
        ],
    ],
    # R02
    [
        [
            [51.625, 14.5], [54.125, 15.25], [56.875, 15.25], [58.375, 16.0],
            [61.375, 15.75], [64.625, 18.0], [67.625, 18.75], [69.625, 20.0],
            [72.125, 22.75], [79.75, 27.875], [80.0, 29.625], [81.5, 31.375],
            [82.5, 34.875], [82.5, 38.625], [83.25, 40.375], [85.375, 42.25],
            [89.75, 43.125], [90.5, 46.125], [88.25, 50.125], [86.75, 54.625],
            [85.875, 55.25], [84.625, 55.25], [83.375, 54.5], [81.125, 54.25],
            [79.375, 55.0], [78.25, 56.875], [77.0, 60.625], [74.125, 63.0],
            [71.875, 64.0], [69.625, 62.25], [66.625, 62.5], [64.875, 61.25],
            [62.125, 61.5], [60.875, 61.0], [57.375, 58.75], [55.75, 57.125],
            [55.25, 55.625], [52.75, 53.375], [49.75, 47.375], [49.5, 42.875],
            [48.0, 39.625], [47.5, 36.625], [46.0, 33.875], [46.25, 18.625],
            [47.375, 17.25], [51.625, 14.5],
        ],
    ],
    # R03
    [
        [
            [92.625, 0.0], [128.0, 0.125], [128.0, 91.375], [124.375, 90.5],
            [123.125, 88.75], [123.0, 84.375], [123.75, 82.125], [123.0, 80.125],
            [123.0, 74.875], [121.25, 71.875], [119.25, 65.125], [116.375, 63.0],
            [113.375, 62.75], [110.75, 56.625], [107.375, 53.25], [104.125, 52.0],
            [100.375, 52.0], [98.875, 50.0], [95.625, 49.25], [93.125, 47.0],
            [90.875, 46.25], [90.25, 45.375], [89.75, 43.375], [90.0, 40.375],
            [89.25, 37.625], [89.25, 32.875], [88.25, 30.375], [91.5, 21.375],
            [91.5, 19.875], [90.0, 16.125], [91.5, 12.625], [91.75, 10.625],
            [93.25, 8.875], [93.25, 5.875], [92.5, 4.375], [92.625, 0.0],
        ],
    ],
    # R04
    [
        [
            [0.125, 35.0], [7.375, 35.0], [9.25, 36.375], [11.0, 40.375],
            [11.0, 46.125], [12.25, 48.625], [12.75, 51.375], [14.25, 53.125],
            [14.5, 54.875], [15.75, 56.125], [16.25, 58.125], [18.625, 59.75],
            [23.875, 65.5], [26.375, 67.0], [28.75, 67.625], [28.75, 71.625],
            [30.125, 73.75], [32.875, 75.5], [35.625, 75.25], [38.875, 77.0],
            [41.875, 79.0], [43.0, 80.375], [43.0, 81.625], [46.25, 84.375],
            [46.75, 87.125], [46.375, 87.75], [43.375, 89.0], [39.875, 89.25],
            [33.875, 94.25], [32.625, 94.5], [31.125, 93.75], [23.125, 93.75],
            [21.625, 94.5], [20.125, 94.5], [18.875, 94.5], [14.875, 92.25],
            [11.375, 92.0], [7.625, 89.5], [5.875, 89.75], [4.625, 91.25],
            [1.125, 93.25], [0.0, 93.125], [0.125, 80.25], [4.625, 82.25],
            [7.125, 82.75], [9.625, 84.25], [14.125, 84.25], [15.625, 83.5],
            [18.625, 83.25], [20.0, 79.875], [19.5, 77.625], [19.5, 69.875],
            [17.75, 65.625], [17.5, 63.625], [15.75, 62.125], [14.75, 59.875],
            [10.5, 56.375], [9.25, 53.875], [7.75, 52.625], [7.25, 47.875],
            [0.0, 42.125], [0.125, 35.0],
        ],
    ],
    # R05
    [
        [
            [0.125, 42.25], [7.25, 47.875], [7.75, 52.625], [9.25, 53.875],
            [10.5, 56.375], [14.75, 59.875], [15.75, 62.125], [17.5, 63.625],
            [17.75, 65.625], [19.5, 69.875], [19.5, 77.625], [20.0, 79.875],
            [18.625, 83.25], [15.625, 83.5], [14.125, 84.25], [9.625, 84.25],
            [7.125, 82.75], [4.625, 82.25], [0.0, 80.125], [0.125, 42.25],
        ],
    ],
    # R06
    [
        [
            [42.125, 33.0], [46.25, 34.125], [49.5, 42.875], [50.25, 48.625],
            [52.75, 53.375], [55.25, 55.625], [55.75, 57.125], [61.375, 61.25],
            [62.625, 61.5], [63.875, 61.0], [66.625, 62.5], [69.625, 62.25],
            [71.75, 63.875], [71.5, 70.625], [68.25, 73.625], [68.5, 79.125],
            [65.125, 82.75], [62.125, 86.25], [59.125, 85.75], [56.875, 86.75],
            [55.75, 86.625], [54.75, 82.875], [54.5, 73.875], [47.0, 59.125],
            [43.0, 54.375], [43.5, 52.875], [41.5, 48.875], [41.25, 46.625],
            [39.5, 42.625], [39.5, 37.875], [42.125, 33.0],
        ],
    ],
    # R07
    [
        [
            [42.875, 54.25], [47.0, 59.125], [54.75, 74.625], [54.75, 82.875],
            [55.75, 86.125], [55.25, 87.625], [51.625, 90.5], [48.625, 91.75],
            [46.5, 86.625], [46.25, 84.375], [43.0, 81.625], [43.0, 80.375],
            [41.875, 79.0], [36.375, 75.5], [32.375, 75.25], [30.125, 73.75],
            [28.75, 71.625], [28.75, 67.625], [34.375, 65.75], [35.75, 64.375],
            [37.5, 61.125], [38.25, 57.875], [41.125, 55.0], [42.875, 54.25],
        ],
    ],
    # R08
    [
        [
            [59.125, 85.75], [62.0, 86.375], [59.75, 89.875], [59.75, 95.875],
            [59.25, 96.875], [56.75, 99.875], [54.625, 101.5], [53.625, 102.0],
            [50.875, 102.0], [47.125, 104.0], [45.75, 105.625], [43.75, 109.875],
            [43.75, 115.625], [46.5, 124.875], [48.0, 126.625], [47.875, 128.0],
            [0.125, 128.0], [0.0, 124.625], [0.125, 93.25], [1.125, 93.25],
            [3.875, 91.75], [5.875, 89.75], [7.625, 89.5], [11.375, 92.0],
            [14.875, 92.25], [18.875, 94.5], [21.625, 94.5], [23.125, 93.75],
            [31.125, 93.75], [32.875, 94.5], [35.875, 93.0], [39.875, 89.25],
            [43.375, 89.0], [46.625, 87.5], [48.0, 89.625], [48.25, 91.375],
            [49.375, 91.75], [53.625, 89.25], [55.875, 86.75], [59.125, 85.75],
        ],
    ],
    # R09
    [
        [
            [113.125, 62.75], [116.375, 63.0], [119.25, 65.125], [121.25, 71.875],
            [123.0, 74.875], [123.0, 80.125], [123.75, 82.125], [123.0, 84.375],
            [123.0, 88.625], [124.375, 90.5], [128.0, 91.625], [128.0, 127.875],
            [111.375, 128.0], [111.25, 122.625], [110.25, 120.125], [110.25, 116.875],
            [108.0, 114.875], [106.375, 114.0], [104.75, 112.375], [104.0, 110.625],
            [100.25, 107.375], [99.75, 104.625], [98.125, 102.25], [97.125, 102.25],
            [94.125, 104.0], [91.375, 104.0], [89.875, 105.0], [87.0, 103.375],
            [86.75, 101.875], [85.0, 99.375], [85.0, 88.625], [86.5, 85.375],
            [88.875, 83.5], [93.625, 86.25], [98.375, 86.25], [103.875, 83.75],
            [105.0, 82.125], [105.25, 79.625], [106.75, 76.625], [108.0, 69.625],
            [110.25, 65.875], [113.125, 62.75],
        ],
    ],
    # R10
    [
        [
            [100.375, 52.0], [104.125, 52.0], [107.375, 53.25], [110.75, 56.625],
            [112.5, 59.875], [113.25, 62.625], [110.25, 65.875], [108.0, 69.625],
            [106.75, 76.625], [105.25, 79.625], [105.25, 81.375], [104.5, 83.125],
            [102.875, 84.5], [98.375, 86.25], [93.625, 86.25], [88.875, 83.5],
            [88.25, 80.875], [86.75, 79.375], [86.0, 75.625], [86.75, 73.875],
            [86.75, 70.375], [88.25, 68.375], [88.5, 64.375], [91.5, 58.375],
            [100.375, 52.0],
        ],
    ],
    # R11
    [
        [
            [65.375, 99.0], [67.125, 99.5], [68.625, 99.0], [70.625, 99.5],
            [72.125, 99.0], [73.875, 99.5], [75.875, 99.0], [77.25, 100.125],
            [78.25, 102.375], [78.25, 105.375], [76.75, 107.125], [76.5, 109.125],
            [75.0, 112.375], [75.0, 114.125], [73.75, 116.875], [67.75, 121.625],
            [65.75, 126.375], [65.625, 128.0], [57.625, 128.0], [57.25, 126.875],
            [55.0, 124.375], [54.75, 122.375], [53.25, 120.625], [53.0, 118.375],
            [53.0, 109.875], [54.5, 106.125], [65.375, 99.0],
        ],
    ],
    # R12
    [
        [
            [71.125, 76.0], [74.375, 76.75], [75.75, 78.375], [75.5, 82.375],
            [77.0, 84.875], [77.0, 86.875], [76.5, 91.375], [75.25, 92.875],
            [75.25, 93.625], [79.75, 97.125], [80.25, 99.125], [81.5, 100.375],
            [81.75, 102.375], [83.25, 105.375], [83.5, 107.875], [85.0, 111.125],
            [85.25, 117.375], [87.25, 119.375], [88.5, 121.875], [91.125, 123.5],
            [94.0, 126.625], [93.875, 128.0], [65.75, 127.875], [65.75, 125.875],
            [67.75, 121.625], [73.75, 116.875], [75.0, 114.125], [75.0, 112.375],
            [76.5, 109.125], [76.75, 107.125], [78.25, 105.375], [78.0, 101.625],
            [76.375, 99.25], [75.25, 99.125], [73.875, 99.5], [72.125, 99.0],
            [70.625, 99.5], [68.625, 99.0], [67.125, 99.5], [65.375, 99.0],
            [54.5, 106.125], [53.0, 109.875], [53.25, 120.625], [54.75, 122.375],
            [55.25, 124.875], [57.25, 126.875], [57.5, 127.875], [48.125, 128.0],
            [47.75, 126.125], [46.5, 124.875], [43.75, 115.625], [44.0, 109.125],
            [47.125, 104.0], [50.125, 102.25], [54.625, 101.5], [58.75, 97.625],
            [59.75, 95.875], [60.0, 89.125], [63.75, 83.875], [66.875, 80.5],
            [69.5, 78.625], [70.0, 76.875], [71.125, 76.0],
        ],
    ],
    # R13
    [
        [
            [97.125, 102.25], [98.125, 102.25], [99.0, 103.125], [100.25, 106.125],
            [100.25, 107.375], [104.0, 110.625], [104.75, 112.375], [106.375, 114.0],
            [110.25, 116.875], [110.25, 120.125], [111.25, 122.625], [111.25, 127.875],
            [102.125, 128.0], [94.0, 127.875], [94.25, 124.875], [93.5, 121.375],
            [91.25, 118.875], [88.5, 113.125], [88.5, 108.125], [90.0, 106.375],
            [89.75, 105.125], [91.375, 104.0], [94.875, 103.75], [97.125, 102.25],
        ],
    ],
    # R14
    [
        [
            [37.125, 0.0], [92.375, 0.0], [93.25, 8.875], [91.75, 10.625],
            [91.5, 12.625], [90.0, 16.125], [91.5, 19.875], [91.5, 21.375],
            [88.25, 30.375], [89.25, 32.875], [89.25, 37.625], [90.0, 42.625],
            [88.375, 43.0], [84.875, 42.0], [82.75, 39.375], [82.5, 36.125],
            [81.5, 31.375], [80.0, 29.625], [79.75, 27.875], [72.125, 22.75],
            [69.625, 20.0], [67.625, 18.75], [64.625, 18.0], [61.375, 15.75],
            [58.375, 16.0], [52.625, 14.5], [51.125, 14.75], [46.5, 18.125],
            [46.25, 32.875], [45.875, 33.75], [42.375, 33.0], [42.75, 29.875],
            [39.0, 22.625], [38.75, 19.875], [37.0, 16.375], [37.0, 12.625],
            [36.25, 9.375], [37.0, 6.625], [37.125, 0.0],
        ],
    ],
    # R15
    [
        [
            [90.625, 46.0], [93.125, 47.0], [96.125, 49.5], [98.875, 50.0],
            [100.25, 52.125], [91.5, 58.375], [88.5, 64.375], [88.25, 68.375],
            [86.75, 70.375], [86.75, 73.875], [86.0, 75.875], [86.75, 79.375],
            [88.25, 80.875], [88.75, 83.625], [86.5, 85.375], [85.0, 88.625],
            [85.0, 99.375], [86.5, 101.375], [87.0, 103.375], [90.0, 105.375],
            [89.75, 106.875], [88.5, 108.125], [88.5, 113.125], [91.25, 118.875],
            [93.5, 121.375], [94.25, 124.875], [94.0, 126.375], [93.375, 126.25],
            [91.125, 123.5], [89.5, 122.625], [85.25, 117.375], [85.0, 111.125],
            [83.5, 107.875], [83.25, 105.375], [81.75, 102.375], [81.5, 100.375],
            [80.25, 99.125], [79.75, 97.125], [75.25, 93.625], [75.25, 92.875],
            [76.5, 91.375], [77.0, 86.875], [77.0, 84.875], [75.5, 82.375],
            [75.75, 78.375], [74.375, 76.75], [71.125, 76.0], [70.0, 76.875],
            [69.5, 78.625], [68.625, 79.0], [68.25, 73.625], [71.5, 70.625],
            [71.75, 64.125], [74.125, 63.0], [77.0, 60.625], [78.25, 57.875],
            [78.25, 56.375], [79.375, 55.0], [81.125, 54.25], [83.375, 54.5],
            [84.625, 55.25], [86.375, 55.0], [87.25, 53.875], [88.25, 50.125],
            [90.625, 46.0],
        ],
    ],
]

# R16 is a separate compartment fully enclosed by R13.
NESTED_R16_CONTOUR = [
    [94.375, 106.5], [98.25, 108.875], [98.75, 113.125], [101.5, 115.125],
    [102.0, 117.125], [103.5, 118.875], [104.25, 123.625], [103.375, 123.75],
    [102.5, 122.625], [101.625, 121.75], [98.875, 122.0], [95.875, 120.25],
    [94.5, 118.625], [91.75, 112.625], [92.0, 109.625], [93.0, 107.625],
    [94.375, 106.5],
]


def generate_layout(nside: int = DEFAULT_NSIDE) -> tuple[RegionGrid, Coordinates]:
    """Return base region IDs (nside, nside) and cell centers (nside, nside, 2).

    Coordinates span the fixed design space, with y increasing downwards.
    The complete square is partitioned into R01–R16. This function neither
    reads nor writes files and does not create figures or activity overlaps.
    """
    if isinstance(nside, bool) or not isinstance(nside, (int, np.integer)) or nside < 32:
        raise ValueError("nside must be an integer >= 32.")

    axis = (np.arange(nside) + 0.5) * SPATIAL_EXTENT / nside
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel()))
    masks = []
    for rings in REGION_CONTOURS:
        inside = np.zeros(len(points), dtype=bool)
        for ring in rings:
            # Even-odd filling preserves holes in surrounding compartments.
            inside ^= PolygonPath(np.asarray(ring)).contains_points(points)
        masks.append(inside.reshape(nside, nside))

    # Simplified neighboring contours may have subpixel gaps or overlaps.
    # Prefer the deepest polygon interior, then fill only uncovered pixels.
    scores = np.stack([distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] for mask in masks])
    regions = (np.argmax(scores, axis=0) + 1).astype(np.int16)
    gaps = np.max(scores, axis=0) == 0
    if gaps.any():
        _, indices = distance_transform_edt(gaps, return_indices=True)
        regions[gaps] = regions[tuple(indices)][gaps]

    inner = PolygonPath(np.asarray(NESTED_R16_CONTOUR)).contains_points(points).reshape(nside, nside)
    regions[inner & (regions == NESTED_PARENT_ID)] = NESTED_REGION_ID
    return regions, np.stack((x, y), axis=-1)


def build_supports(
    regions: RegionGrid,
    coords: Coordinates,
) -> tuple[SupportGrid, SupportGrid, SupportGrid]:
    """Return region, activity and intrusion masks without changing the inputs.

    Pass the two arrays returned by generate_layout(). The three boolean arrays
    have shapes (nside, nside, 16), (nside, nside, 11) and (nside, nside, 5).
    Their last axes follow REGION_IDS, ACTIVITY_GROUPS and OVERLAP_SPECS,
    respectively. Intrusions add source support while retaining target support.
    No files, plots, random sampling or expression generation are involved.
    """
    if regions.ndim != 2 or regions.shape[0] != regions.shape[1]:
        raise ValueError("regions must be a square two-dimensional array.")
    if coords.shape != regions.shape + (2,) or not np.isfinite(coords).all():
        raise ValueError("coords must contain finite x/y centers for each region-grid position.")
    if set(np.unique(regions)) != set(REGION_IDS):
        raise ValueError("regions must contain all base IDs R01–R16, with no other values.")

    assigned = [region_id for ids in ACTIVITY_GROUPS.values() for region_id in ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(REGION_IDS):
        raise ValueError("ACTIVITY_GROUPS must assign each base region exactly once.")

    x, y = coords[:, :, 0], coords[:, :, 1]
    spacing = SPATIAL_EXTENT / regions.shape[0]
    support = np.stack([regions == region_id for region_id in REGION_IDS], axis=-1)
    overlaps = []
    for spec in OVERLAP_SPECS:
        center_x, center_y = spec["center"]
        radius_x, radius_y = spec["radii"]
        bump = np.clip(1 - ((x - center_x) / radius_x) ** 2 - ((y - center_y) / radius_y) ** 2, 0, 1)
        distance = distance_transform_edt(regions != spec["source"]) * spacing
        extension = (regions == spec["target"]) & (bump > 0) & (distance <= spec["depth"] * bump)
        pair = f"R{spec['source']:02d} -> R{spec['target']:02d}"
        if not extension.any():
            raise ValueError(f"Empty intrusion: {pair}.")
        components, _ = label(extension | (regions == spec["source"]))
        source_components = np.unique(components[regions == spec["source"]])
        if not np.isin(components[extension], source_components).all():
            raise ValueError(f"Intrusion is disconnected from its source: {pair}.")
        support[:, :, spec["source"] - 1] |= extension
        overlaps.append(extension)

    activity_support = np.stack(
        [np.any(support[:, :, np.array(ids) - 1], axis=-1) for ids in ACTIVITY_GROUPS.values()],
        axis=-1,
    )
    return support, activity_support, np.stack(overlaps, axis=-1)


def _create_canvas() -> tuple[Figure, Axes]:
    """Create the shared layout used by both overview figures."""
    fig = plt.figure(figsize=(10.5, 8.6), facecolor="white")
    ax = fig.add_axes([0.045, 0.055, 0.65, 0.89])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def _label_regions(ax: Axes, regions: RegionGrid, coords: Coordinates, texts: dict[int, str]) -> None:
    """Place labels at the deepest point inside each base region."""
    for region_id, text in texts.items():
        distance = distance_transform_edt(np.pad(regions == region_id, 1))[1:-1, 1:-1]
        row, col = np.unravel_index(np.argmax(distance), distance.shape)
        ax.text(
            coords[row, col, 0], coords[row, col, 1], text,
            ha="center", va="center", fontsize=8, color="#243340",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.8},
        )


def plot_activity_groups(
    regions: RegionGrid,
    coords: Coordinates,
    activity_support: SupportGrid,
    overlap_masks: SupportGrid,
    out: Path,
) -> None:
    """Plot all activities with instance labels and explicitly marked overlaps."""
    with mpl.rc_context(PLOT_STYLE):
        palette = np.array([to_rgb(color) for color in ACTIVITY_COLORS])
        image = np.einsum("ijk,kc->ijc", activity_support.astype(float), palette)
        image /= activity_support.sum(axis=-1, keepdims=True)
        fig, ax = _create_canvas()
        ax.imshow(image, origin="upper", extent=(0, SPATIAL_EXTENT, SPATIAL_EXTENT, 0), interpolation="nearest")
        for region_id in REGION_IDS:
            ax.contour(
                coords[:, :, 0], coords[:, :, 1], (regions == region_id).astype(float),
                levels=[0.5], colors=["white"], linewidths=0.5,
            )
        texts = {
            region_id: f"{activity}-{instance}\nR{region_id:02d}" if len(ids) > 1 else f"{activity}\nR{region_id:02d}"
            for activity, ids in ACTIVITY_GROUPS.items()
            for instance, region_id in enumerate(ids, 1)
        }
        # Follow region order, independent of the activity grouping order.
        _label_regions(ax, regions, coords, {region_id: texts[region_id] for region_id in REGION_IDS})
        for index in range(overlap_masks.shape[-1]):
            mask = overlap_masks[:, :, index].astype(float)
            ax.contourf(
                coords[:, :, 0], coords[:, :, 1], mask, levels=[0.5, 1.5], colors="none", hatches=["////"],
            )
            ax.contour(
                coords[:, :, 0], coords[:, :, 1], mask, levels=[0.5], colors=["#34414B"],
                linewidths=0.8, linestyles="--",
            )
        handles = [
            Patch(facecolor=color, label=activity + ": " + ", ".join(f"R{region_id:02d}" for region_id in ids))
            for color, (activity, ids) in zip(ACTIVITY_COLORS, ACTIVITY_GROUPS.items())
        ]
        handles.append(Patch(facecolor="white", edgecolor="#34414B", hatch="////", label="Local overlap: two activities"))
        fig.legend(
            handles=handles, loc="center left", bbox_to_anchor=(0.71, 0.52), frameon=False,
            fontsize=10, labelspacing=0.95, handlelength=1.2,
        )
        try:
            fig.savefig(out / "activity_groups.png", dpi=300, facecolor="white", bbox_inches="tight")
        finally:
            plt.close(fig)


def save_layout(out: str | Path) -> None:
    """Export the default design arrays, JSON notes and one activity-overview PNG."""
    out = Path(out)
    regions, coords = generate_layout()
    if set(np.unique(regions)) != set(REGION_IDS) or regions.size != DEFAULT_NSIDE**2:
        raise RuntimeError("The base layout must cover the default grid and contain every region.")
    if len(ACTIVITY_COLORS) != len(ACTIVITY_GROUPS):
        raise ValueError("The activity palette must match the activity IDs.")
    region_support, activity_support, overlap_masks = build_supports(regions, coords)
    base_activity_support = np.stack([np.isin(regions, ids) for ids in ACTIVITY_GROUPS.values()], axis=-1)
    if not np.all(activity_support | ~base_activity_support):
        raise RuntimeError("Intrusions must retain all original activity support.")
    assigned = {region_id for ids in ACTIVITY_GROUPS.values() for region_id in ids}
    unassigned = sorted(set(REGION_IDS) - assigned)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "region_layout.npz", regions=regions, coordinates=coords,
        region_ids=np.array([f"R{region_id:02d}" for region_id in REGION_IDS]),
        activity_ids=np.array(list(ACTIVITY_GROUPS)), activity_support=activity_support,
        base_activity_support=base_activity_support, region_support=region_support,
        overlap_masks=overlap_masks, unassigned_region_ids=np.array(unassigned, dtype=np.int16),
    )
    summary = {
        f"R{region_id:02d}": {
            "n_positions": int((regions == region_id).sum()),
            "n_components_4_neighbor": int(label(regions == region_id)[1]),
        }
        for region_id in REGION_IDS
    }
    report = {
        "grid_side": DEFAULT_NSIDE,
        "n_regions": len(REGION_IDS),
        "coordinate_extent": [0, int(SPATIAL_EXTENT), 0, int(SPATIAL_EXTENT)],
        "n_tissue_positions": int(regions.size),
        "regions": summary,
        "geometry_design": DESIGN_NOTES,
        "activity_groups": ACTIVITY_GROUPS,
        "unassigned_region_ids": unassigned,
        "overlaps": [
            dict(spec, n_positions=int(overlap_masks[:, :, index].sum()))
            for index, spec in enumerate(OVERLAP_SPECS)
        ],
        "n_overlapping_positions": int((activity_support.sum(axis=-1) > 1).sum()),
        "qa": "One activity-overview PNG at 300 dpi; larger canvas intentional for layout discussion.",
    }
    (out / "region_layout.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_activity_groups(regions, coords, activity_support, overlap_masks, out)
    print(json.dumps({
        "output": str(out),
        "n_regions": len(REGION_IDS),
        "region_sizes": [int((regions == region_id).sum()) for region_id in REGION_IDS],
    }, indent=2))


def main() -> None:
    save_layout(Path(__file__).resolve().parents[1] / "outputs/resolution/region_layout")


if __name__ == "__main__":
    main()
