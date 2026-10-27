from __future__ import annotations

import re

import numpy as np
from sklearn.neighbors import NearestNeighbors


def jaccard(a: set[int] | set[str], b: set[int] | set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def build_spatial_neighbors(spatial: np.ndarray, knn: int) -> np.ndarray:
    coords = np.asarray(spatial, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.empty((0, 0), dtype=np.int64)
    n_neighbors = max(2, min(int(knn) + 1, coords.shape[0]))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords[:, :2])
    return nn.kneighbors(coords[:, :2], return_distance=False)[:, 1:]


def moran_matrix(matrix: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.size == 0:
        return np.array([], dtype=np.float32)
    if neighbors.size == 0:
        return np.zeros(values.shape[0], dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    denominator = np.sum(centered * centered, axis=1)
    neighbor_sum = centered[:, neighbors].sum(axis=2)
    numerator = np.sum(centered * neighbor_sum, axis=1)
    n = float(values.shape[1])
    w = float(neighbors.size)
    moran = np.zeros(values.shape[0], dtype=np.float32)
    valid = denominator > 0
    moran[valid] = (n / w) * (numerator[valid] / denominator[valid])
    moran[~np.isfinite(moran)] = 0.0
    return moran


def connected_component_indices(mask: np.ndarray, neighbors: np.ndarray) -> list[np.ndarray]:
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    if active.size == 0:
        return []
    active_set = set(active.tolist())
    visited: set[int] = set()
    components: list[np.ndarray] = []
    for start in active:
        start_int = int(start)
        if start_int in visited:
            continue
        stack = [start_int]
        visited.add(start_int)
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for nb in neighbors[node]:
                nb_int = int(nb)
                if nb_int in active_set and nb_int not in visited:
                    visited.add(nb_int)
                    stack.append(nb_int)
        components.append(np.asarray(nodes, dtype=np.int64))
    return components


def largest_component_fraction(mask: np.ndarray, neighbors: np.ndarray) -> float:
    active = int(np.count_nonzero(mask))
    if active <= 0:
        return 0.0
    largest = max((component.size for component in connected_component_indices(mask, neighbors)), default=0)
    return float(largest / active)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return float(np.sum(x * y) / denom)


def positive_quantile(values: np.ndarray, quantile: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    pos = arr[arr > 0]
    return float(np.quantile(pos, float(quantile))) if pos.size else 0.0


def activation_active_mask(
    values: np.ndarray,
    neighbors: np.ndarray | None,
    *,
    min_peak_fraction: float,
    quantile: float,
    min_abs: float,
    min_component_size: int,
    fallback_to_base: bool = False,
) -> tuple[np.ndarray, float]:
    arr = np.clip(np.asarray(values, dtype=np.float32).reshape(-1), 0.0, None)
    if arr.size == 0:
        return np.zeros(0, dtype=bool), 0.0
    peak = float(np.max(arr))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros(arr.shape[0], dtype=bool), 0.0
    base_threshold = max(
        float(max(0.0, min_abs)),
        float(max(0.0, min_peak_fraction)) * peak,
        _activation_noise_floor(arr, quantile),
    )
    threshold = _activation_elbow_threshold(arr, base_threshold, peak)
    mask = _filter_active_components(
        arr >= threshold,
        neighbors,
        min_component_size=min_component_size,
    )
    if fallback_to_base and not np.any(mask):
        base_mask = _filter_active_components(
            arr >= base_threshold,
            neighbors,
            min_component_size=min_component_size,
        )
        if np.any(base_mask):
            threshold = base_threshold
            mask = base_mask
    return mask.astype(bool, copy=False), float(threshold)


def _filter_active_components(
    mask: np.ndarray,
    neighbors: np.ndarray | None,
    *,
    min_component_size: int,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if neighbors is None or not neighbors.size or int(min_component_size) <= 1:
        return mask
    keep = np.zeros(mask.shape[0], dtype=bool)
    for component in connected_component_indices(mask, neighbors):
        if component.size >= int(min_component_size):
            keep[component] = True
    return keep


def _activation_noise_floor(values: np.ndarray, quantile: float) -> float:
    # Quantile is a low-tail denoising floor, not a top-active fraction.
    q = float(quantile)
    if q <= 0.0 or q > 0.5:
        return 0.0
    pos = np.asarray(values[values > 0], dtype=np.float32)
    if pos.size == 0:
        return 0.0
    return float(np.quantile(pos, q))


def _activation_elbow_threshold(values: np.ndarray, base_threshold: float, peak: float) -> float:
    supported = np.sort(np.asarray(values[values >= base_threshold], dtype=np.float32))[::-1]
    if supported.size <= 2:
        return float(base_threshold)
    drops = supported[:-1] - supported[1:]
    if drops.size == 0:
        return float(base_threshold)
    idx = int(np.argmax(drops))
    largest_drop = float(drops[idx])
    relative_drop = largest_drop / max(float(peak), 1e-12)
    local_scale = float(np.median(drops) + 3.0 * np.std(drops))
    if relative_drop < 0.05 or largest_drop < max(local_scale, 1e-8):
        return float(base_threshold)
    return float(max(base_threshold, 0.5 * (float(supported[idx]) + float(supported[idx + 1]))))


def gene_mask(genes: np.ndarray, prefixes: tuple[str, ...], exact: tuple[str, ...]) -> np.ndarray:
    prefix_upper = tuple(str(x).upper() for x in prefixes)
    exact_upper = {str(x).upper() for x in exact}
    out = np.zeros(np.asarray(genes).shape[0], dtype=bool)
    for idx, gene in enumerate(np.asarray(genes, dtype=str)):
        gu = str(gene).upper()
        out[idx] = gu in exact_upper or (bool(prefix_upper) and gu.startswith(prefix_upper))
    return out


def low_info_gene_mask(genes: np.ndarray, prefixes: tuple[str, ...]) -> np.ndarray:
    prefix_upper = tuple(str(x).upper() for x in prefixes)
    accession_prefixes = {"AC", "AL", "AP"}.intersection(prefix_upper)
    other_prefixes = tuple(p for p in prefix_upper if p not in accession_prefixes)
    out = np.zeros(np.asarray(genes).shape[0], dtype=bool)
    for idx, gene in enumerate(np.asarray(genes, dtype=str)):
        gu = str(gene).upper()
        # Numbered accessions must not match gene symbols such as ACTA2 or ALDOB.
        accession = re.fullmatch(r"(AC|AL|AP)\d+(?:\.\d+)?", gu)
        out[idx] = (
            gu.startswith(other_prefixes)
            or (accession is not None and accession.group(1) in accession_prefixes)
            or bool(re.match(r"^[A-Z]\d{5,}(?:\.\d+)?$", gu))
        )
    return out
