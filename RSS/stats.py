from __future__ import annotations

import numpy as np


def quantiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}

    return {
        "p10": float(np.nanpercentile(values, 10)),
        "p50": float(np.nanpercentile(values, 50)),
        "p90": float(np.nanpercentile(values, 90)),
    }


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def neighbors_to_sets(neighbor_idx: np.ndarray) -> list[set[int]]:
    sets = []
    for i in range(neighbor_idx.shape[0]):
        valid = neighbor_idx[i][neighbor_idx[i] >= 0]
        sets.append(set(int(x) for x in valid.tolist()))
    return sets
