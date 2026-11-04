"""Deterministic spatial geometry for the tissue-scale simulation layout."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def ellipse_mask(coords: np.ndarray, shape: dict) -> np.ndarray:
    cx, cy, rx, ry, degrees = shape['ellipse']
    angle = np.deg2rad(degrees)
    dx, dy = coords[:, 0] - cx, coords[:, 1] - cy
    u = (np.cos(angle) * dx + np.sin(angle) * dy) / rx
    v = (-np.sin(angle) * dx + np.cos(angle) * dy) / ry
    theta = np.arctan2(v, u)
    phase = shape.get('phase', 0.0)
    ripple = shape.get('ripple', 0.06)
    boundary = 1 + ripple * np.sin(3 * theta + phase) + 0.5 * ripple * np.cos(5 * theta - phase)
    return np.hypot(u, v) < boundary


def primary_components(c: dict, coords: np.ndarray) -> dict[str, list[np.ndarray]]:
    if 'landscape_layout' not in c:
        return {}
    spec = c['landscape_layout']
    x, y = coords.T
    ids = spec['primary_activity_ids']
    lower = y < 36.6 + 1.8 * np.sin(x / 9) + 0.7 * np.cos(x / 4)
    bottom_band = ((x >= 16 + 1.2*np.sin(y/7)).astype(int)
                   + (x >= 32 + 1.5*np.sin(y/9+.5)).astype(int)
                   + (x >= 48 + 1.2*np.cos(y/8)).astype(int))
    top_band = ((x >= 64/3 + 1.5*np.sin(y/8)).astype(int)
                + (x >= 128/3 + 1.5*np.cos(y/9)).astype(int))
    labels = np.where(lower, bottom_band, 4 + top_band)
    for patch in spec['remote_patches']:
        mask = ellipse_mask(coords, patch)
        labels[mask] = ids.index(patch['activity_id'])
    # ACT04 and ACT09 are adjacent backbone regions, never a nested pair.
    offset, amplitude, period, phase = spec['act04_act09_boundary']
    split = (labels == ids.index('ACT04')) & (y < offset + amplitude*np.sin(x/period+phase))
    labels[split] = ids.index('ACT09')
    result = {}
    for j, activity_id in enumerate(ids):
        components, count = ndimage.label((labels == j).reshape(c['nside'], c['nside']))
        masks = [components.ravel() == label for label in range(1, count + 1)]
        result[activity_id] = sorted(masks, key=lambda mask: -int(mask.sum()))
    return result


def child_components(c: dict, coords: np.ndarray, primary: dict) -> dict[str, list[np.ndarray]]:
    result = {}
    x, y = coords.T
    for spec in c.get('heterogeneous_partitions', []):
        parent = np.logical_or.reduce(primary[spec['parent']])
        offset, amplitude, period, phase = spec['boundary']
        low = (x < offset + amplitude*np.sin(y/period+phase) if spec['axis'] == 'x'
               else y < offset + amplitude*np.sin(x/period+phase))
        labels, count = ndimage.label((parent & low).reshape(c['nside'], c['nside']))
        result[spec['children'][0]] = [labels.ravel() == j for j in range(1, count+1)]
    return result
