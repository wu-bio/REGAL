from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from utils.dataset_registry import get_dataset


@dataclass(frozen=True)
class BundlePaths:
    dataset_id: str
    sample_id: str
    sample_dir: Path
    rss_bundle: Path
    rsp_bundle: Path
    rsmd_bundle: Path


def resolve_sample_bundle_paths(dataset_id: str, sample_id: str) -> BundlePaths:
    dataset = get_dataset(dataset_id)
    sample_dir = dataset.sample_dir(sample_id)
    return BundlePaths(
        dataset_id=dataset.dataset_id,
        sample_id=str(sample_id),
        sample_dir=sample_dir,
        rss_bundle=sample_dir / "rss_bundle",
        rsp_bundle=sample_dir / "rsp_bundle",
        rsmd_bundle=sample_dir / "rsmd_bundle",
    )


def ensure_existing_dir(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_rss_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = ensure_existing_dir(bundle_dir, "rss_bundle")
    return {
        "manifest": _read_json(bundle_dir / "manifest.json"),
        "rss_sparse": _read_optional_parquet(bundle_dir / "rss" / "rss_sparse.parquet"),
    }


def load_rsp_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = ensure_existing_dir(bundle_dir, "rsp_bundle")
    return {
        "manifest": _read_json(bundle_dir / "manifest.json"),
        "rsp": _read_parquet(bundle_dir / "rsp.parquet"),
        "activation": _read_parquet(bundle_dir / "rsp_activation.parquet"),
        "gene_loadings": _read_parquet(bundle_dir / "rsp_gene_loadings.parquet"),
        "qc": _read_parquet(bundle_dir / "rsp_qc.parquet"),
        "gene_filter_metrics": _read_optional_parquet(bundle_dir / "gene_filter_metrics.parquet"),
    }


def load_rsmd_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = ensure_existing_dir(bundle_dir, "rsmd_bundle")
    return {
        "manifest": _read_json(bundle_dir / "manifest.json"),
        "metadomains": _read_optional_parquet(bundle_dir / "metadomains.parquet"),
        "domains": _read_optional_parquet(bundle_dir / "domains.parquet"),
        "membership": _read_optional_parquet(bundle_dir / "domain_spot_membership.parquet"),
        "graph": _read_optional_parquet(bundle_dir / "domain_graph.parquet"),
        "rsp_map": _read_optional_parquet(bundle_dir / "domain_rsp_map.parquet"),
    }


def summarize_rsp_by_label(
    activation: pd.DataFrame,
    labels: pd.Series,
    *,
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    """Summarize RSP activation against a categorical annotation."""

    if "spot_id" not in activation.columns:
        raise ValueError("activation must contain a spot_id column.")
    if "rsp_id" in activation.columns and "activation" in activation.columns:
        matrix = activation.pivot_table(index="spot_id", columns="rsp_id", values="activation", aggfunc="first")
    else:
        matrix = activation.set_index("spot_id")

    matrix.index = matrix.index.astype(str)
    labels = labels.copy()
    labels.index = labels.index.astype(str)
    common = matrix.index.intersection(labels.index)
    if common.empty:
        raise ValueError("No common spots between activation and labels.")

    matrix = matrix.loc[common].astype(float)
    labels = labels.loc[common].astype("object").astype(str)
    valid = ~labels.str.lower().isin({"", "nan", "none"})
    matrix = matrix.loc[valid]
    labels = labels.loc[valid]
    if matrix.empty:
        raise ValueError("No labeled spots available after filtering empty labels.")

    rows: list[dict] = []
    for rsp_id in matrix.columns.astype(str):
        values = matrix[rsp_id].astype(float)
        top_mask = _top_mask(values, top_fraction)
        global_mean = float(values.mean())
        for label, label_idx in labels.groupby(labels).groups.items():
            in_label = values.index.isin(label_idx)
            in_values = values.loc[in_label].to_numpy(dtype=float)
            out_values = values.loc[~in_label].to_numpy(dtype=float)
            rows.append(
                {
                    "rsp_id": rsp_id,
                    "label": str(label),
                    "n_label_spots": int(in_label.sum()),
                    "n_other_spots": int((~in_label).sum()),
                    "mean_activation": float(np.nanmean(in_values)) if in_values.size else np.nan,
                    "other_mean_activation": float(np.nanmean(out_values)) if out_values.size else np.nan,
                    "global_mean_activation": global_mean,
                    "log2_mean_enrichment": _log2_ratio(np.nanmean(in_values), np.nanmean(out_values)),
                    "top_spot_fraction": float(np.mean(top_mask[in_label])) if int(in_label.sum()) else np.nan,
                    "other_top_spot_fraction": float(np.mean(top_mask[~in_label])) if int((~in_label).sum()) else np.nan,
                    "top_spot_enrichment": _ratio(np.mean(top_mask[in_label]), np.mean(top_mask[~in_label])),
                    "mannwhitney_pvalue": _mannwhitney_pvalue(in_values, out_values),
                }
            )
    return pd.DataFrame(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path)


def _read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _top_mask(values: pd.Series, top_fraction: float) -> np.ndarray:
    frac = float(top_fraction)
    if not 0 < frac < 1:
        raise ValueError("top_fraction must be between 0 and 1.")
    n_top = max(1, int(np.ceil(values.shape[0] * frac)))
    top_ids = values.sort_values(ascending=False, kind="mergesort").index[:n_top]
    return values.index.isin(top_ids)


def _ratio(numerator: float, denominator: float, eps: float = 1e-9) -> float:
    return float((float(numerator) + eps) / (float(denominator) + eps))


def _log2_ratio(numerator: float, denominator: float, eps: float = 1e-9) -> float:
    return float(np.log2(_ratio(numerator, denominator, eps=eps)))


def _mannwhitney_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    try:
        return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan
