from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.dataset_registry import DatasetSpec


@dataclass(frozen=True)
class H5ADSchemaCheckResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]

    def format_report(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"H5AD schema check: {status}"]
        for key, value in self.summary.items():
            lines.append(f"  {key}: {value}")
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        for error in self.errors:
            lines.append(f"  ERROR: {error}")
        return "\n".join(lines)


def check_rss_h5ad_schema(
    h5ad_path: str | Path,
    dataset: DatasetSpec,
    sample_id: str,
    *,
    sample_rows: int = 200,
    sample_cols: int = 500,
) -> H5ADSchemaCheckResult:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("anndata is required for H5AD schema checking.") from exc

    h5ad_path = Path(h5ad_path)
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "dataset_id": dataset.dataset_id,
        "sample_id": sample_id,
        "path": str(h5ad_path),
    }

    if not h5ad_path.exists():
        errors.append(f"H5AD file does not exist: {h5ad_path}")
        return H5ADSchemaCheckResult(False, tuple(errors), tuple(warnings), summary)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    summary["shape"] = (int(adata.n_obs), int(adata.n_vars))
    summary["data_layer"] = dataset.data_layer
    summary["spatial_obsm_key"] = dataset.spatial_obsm_key
    summary["spot_id_field"] = dataset.spot_id_field or "obs_names"
    summary["gene_id_source"] = dataset.gene_id_source
    summary["expression_state"] = dataset.expression_state

    if adata.n_obs == 0:
        errors.append("AnnData has no observations/spots.")
    if adata.n_vars == 0:
        errors.append("AnnData has no variables/genes.")

    _check_expression_matrix(adata, dataset, errors, warnings, summary, sample_rows, sample_cols)
    _check_spatial_coordinates(adata, dataset, errors, summary)
    _check_spot_ids(adata, dataset, errors, warnings, summary)
    _check_gene_ids(adata, dataset, errors, warnings, summary)

    return H5ADSchemaCheckResult(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        summary=summary,
    )


def require_rss_h5ad_schema(
    h5ad_path: str | Path,
    dataset: DatasetSpec,
    sample_id: str,
) -> H5ADSchemaCheckResult:
    result = check_rss_h5ad_schema(h5ad_path, dataset, sample_id)
    if not result.ok:
        raise ValueError(result.format_report())
    return result


def _check_expression_matrix(
    adata: Any,
    dataset: DatasetSpec,
    errors: list[str],
    warnings: list[str],
    summary: dict[str, Any],
    sample_rows: int,
    sample_cols: int,
) -> None:
    if dataset.data_layer == "X":
        matrix = adata.X
        layer_available = matrix is not None
    else:
        layer_available = dataset.data_layer in adata.layers
        matrix = adata.layers[dataset.data_layer] if layer_available else None

    summary["data_layer_available"] = bool(layer_available)
    if not layer_available:
        errors.append(f"Data layer {dataset.data_layer!r} is not available.")
        return

    rows = max(1, min(sample_rows, int(adata.n_obs)))
    cols = max(1, min(sample_cols, int(adata.n_vars)))
    sample = matrix[:rows, :cols]
    arr = sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)
    finite = bool(np.isfinite(arr).all())
    nonnegative = bool((arr >= 0).all())
    nonzero = arr[arr > 0]
    integer_like = bool(nonzero.size == 0 or np.allclose(nonzero, np.rint(nonzero)))

    summary["expression_sample_shape"] = (int(rows), int(cols))
    summary["expression_dtype"] = str(getattr(arr, "dtype", "unknown"))
    summary["expression_finite"] = finite
    summary["expression_nonnegative"] = nonnegative
    summary["expression_integer_like_nonzero"] = integer_like

    if not finite:
        errors.append("Expression matrix sample contains NaN or infinite values.")
    if not nonnegative:
        errors.append("Expression matrix sample contains negative values.")
    if dataset.expression_state == "raw_counts" and not integer_like:
        warnings.append(
            "Dataset is marked raw_counts, but sampled nonzero expression values are not integer-like."
        )


def _check_spatial_coordinates(
    adata: Any,
    dataset: DatasetSpec,
    errors: list[str],
    summary: dict[str, Any],
) -> None:
    key = dataset.spatial_obsm_key
    available = key in adata.obsm
    summary["spatial_available"] = bool(available)
    if not available:
        errors.append(f"Missing adata.obsm[{key!r}], required for RSS MicroDomain construction.")
        return
    if key != "spatial":
        errors.append(
            f"Dataset uses spatial_obsm_key={key!r}, but current RSS pipeline expects 'spatial'."
        )
        return

    coords = np.asarray(adata.obsm[key])
    summary["spatial_shape"] = tuple(int(x) for x in coords.shape)
    summary["spatial_dtype"] = str(coords.dtype)
    summary["spatial_finite"] = bool(np.isfinite(coords).all())

    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        errors.append(
            f"Spatial coordinates must be a 2D array with shape (n_spots, >=2); got {coords.shape}."
        )
    if not np.isfinite(coords).all():
        errors.append("Spatial coordinates contain NaN or infinite values.")


def _check_spot_ids(
    adata: Any,
    dataset: DatasetSpec,
    errors: list[str],
    warnings: list[str],
    summary: dict[str, Any],
) -> None:
    if dataset.spot_id_field:
        available = dataset.spot_id_field in adata.obs.columns
        summary["spot_id_field_available"] = bool(available)
        if not available:
            errors.append(f"Configured spot_id_field {dataset.spot_id_field!r} is missing.")
            return
        values = adata.obs[dataset.spot_id_field].astype(str)
        unique = bool(values.is_unique)
        null_free = bool(not adata.obs[dataset.spot_id_field].isna().any())
        summary["spot_id_unique"] = unique
        summary["spot_id_null_free"] = null_free
        if not unique:
            errors.append(f"Configured spot_id_field {dataset.spot_id_field!r} is not unique.")
        if not null_free:
            errors.append(f"Configured spot_id_field {dataset.spot_id_field!r} contains null values.")
        return

    obs_unique = bool(adata.obs_names.is_unique)
    summary["obs_names_unique"] = obs_unique
    if not obs_unique:
        warnings.append("obs_names are not unique; RSS will generate synthetic spot ids.")


def _check_gene_ids(
    adata: Any,
    dataset: DatasetSpec,
    errors: list[str],
    warnings: list[str],
    summary: dict[str, Any],
) -> None:
    if dataset.gene_id_source == "var_names":
        unique = bool(adata.var_names.is_unique)
        summary["var_names_unique"] = unique
        summary["gene_id_example"] = tuple(map(str, adata.var_names[:5]))
        if not unique:
            warnings.append("var_names are not unique; RSS will make them unique before computing.")
        return

    available = dataset.gene_id_source in adata.var.columns
    summary["gene_id_source_available"] = bool(available)
    if not available:
        errors.append(f"Configured gene_id_source {dataset.gene_id_source!r} is missing.")
