from __future__ import annotations

import logging
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .ops import discover_rsp_from_rss
from .schema import (
    RSPDiscoveryConfig,
    RSPInputConfig,
    RSPPipelineConfig,
    RSPScoringConfig,
    RSPStabilityConfig,
)

logger = logging.getLogger(__name__)

RSP_OUTPUT_DIR = "rsp_out"

__all__ = [
    "RSPPipelineConfig",
    "RSPInputConfig",
    "RSPDiscoveryConfig",
    "RSPStabilityConfig",
    "RSPScoringConfig",
    "RSP_OUTPUT_DIR",
    "run_rsp_pipeline",
]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Torch seeding is opt-in to keep this pipeline lightweight and avoid
    # runtime issues on environments without stable torch/OpenMP support.
    if os.environ.get("REGIONAL_SPECIFICITY_PROGRAM_SEED_TORCH", "0") == "1":
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:  # noqa: BLE001
            pass


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to write parquet file: {path}. Install a parquet engine such as pyarrow."
        ) from exc


def ensure_bundle_dirs(bundle_root: Path) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)


def promote_bundle(tmp_bundle: Path, final_bundle: Path) -> None:
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    backup = final_bundle.parent / f"{final_bundle.name}.__bak__{int(time.time())}"

    if final_bundle.exists():
        if backup.exists():
            shutil.rmtree(backup)
        final_bundle.rename(backup)

    tmp_bundle.rename(final_bundle)

    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_code_version(repo_root: Path, override: str | None) -> str:
    if override:
        return override

    cmd = ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            commit = proc.stdout.strip()
            if commit:
                return commit
    except Exception:  # noqa: BLE001
        pass

    fallback = hashlib.sha1(f"{time.time()}-{os.getpid()}".encode("utf-8")).hexdigest()[:12]
    return f"local-{fallback}"


def load_spot_ids_and_spatial_coords(
    rss_manifest: dict,
    input_cfg: RSPInputConfig,
) -> tuple[np.ndarray, np.ndarray]:
    h5ad_path = _resolve_h5ad_path(rss_manifest=rss_manifest, input_cfg=input_cfg)
    spot_field = str(rss_manifest.get("inputs", {}).get("spot_id_field", "obs_names")).strip() or "obs_names"
    spatial_key = str(input_cfg.spatial_obsm_key).strip() or "spatial"

    try:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path, backed="r")
        if spot_field == "obs_names":
            spot_ids = adata.obs_names.astype(str).to_numpy()
        elif spot_field in adata.obs.columns:
            raw = adata.obs[spot_field]
            if raw.isna().any():
                raise ValueError(f"Spot id field contains NaN values: {spot_field}")
            spot_ids = raw.astype(str).to_numpy()
        else:
            spot_ids = adata.obs_names.astype(str).to_numpy()
        if spatial_key not in adata.obsm:
            raise ValueError(f"Raw h5ad is missing adata.obsm[{spatial_key!r}].")
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    except Exception:
        if spot_field != "obs_names":
            raise
        import h5py

        with h5py.File(h5ad_path, "r") as f:
            if "obs" not in f:
                raise ValueError("Raw h5ad missing obs group.")
            obs = f["obs"]
            key = "_index" if "_index" in obs else "index" if "index" in obs else None
            if key is None:
                raise ValueError("Raw h5ad missing obs index for spot alignment.")
            raw = obs[key][:]
            if raw.dtype.kind in {"S", "O"}:
                vals = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in raw.tolist()]
            else:
                vals = [str(x) for x in raw.tolist()]
            spot_ids = np.asarray(vals, dtype=str)
            if "obsm" not in f or spatial_key not in f["obsm"]:
                raise ValueError(f"Raw h5ad missing obsm/{spatial_key}.")
            coords = np.asarray(f["obsm"][spatial_key][:], dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != spot_ids.shape[0]:
        raise ValueError("Invalid raw spatial coordinates; expected shape (n_spots, >=2) aligned with spot ids.")
    if len(spot_ids) != len(set(spot_ids.tolist())):
        raise ValueError("Spot IDs in raw h5ad are not unique.")
    return spot_ids.astype(str), coords[:, :2].copy()


def _resolve_h5ad_path(rss_manifest: dict, input_cfg: RSPInputConfig) -> Path:
    if input_cfg.h5ad_path_override:
        raw = str(input_cfg.h5ad_path_override).strip()
    else:
        raw = str(rss_manifest.get("inputs", {}).get("h5ad_path", "")).strip()
    if not raw:
        raise ValueError("Missing h5ad path. Set RSPInputConfig.h5ad_path_override or provide it in rss manifest.")
    path = Path(raw)
    if (not path.exists()) or (not path.is_file()):
        raise FileNotFoundError(f"h5ad file not found: {path}")
    return path


def run_rsp_pipeline(
    rss_out_path: str | Path,
    out_root: str | Path,
    sample_id: str,
    config: RSPPipelineConfig | None = None,
) -> Path:
    cfg = config or RSPPipelineConfig()
    set_random_seed(cfg.random_seed)

    rss_out_path = Path(rss_out_path)
    out_root = Path(out_root)
    sample_dir = out_root / sample_id
    final_bundle = sample_dir / RSP_OUTPUT_DIR
    tmp_bundle = sample_dir / f"{RSP_OUTPUT_DIR}.__tmp__{int(time.time())}_{os.getpid()}"

    if tmp_bundle.exists():
        shutil.rmtree(tmp_bundle)
    ensure_bundle_dirs(tmp_bundle)

    try:
        manifest_path = rss_out_path / cfg.input.rss_manifest_relpath
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing RSS manifest: {manifest_path}")
        rss_manifest = read_json(manifest_path)

        rss_sparse_path = rss_out_path / cfg.input.rss_sparse_relpath
        if not rss_sparse_path.exists():
            raise FileNotFoundError(f"Missing required RSS sparse file: {rss_sparse_path}")

        spot_ids, spatial = load_spot_ids_and_spatial_coords(
            rss_manifest=rss_manifest,
            input_cfg=cfg.input,
        )
        payload = discover_rsp_from_rss(
            rss_sparse_path=rss_sparse_path,
            spot_ids=spot_ids,
            spatial=spatial,
            discovery_cfg=cfg.discovery,
            stability_cfg=cfg.stability,
            scoring_cfg=cfg.scoring,
            random_seed=cfg.random_seed,
        )

        write_parquet(payload["rsp"], tmp_bundle / "rsp.parquet")
        write_parquet(payload["rsp_activation"], tmp_bundle / "rsp_activation.parquet")
        write_parquet(payload["rsp_gene_loadings"], tmp_bundle / "rsp_gene_loadings.parquet")
        write_parquet(payload["rsp_core_gene_loadings"], tmp_bundle / "rsp_core_gene_loadings.parquet")
        write_parquet(payload["rsp_family_members"], tmp_bundle / "rsp_family_members.parquet")
        write_parquet(payload["rsp_family_member_activation"], tmp_bundle / "rsp_family_member_activation.parquet")
        write_parquet(payload["rsp_qc"], tmp_bundle / "rsp_qc.parquet")
        write_parquet(payload["candidate_diagnostics"], tmp_bundle / "candidate_diagnostics.parquet")
        write_parquet(payload["gene_filter_metrics"], tmp_bundle / "gene_filter_metrics.parquet")
        write_json(tmp_bundle / "rss_matrix_metadata.json", payload["rss_matrix_metadata"])

        discovery_backend = str(payload["summary"].get("discovery_backend", cfg.discovery.decomposition_backend))

        rsp_meta = {
            "sample_id": str(sample_id),
            "n_spots": int(len(spot_ids)),
            "rsp_count": int(payload["rsp"].shape[0]),
            "discovery_backend": discovery_backend,
            "rsp_ids": payload["rsp"]["rsp_id"].astype(str).tolist() if not payload["rsp"].empty else [],
            "score_semantics": "Continuous 0-1 score; no confidence categories are emitted.",
            "core_gene_count_summary": {
                "min": payload["summary"].get("core_gene_count_min", 0),
                "median": payload["summary"].get("core_gene_count_median", 0.0),
                "max": payload["summary"].get("core_gene_count_max", 0),
            },
            "summary": payload["summary"],
        }
        write_json(tmp_bundle / "rsp_meta.json", rsp_meta)

        qc_report = {
            "sample_id": str(sample_id),
            "inputs_summary": {
                "rss_out_path": str(rss_out_path.resolve()),
                "rss_sparse_path": str(rss_sparse_path.resolve()),
                "rss_sparse_hash": hash_file(rss_sparse_path),
                "n_spots": int(len(spot_ids)),
                "rss_schema_version": rss_manifest.get("schema_version", None),
            },
            "rsp_summary": payload["summary"],
            "acceptance": {
                "rsp_entity": "Regional Specificity Program",
                "score_formula": (
                    "candidate quality from spatial structure, recurrence, gene quality, "
                    "reliability-weighted RSS reconstruction, activation quality, redundancy, "
                    "and confounder penalties"
                ),
            },
        }
        write_json(tmp_bundle / "qc_report.json", qc_report)

        manifest = {
            "schema_version": cfg.schema_version,
            "created_at": iso_now(),
            "sample_id": str(sample_id),
            "input_dir": str(rss_out_path.name),
            "output_dir": RSP_OUTPUT_DIR,
            "code_version": get_code_version(
                repo_root=Path(__file__).resolve().parents[1],
                override=cfg.code_version_override,
            ),
            "random_seed": int(cfg.random_seed),
            "discovery_backend": discovery_backend,
            "inputs": {
                "rss_out_path": str(rss_out_path.resolve()),
                "rss_manifest_path": str(manifest_path.resolve()),
                "rss_sparse_path": str(rss_sparse_path.resolve()),
                "rss_sparse_hash": hash_file(rss_sparse_path),
                "n_spots": int(len(spot_ids)),
                "rss_schema_version": rss_manifest.get("schema_version", None),
            },
            "params": asdict(cfg),
            "outputs": {
                "rsp": "rsp.parquet",
                "rsp_activation": "rsp_activation.parquet",
                "rsp_gene_loadings": "rsp_gene_loadings.parquet",
                "rsp_core_gene_loadings": "rsp_core_gene_loadings.parquet",
                "rsp_family_members": "rsp_family_members.parquet",
                "rsp_family_member_activation": "rsp_family_member_activation.parquet",
                "rsp_qc": "rsp_qc.parquet",
                "candidate_diagnostics": "candidate_diagnostics.parquet",
                "gene_filter_metrics": "gene_filter_metrics.parquet",
                "rss_matrix_metadata": "rss_matrix_metadata.json",
                "rsp_meta": "rsp_meta.json",
                "qc_report": "qc_report.json",
            },
            "output_semantics": {
                "rsp": {
                    "entity": "Regional Specificity Program",
                    "score_column": "rsp_score",
                    "no_confidence_level": True,
                },
                "rsp_gene_loadings": {
                    "full_loading_table": True,
                    "core_support_footprint": "retained_rsp_activation",
                    "core_gene_columns": ["core_eligible", "is_core_gene", "core_rank", "core_score", "core_drop_reason"],
                    "core_support_columns": [
                        "loading_strength",
                        "local_rss_strength",
                        "local_rss_coverage",
                        "local_rss_lift",
                        "local_rss_lift_scaled",
                        "program_specificity",
                    ],
                },
                "rsp_core_gene_loadings": {
                    "interpretation_table": True,
                    "subset_of": "rsp_gene_loadings.parquet",
                },
                "rsp_family_members": {
                    "non_dropped_family_candidates": True,
                    "includes_retained_and_redundant_candidates": True,
                },
                "rsp_family_member_activation": {
                    "spot_level_activation_for_non_dropped_family_candidates": True,
                    "not_used_by_downstream_rsmd": True,
                },
                "candidate_diagnostics": {
                    "stability_ready_candidates": True,
                    "pre_stability_and_post_stability_qc": True,
                },
            },
            "timestamps": {"finished_at": iso_now()},
        }
        write_json(tmp_bundle / "manifest.json", manifest)

        promote_bundle(tmp_bundle, final_bundle)
        logger.info("RSP bundle written: %s", final_bundle)
        return final_bundle
    except Exception:
        shutil.rmtree(tmp_bundle, ignore_errors=True)
        raise
