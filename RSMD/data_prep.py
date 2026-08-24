from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .bundle_io import read_json
from .domain_ops import build_spot_graph_from_coords_local, build_spot_graph_from_neighbors
from .schema import RSMDPipelineConfig


def _resolve_rsp_allowlist(cfg: RSMDPipelineConfig) -> tuple[set[str], dict]:
    selected: list[str] = []
    inline = list(getattr(cfg.input, "rsp_allowlist", ()) or ())
    selected.extend([str(x).strip() for x in inline if str(x).strip()])
    file_path_raw = str(getattr(cfg.input, "rsp_allowlist_file", "") or "").strip()
    if file_path_raw:
        path = Path(file_path_raw)
        if not path.exists():
            raise FileNotFoundError(f"Configured rsp_allowlist_file does not exist: {path}")
        selected.extend([line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    allow = {str(x) for x in selected if str(x)}
    return allow, {
        "enabled": bool(allow),
        "requested_count": int(len(selected)),
        "unique_count": int(len(allow)),
        "file": file_path_raw or None,
    }


def _resolve_rsp_qc_selection(
    rsp_df: pd.DataFrame,
    activation_df: pd.DataFrame,
    rsp_qc_df: pd.DataFrame,
    cfg: RSMDPipelineConfig,
) -> tuple[set[str], dict]:
    known_rsps = {str(x) for x in rsp_df["rsp_id"].astype(str).unique().tolist()}
    if not bool(getattr(cfg.input, "use_rsp_score_filter", False)):
        return known_rsps, {
            "enabled": False,
            "selected_count": int(len(known_rsps)),
            "dropped_count": 0,
        }

    if "rsp_score" not in rsp_df.columns:
        raise ValueError("rsp.parquet must include 'rsp_score' when use_rsp_score_filter=True")

    min_score = float(np.clip(float(getattr(cfg.input, "min_rsp_score", 0.0)), 0.0, 1.0))
    score = pd.to_numeric(rsp_df["rsp_score"], errors="coerce").fillna(0.0)
    selected_rsps = set(rsp_df.loc[score >= min_score, "rsp_id"].astype(str).tolist())
    summary: dict[str, object] = {
        "enabled": True,
        "min_rsp_score": min_score,
        "score_source": "rsp.rsp_score",
    }

    if not selected_rsps and known_rsps:
        selected_rsps = known_rsps
        summary["fallback"] = "empty_after_score_filter"

    dropped_rsps = sorted(known_rsps - selected_rsps)
    summary["selected_count"] = int(len(selected_rsps))
    summary["dropped_count"] = int(len(dropped_rsps))
    summary["dropped_examples"] = dropped_rsps[:10]
    summary["activation_rows_before"] = int(activation_df.shape[0])
    return selected_rsps, summary


def _load_h5ad_spot_ids_and_coords(rss_manifest: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    h5ad_path = str(rss_manifest.get("inputs", {}).get("h5ad_path", "")).strip()
    if not h5ad_path:
        return None, None

    path = Path(h5ad_path)
    if not path.exists():
        return None, None

    try:
        import scanpy as sc
    except Exception:  # noqa: BLE001
        return None, None

    try:
        adata = sc.read_h5ad(path)
    except Exception:  # noqa: BLE001
        return None, None

    spot_field = str(rss_manifest.get("inputs", {}).get("spot_id_field", "obs_names"))
    if spot_field == "obs_names":
        spot_ids = adata.obs_names.astype(str).to_numpy()
    elif spot_field in adata.obs.columns:
        raw = adata.obs[spot_field]
        if raw.isna().any():
            return None, None
        spot_ids = raw.astype(str).to_numpy()
    else:
        return None, None

    if len(spot_ids) != len(set(spot_ids.tolist())):
        return None, None

    coords = None
    if "spatial" in adata.obsm:
        try:
            coords_arr = np.asarray(adata.obsm["spatial"], dtype=np.float64)
            if coords_arr.ndim == 2 and coords_arr.shape[1] >= 2 and coords_arr.shape[0] == spot_ids.shape[0]:
                coords = coords_arr[:, :2].copy()
        except Exception:  # noqa: BLE001
            coords = None
    return spot_ids.astype(str), coords


def _resolve_rss_out_path(rsp_out_path: Path, cfg: RSMDPipelineConfig, rsp_manifest: dict) -> Path:
    if cfg.input.rss_out_path_override:
        path = Path(cfg.input.rss_out_path_override)
        if not path.exists():
            raise FileNotFoundError(f"Configured rss_out_path_override does not exist: {path}")
        return path

    inputs = rsp_manifest.get("inputs", {})
    rss_path = inputs.get("rss_out_path", None)
    if rss_path:
        path = Path(str(rss_path))
        if path.exists():
            return path

    # Fallback for local layout where rsp_out sits next to rss_out.
    sibling = rsp_out_path.parent / "rss_out"
    if sibling.exists():
        return sibling

    raise FileNotFoundError(
        "Unable to locate rss_out path from rsp manifest. "
        "Set RSMDPipelineConfig.input.rss_out_path_override explicitly."
    )


def _read_required_rsp_inputs(
    rsp_out_path: Path,
    cfg: RSMDPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, Path]:
    manifest_path = rsp_out_path / cfg.input.rsp_manifest_relpath
    rsp_path = rsp_out_path / cfg.input.rsp_relpath
    activation_path = rsp_out_path / cfg.input.rsp_activation_relpath
    rsp_qc_path = rsp_out_path / cfg.input.rsp_qc_relpath

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing rsp manifest: {manifest_path}")
    if not rsp_path.exists():
        raise FileNotFoundError(f"Missing rsp parquet: {rsp_path}")
    if not activation_path.exists():
        raise FileNotFoundError(f"Missing rsp activation parquet: {activation_path}")

    manifest = read_json(manifest_path)
    rsp_df = pd.read_parquet(rsp_path)
    activation_df = pd.read_parquet(activation_path)
    rsp_qc_df = pd.read_parquet(rsp_qc_path) if rsp_qc_path.exists() else pd.DataFrame()
    return rsp_df, activation_df, manifest, rsp_qc_df, rsp_qc_path


def _resolve_spot_order(
    rss_out_path: Path,
    rss_manifest: dict,
    activation_df: pd.DataFrame,
    cfg: RSMDPipelineConfig,
    n_spots_neighbors: int,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    spot_h5ad = None
    coords_h5ad = None
    if cfg.input.try_load_coords_from_h5ad:
        spot_h5ad, coords_h5ad = _load_h5ad_spot_ids_and_coords(rss_manifest)
    if spot_h5ad is not None:
        if int(spot_h5ad.shape[0]) != int(n_spots_neighbors):
            raise ValueError(
                f"h5ad spot count mismatch with neighbors rows: {spot_h5ad.shape[0]} vs {n_spots_neighbors}"
            )
        return spot_h5ad, coords_h5ad, "h5ad"

    raise ValueError(
        "Cannot recover canonical spot order. "
        "Ensure RSS manifest points to a readable h5ad with resolvable spot order."
    )


def _build_activation_matrix(
    activation_df: pd.DataFrame,
    spot_ids: np.ndarray,
    rsp_ids: np.ndarray,
    activation_col: str = "activation",
) -> sp.csr_matrix:
    spot_to_idx = {str(s): i for i, s in enumerate(spot_ids)}
    rsp_to_idx = {str(p): i for i, p in enumerate(rsp_ids)}

    row = activation_df["spot_id"].map(spot_to_idx)
    col = activation_df["rsp_id"].map(rsp_to_idx)
    if row.isna().any():
        unknown = activation_df.loc[row.isna(), "spot_id"].astype(str).unique().tolist()
        raise ValueError(f"Activation contains unknown spot_id entries: {unknown[:5]}")
    if col.isna().any():
        unknown = activation_df.loc[col.isna(), "rsp_id"].astype(str).unique().tolist()
        raise ValueError(f"Activation contains unknown rsp_id entries: {unknown[:5]}")

    data = activation_df[activation_col].to_numpy(dtype=np.float32)
    mat = sp.csr_matrix(
        (data, (row.to_numpy(dtype=np.int64), col.to_numpy(dtype=np.int64))),
        shape=(spot_ids.shape[0], rsp_ids.shape[0]),
        dtype=np.float32,
    )
    mat.eliminate_zeros()
    return mat


def _resolve_rsp_score_table(
    rsp_ids: np.ndarray,
    rsp_df: pd.DataFrame,
    cfg: RSMDPipelineConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict]:
    pids = [str(pid) for pid in np.asarray(rsp_ids, dtype=object).tolist()]
    rows: list[dict] = []
    if not pids:
        empty = pd.DataFrame(
            columns=[
                "rsp_id",
                "rsp_score",
                "rsp_score_used",
                "rsp_score_weight",
            ]
        )
        return (
            empty,
            {},
            {
                "enabled": True,
                "rsp_count": 0,
                "missing_rsp_score_count": 0,
            },
        )

    if "rsp_score" not in rsp_df.columns:
        raise ValueError("rsp.parquet must include 'rsp_score' for RSMD")

    score_df = rsp_df.loc[:, ["rsp_id", "rsp_score"]].copy()
    score_df["rsp_id"] = score_df["rsp_id"].astype(str)
    score_df["rsp_score"] = pd.to_numeric(score_df["rsp_score"], errors="coerce")
    score_df = score_df.dropna(subset=["rsp_score"]).groupby("rsp_id", as_index=False)["rsp_score"].max()
    score_map = {
        str(pid): float(np.clip(float(val), 0.0, 1.0))
        for pid, val in zip(score_df["rsp_id"].tolist(), score_df["rsp_score"].tolist())
    }
    missing_pids = sorted([pid for pid in pids if pid not in score_map])

    rsp_score_info: dict[str, dict[str, float]] = {}
    for pid in pids:
        score_val = float(score_map.get(pid, 1.0))
        rsp_score_info[pid] = {
            "rsp_score": score_val,
            "rsp_score_used": score_val,
            "rsp_score_weight": score_val,
        }
        rows.append({"rsp_id": pid, **rsp_score_info[pid]})

    out_df = pd.DataFrame(rows).sort_values("rsp_id").reset_index(drop=True)
    scores = out_df["rsp_score"].to_numpy(dtype=np.float64)
    summary = {
        "enabled": True,
        "score_col": "rsp_score",
        "rsp_count": int(out_df.shape[0]),
        "missing_rsp_score_count": int(len(missing_pids)),
        "score_quantiles": {
            "p10": float(np.quantile(scores, 0.10)),
            "p50": float(np.quantile(scores, 0.50)),
            "p90": float(np.quantile(scores, 0.90)),
        },
    }
    return out_df, rsp_score_info, summary


def load_rsmd_inputs(
    rsp_out_path: Path,
    cfg: RSMDPipelineConfig,
) -> dict:
    rsp_df, activation_df, rsp_manifest, rsp_qc_df, rsp_qc_path = _read_required_rsp_inputs(
        rsp_out_path,
        cfg,
    )
    rsp_allowlist, allowlist_summary = _resolve_rsp_allowlist(cfg)

    required_activation_cols = {"rsp_id", "spot_id", "activation"}
    missing = required_activation_cols - set(activation_df.columns)
    if missing:
        raise ValueError(
            f"rsp_activation.parquet missing required columns: {sorted(missing)}; "
            f"found={list(activation_df.columns)}"
        )

    activation_value_col = "activation"
    base_cols = ["rsp_id", "spot_id", "activation"]
    extra_cols = [
        c
        for c in [
            "activation_raw",
            "activation_denoised",
            "rank_in_spot",
        ]
        if c in activation_df.columns
    ]
    ordered_cols = []
    for col in base_cols + extra_cols:
        if col in activation_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    activation_df = activation_df.loc[:, ordered_cols].copy()
    activation_df["rsp_id"] = activation_df["rsp_id"].astype(str)
    activation_df["spot_id"] = activation_df["spot_id"].astype(str)
    if activation_value_col != "activation":
        activation_df["activation"] = activation_df[activation_value_col].to_numpy(dtype=np.float32)
    else:
        activation_df["activation"] = activation_df["activation"].to_numpy(dtype=np.float32)
    if "activation_raw" in activation_df.columns:
        activation_df["activation_raw"] = activation_df["activation_raw"].to_numpy(dtype=np.float32)
    if "activation_denoised" in activation_df.columns:
        activation_df["activation_denoised"] = activation_df["activation_denoised"].to_numpy(dtype=np.float32)
    activation_df = activation_df[activation_df["activation"] > 0].reset_index(drop=True)

    if "rsp_id" not in rsp_df.columns:
        raise ValueError("rsp.parquet must include 'rsp_id' column")
    rsp_df = rsp_df.copy()
    rsp_df["rsp_id"] = rsp_df["rsp_id"].astype(str)
    if "gene" in rsp_df.columns:
        rsp_df["gene"] = rsp_df["gene"].astype(str)
    qc_selected_rsps, qc_selection_summary = _resolve_rsp_qc_selection(
        rsp_df=rsp_df,
        activation_df=activation_df,
        rsp_qc_df=rsp_qc_df,
        cfg=cfg,
    )
    rsp_df = rsp_df.loc[rsp_df["rsp_id"].isin(qc_selected_rsps)].reset_index(drop=True)
    activation_df = activation_df.loc[
        activation_df["rsp_id"].astype(str).isin(qc_selected_rsps)
    ].reset_index(drop=True)
    if not rsp_qc_df.empty and "rsp_id" in rsp_qc_df.columns:
        rsp_qc_df = rsp_qc_df.loc[
            rsp_qc_df["rsp_id"].astype(str).isin(qc_selected_rsps)
        ].reset_index(drop=True)
    if rsp_allowlist:
        known_rsps = set(rsp_df["rsp_id"].unique().tolist())
        missing_rsps = sorted(rsp_allowlist - known_rsps)
        if missing_rsps:
            raise ValueError(
                f"Requested rsp_allowlist contains unknown rsp_id: {missing_rsps[:10]}"
            )
        rsp_df = rsp_df.loc[rsp_df["rsp_id"].isin(rsp_allowlist)].reset_index(drop=True)
        activation_df = activation_df.loc[
            activation_df["rsp_id"].astype(str).isin(rsp_allowlist)
        ].reset_index(drop=True)
        if not rsp_qc_df.empty and "rsp_id" in rsp_qc_df.columns:
            rsp_qc_df = rsp_qc_df.loc[
                rsp_qc_df["rsp_id"].astype(str).isin(rsp_allowlist)
            ].reset_index(drop=True)

    rsp_ids = np.sort(rsp_df["rsp_id"].unique().astype(str))
    if rsp_ids.size == 0:
        # Keep matrix shape valid for no-rsp samples.
        rsp_ids = np.asarray([], dtype=object)

    # Activation rsp_id must be subset of rsp list.
    extra_rsps = set(activation_df["rsp_id"].unique().tolist()) - set(rsp_ids.tolist())
    if extra_rsps:
        raise ValueError(f"Activation contains rsp_id not in rsp.parquet: {sorted(extra_rsps)[:5]}")
    rsp_score_df, rsp_score_info, rsp_score_summary = _resolve_rsp_score_table(
        rsp_ids=rsp_ids,
        rsp_df=rsp_df,
        cfg=cfg,
    )
    score_map = {str(r["rsp_id"]): float(r["rsp_score"]) for r in rsp_score_df.to_dict(orient="records")}
    weight_map = {
        str(r["rsp_id"]): float(r["rsp_score_weight"]) for r in rsp_score_df.to_dict(orient="records")
    }
    activation_df["activation_raw"] = activation_df["activation"].to_numpy(dtype=np.float32)
    activation_df["rsp_score"] = activation_df["rsp_id"].map(score_map).fillna(1.0).to_numpy(dtype=np.float32)
    activation_df["rsp_score_weight"] = activation_df["rsp_id"].map(weight_map).fillna(1.0).to_numpy(
        dtype=np.float32
    )
    activation_df["activation_effective"] = activation_df["activation_raw"].to_numpy(dtype=np.float32)
    activation_df["activation"] = activation_df["activation_effective"].to_numpy(dtype=np.float32)

    rss_out_path = _resolve_rss_out_path(rsp_out_path, cfg, rsp_manifest)
    rss_manifest_path = rss_out_path / cfg.input.rss_manifest_relpath
    if not rss_manifest_path.exists():
        raise FileNotFoundError(f"Missing rss manifest: {rss_manifest_path}")
    rss_manifest = read_json(rss_manifest_path)

    microdomain_idx_path = rss_out_path / cfg.input.microdomain_idx_relpath
    microdomain_meta_path = rss_out_path / cfg.input.microdomain_meta_relpath
    if not microdomain_idx_path.exists():
        raise FileNotFoundError(f"Missing microdomain idx: {microdomain_idx_path}")
    if not microdomain_meta_path.exists():
        raise FileNotFoundError(f"Missing microdomain meta: {microdomain_meta_path}")

    microdomain_idx = np.load(microdomain_idx_path, allow_pickle=False)
    if microdomain_idx.ndim != 2:
        raise ValueError(f"microdomain.idx.npy must be 2D, got shape={microdomain_idx.shape}")
    microdomain_meta = read_json(microdomain_meta_path)

    spot_ids, coords, spot_order_source = _resolve_spot_order(
        rss_out_path=rss_out_path,
        rss_manifest=rss_manifest,
        activation_df=activation_df,
        cfg=cfg,
        n_spots_neighbors=int(microdomain_idx.shape[0]),
    )

    activation_mat_raw = _build_activation_matrix(
        activation_df,
        spot_ids=spot_ids,
        rsp_ids=rsp_ids,
        activation_col="activation_raw",
    )
    activation_mat = _build_activation_matrix(
        activation_df,
        spot_ids=spot_ids,
        rsp_ids=rsp_ids,
        activation_col="activation_effective",
    )
    dense_activation_raw = np.asarray(activation_mat_raw.toarray(), dtype=np.float32)
    dense_activation = np.asarray(activation_mat.toarray(), dtype=np.float32)

    rss_adjacency, rss_edge_array = build_spot_graph_from_neighbors(neighbor_idx=microdomain_idx)

    flow_graph_mode = str(cfg.potential.flow_graph_mode)
    if flow_graph_mode == "rss":
        adjacency, edge_array = rss_adjacency, rss_edge_array
        graph_source = "rss_microdomain_idx"
    elif flow_graph_mode == "spatial":
        if coords is None:
            raise ValueError(
                "flow_graph_mode='spatial' requires coords, but no spatial coordinates were resolved "
                "(check h5ad path / spot order mapping)."
            )
        adjacency, edge_array = build_spot_graph_from_coords_local(
            coords=coords,
            k=int(cfg.potential.spatial_graph_k),
            max_distance_factor=float(cfg.potential.spatial_graph_max_distance_factor),
        )
        graph_source = "spatial_local_radius_from_coords"
    else:
        raise ValueError(f"Unsupported flow_graph_mode: {flow_graph_mode}")

    return {
        "rsp_df": rsp_df,
        "activation_df": activation_df,
        "rsp_manifest": rsp_manifest,
        "rss_manifest": rss_manifest,
        "microdomain_meta": microdomain_meta,
        "rss_out_path": rss_out_path,
        "microdomain_idx_path": microdomain_idx_path,
        "microdomain_meta_path": microdomain_meta_path,
        "spot_ids": spot_ids,
        "rsp_ids": rsp_ids,
        "coords": coords,
        "spot_order_source": spot_order_source,
        "activation_matrix": activation_mat,
        "activation_matrix_raw": activation_mat_raw,
        "dense_activation": dense_activation,
        "dense_activation_raw": dense_activation_raw,
        "rsp_score_table": rsp_score_df,
        "rsp_score_info": rsp_score_info,
        "rsp_score_summary": rsp_score_summary,
        "rsp_allowlist_summary": allowlist_summary,
        "rsp_qc_selection_summary": qc_selection_summary,
        "rsp_qc_path": rsp_qc_path,
        "adjacency": adjacency,
        "spot_edges": edge_array,
        "graph_source": graph_source,
        "flow_graph_mode": flow_graph_mode,
    }


def load_rsmd_visualization_inputs(
    rsp_out_path: Path,
    cfg: RSMDPipelineConfig,
) -> dict:
    cfg_vis = deepcopy(cfg)
    cfg_vis.input.use_rsp_score_filter = False
    return load_rsmd_inputs(rsp_out_path=rsp_out_path, cfg=cfg_vis)
