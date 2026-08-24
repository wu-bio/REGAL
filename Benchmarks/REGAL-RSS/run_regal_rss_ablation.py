#!/usr/bin/env python
"""Run RSP on STAMP-selected HVG expression without using any RSS artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from RSP.pipeline import run_rsp_pipeline
from RSP.schema import RSPPipelineConfig


DEFAULT_INPUT = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
)
DEFAULT_SAMPLE_DIR = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1"
)
DEFAULT_STAMP_H5AD = DEFAULT_SAMPLE_DIR / "STAMP" / "K20" / "stamp.h5ad"
DEFAULT_OUTPUT_ROOT = DEFAULT_SAMPLE_DIR / "REGAL-RSS"
EXPRESSION_VALUE_COLUMN = "expression_value"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--stamp-h5ad", type=Path, default=DEFAULT_STAMP_H5AD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=("prepare", "smoke", "full"), default="smoke")
    parser.add_argument("--target-sum", type=float, default=1e4)
    args = parser.parse_args()
    if args.target_sum <= 0:
        parser.error("--target-sum must be positive.")
    return args


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def validate_paths(args: argparse.Namespace) -> None:
    native_rsp = (args.sample_dir / "rsp_out").resolve()
    output_root = args.output_root.resolve()
    for path in (args.input, args.stamp_h5ad):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not native_rsp.is_dir():
        raise FileNotFoundError(native_rsp)
    if output_root == native_rsp or native_rsp in output_root.parents:
        raise ValueError("Ablation output must not be the native rsp_out directory.")
    if args.sample_dir.resolve() not in output_root.parents:
        raise ValueError("Ablation output must remain below the native sample directory.")


def load_stamp_counts(path: Path) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
    adata = ad.read_h5ad(path, backed="r")
    matrix = adata.X.to_memory() if hasattr(adata.X, "to_memory") else adata.X[:]
    counts = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix))
    obs_names = pd.Index(adata.obs_names.astype(str))
    var_names = pd.Index(adata.var_names.astype(str))
    if adata.file is not None:
        adata.file.close()

    if counts.shape != (len(obs_names), len(var_names)):
        raise ValueError("STAMP h5ad axes do not match its expression matrix.")
    if counts.data.size and (
        not np.isfinite(counts.data).all()
        or np.min(counts.data) < 0
        or not np.allclose(counts.data, np.rint(counts.data), atol=1e-8)
    ):
        raise ValueError("STAMP h5ad must contain non-negative integer-like raw counts.")
    if obs_names.has_duplicates or var_names.has_duplicates:
        raise ValueError("STAMP h5ad contains duplicate spot or gene identifiers.")
    return counts.astype(np.float32), obs_names, var_names


def validate_alignment(input_path: Path, stamp_spots: pd.Index) -> None:
    adata = ad.read_h5ad(input_path, backed="r")
    input_spots = pd.Index(adata.obs_names.astype(str))
    if adata.file is not None:
        adata.file.close()
    if not input_spots.equals(stamp_spots):
        raise ValueError("STAMP and native BREAST h5ad spot axes are not identical and ordered.")


def normalize_log1p(counts: sp.csr_matrix, target_sum: float) -> sp.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).reshape(-1).astype(np.float64)
    if np.any(totals <= 0):
        raise ValueError("STAMP-selected count matrix contains an empty spot.")
    scale = np.asarray(float(target_sum) / totals, dtype=np.float32)
    matrix = sp.diags(scale) @ counts
    matrix = matrix.tocsr().astype(np.float32)
    matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
    matrix.eliminate_zeros()
    return matrix


def expression_long_table(
    expression: sp.csr_matrix,
    spots: pd.Index,
    genes: pd.Index,
) -> pd.DataFrame:
    coo = expression.tocoo(copy=False)
    table = pd.DataFrame(
        {
            "spot_id": spots.to_numpy(dtype=str)[coo.row],
            "gene": genes.to_numpy(dtype=str)[coo.col],
            EXPRESSION_VALUE_COLUMN: coo.data.astype(np.float32, copy=False),
        }
    )
    if table.empty or not np.isfinite(table[EXPRESSION_VALUE_COLUMN]).all():
        raise ValueError("Prepared no-RSS expression table is empty or non-finite.")
    if table.duplicated(["spot_id", "gene"]).any():
        raise ValueError("Prepared no-RSS expression table contains duplicate pairs.")
    return table


def prepare_no_rss_input(args: argparse.Namespace) -> Path:
    input_root = args.output_root / "input"
    sparse_path = input_root / "expression" / "expression_sparse.parquet"
    manifest_path = input_root / "manifest.json"
    metadata_path = input_root / "no_rss_input_metadata.json"

    if sparse_path.is_file() and manifest_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if (
            float(metadata.get("target_sum", -1)) == float(args.target_sum)
            and Path(metadata.get("stamp_h5ad", "")).resolve() == args.stamp_h5ad.resolve()
        ):
            print(f"Reusing prepared no-RSS input: {sparse_path}")
            return input_root
        raise FileExistsError("Prepared no-RSS input uses different parameters; use a new output root.")
    if sparse_path.exists() or manifest_path.exists() or metadata_path.exists():
        raise FileExistsError("Incomplete no-RSS input exists; use a new output root.")

    counts, spots, genes = load_stamp_counts(args.stamp_h5ad)
    validate_alignment(args.input, spots)
    expression = normalize_log1p(counts, args.target_sum)
    table = expression_long_table(expression, spots, genes)

    sparse_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sparse_path.with_name(f"{sparse_path.stem}.__tmp__{os.getpid()}{sparse_path.suffix}")
    table.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(sparse_path)

    write_json(
        manifest_path,
        {
            "schema_version": "no-rss-expression-input.v1",
            "inputs": {
                "h5ad_path": str(args.input.resolve()),
                "spot_id_field": "obs_names",
            },
            "ablation": {
                "type": "complete_rss_frontend_replacement",
                "gene_universe": "STAMP K20 h5ad genes after native filtering and 6000-HVG selection",
                "matrix": "library-size-normalized log1p expression",
                "target_sum": float(args.target_sum),
                "rss_artifacts_used": False,
                "downstream": "unchanged RSP cross-K/seed discovery and stability workflow",
            },
        },
    )
    write_json(
        metadata_path,
        {
            "input_h5ad": str(args.input.resolve()),
            "stamp_h5ad": str(args.stamp_h5ad.resolve()),
            "target_sum": float(args.target_sum),
            "n_spots": int(len(spots)),
            "n_hvgs": int(len(genes)),
            "n_positive_rows": int(table.shape[0]),
            "positive_fraction": float(table.shape[0] / (len(spots) * len(genes))),
            "columns": table.columns.tolist(),
            "rss_artifacts_used": [],
        },
    )
    print(
        f"Prepared no-RSS input with {len(spots):,} spots, {len(genes):,} STAMP-selected "
        f"HVGs and {table.shape[0]:,} positive expression values."
    )
    return input_root


def update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            update_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)


def native_config(sample_dir: Path) -> RSPPipelineConfig:
    manifest = read_json(sample_dir / "rsp_out" / "manifest.json")
    config = RSPPipelineConfig()
    update_dataclass(config, manifest.get("params", {}))
    return config


def run_ablation(args: argparse.Namespace, input_root: Path) -> Path:
    config = native_config(args.sample_dir)
    config.code_version_override = "r1-no-rss-stamp-hvg-expression"
    config.input.rss_sparse_relpath = "expression/expression_sparse.parquet"
    config.input.rss_manifest_relpath = "manifest.json"
    config.input.h5ad_path_override = str(args.input.resolve())
    config.discovery.rss_value_col = EXPRESSION_VALUE_COLUMN
    config.discovery.rss_rank_cutoff = None
    config.discovery.reliability_columns = ()
    # STAMP's native filtering plus 6000-HVG selection replaces the RSS-specific
    # gene-evidence prefilter. Keep only the shared confounder-gene exclusion in
    # RSP, then let decomposition and cross-K/seed stability evaluate programs.
    config.discovery.enable_adaptive_gene_selection = False
    config.discovery.max_genes_for_decomposition = None
    config.discovery.min_nonzero_rss_spot_fraction = 0.0
    config.discovery.min_nonzero_rss_spot_count = 0
    config.discovery.max_nonzero_rss_spot_fraction = None
    config.discovery.min_spatial_autocorrelation_score = -1.0
    config.discovery.min_structured_component_support = 0.0
    config.discovery.min_local_support_score = 0.0

    if args.mode == "smoke":
        config.discovery.k_grid = (4, 6)
        config.discovery.seed_grid = (0, 1)
        out_root = args.output_root
        sample_id = "smoke_all_hvg"
    elif args.mode == "full":
        out_root = args.output_root.parent
        sample_id = args.output_root.name
    else:
        raise ValueError(f"Unsupported run mode: {args.mode}")

    expected = out_root / sample_id / "rsp_out"
    native_rsp = (args.sample_dir / "rsp_out").resolve()
    if expected.resolve() == native_rsp:
        raise RuntimeError("Refusing to overwrite the native RSP bundle.")
    if expected.exists():
        raise FileExistsError(f"No-RSS output already exists: {expected}")

    write_json(
        args.output_root / f"{args.mode}_run_request.json",
        {
            "mode": args.mode,
            "input_root": str(input_root.resolve()),
            "expected_output": str(expected.resolve()),
            "native_rsp_read_only": str(native_rsp),
            "rss_artifacts_used": False,
            "gene_selection_control": (
                "STAMP native filtering and 6000-HVG selection replace RSS-specific "
                "prefiltering and adaptive gene selection"
            ),
            "config": asdict(config),
        },
    )
    return run_rsp_pipeline(
        rss_out_path=input_root,
        out_root=out_root,
        sample_id=sample_id,
        config=config,
    )


def main() -> None:
    args = parse_args()
    validate_paths(args)
    input_root = prepare_no_rss_input(args)
    if args.mode == "prepare":
        return
    output = run_ablation(args, input_root)
    print(f"Completed {args.mode} no-RSS HVG-expression ablation: {output}")


if __name__ == "__main__":
    main()
