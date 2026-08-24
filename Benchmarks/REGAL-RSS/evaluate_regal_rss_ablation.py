#!/usr/bin/env python
"""Evaluate native REGAL against the complete RSS-front-end ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parents[1]
    sys.path.insert(0, str(repository_root))

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from studies.R1_regional_activity.scripts.program_metrics import (
    component_metrics,
    expression_lookup,
    module_diversity,
    spatial_graph,
    top_gene_lists,
)


DEFAULT_INPUT = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
)
DEFAULT_SAMPLE_DIR = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SAMPLE_DIR / "REGAL-RSS" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--native-bundle", type=Path)
    parser.add_argument("--ablation-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--spatial-neighbors", type=int, default=6)
    parser.add_argument("--high-field-fraction", type=float, default=0.10)
    parser.add_argument(
        "--interpretation",
        default=(
            "complete RSS-front-end replacement by STAMP-selected 6000-HVG "
            "log-normalized expression; no RSS artifacts used; unchanged RSP "
            "decomposition and cross-K/seed stability workflow"
        ),
        help="Experiment-specific interpretation recorded in the comparison manifest.",
    )
    args = parser.parse_args()
    if args.top_n < 2 or args.spatial_neighbors < 1:
        parser.error("--top-n must be >=2 and --spatial-neighbors must be positive.")
    if not 0 < args.high_field_fraction < 1:
        parser.error("--high-field-fraction must lie in (0, 1).")
    return args


def load_rsp_objects(bundle: Path, obs_names: pd.Index) -> dict[str, pd.DataFrame]:
    loadings = pd.read_parquet(bundle / "rsp_gene_loadings.parquet").pivot(
        index="gene", columns="rsp_id", values="loading"
    ).fillna(0)
    fields = pd.read_parquet(bundle / "rsp_activation.parquet").pivot(
        index="spot_id", columns="rsp_id", values="activation"
    )
    missing = obs_names.difference(fields.index)
    if len(missing):
        raise ValueError(f"{bundle} is missing {len(missing)} spots.")
    loadings = loadings.apply(pd.to_numeric, errors="coerce").fillna(0)
    fields = fields.loc[obs_names, loadings.columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return {"loadings": loadings, "fields": fields}


def top_jaccard(first: list[str], second: list[str]) -> float:
    left, right = set(first), set(second)
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def match_components(
    native: dict[str, pd.DataFrame],
    ablation: dict[str, pd.DataFrame],
    native_top: dict[str, list[str]],
    ablation_top: dict[str, list[str]],
) -> pd.DataFrame:
    native_ids = list(native["fields"].columns.astype(str))
    ablation_ids = list(ablation["fields"].columns.astype(str))
    records: list[dict[str, object]] = []
    score = np.zeros((len(native_ids), len(ablation_ids)), dtype=float)
    for row, native_id in enumerate(native_ids):
        for column, ablation_id in enumerate(ablation_ids):
            spatial_rho = float(
                spearmanr(
                    native["fields"][native_id],
                    ablation["fields"][ablation_id],
                ).statistic
            )
            gene_jaccard = top_jaccard(native_top[native_id], ablation_top[ablation_id])
            combined = 0.75 * max(0.0, spatial_rho) + 0.25 * gene_jaccard
            score[row, column] = combined
            records.append(
                {
                    "native_rsp": native_id,
                    "ablation_rsp": ablation_id,
                    "spatial_spearman": spatial_rho,
                    "top_gene_jaccard": gene_jaccard,
                    "combined_match_score": combined,
                    "is_hungarian_match": False,
                }
            )
    matched_rows, matched_columns = linear_sum_assignment(-score)
    match_keys = {
        (native_ids[row], ablation_ids[column])
        for row, column in zip(matched_rows, matched_columns)
    }
    table = pd.DataFrame(records)
    table["is_hungarian_match"] = [
        (native_id, ablation_id) in match_keys
        for native_id, ablation_id in zip(table["native_rsp"], table["ablation_rsp"])
    ]
    return table


def paired_metric_table(matches: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    selected = matches.loc[matches["is_hungarian_match"]].copy()
    native = metrics.loc[metrics["model"].eq("REGAL-RSS")].set_index("component")
    ablation = metrics.loc[metrics["model"].eq("REGAL-expression")].set_index("component")
    metric_columns = [
        "top50_loading_fraction",
        "effective_gene_number",
        "program_score_field_spearman",
        "moran_i",
        "local_contrast_z",
        "largest_component_fraction",
    ]
    rows: list[dict[str, object]] = []
    for record in selected.to_dict("records"):
        native_id = str(record["native_rsp"])
        ablation_id = str(record["ablation_rsp"])
        row: dict[str, object] = dict(record)
        for metric in metric_columns:
            native_value = float(native.loc[native_id, metric])
            ablation_value = float(ablation.loc[ablation_id, metric])
            row[f"native_{metric}"] = native_value
            row[f"ablation_{metric}"] = ablation_value
            row[f"delta_native_minus_ablation_{metric}"] = native_value - ablation_value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    native_bundle = args.native_bundle or (args.sample_dir / "rsp_out")
    ablation_bundle = args.ablation_bundle or (args.sample_dir / "REGAL-RSS" / "rsp_out")
    for path in (args.input, native_bundle / "manifest.json", ablation_bundle / "manifest.json"):
        if not path.exists():
            raise FileNotFoundError(path)

    adata = ad.read_h5ad(args.input)
    outputs = {
        "REGAL-RSS": load_rsp_objects(native_bundle, adata.obs_names),
        "REGAL-expression": load_rsp_objects(ablation_bundle, adata.obs_names),
    }
    valid_genes = set(adata.var_names.astype(str))
    top_lists = {
        model: top_gene_lists(objects["loadings"], args.top_n, valid_genes)
        for model, objects in outputs.items()
    }
    required_genes = sorted(
        {
            gene
            for model_lists in top_lists.values()
            for genes in model_lists.values()
            for gene in genes
        }
    )
    gene_index, counts = expression_lookup(adata, required_genes)
    graph = spatial_graph(np.asarray(adata.obsm["spatial"])[:, :2], args.spatial_neighbors)
    records: list[dict[str, object]] = []
    for model, objects in outputs.items():
        diversity = module_diversity(top_lists[model])
        records.extend(
            component_metrics(
                model=model,
                loadings=objects["loadings"],
                fields=objects["fields"],
                top_lists=top_lists[model],
                diversity=diversity,
                gene_index=gene_index,
                counts=counts,
                graph=graph,
                high_fraction=args.high_field_fraction,
            )
        )
    metrics = pd.DataFrame(records)
    matches = match_components(
        outputs["REGAL-RSS"],
        outputs["REGAL-expression"],
        top_lists["REGAL-RSS"],
        top_lists["REGAL-expression"],
    )
    paired = paired_metric_table(matches, metrics)
    summary_columns = [
        "top50_loading_fraction",
        "effective_gene_number",
        "program_score_field_spearman",
        "moran_i",
        "local_contrast_z",
        "largest_component_fraction",
    ]
    summary = metrics.groupby("model", observed=True)[summary_columns].agg(
        ["count", "median", "mean", "std"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "component_metrics.csv", index=False)
    matches.to_csv(args.output_dir / "all_pair_similarities.csv", index=False)
    paired.to_csv(args.output_dir / "matched_component_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "metric_summary.csv", index=False)
    with (args.output_dir / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "native_bundle": str(native_bundle.resolve()),
                "ablation_bundle": str(ablation_bundle.resolve()),
                "top_n": int(args.top_n),
                "spatial_neighbors": int(args.spatial_neighbors),
                "high_field_fraction": float(args.high_field_fraction),
                "matching": "Hungarian assignment on 0.75*positive spatial Spearman + 0.25*Top-N gene Jaccard",
                "interpretation": str(args.interpretation),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved native-versus-expression comparison metrics to {args.output_dir}.")


if __name__ == "__main__":
    main()
