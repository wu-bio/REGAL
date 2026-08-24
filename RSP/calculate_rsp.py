from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RSP.pipeline import RSPPipelineConfig, run_rsp_pipeline
from utils.dataset_registry import get_dataset


DEFAULT_DATASET_ID = "WU"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Regional Specificity Programs from rss_out.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="Dataset id from utils.dataset_registry.")
    parser.add_argument("--sample", default=None, help="Single sample id. Defaults to all samples in dataset.")
    parser.add_argument(
        "--rss-rank-cutoff",
        type=int,
        help="Optional per-spot RSS rank cutoff applied when RSP reads rss_sparse.parquet.",
    )
    parser.add_argument(
        "--decomposition-backend",
        choices=("legacy_rss_spatial_nmf", "rss_anchored_cd_l1_sparse_factorization"),
        help="Optional RSP decomposition backend override.",
    )
    parser.add_argument(
        "--disable-adaptive-gene-selection",
        action="store_true",
        help=(
            "Use all hard-kept RSS genes for decomposition instead of applying the "
            "adaptive gene_evidence_score threshold."
        ),
    )
    parser.add_argument(
        "--max-genes-for-decomposition",
        type=int,
        help="Optional cap on hard-kept genes passed into decomposition.",
    )
    parser.add_argument(
        "--min-structured-component-support",
        type=float,
        help="Optional hard-filter threshold for structured RSS component support.",
    )
    parser.add_argument(
        "--min-local-support-score",
        type=float,
        help="Optional hard-filter threshold for local RSS support score.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Optional pipeline random seed override.",
    )
    parser.add_argument(
        "--k-grid",
        help="Comma-separated decomposition k values, e.g. 12,16,20,24.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = get_dataset(args.dataset)
    pipeline_config = RSPPipelineConfig()
    if args.rss_rank_cutoff is not None:
        pipeline_config.discovery.rss_rank_cutoff = args.rss_rank_cutoff
    if args.decomposition_backend is not None:
        pipeline_config.discovery.decomposition_backend = args.decomposition_backend
    if args.disable_adaptive_gene_selection:
        pipeline_config.discovery.enable_adaptive_gene_selection = False
    if args.max_genes_for_decomposition is not None:
        pipeline_config.discovery.max_genes_for_decomposition = int(args.max_genes_for_decomposition)
    if args.min_structured_component_support is not None:
        pipeline_config.discovery.min_structured_component_support = float(args.min_structured_component_support)
    if args.min_local_support_score is not None:
        pipeline_config.discovery.min_local_support_score = float(args.min_local_support_score)
    if args.random_seed is not None:
        pipeline_config.random_seed = int(args.random_seed)
    if args.k_grid is not None:
        pipeline_config.discovery.k_grid = tuple(
            int(value.strip()) for value in str(args.k_grid).split(",") if value.strip()
        )

    sample_ids = [args.sample] if args.sample else dataset.sample_ids
    for sample_id in sample_ids:
        print(f"=== Processing {dataset.dataset_id} / {sample_id} ===")
        rss_out = dataset.sample_dir(sample_id) / "rss_out"
        out_root = Path(dataset.out_root())
        rsp_out = run_rsp_pipeline(
            rss_out_path=rss_out,
            out_root=out_root,
            sample_id=str(sample_id),
            config=pipeline_config,
        )
        print(f"Done: {rsp_out}")


if __name__ == "__main__":
    main()
