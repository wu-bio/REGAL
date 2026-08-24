from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# Ensure package-style import works even when running this file directly:
# `python /.../RSS/calculate_rss.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RSS.pipeline import RSSPipelineConfig, run_rss_pipeline
from utils.dataset_registry import get_dataset
from utils.h5ad_schema import require_rss_h5ad_schema

warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


dataset = get_dataset("WU")
sample_ids = dataset.sample_ids


pipeline_config = RSSPipelineConfig()
pipeline_config.preprocess.data_layer = dataset.data_layer
pipeline_config.preprocess.spot_id_field = dataset.spot_id_field
pipeline_config.preprocess.hvg = 3000

# Spatial graph attention encoder parameters.
pipeline_config.latent.epochs = 500
pipeline_config.latent.spatial_graph_mode = "knn"
pipeline_config.latent.graph_k = 6

# MicroDomain parameters.
pipeline_config.neighbors.normal_k = 12
pipeline_config.neighbors.min_k = 6
pipeline_config.neighbors.candidate_k = 61
pipeline_config.rss.sparsify_rule = "topM"
pipeline_config.rss.top_m = 3000
pipeline_config.qc.bootstrap_repeats = 5


def main() -> None:
    for sample_id in sample_ids:
        print(f"Processing {dataset.dataset_id} / {sample_id} ...")
        h5ad_path = dataset.h5ad_path(sample_id)
        out_root = dataset.out_root()
        schema_report = require_rss_h5ad_schema(
            h5ad_path=h5ad_path,
            dataset=dataset,
            sample_id=sample_id,
        )
        print(schema_report.format_report())

        bundle_path = run_rss_pipeline(
            h5ad_path=h5ad_path,
            out_root=out_root,
            sample_id=sample_id,
            config=pipeline_config,
        )
        print(f"Done: {bundle_path}")


if __name__ == "__main__":
    main()
