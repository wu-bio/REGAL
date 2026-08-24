from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Keep package imports working when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RSMD.pipeline import RSMDPipelineConfig, run_rsmd_pipeline
from utils.dataset_registry import get_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


dataset = get_dataset("WU")
sample_ids = dataset.sample_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate RSMD domains from RSP outputs.")
    parser.add_argument(
        "--activation-threshold-quantile",
        type=float,
        default=None,
        help="Manual activation quantile used to threshold each RSP footprint.",
    )
    args = parser.parse_args()

    pipeline_config = RSMDPipelineConfig()
    if args.activation_threshold_quantile is not None:
        pipeline_config.filtration.activation_threshold_quantile = float(args.activation_threshold_quantile)

    for sample_id in sample_ids:
        rsp_out = dataset.sample_dir(sample_id) / "rsp_out"
        out_root = dataset.out_root()
        print(f"Processing {dataset.dataset_id} / {sample_id} ...")
        rsmd_out = run_rsmd_pipeline(
            rsp_out_path=rsp_out,
            out_root=out_root,
            sample_id=sample_id,
            config=pipeline_config,
        )
        print(f"Done: {rsmd_out}")


if __name__ == "__main__":
    main()
