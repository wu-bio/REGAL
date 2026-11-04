r"""Run REGAL stages for registered datasets or configured simulation samples.

Examples (from the repository root, in the REGAL environment)::

    python run_regal.py --dataset SIMULATION --no-adaptive-gene-selection --dry-run
    python run_regal.py --dataset SIMULATION \
        --sample regional_activity_11_resolution_seed2026_128 --no-adaptive-gene-selection
    python run_regal.py --dataset LIVER --stage rsp --all-samples \
        --no-adaptive-gene-selection --output-root /path/to/new/experiment
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import logging
import math
from pathlib import Path

from Utils.dataset_registry import DatasetSpec, get_dataset

STAGES = ("rss", "rsp", "rsmd")


def finite_correlation(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not -1 <= number <= 1:
        raise argparse.ArgumentTypeError("Expected a finite value in [-1, 1].")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def parse_args(argv: list[str] | None = None, *, fixed_stage: str | None = None) -> argparse.Namespace:
    description = __doc__ if fixed_stage is None else f"Run the REGAL {fixed_stage.upper()} stage."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Dataset ID in Utils/dataset_registry.py.")
    source.add_argument("--config", type=Path,
                        help="Resolution simulation config containing data_root, sample_prefix and resolutions.")
    samples = parser.add_mutually_exclusive_group()
    samples.add_argument("--sample", help="One sample; defaults to the first sample in the dataset/config.")
    samples.add_argument("--all-samples", action="store_true", help="Run all samples in the dataset/config.")
    if fixed_stage is None:
        parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    else:
        parser.set_defaults(stage=fixed_stage)
    parser.add_argument("--work-dir", type=Path, help="Override the SC-ST data root (parent of dataset directories).")
    parser.add_argument("--input-root", type=Path,
                        help="Root containing existing <sample>/rss_out or rsp_out; defaults to the dataset ST directory.")
    parser.add_argument("--output-root", type=Path,
                        help="Root for new <sample>/<stage>_out bundles; defaults to the dataset ST directory.")
    parser.add_argument("--adaptive-gene-selection", action=argparse.BooleanOptionalAction, default=None,
                        help="Override the RSP module default (enabled). "
                             "Use --no-adaptive-gene-selection for the old revision runners.")
    parser.add_argument("--rss-rank-cutoff", type=positive_int,
                        help="Override the RSP RSS reading limit (default: 1000).")
    parser.add_argument("--min-spatial-autocorrelation-score", type=finite_correlation,
                        help="Override the RSP gene RSS Moran I filter (default: 0.30).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and targets, print settings, and write nothing.")
    args = parser.parse_args(argv)
    if args.stage not in ("rsp", "all") and any(
        getattr(args, field) is not None for field in
        ("adaptive_gene_selection", "rss_rank_cutoff", "min_spatial_autocorrelation_score")
    ):
        parser.error("RSP parameter overrides require --stage rsp or all.")
    return args


def load_dataset(args: argparse.Namespace) -> DatasetSpec:
    if args.config is None:
        dataset = get_dataset(args.dataset)
    else:
        config = json.loads(args.config.read_text())
        data_root = Path(config["data_root"]).expanduser().resolve()
        dataset = DatasetSpec(
            "external_dataset", "SIMULATION",
            tuple(f"{config['sample_prefix']}_{side}" for side in config["resolutions"]),
            species="synthetic", condition="simulated regional gene activities",
            work_dir=data_root.parent.parent, expression_state="raw_counts",
        )
        if dataset.out_root() != data_root:
            raise ValueError("Configured data_root must end with SIMULATION/ST.")
    if args.work_dir is not None:
        dataset = replace(dataset, work_dir=args.work_dir.expanduser().resolve())
    if not dataset.sample_ids:
        raise ValueError("No samples configured.")
    return dataset


def build_configs(dataset: DatasetSpec, args: argparse.Namespace) -> dict:
    from RSS.schema import RSSPipelineConfig
    from RSP.schema import RSPPipelineConfig
    from RSMD.schema import RSMDPipelineConfig

    rss = RSSPipelineConfig()
    rss.preprocess.data_layer = dataset.data_layer
    rss.preprocess.spot_id_field = dataset.spot_id_field
    rsp = RSPPipelineConfig()
    for option in ("adaptive_gene_selection", "rss_rank_cutoff", "min_spatial_autocorrelation_score"):
        value = getattr(args, option)
        if value is not None:
            field = "enable_adaptive_gene_selection" if option == "adaptive_gene_selection" else option
            setattr(rsp.discovery, field, value)
    return {"rss": rss, "rsp": rsp, "rsmd": RSMDPipelineConfig()}


def reject_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def require_files(root: Path, relatives: tuple[str, ...]) -> None:
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {path}")


def prepare_sample(dataset: DatasetSpec, sample: str, args: argparse.Namespace) -> tuple[dict, dict]:
    stages = STAGES if args.stage == "all" else (args.stage,)
    input_root = (args.input_root or dataset.out_root()).expanduser().resolve()
    output_root = (args.output_root or dataset.out_root()).expanduser().resolve()
    outputs = {stage: output_root / sample / f"{stage}_out" for stage in stages}
    for target in outputs.values():
        reject_existing(target)
    configs = build_configs(dataset, args)
    rss_input = outputs.get("rss", input_root / sample / "rss_out")
    rsp_input = outputs.get("rsp", input_root / sample / "rsp_out")
    if "rss" in stages:
        from Utils.h5ad_schema import require_rss_h5ad_schema

        report = require_rss_h5ad_schema(dataset.h5ad_path(sample), dataset, sample)
        print(report.format_report(), flush=True)
    if "rsp" in stages and "rss" not in stages:
        import numpy as np
        import pyarrow.parquet as pq
        from RSP.pipeline import load_spot_ids_and_spatial_coords

        cfg = configs["rsp"]
        require_files(rss_input, (cfg.input.rss_manifest_relpath, cfg.input.rss_sparse_relpath))
        required = {"spot_id", "gene", cfg.discovery.rss_value_col}
        if cfg.discovery.rss_rank_cutoff is not None:
            required.add("rss_rank_in_spot")
        columns = set(pq.read_schema(rss_input / cfg.input.rss_sparse_relpath).names)
        if missing := required - columns:
            raise ValueError(f"RSS parquet is missing columns: {sorted(missing)}")
        manifest = json.loads((rss_input / cfg.input.rss_manifest_relpath).read_text())
        spots, spatial = load_spot_ids_and_spatial_coords(manifest, cfg.input)
        if len(spots) == 0 or not np.isfinite(spatial).all():
            raise ValueError(f"Empty positions or nonfinite coordinates: {sample}")
    if "rsmd" in stages:
        cfg = configs["rsmd"]
        cfg.input.rss_out_path_override = str(rss_input)
        if "rsp" not in stages:
            require_files(rsp_input, (cfg.input.rsp_manifest_relpath, cfg.input.rsp_relpath,
                                      cfg.input.rsp_activation_relpath))
        if "rss" not in stages:
            require_files(rss_input, (cfg.input.rss_manifest_relpath, cfg.input.microdomain_idx_relpath,
                                      cfg.input.microdomain_meta_relpath))
    return {
        "dataset": dataset.dataset_id, "sample_id": sample,
        "h5ad_path": str(dataset.h5ad_path(sample)), "out_root": str(output_root),
        "rss_input": str(rss_input), "rsp_input": str(rsp_input), "stages": stages,
        "outputs": {stage: str(path) for stage, path in outputs.items()},
        "configs": {stage: asdict(configs[stage]) for stage in stages},
    }, configs


def run_sample(plan: dict, configs: dict) -> None:
    for stage in plan["stages"]:
        reject_existing(Path(plan["outputs"][stage]))
        print(f"Running {stage.upper()}: {plan['dataset']} / {plan['sample_id']}", flush=True)
        if stage == "rss":
            from RSS.pipeline import run_rss_pipeline

            output = run_rss_pipeline(h5ad_path=plan["h5ad_path"], out_root=plan["out_root"],
                                      sample_id=plan["sample_id"], config=configs[stage])
        elif stage == "rsp":
            from RSP.pipeline import run_rsp_pipeline

            output = run_rsp_pipeline(rss_out_path=plan["rss_input"], out_root=plan["out_root"],
                                      sample_id=plan["sample_id"], config=configs[stage])
        else:
            from RSMD.pipeline import run_rsmd_pipeline

            output = run_rsmd_pipeline(rsp_out_path=plan["rsp_input"], out_root=plan["out_root"],
                                       sample_id=plan["sample_id"], config=configs[stage])
        if not (Path(output) / "manifest.json").is_file():
            raise RuntimeError(f"{stage.upper()} did not produce an output manifest: {output}")
        print(f"Done: {output}", flush=True)


def main(argv: list[str] | None = None, *, fixed_stage: str | None = None) -> None:
    args = parse_args(argv, fixed_stage=fixed_stage)
    try:
        dataset = load_dataset(args)
        samples = dataset.sample_ids if args.all_samples else (args.sample or dataset.sample_ids[0],)
        if unknown := set(samples) - set(dataset.sample_ids):
            raise ValueError(f"Unregistered samples for {dataset.dataset_id}: {sorted(unknown)}")
        # Check every sample before dispatching the first computation.
        prepared = [prepare_sample(dataset, sample, args) for sample in samples]
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"dry_run": args.dry_run, "samples": [plan for plan, _ in prepared]}, indent=2), flush=True)
    if args.dry_run:
        return
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    for plan, configs in prepared:
        run_sample(plan, configs)


if __name__ == "__main__":
    main()
