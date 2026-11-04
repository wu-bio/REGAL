# Simulation datasets

Two synthetic designs provide expression counts and known regional activity structure for evaluating REGAL.

| Design | Spatial grid | Genes | Activities | Configuration |
| --- | --- | --- | --- | --- |
| Heterogeneous activities | 64 × 64 | 5,000 | 16 | [heterogeneous.json](heterogeneous/configs/heterogeneous.json) |
| Resolution series | 128 × 128, 64 × 64, 32 × 32 | 5,000 | 11 | [simulation.json](resolution/configs/simulation.json) |

The resolution series starts with one 128 × 128 count realization. Each coarser grid sums adjacent 2 × 2 positions and retains fractional activity coverage. Layouts, gene profiles, noise parameters, and random seeds are specified by the configuration files and layout helpers.

## Generate datasets

Run from the repository root after activating the REGAL environment. Specify destinations with `--output-root` for the heterogeneous sample and `--data-root` for the resolution series:

```bash
python Simulation/heterogeneous/generate_simulation.py \
  --config Simulation/heterogeneous/configs/heterogeneous.json \
  --output-root "$HOME/data/SIMULATION/ST/regional_activity_16_heterogeneous_seed2026"

python Simulation/resolution/generate_resolution_simulation.py \
  --config Simulation/resolution/configs/simulation.json \
  --data-root "$HOME/data/SIMULATION/ST"
```

Append `--preflight` to validate the design and inspect dimensions before generation.

## Generated files

Each sample produces an h5ad and an adjacent directory:

```text
SIMULATION/ST/
├── <sample_id>.h5ad
└── <sample_id>/
    └── simulation/
        ├── config.json
        ├── coordinates.csv
        └── truth/
```

The h5ad provides observed expression counts and spatial coordinates. The `truth/` directory records activity memberships, spatial instances, gene profiles, and latent fields. The heterogeneous sample ID is `regional_activity_16_heterogeneous_seed2026`; resolution sample IDs are `regional_activity_11_resolution_seed2026_{128,64,32}`.

## Validate generated data

```bash
python Simulation/heterogeneous/check_simulation.py \
  --root "$HOME/data/SIMULATION/ST/regional_activity_16_heterogeneous_seed2026"

python Simulation/resolution/check_simulation.py \
  --data-root "$HOME/data/SIMULATION/ST"
```

Validation produces structural checks, summary tables, and diagnostic figures. Heterogeneous-sample QC is saved under `<sample_id>/simulation/qc/`; resolution QC is saved under `Simulation/outputs/resolution/simulation_check/`. Resolution generation also writes `Simulation/outputs/resolution/generation_audit.json`.

## Analyze with REGAL

The `SIMULATION` dataset in [Utils/dataset_registry.py](../Utils/dataset_registry.py) registers all four samples. For example, after computing RSS for the 128 × 128 sample, run RSP with:

```bash
python run_regal.py --dataset SIMULATION \
  --sample regional_activity_11_resolution_seed2026_128 \
  --work-dir "$HOME/data" --stage rsp --no-adaptive-gene-selection
```

See the [main README](../README.md) for the RSS → RSP → RSMD workflow.
