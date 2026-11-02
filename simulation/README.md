# REGAL simulation datasets

Run from the repository root in the `REGAL` conda environment. These scripts generate simulation inputs and truth independently of model fitting.

| Directory | Design | Configuration |
| --- | --- | --- |
| `benchmark/` | 64×64 positions, 5,000 genes, 16 activities | `benchmark/configs/heterogeneous.json` |
| `resolution/` | One 128×128 sample aggregated by count summation to 64×64 and 32×32; 5,000 genes, 11 activities | `resolution/configs/simulation.json` |

Each directory contains a generator, a layout helper, and `check_simulation.py` for saved-input validation and QC figures. Configurations preserve the original parameters and random seeds. The resolution geometry uses fixed contours derived from BREAST annotations; no BREAST files are needed at runtime.

## Paths and existing artifacts

Moving the source code does not move datasets, truth, QC reports, or model results. Default dataset locations remain:

- Benchmark: `/Users/wuyang/Documents/SC-ST data/SIMULATION/ST/regional_activity_16_heterogeneous_seed2026.h5ad` and the adjacent sample directory of the same stem.
- Resolution: `/Users/wuyang/Documents/SC-ST data/SIMULATION/ST/regional_activity_11_resolution_seed2026_{128,64,32}.h5ad` and the adjacent sample directories of the same stems.

Benchmark QC remains under the sample's `simulation/qc/`. Resolution auxiliary outputs are written under `simulation/outputs/resolution/`: `region_layout/` for optional layout previews, `generation_audit.json` for the generation audit, and `simulation_check/` for input QC. This output directory is ignored by Git. Simulation scripts do not write to `studies/` by default. The existing generation audit and input QC results have also been moved here without changing their contents. Run logs, recovery evaluations, and historical figure-style validation remain in `studies/`.

Generated data and results are not included in this directory. Reuse existing artifacts. Generators refuse to overwrite existing dataset targets. QC commands write reports and figures and should only be run when those writes are intended.

## Commands

Inspect command options without generating data or writing reports:

```bash
conda run -n REGAL python simulation/benchmark/generate_simulation.py --help
conda run -n REGAL python simulation/benchmark/check_simulation.py --help
conda run -n REGAL python simulation/resolution/generate_resolution_simulation.py --help
conda run -n REGAL python simulation/resolution/check_simulation.py --help
```

Generate only when the destination does not already contain the dataset:

```bash
conda run -n REGAL python simulation/benchmark/generate_simulation.py
conda run -n REGAL python simulation/resolution/generate_resolution_simulation.py
```

Use `--output-root` for a new benchmark destination or `--data-root` for a new resolution destination on another machine. Model-running and recovery-analysis scripts remain in `studies/` locally and are not part of this source directory.
