# REGAL

**Regional Gene Activity Landscapes**

REGAL characterizes regional gene activity in spatial transcriptomics. It computes gene-level regional specificity scores, identifies stable programs with gene-loading profiles and spatial activation fields, and delineates their connected spatial instances. Together, these activities describe how tissue organization recurs, overlaps, and varies across local contexts.

## Workflow

| Stage | Function | Output |
| --- | --- | --- |
| RSS | Construct local MicroDomains and quantify gene-level regional specificity | Regional specificity scores |
| RSP | Discover programs across ranks and random seeds, assess stability, and score activity | Gene-loading profiles and spatial activation fields |
| RSMD | Delineate connected regional specificity domains (RSDs) and group instances of each program | Regional specificity metadomains |

## Installation

From the repository root, install REGAL in a Python 3.10 environment:

```bash
conda create -n REGAL python=3.10
conda activate REGAL
pip install -e .
```

Dependencies are specified in [pyproject.toml](pyproject.toml).

## Input data

Prepare an AnnData (`.h5ad`) object with:

- a non-negative expression matrix in `adata.X` or a selected layer;
- finite spatial coordinates in `adata.obsm["spatial"]`, with at least two columns;
- unique observation identifiers;
- gene identifiers in `adata.var_names` or a selected `adata.var` column.

Register dataset and sample identifiers in [Utils/dataset_registry.py](Utils/dataset_registry.py). Each `DatasetSpec` defines its data directory, sample IDs, expression layer, and coordinate fields. For example, the registered Breast sample uses this layout:

```text
~/data/
└── BREAST/
    └── ST/
        └── Breast_BlockA_Section1.h5ad
```

## Run REGAL

Run the stages in order from the repository root:

```bash
python run_regal.py --dataset BREAST --work-dir "$HOME/data" --stage rss
python run_regal.py --dataset BREAST --work-dir "$HOME/data" --stage rsp
python run_regal.py --dataset BREAST --work-dir "$HOME/data" --stage rsmd
```

Use `--sample SAMPLE_ID` to select a sample or `--all-samples` to process every registered sample. Append `--dry-run` to validate the selected stage's inputs and settings without computation.

Results are saved beside the input h5ad:

```text
BREAST/ST/Breast_BlockA_Section1/
├── rss_out/
├── rsp_out/
└── rsmd_out/
```

Use `--input-root` to read existing upstream results and `--output-root` to save a new analysis elsewhere. Both roots contain sample directories. See `python run_regal.py --help` for command-line options.

Default parameters are defined in [RSS/schema.py](RSS/schema.py), [RSP/schema.py](RSP/schema.py), and [RSMD/schema.py](RSMD/schema.py). For the manuscript analyses, Supplementary Table S9 specifies the dataset settings: disable adaptive gene selection for Liver, Brain, and simulations with `--no-adaptive-gene-selection`; set `--rss-rank-cutoff 1500` for Puck and Stereo. These flags apply to the RSP stage.

## Repository guide

| Directory | Contents |
| --- | --- |
| [RSS/](RSS/) | MicroDomain construction and regional specificity scoring |
| [RSP/](RSP/) | Program discovery, stability selection, and activity scoring |
| [RSMD/](RSMD/) | Spatial domain delineation and metadomain construction |
| [Utils/](Utils/) | Dataset registration, input schema checks, and shared plotting utilities |
| [Simulation/](Simulation/README.md) | Synthetic datasets and validation workflows |
| [Plotting/](Plotting/README.md) | Intermediate-table calculations and manuscript Figures 3–8 |
| [Benchmarks/](Benchmarks/) | Comparator and RSS-ablation workflows |
| [Deconvolution/](Deconvolution/) | Cell-type deconvolution workflows |

## Citation

Wu Y, Zeng Z, Fan H, et al. *Regional gene activity landscapes reveal recurrent programs with context-dependent realization*. Manuscript.

## License

[BSD 3-Clause](LICENSE).
