# Manuscript figures

Scripts for computing figure source tables and drawing REGAL Figures 3–8.

| Directory | Figure content |
| --- | --- |
| [fig03/](fig03/) | Regional gene activity, method comparisons, and program stability |
| [fig04/](fig04/) | Liver zonation, Brain anatomy, and MOB laminar programs |
| [fig05/](fig05/) | Tumor activity landscapes and local molecular variation |
| [fig06/](fig06/) | DCIS structures, cell environments, and regional activity |
| [fig07/](fig07/) | Invasive contexts and local carrier disruption |
| [fig08/](fig08/) | Single-cell hypothesis queries and spatial molecular interfaces |

Each directory contains `prepare_fig0n.py` to compute intermediate tables and `plot_fig0n.py` to draw the complete figure.

## Configure inputs

Set external data paths in [config.yaml](config.yaml):

- `roots` defines shared data directories, starting with `data: ${HOME}/data`.
- Each figure's `paths` specifies its h5ad files, saved model results, and auxiliary data.
- `options` specifies sample identifiers and analysis parameters.

`${HOME}` expands to the current user's home directory. Other variables reference earlier entries in `roots`; relative paths resolve from the configuration file's directory.

| Figure | External inputs |
| --- | --- |
| Fig03 | Breast h5ad; SPARK-X results; STAGATE h5ad; NMF, NNMF, STAMP, REGAL, and RSS-ablation results |
| Fig04 | Liver, Brain, Puck, and Stereo h5ads and RSP results |
| Fig05 | Breast h5ad with cell-type/pathology annotations; RSP and RSMD results |
| Fig06–07 | Janesick h5ad; RSP and RSMD results; Xenium-to-Visium cell mapping |
| Fig08 | Wu spatial h5ad and RSP results; single-cell reference h5ad; Stereoscope cell-type proportions |

Model result directories contain the saved gene loadings, spatial activities, and domain memberships used by each figure. Stereoscope proportions are supplied as a parquet table indexed by spot, with cell types as columns.

## Run

From the repository root, activate the REGAL environment, compute the tables, and draw the figure:

```bash
python Plotting/fig03/prepare_fig03.py --config Plotting/config.yaml
python Plotting/fig03/plot_fig03.py --config Plotting/config.yaml
```

Replace `fig03` with `fig04`–`fig08` for other figures. When the intermediate tables are already available, run the plotting script directly.

Outputs are saved relative to `Plotting/`:

```text
output/
├── tables/
│   ├── fig03/
│   └── ...
└── figures/
    ├── fig03.png
    └── ...
```

Each plotting script saves one complete PNG. Table filenames are listed in the corresponding preparation script's `TABLE_NAMES`.

## Figure definitions and resources

Selected RSPs/RSDs, marker panels, pathway labels, colors, and layouts are defined in the plotting scripts. Preparation scripts reuse the same figure definitions. Update these selections when adapting a figure to other samples.

Small reference resources are included in [resources/](resources/):

| Directory | Resources |
| --- | --- |
| `shared/` | Reactome and BioPlanet gene sets; FFPE marker table |
| `fig04/` | MOB anatomical reference images |
| `fig05/` | GO gene sets; published DeepST gene table |
| `fig06_7/` | Literature annotations; Hallmark gene sets |
| `fig08/` | Single-cell and spatial hypothesis definitions |

Resource paths are defined in each plotting script's `RESOURCE_PATHS`. Spatial maps and figure styling use shared functions from `Utils/` and `RSP/visualize_rsp.py`.
