# REGAL

REGAL identifies regional gene activity objects from spatial transcriptomics
data and supports downstream study-specific interpretation through the
`RSS -> RSP -> RSMD` object chain.

## Layout

- `RSS/`: regional specificity score calculation.
- `RSP/`: regional specificity program discovery and scoring.
- `RSMD/`: regional specificity metadomain construction.
- `utils/`: shared dataset, h5ad, and plotting helpers.
- `deconvolution/`: optional deconvolution utilities used by selected studies.
- `studies/`: dataset-specific interpretation and figure workflows.

## Data Contract

REGAL keeps the existing data and artifact layout used by the migrated
analyses. Inputs and generated objects remain under the relevant sample
directory in `/Users/wuyang/Documents/SC-ST data/`, such as:

```text
/Users/wuyang/Documents/SC-ST data/Janesick/ST/Janesick_Visium_Sample1/
```

Existing `rss_out`, `rsp_out`, `rsmd_out` and `vis` artifacts
remain valid when their file schema matches the current code.
