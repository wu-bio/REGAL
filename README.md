# REGAL

**Regional Gene Activity Landscapes**

REGAL is a framework for representing spatial transcriptomic tissue organization as a regional gene activity landscape. It quantifies gene-level regional specificity within locally coherent tissue contexts, organizes this evidence into stable and gene-traceable regional specificity programs (RSPs), and delineates the spatial instances through which each program is realized.

REGAL is described in:

> *Regional gene activity landscapes reveal recurrent programs with context-dependent realization*

## Overview

REGAL links gene-level evidence to tissue-level spatial organization through the following object hierarchy:

1. **MicroDomain** - a local context defined by spatial proximity and transcriptional similarity.
2. **Regional specificity score (RSS)** - gene-by-location evidence that expression has stronger local support than expected from its tissue-wide background.
3. **Regional specificity program (RSP)** - a stable regional gene activity linking a gene-loading profile to a continuous spatial activation field.
4. **Regional specificity domain (RSD)** - a spatially connected instance derived from an RSP activation field.
5. **Regional specificity metadomain (RSMD)** - the collection of RSDs belonging to the same RSP.

Together, the spatial deployment and overlap of retained RSPs form the regional gene activity landscape. This representation supports questions about which regional activities occur in a tissue, where the same activity recurs, and how its local realization varies with tissue context.

## Repository structure

- `RSS/` constructs MicroDomains and computes sparse gene-by-location RSS evidence.
- `RSP/` discovers candidate programs across factorization ranks and random seeds, assembles stability families, and retains representative RSPs.
- `RSMD/` smooths and thresholds RSP activation fields to delineate connected RSDs and construct RSMDs.
- `utils/` provides dataset registration, AnnData schema checks, bundle I/O, and shared utilities.
- `Benchmarks/` contains comparator and RSS-ablation workflows.
- `Deconvolution/` contains optional deconvolution utilities used for selected datasets.

## Installation

REGAL currently targets Python 3.10.

```bash
conda create -n REGAL python=3.10
conda activate REGAL
pip install -e .
```

## Input data

The primary input is an AnnData (`.h5ad`) object containing:

- a non-negative expression matrix in `adata.X` or a configured layer;
- finite two-dimensional spatial coordinates in `adata.obsm["spatial"]`;
- unique observation identifiers;
- gene identifiers in `adata.var_names` or a configured `adata.var` column.

Datasets and sample paths are registered in `utils/dataset_registry.py`. Raw and processed datasets are not distributed with this repository.

## Basic workflow

After registering a dataset and confirming its AnnData schema, run the three stages in order:

```bash
python RSS/calculate_rss.py
python RSP/calculate_rsp.py --dataset <DATASET_ID> --sample <SAMPLE_ID>
python RSMD/calculate_rsmd.py
```

The current RSS and RSMD command-line entry points use the dataset selected in their respective scripts. RSP accepts dataset and sample arguments directly. Pipeline classes in each module provide programmatic configuration for other datasets and parameter settings.

By default, the stages write reusable artifacts under the corresponding sample directory:

```text
<sample_dir>/rss_out/
<sample_dir>/rsp_out/
<sample_dir>/rsmd_out/
```

The pipelines perform schema and consistency checks before consuming upstream objects. Existing artifacts can be reused when their schema matches the current implementation.

## Scope

REGAL provides a mesoscopic and traceable representation of regional gene activity. RSPs and RSDs are transcriptomic spatial objects; their biological interpretation should be supported by appropriate anatomical, molecular, cellular, or external-reference evidence.

## Citation

If you use REGAL, please cite:

> Wu Y, Zeng Z, Fan H, et al. *Regional gene activity landscapes reveal recurrent programs with context-dependent realization*. Manuscript.

## License

No software license has been specified yet. Please contact the authors before redistributing or incorporating the code into another project.
