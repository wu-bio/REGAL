#!/usr/bin/env Rscript

# Run the official SPARK-X implementation on raw counts stored in an h5ad file.

SPARK_SOURCE_COMMIT <- "a8b4bf27b804604dfda53da42992f100b8e4e727"
DEFAULT_INPUT <- "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
DEFAULT_OUTPUT_DIR <- "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/SPARKX"

usage <- function() {
  cat(paste0(
    "Usage: run_sparkx.R [options]\n\n",
    "Identify spatially expressed genes with the official SPARK::sparkx() implementation.\n\n",
    "Options:\n",
    "  --input PATH          Input h5ad containing raw counts and spatial coordinates\n",
    "                        (default: ", DEFAULT_INPUT, ")\n",
    "  --output-dir PATH     Directory for SPARK-X result files\n",
    "                        (default: ", DEFAULT_OUTPUT_DIR, ")\n",
    "  --spatial-key KEY     Key in adata.obsm (default: spatial)\n",
    "  --num-cores N         Number of R worker processes (default: 4)\n",
    "  --option MODE         Kernel mode: mixture or single (default: mixture)\n",
    "  -h, --help            Show this help message\n"
  ))
}

parse_args <- function(argv) {
  out <- list(
    input = DEFAULT_INPUT,
    output_dir = DEFAULT_OUTPUT_DIR,
    spatial_key = "spatial",
    num_cores = 4L,
    option = "mixture"
  )
  key_map <- c(
    "--input" = "input",
    "--output-dir" = "output_dir",
    "--spatial-key" = "spatial_key",
    "--num-cores" = "num_cores",
    "--option" = "option"
  )

  i <- 1L
  while (i <= length(argv)) {
    token <- argv[[i]]
    if (token %in% c("-h", "--help")) {
      usage()
      quit(status = 0L)
    }
    if (!token %in% names(key_map)) {
      stop("Unknown argument: ", token, call. = FALSE)
    }
    if (i == length(argv)) {
      stop("Missing value for ", token, call. = FALSE)
    }
    out[[key_map[[token]]]] <- argv[[i + 1L]]
    i <- i + 2L
  }

  out$num_cores <- suppressWarnings(as.integer(out$num_cores))
  if (is.na(out$num_cores) || out$num_cores < 1L) {
    stop("--num-cores must be a positive integer.", call. = FALSE)
  }
  if (!out$option %in% c("mixture", "single")) {
    stop("--option must be either 'mixture' or 'single'.", call. = FALSE)
  }
  out
}

require_package <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("Missing R package '", package, "' in the active environment.", call. = FALSE)
  }
}

configure_reticulate_python <- function() {
  conda_prefix <- Sys.getenv("CONDA_PREFIX", unset = "")
  if (nzchar(conda_prefix)) {
    env_python <- file.path(conda_prefix, "bin", "python")
    if (file.exists(env_python)) {
      Sys.setenv(RETICULATE_PYTHON = env_python)
    }
  }
}

read_h5ad_input <- function(input, spatial_key) {
  anndata <- reticulate::import("anndata", convert = FALSE)
  numpy <- reticulate::import("numpy", convert = FALSE)
  scipy_sparse <- reticulate::import("scipy.sparse", convert = FALSE)
  adata <- anndata$read_h5ad(input)

  has_spatial <- reticulate::py_to_r(adata$obsm$`__contains__`(spatial_key))
  if (!isTRUE(has_spatial)) {
    stop("Missing adata.obsm['", spatial_key, "'] in input h5ad.", call. = FALSE)
  }

  x_python <- adata$X
  is_sparse <- reticulate::py_to_r(scipy_sparse$issparse(x_python))
  if (isTRUE(is_sparse)) {
    # R's sparse Matrix classes require a double-valued x slot. astype preserves sparsity.
    x_spot_gene <- reticulate::py_to_r(x_python$astype(numpy$float64))
  } else {
    x_spot_gene <- Matrix::Matrix(
      reticulate::py_to_r(x_python$astype(numpy$float64)),
      sparse = TRUE
    )
  }

  gene_names <- unlist(reticulate::py_to_r(adata$var_names$to_list()), use.names = FALSE)
  spot_names <- unlist(reticulate::py_to_r(adata$obs_names$to_list()), use.names = FALSE)
  coordinates <- as.matrix(reticulate::py_to_r(adata$obsm$get(spatial_key)))
  x_spot_gene <- methods::as(x_spot_gene, "CsparseMatrix")
  counts <- methods::as(Matrix::t(x_spot_gene), "CsparseMatrix")

  list(
    counts = counts,
    coordinates = coordinates,
    gene_names = as.character(gene_names),
    spot_names = as.character(spot_names)
  )
}

validate_and_filter <- function(data) {
  counts <- data$counts
  coordinates <- data$coordinates

  if (nrow(counts) != length(data$gene_names) || ncol(counts) != length(data$spot_names)) {
    stop("Count matrix dimensions do not match h5ad gene or spot names.", call. = FALSE)
  }
  if (anyDuplicated(data$gene_names)) {
    stop("Gene names must be unique before running SPARK-X.", call. = FALSE)
  }
  if (nrow(coordinates) != ncol(counts) || ncol(coordinates) < 2L) {
    stop("Spatial coordinates must have shape (n_spots, >=2).", call. = FALSE)
  }
  if (length(counts@x) && (!all(is.finite(counts@x)) || any(counts@x < 0))) {
    stop("Counts contain non-finite or negative values.", call. = FALSE)
  }
  if (length(counts@x) && any(abs(counts@x - round(counts@x)) > 1e-8)) {
    stop("Input X is not integer-like. SPARK-X must receive raw counts, not normalized expression.", call. = FALSE)
  }
  if (!all(is.finite(coordinates))) {
    stop("Spatial coordinates contain non-finite values.", call. = FALSE)
  }

  rownames(counts) <- data$gene_names
  colnames(counts) <- data$spot_names
  gene_totals <- Matrix::rowSums(counts)
  detected_spots <- Matrix::rowSums(counts > 0)
  spot_totals <- Matrix::colSums(counts)
  mitochondrial <- grepl("^MT-", data$gene_names, ignore.case = TRUE)
  keep_gene <- gene_totals > 0 & !mitochondrial
  keep_spot <- spot_totals > 0

  if (sum(keep_gene) < 2L || sum(keep_spot) < 3L) {
    stop("Too few non-zero genes or spots remain after filtering.", call. = FALSE)
  }

  list(
    counts = counts[keep_gene, keep_spot, drop = FALSE],
    coordinates = coordinates[keep_spot, , drop = FALSE],
    gene_summary = data.frame(
      gene = data$gene_names[keep_gene],
      total_counts = as.numeric(gene_totals[keep_gene]),
      detected_spots = as.integer(detected_spots[keep_gene]),
      stringsAsFactors = FALSE
    ),
    n_genes_input = nrow(counts),
    n_spots_input = ncol(counts),
    n_zero_genes_removed = sum(gene_totals == 0),
    n_mito_genes_removed = sum(mitochondrial & gene_totals > 0),
    n_zero_spots_removed = sum(!keep_spot)
  )
}

run_sparkx <- function(args) {
  require_package("SPARK")
  require_package("Matrix")
  require_package("reticulate")
  require_package("jsonlite")
  configure_reticulate_python()

  if (!file.exists(args$input)) {
    stop("Input h5ad does not exist: ", args$input, call. = FALSE)
  }
  input_path <- normalizePath(args$input, mustWork = TRUE)
  loaded <- read_h5ad_input(input_path, args$spatial_key)
  filtered <- validate_and_filter(loaded)

  cat(
    "Prepared", nrow(filtered$counts), "genes and", ncol(filtered$counts),
    "spots for SPARK-X.\n"
  )
  result <- SPARK::sparkx(
    count_in = filtered$counts,
    locus_in = filtered$coordinates,
    X_in = NULL,
    numCores = args$num_cores,
    option = args$option,
    verbose = TRUE
  )

  main_result <- data.frame(
    gene = rownames(result$res_mtest),
    combined_pvalue = result$res_mtest[, "combinedPval"],
    adjusted_pvalue = result$res_mtest[, "adjustedPval"],
    stringsAsFactors = FALSE
  )
  summary_index <- match(main_result$gene, filtered$gene_summary$gene)
  main_result$total_counts <- filtered$gene_summary$total_counts[summary_index]
  main_result$detected_spots <- filtered$gene_summary$detected_spots[summary_index]
  main_result <- main_result[
    order(main_result$adjusted_pvalue, main_result$combined_pvalue, main_result$gene),
    ,
    drop = FALSE
  ]
  rownames(main_result) <- NULL

  output_dir <- path.expand(args$output_dir)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  result_csv <- file.path(output_dir, "sparkx_results.csv")
  result_rds <- file.path(output_dir, "sparkx_full_results.rds")
  metadata_json <- file.path(output_dir, "run_metadata.json")
  utils::write.csv(main_result, result_csv, row.names = FALSE)
  saveRDS(result, result_rds)

  metadata <- list(
    input = input_path,
    created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    source_repository = "https://github.com/xzhoulab/SPARK",
    source_commit = SPARK_SOURCE_COMMIT,
    r_version = R.version.string,
    spark_version = as.character(utils::packageVersion("SPARK")),
    python = reticulate::py_config()$version_string,
    anndata_version = reticulate::py_to_r(
      reticulate::import("anndata", convert = FALSE)$`__version__`
    ),
    spatial_key = args$spatial_key,
    option = args$option,
    num_cores = args$num_cores,
    n_genes_input = filtered$n_genes_input,
    n_spots_input = filtered$n_spots_input,
    n_genes_analyzed = nrow(filtered$counts),
    n_spots_analyzed = ncol(filtered$counts),
    n_zero_genes_removed = filtered$n_zero_genes_removed,
    n_mito_genes_removed = filtered$n_mito_genes_removed,
    n_zero_spots_removed = filtered$n_zero_spots_removed,
    n_significant_by_0_05 = sum(main_result$adjusted_pvalue <= 0.05, na.rm = TRUE),
    outputs = list(
      ranked_gene_results = result_csv,
      full_kernel_results = result_rds
    )
  )
  jsonlite::write_json(metadata, metadata_json, auto_unbox = TRUE, pretty = TRUE)

  cat(
    "Saved", nrow(main_result), "gene-level results to", output_dir,
    "with", metadata$n_significant_by_0_05, "genes at adjusted P <= 0.05.\n"
  )
}

run_sparkx(parse_args(commandArgs(trailingOnly = TRUE)))
