#!/usr/bin/env Rscript

# Run the official neighborhood non-negative matrix factorization implementation.

NNMF_SOURCE_COMMIT <- "739ac095460e67c72db102ce3b192bdd89583086"
DEFAULT_INPUT <- "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
DEFAULT_OUTPUT_ROOT <- "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/NNMF"
DEFAULT_K <- 24L
# Official estimate_lengthscale criterion on the native 10% detection-filtered BREAST input:
# minimum test MSE on the 10-point grid with max_avg_nn = 20.
DEFAULT_LENGTHSCALE <- 324.128494

usage <- function() {
  cat(paste0(
    "Usage: run_nnmf.R [options]\n\n",
    "Learn native NNMF weights/signatures and save base-threshold spatial coverage.\n\n",
    "Model rank:\n",
    "  --k INTEGER               Number of signatures (default: ", DEFAULT_K, ";\n",
    "                            BREAST AIC scan with seed sensitivity review)\n\n",
    "Input/output:\n",
    "  --input PATH              Input h5ad containing raw counts and coordinates\n",
    "                            (default: ", DEFAULT_INPUT, ")\n",
    "  --output-dir PATH         Output directory (default: ", DEFAULT_OUTPUT_ROOT, "/K<k>)\n",
    "  --spatial-key KEY         Key in adata.obsm (default: spatial)\n",
    "  --gene-list PATH          Optional one-gene-per-line override for sensitivity analysis\n",
    "  --min-detection-rate X    Native Visium detection filter (default: 0.10)\n\n",
    "Official NNMF parameters:\n",
    "  --lengthscale X           Gaussian length scale (default: ", DEFAULT_LENGTHSCALE, ";\n",
    "                            BREAST estimate_lengthscale result with max_avg_nn=20)\n",
    "  --kernel-cutoff X         Kernel entries below this value are zeroed (default: 0.1)\n",
    "  --maxiter N               Maximum multiplicative-update iterations (default: 1000)\n",
    "  --tolerance X             Convergence tolerance (default: 1e-10)\n",
    "  --initial N               Warm-start initializations (default: 3)\n",
    "  --small-iter N            NMF iterations per warm start (default: 50)\n",
    "  --seed N                  R random seed (default: 0)\n",
    "  -h, --help                Show this help message\n"
  ))
}

parse_args <- function(argv) {
  out <- list(
    k = DEFAULT_K,
    input = DEFAULT_INPUT,
    output_dir = NULL,
    spatial_key = "spatial",
    gene_list = NULL,
    min_detection_rate = 0.10,
    lengthscale = DEFAULT_LENGTHSCALE,
    kernel_cutoff = 0.1,
    maxiter = 1000L,
    tolerance = 1e-10,
    initial = 3L,
    small_iter = 50L,
    seed = 0L
  )
  key_map <- c(
    "--k" = "k",
    "--input" = "input",
    "--output-dir" = "output_dir",
    "--spatial-key" = "spatial_key",
    "--gene-list" = "gene_list",
    "--min-detection-rate" = "min_detection_rate",
    "--lengthscale" = "lengthscale",
    "--kernel-cutoff" = "kernel_cutoff",
    "--maxiter" = "maxiter",
    "--tolerance" = "tolerance",
    "--initial" = "initial",
    "--small-iter" = "small_iter",
    "--seed" = "seed"
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

  integer_fields <- c("k", "maxiter", "initial", "small_iter", "seed")
  for (field in integer_fields) {
    if (!is.null(out[[field]])) {
      out[[field]] <- suppressWarnings(as.integer(out[[field]]))
    }
  }
  numeric_fields <- c("min_detection_rate", "lengthscale", "kernel_cutoff", "tolerance")
  for (field in numeric_fields) {
    if (!is.null(out[[field]])) {
      out[[field]] <- suppressWarnings(as.numeric(out[[field]]))
    }
  }

  if (is.null(out$k) || is.na(out$k) || out$k < 2L) {
    stop("--k is required and must be an integer of at least 2.", call. = FALSE)
  }
  if (is.na(out$min_detection_rate) || out$min_detection_rate <= 0 || out$min_detection_rate > 1) {
    stop("--min-detection-rate must be in (0, 1].", call. = FALSE)
  }
  if (!is.null(out$lengthscale) && (is.na(out$lengthscale) || out$lengthscale <= 0)) {
    stop("--lengthscale must be positive when supplied.", call. = FALSE)
  }
  if (is.na(out$kernel_cutoff) || out$kernel_cutoff < 0 || out$kernel_cutoff >= 1) {
    stop("--kernel-cutoff must be in [0, 1).", call. = FALSE)
  }
  if (is.na(out$tolerance) || out$tolerance <= 0) {
    stop("--tolerance must be positive.", call. = FALSE)
  }
  for (field in c("maxiter", "initial", "small_iter")) {
    if (is.na(out[[field]]) || out[[field]] < 1L) {
      stop("--", gsub("_", "-", field), " must be a positive integer.", call. = FALSE)
    }
  }
  if (is.na(out$seed) || out$seed < 0L) {
    stop("--seed must be a non-negative integer.", call. = FALSE)
  }
  if (is.null(out$output_dir)) {
    out$output_dir <- file.path(DEFAULT_OUTPUT_ROOT, paste0("K", out$k))
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

read_gene_list <- function(path) {
  if (!file.exists(path)) {
    stop("Gene-list file does not exist: ", path, call. = FALSE)
  }
  genes <- trimws(readLines(path, warn = FALSE))
  genes <- genes[nzchar(genes) & !startsWith(genes, "#")]
  if (!length(genes)) {
    stop("Gene-list file is empty: ", path, call. = FALSE)
  }
  if (anyDuplicated(genes)) {
    stop("Gene-list file contains duplicated genes.", call. = FALSE)
  }
  genes
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
    x_spot_gene <- reticulate::py_to_r(x_python$astype(numpy$float64))
  } else {
    x_spot_gene <- Matrix::Matrix(
      reticulate::py_to_r(x_python$astype(numpy$float64)),
      sparse = TRUE
    )
  }
  x_spot_gene <- methods::as(x_spot_gene, "CsparseMatrix")

  list(
    counts = x_spot_gene,
    coordinates = as.matrix(reticulate::py_to_r(adata$obsm$get(spatial_key))),
    gene_names = as.character(unlist(reticulate::py_to_r(adata$var_names$to_list()), use.names = FALSE)),
    spot_names = as.character(unlist(reticulate::py_to_r(adata$obs_names$to_list()), use.names = FALSE))
  )
}

prepare_nnmf_input <- function(data, args) {
  counts <- data$counts
  if (nrow(counts) != length(data$spot_names) || ncol(counts) != length(data$gene_names)) {
    stop("Count matrix dimensions do not match h5ad spot or gene names.", call. = FALSE)
  }
  if (anyDuplicated(data$gene_names)) {
    stop("Gene names must be unique before running NNMF.", call. = FALSE)
  }
  if (nrow(data$coordinates) != nrow(counts) || ncol(data$coordinates) < 2L) {
    stop("Spatial coordinates must have shape (n_spots, >=2).", call. = FALSE)
  }
  if (length(counts@x) && (!all(is.finite(counts@x)) || any(counts@x < 0))) {
    stop("Counts contain non-finite or negative values.", call. = FALSE)
  }
  if (length(counts@x) && any(abs(counts@x - round(counts@x)) > 1e-8)) {
    stop("Input X is not integer-like. NNMF must receive raw counts.", call. = FALSE)
  }
  if (!all(is.finite(data$coordinates))) {
    stop("Spatial coordinates contain non-finite values.", call. = FALSE)
  }

  spot_totals <- Matrix::rowSums(counts)
  keep_spot <- spot_totals > 0
  counts <- counts[keep_spot, , drop = FALSE]
  coordinates <- data$coordinates[keep_spot, , drop = FALSE]
  spot_names <- data$spot_names[keep_spot]
  detected_spots <- Matrix::colSums(counts > 0)
  gene_totals <- Matrix::colSums(counts)

  if (!is.null(args$gene_list)) {
    requested_genes <- read_gene_list(args$gene_list)
    missing_genes <- setdiff(requested_genes, data$gene_names)
    if (length(missing_genes)) {
      stop(
        "Gene list contains ", length(missing_genes), " genes absent from the h5ad; first missing: ",
        paste(utils::head(missing_genes, 5L), collapse = ", "),
        call. = FALSE
      )
    }
    requested_totals <- gene_totals[match(requested_genes, data$gene_names)]
    zero_genes <- requested_genes[requested_totals == 0]
    if (length(zero_genes)) {
      stop(
        "Gene list contains ", length(zero_genes), " all-zero genes after spot filtering; first: ",
        paste(utils::head(zero_genes, 5L), collapse = ", "),
        call. = FALSE
      )
    }
    selected_genes <- requested_genes
    selection_mode <- "explicit_gene_list_sensitivity_analysis"
  } else {
    minimum_spots <- ceiling(args$min_detection_rate * nrow(counts))
    selected_genes <- data$gene_names[detected_spots >= minimum_spots & gene_totals > 0]
    selection_mode <- paste0(
      "native_visium_detected_in_at_least_",
      args$min_detection_rate,
      "_of_spots"
    )
  }

  if (length(selected_genes) <= args$k) {
    stop("The selected gene universe must contain more genes than --k.", call. = FALSE)
  }
  gene_index <- match(selected_genes, data$gene_names)
  dense_counts <- as.matrix(counts[, gene_index, drop = FALSE])
  rownames(dense_counts) <- spot_names
  colnames(dense_counts) <- selected_genes

  list(
    counts = dense_counts,
    coordinates = coordinates,
    genes = selected_genes,
    spots = spot_names,
    selection_mode = selection_mode,
    n_genes_input = length(data$gene_names),
    n_spots_input = length(data$spot_names),
    n_zero_spots_removed = sum(!keep_spot)
  )
}

validate_result <- function(result, n_spots, n_genes, k) {
  if (!identical(dim(result$weights), c(as.integer(n_spots), as.integer(k)))) {
    stop("NNMF returned weights with unexpected dimensions.", call. = FALSE)
  }
  if (!identical(dim(result$signatures), c(as.integer(k), as.integer(n_genes)))) {
    stop("NNMF returned signatures with unexpected dimensions.", call. = FALSE)
  }
  if (any(!is.finite(result$weights)) || any(result$weights < 0) ||
      any(!is.finite(result$signatures)) || any(result$signatures < 0)) {
    stop("NNMF returned invalid non-negative factor matrices.", call. = FALSE)
  }
  if (max(abs(rowSums(result$signatures) - 1)) > 1e-6) {
    stop("NNMF signatures do not have the documented row-sum normalization.", call. = FALSE)
  }
  if (!is.finite(result$error)) {
    stop("NNMF returned a non-finite GKL error.", call. = FALSE)
  }
}

# Per-signature spatial cleanup; no adaptive elbow or truth-dependent threshold.
# Neighbor-distance ties use spot order for a deterministic native R graph.
spatial_activation <- function(weights, coordinates) {
  weights <- as.matrix(weights)
  coordinates <- as.matrix(coordinates)
  n <- nrow(weights)
  if (n < 2L || nrow(coordinates) != n || ncol(coordinates) < 2L ||
      any(!is.finite(coordinates)) || any(!is.finite(weights)) || any(weights < 0)) {
    stop("Spatial postprocessing requires finite aligned coordinates and nonnegative weights.", call. = FALSE)
  }
  neighbors <- matrix(0L, n, min(6L, n - 1L))
  xy <- coordinates[, 1:2, drop = FALSE]
  for (i in seq_len(n)) {
    distances <- rowSums(sweep(xy, 2L, xy[i, ], "-")^2)
    distances[i] <- Inf
    neighbors[i, ] <- order(distances, method = "radix")[seq_len(ncol(neighbors))]
  }
  filter_components <- function(mask, minimum) {
    keep <- visited <- rep(FALSE, n)
    for (start in which(mask)) {
      if (visited[start]) next
      stack <- start
      visited[start] <- TRUE
      nodes <- integer(0)
      while (length(stack)) {
        node <- stack[length(stack)]
        stack <- stack[-length(stack)]
        nodes <- c(nodes, node)
        next_nodes <- neighbors[node, ]
        next_nodes <- next_nodes[mask[next_nodes] & !visited[next_nodes]]
        visited[next_nodes] <- TRUE
        stack <- c(stack, next_nodes)
      }
      if (length(nodes) >= minimum) keep[nodes] <- TRUE
    }
    keep
  }
  clean <- activation <- matrix(0, n, ncol(weights), dimnames = dimnames(weights))
  active <- matrix(FALSE, n, ncol(weights), dimnames = dimnames(weights))
  tables <- vector("list", ncol(weights))
  for (j in seq_len(ncol(weights))) {
    v <- weights[, j]
    v[v < max(1e-6, .05 * max(v))] <- 0
    v[!filter_components(v > 0, 5L)] <- 0
    clean[, j] <- v
    v[v < max(1e-6, .05 * max(v))] <- 0
    v[!filter_components(v > 0, 20L)] <- 0
    activation[, j] <- v
    positive <- v[v > 0]
    floor <- if (length(positive)) as.numeric(stats::quantile(positive, .05, names = FALSE, type = 7L)) else 0
    threshold <- max(1e-6, .10 * max(v), floor)
    mask <- filter_components(v > 0 & v >= threshold, 20L)
    active[, j] <- mask
    ordering <- order(-v, seq_len(n))
    ranks <- active_ranks <- integer(n)
    ranks[ordering] <- seq_len(n)
    active_order <- ordering[mask[ordering]]
    active_ranks[active_order] <- seq_along(active_order)
    tables[[j]] <- data.frame(
      component_id = colnames(weights)[j], spot_id = rownames(weights),
      activation_raw = weights[, j], activation = v, rank_in_spot = ranks,
      activation_active = mask, activation_active_rank = active_ranks,
      activation_active_threshold = threshold, row.names = NULL
    )
  }
  list(activation_clean = clean, activation = activation, activation_active = active,
       table = do.call(rbind, tables), parameters = list(
         method = "RSP_cleanup_with_base_threshold_only", activation_active_elbow_enabled = FALSE,
         active_threshold_rule = "max(min_abs, min_peak_fraction * peak, positive_quantile)",
         spatial_knn = 6L, moran_knn = 6L, neighbor_distance_ties = "ascending spot order",
         candidate_cleanup = list(activation_cleanup_min_abs = 1e-6,
                                 activation_cleanup_min_peak_fraction = .05,
                                 activation_cleanup_min_component_size = 5L),
         output_and_active_region = list(activation_trim_enabled = TRUE,
           activation_trim_fraction_of_peak = .05, activation_trim_min_abs = 1e-6,
           activation_trim_min_component_size = 20L, activation_active_min_peak_fraction = .10,
           activation_active_quantile = .05, activation_active_min_abs = 1e-6,
           activation_active_min_component_size = 20L, activation_active_fallback_to_base = FALSE),
         uses_truth = FALSE, coverage_field = "activation_active", raw_weights_field = "weights",
         activation_field = "activation", coverage_object_field = "activation_active"))
}

run_nnmf <- function(args) {
  require_package("NNMF")
  require_package("Matrix")
  require_package("reticulate")
  require_package("jsonlite")
  configure_reticulate_python()

  if (!file.exists(args$input)) {
    stop("Input h5ad does not exist: ", args$input, call. = FALSE)
  }
  input_path <- normalizePath(args$input, mustWork = TRUE)
  loaded <- read_h5ad_input(input_path, args$spatial_key)
  prepared <- prepare_nnmf_input(loaded, args)
  if (nrow(prepared$counts) <= args$k) {
    stop("The number of spots must exceed --k.", call. = FALSE)
  }

  dense_gb <- as.numeric(object.size(prepared$counts)) / 1024^3
  cat(
    "Prepared", nrow(prepared$counts), "spots and", ncol(prepared$counts),
    "genes for NNMF; dense input uses", sprintf("%.2f", dense_gb), "GB.\n"
  )
  set.seed(args$seed)
  result <- NNMF::nnmf(
    data = prepared$counts,
    noSignatures = args$k,
    location = prepared$coordinates,
    lengthscale = args$lengthscale,
    batch = 1,
    maxiter = args$maxiter,
    tolerance = args$tolerance,
    initial = args$initial,
    smallIter = args$small_iter,
    kernel_cutoff = args$kernel_cutoff,
    normalize = TRUE,
    not_sc = FALSE
  )
  validate_result(result, nrow(prepared$counts), ncol(prepared$counts), args$k)

  signature_names <- paste0("signature_", seq_len(args$k))
  rownames(result$weights) <- prepared$spots
  colnames(result$weights) <- signature_names
  rownames(result$signatures) <- signature_names
  colnames(result$signatures) <- prepared$genes

  spatial <- spatial_activation(result$weights, prepared$coordinates)
  result$activation_clean <- spatial$activation_clean
  result$activation <- spatial$activation
  result$activation_active <- spatial$activation_active
  result$spatial_postprocessing <- spatial$parameters

  output_dir <- path.expand(args$output_dir)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  weights_csv <- file.path(output_dir, "nnmf_weights.csv")
  signatures_csv <- file.path(output_dir, "nnmf_signatures.csv")
  full_rds <- file.path(output_dir, "nnmf_full_results.rds")
  activation_csv <- file.path(output_dir, "nnmf_activation.csv")
  metadata_json <- file.path(output_dir, "run_metadata.json")

  weights_table <- data.frame(spot_id = prepared$spots, result$weights, check.names = FALSE)
  signatures_table <- data.frame(
    gene = prepared$genes,
    Matrix::t(result$signatures),
    check.names = FALSE
  )
  utils::write.csv(weights_table, weights_csv, row.names = FALSE)
  utils::write.csv(signatures_table, signatures_csv, row.names = FALSE)
  utils::write.csv(spatial$table, activation_csv, row.names = FALSE)
  saveRDS(result, full_rds)

  metadata <- list(
    input = input_path,
    created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    source_repository = "https://github.com/ragnhildlaursen/NNMF",
    source_commit = NNMF_SOURCE_COMMIT,
    r_version = R.version.string,
    nnmf_version = as.character(utils::packageVersion("NNMF")),
    python = reticulate::py_config()$version_string,
    anndata_version = reticulate::py_to_r(
      reticulate::import("anndata", convert = FALSE)$`__version__`
    ),
    spatial_key = args$spatial_key,
    k = args$k,
    seed = args$seed,
    gene_selection = prepared$selection_mode,
    gene_list = if (is.null(args$gene_list)) NULL else normalizePath(args$gene_list, mustWork = TRUE),
    min_detection_rate = if (is.null(args$gene_list)) args$min_detection_rate else NULL,
    n_genes_input = prepared$n_genes_input,
    n_spots_input = prepared$n_spots_input,
    n_genes_analyzed = ncol(prepared$counts),
    n_spots_analyzed = nrow(prepared$counts),
    n_zero_spots_removed = prepared$n_zero_spots_removed,
    normalize = TRUE,
    not_sc = FALSE,
    requested_lengthscale = args$lengthscale,
    fitted_lengthscale = result$lengthscale,
    kernel_cutoff = args$kernel_cutoff,
    average_neighbors = result$avg_nn,
    maxiter = args$maxiter,
    tolerance = args$tolerance,
    initial = args$initial,
    small_iter = args$small_iter,
    final_gkl = result$error,
    spatial_postprocessing = spatial$parameters,
    outputs = list(
      spot_weights = weights_csv,
      gene_signatures = signatures_csv,
      full_results = full_rds,
      activation = activation_csv
    )
  )
  jsonlite::write_json(metadata, metadata_json, auto_unbox = TRUE, pretty = TRUE, null = "null")

  cat(
    "Saved NNMF K=", args$k, " outputs to ", output_dir,
    " with final GKL ", signif(result$error, 6), ".\n",
    sep = ""
  )
}

run_nnmf(parse_args(commandArgs(trailingOnly = TRUE)))
