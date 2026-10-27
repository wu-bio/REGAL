#!/usr/bin/env Rscript
# Select NNMF rank by the paper's AIC, using only counts and coordinates.
argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) != 2L) stop("Usage: run_nnmf_aic.R INPUT_H5AD OUTPUT_DIR")
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
entry <- file.path(dirname(script_path), "run_nnmf.R")
expressions <- parse(entry)
stopifnot(identical(expressions[[length(expressions)]],
                    quote(run_nnmf(parse_args(commandArgs(trailingOnly = TRUE))))))
for (expr in expressions[-length(expressions)]) eval(expr, envir = .GlobalEnv)
args <- parse_args(c("--input", argv[[1]], "--output-dir", argv[[2]], "--k", "2"))
if (dir.exists(args$output_dir)) stop("Refusing to overwrite output: ", args$output_dir)
for (pkg in c("NNMF", "Matrix", "reticulate", "jsonlite")) require_package(pkg)
configure_reticulate_python()
cat("R runtime:", R.home(), "\nPython:", Sys.getenv("RETICULATE_PYTHON"), "\n")
prepared <- prepare_nnmf_input(read_h5ad_input(args$input, args$spatial_key), args)
dir.create(args$output_dir, recursive = TRUE)
writeLines(prepared$genes, file.path(args$output_dir, "analyzed_genes.txt"))
cat("Estimating length scale on", nrow(prepared$counts), "spots and",
    ncol(prepared$counts), "genes.\n")
ls_grid <- NNMF::estimate_lengthscale(prepared$counts, prepared$coordinates, max_avg_nn = 20)
write.csv(ls_grid, file.path(args$output_dir, "lengthscale_selection.csv"), row.names = FALSE)
args$lengthscale <- ls_grid$lengthscale[[which.min(ls_grid$test_error)]]
cat("Selected lengthscale:", args$lengthscale, "\n")
output_root <- args$output_dir
scan <- data.frame()
fit_rank <- function(k) {
  if (nrow(scan) && k %in% scan$k) return(invisible(NULL))
  cat("\n=== Fitting K=", k, " ===\n", sep = "")
  run_args <- args
  run_args$k <- as.integer(k)
  run_args$output_dir <- file.path(output_root, paste0("K", k))
  started <- Sys.time()
  run_nnmf(run_args)
  meta <- jsonlite::read_json(file.path(run_args$output_dir, "run_metadata.json"), simplifyVector = TRUE)
  stopifnot(meta$n_spots_analyzed == nrow(prepared$counts),
            meta$n_genes_analyzed == ncol(prepared$counts))
  # Paper: n_prm counts all entries of W (N x K) and H (K x G).
  row <- data.frame(k = k, seed = args$seed, final_gkl = meta$final_gkl,
                    n_parameters = k * (meta$n_spots_analyzed + meta$n_genes_analyzed),
                    aic = 2 * meta$final_gkl + 2 * k * (meta$n_spots_analyzed + meta$n_genes_analyzed),
                    lengthscale = args$lengthscale,
                    elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")))
  scan <<- rbind(scan, row)
  scan <<- scan[order(scan$k), ]
  write.csv(scan, file.path(output_root, "parameter_selection.csv"), row.names = FALSE)
  cat("K=", k, " AIC=", format(row$aic, digits = 12), "\n", sep = "")
}
for (k in seq(20L, 30L, 2L)) fit_rank(k)
repeat {
  best <- scan$k[[which.min(scan$aic)]]
  if (best == max(scan$k)) {
    for (k in seq(max(scan$k) + 2L, max(scan$k) + 10L, 2L)) fit_rank(k)
  } else break
}
best <- scan$k[[which.min(scan$aic)]]
scan$selected_formal_run <- scan$k == best
write.csv(scan, file.path(output_root, "parameter_selection.csv"), row.names = FALSE)
jsonlite::write_json(list(
  selection = "Minimum paper AIC: 2 * GKL + 2 * K * (n_spots + n_genes)",
  selected_k = best, selected_output = file.path(output_root, paste0("K", best)),
  seed = args$seed, lengthscale = args$lengthscale,
  n_spots = nrow(prepared$counts), n_genes = ncol(prepared$counts),
  input = normalizePath(args$input), conda_prefix = Sys.getenv("CONDA_PREFIX"),
  r_home = R.home(), source_commit = NNMF_SOURCE_COMMIT,
  truth_used_for_selection = FALSE
), file.path(output_root, "selection.json"), auto_unbox = TRUE, pretty = TRUE)
cat("\nAIC-selected K:", best, "\n")
