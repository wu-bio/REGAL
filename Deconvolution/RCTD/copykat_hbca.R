#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(copykat)
  library(jsonlite)
})

read_lines_gz <- function(path) {
  con <- gzfile(path, open = "rt")
  on.exit(close(con))
  readLines(con)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop("Usage: copykat_hbca.R <obs_mtx> <obs_genes> <obs_barcodes> <ref_mtx> <ref_genes> <ref_barcodes> <outdir> <params_json>")
}

obs_mtx <- args[[1]]
obs_genes <- args[[2]]
obs_barcodes <- args[[3]]
ref_mtx <- args[[4]]
ref_genes <- args[[5]]
ref_barcodes <- args[[6]]
outdir <- args[[7]]
params_json <- args[[8]]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
outdir <- normalizePath(outdir, mustWork = TRUE)

params <- fromJSON(params_json)
sample_name <- if (!is.null(params$sample_name)) params$sample_name else "copykat_hbca"
id_type <- if (!is.null(params$id_type)) params$id_type else "S"
n_cores <- if (!is.null(params$n_cores)) as.integer(params$n_cores) else 1
ngene_chr <- if (!is.null(params$ngene_chr)) as.integer(params$ngene_chr) else 1
win_size <- if (!is.null(params$win_size)) as.integer(params$win_size) else 25
ks_cut <- if (!is.null(params$ks_cut)) as.numeric(params$ks_cut) else 0.2
force_rerun <- if (!is.null(params$force_rerun)) isTRUE(params$force_rerun) else FALSE

md5_text <- function(x) {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeLines(as.character(x), tmp, useBytes = TRUE)
  tools::md5sum(tmp)[[1]]
}

message("[R] Reading observed matrix...")
obs <- as(readMM(gzfile(obs_mtx)), "CsparseMatrix")
obs_gene_v <- read_lines_gz(obs_genes)
obs_bc_v <- read_lines_gz(obs_barcodes)
stopifnot(nrow(obs) == length(obs_gene_v), ncol(obs) == length(obs_bc_v))
dimnames(obs) <- list(obs_gene_v, obs_bc_v)

message("[R] Reading HBCA reference matrix...")
ref <- as(readMM(gzfile(ref_mtx)), "CsparseMatrix")
ref_gene_v <- read_lines_gz(ref_genes)
ref_bc_v <- read_lines_gz(ref_barcodes)
stopifnot(nrow(ref) == length(ref_gene_v), ncol(ref) == length(ref_bc_v))
dimnames(ref) <- list(ref_gene_v, ref_bc_v)

genes <- intersect(rownames(obs), rownames(ref))
if (length(genes) < 1000) stop(sprintf("Too few intersecting genes: %d", length(genes)))
obs <- obs[genes, , drop = FALSE]
ref <- ref[genes, , drop = FALSE]
colnames(ref) <- paste0("hbca_ref_", seq_len(ncol(ref)))
mat <- cbind(obs, ref)
mat <- as.matrix(mat)

message(sprintf("[R] Running CopyKAT: %d genes x %d observed + %d HBCA ref cells",
                nrow(obs), ncol(obs), ncol(ref)))
wd_orig <- getwd()
setwd(outdir)
on.exit(setwd(wd_orig), add = TRUE)

copykat_rds <- file.path(outdir, "copykat_hbca.rds")
fingerprint_path <- file.path(outdir, "copykat_input_fingerprint.json")
fingerprint <- list(
  obs_barcodes_md5 = md5_text(obs_bc_v),
  ref_barcodes_md5 = md5_text(ref_bc_v),
  intersect_genes_md5 = md5_text(genes),
  n_observed = ncol(obs),
  n_hbca_ref = ncol(ref),
  n_genes = nrow(obs),
  sample_name = sample_name,
  id_type = id_type,
  ngene_chr = ngene_chr,
  win_size = win_size,
  ks_cut = ks_cut
)

if (file.exists(copykat_rds) && file.exists(fingerprint_path) && !force_rerun) {
  old_fingerprint <- fromJSON(fingerprint_path, simplifyVector = FALSE)
  if (!identical(old_fingerprint, fingerprint)) {
    stop(
      "Existing copykat_hbca.rds was generated from different inputs/parameters. ",
      "Use --force-rerun or a new outdir."
    )
  }
}

if (file.exists(copykat_rds) && !force_rerun) {
  message("[R] Loading existing CopyKAT object: ", copykat_rds)
  cpkt <- readRDS(copykat_rds)
} else {
  if (file.exists(copykat_rds) && force_rerun) {
    message("[R] force_rerun=TRUE; ignoring existing CopyKAT object.")
  }
  cpkt <- copykat(
    mat,
    id.type = id_type,
    cell.line = "no",
    ngene.chr = ngene_chr,
    win.size = win_size,
    KS.cut = ks_cut,
    sam.name = sample_name,
    plot.genes = FALSE,
    n.cores = n_cores
  )
  saveRDS(cpkt, copykat_rds)
  write_json(fingerprint, fingerprint_path, pretty = TRUE, auto_unbox = TRUE)
}

prediction <- as.data.frame(cpkt$prediction)
write.csv(prediction, gzfile(file.path(outdir, "copykat_prediction_raw.csv.gz")), quote = FALSE)

prediction <- prediction[!grepl("^X", rownames(prediction)), , drop = FALSE]
default <- setNames(as.character(prediction$copykat.pred), prediction$cell.names)
default <- default[obs_bc_v]
default[is.na(default)] <- "Unknown"

decide_aneuploid <- function(X, idx_ref, idx_obs) {
  mat_obs <- X[, idx_obs, drop = FALSE]
  mat_ref <- X[, idx_ref, drop = FALSE]

  cnv_score_ref <- sqrt(colSums(mat_ref^2, na.rm = TRUE))
  cnv_score_obs <- sqrt(colSums(mat_obs^2, na.rm = TRUE))
  norm_window <- quantile(cnv_score_ref, c(0.005, 0.995), na.rm = TRUE)
  denom <- diff(norm_window)
  if (is.na(denom) || denom == 0) denom <- 1
  cnv_score_ref <- (cnv_score_ref - norm_window[1]) / denom
  cnv_score_obs <- (cnv_score_obs - norm_window[1]) / denom

  mat_ref[mat_ref == log2(1)] <- log2(1 + 1e-4)
  mat_obs[mat_obs == log2(1)] <- log2(1 + 1e-4)
  idx_top_ref <- head(order(cnv_score_ref, decreasing = FALSE), max(1, ceiling(0.01 * ncol(mat_ref))))
  idx_top_obs <- head(order(cnv_score_obs, decreasing = TRUE), max(1, ceiling(0.01 * ncol(mat_obs))))
  model_ref <- rowMeans(mat_ref[, idx_top_ref, drop = FALSE], na.rm = TRUE)
  model_obs <- rowMeans(mat_obs[, idx_top_obs, drop = FALSE], na.rm = TRUE)
  model_ref[is.na(model_ref)] <- 0
  model_obs[is.na(model_obs)] <- 0

  cor_ref_obs <- apply(mat_ref, 2, function(v) cor(v, model_obs, method = "pearson"))
  cnvcor_on_obs <- apply(mat_obs, 2, function(v) cor(v, model_obs, method = "pearson"))

  score_threshold <- quantile(cnv_score_ref, 0.99, na.rm = TRUE)
  if (is.na(score_threshold) || score_threshold < 1) score_threshold <- 1
  cor_threshold <- quantile(cor_ref_obs, 0.99, na.rm = TRUE)
  if (is.na(cor_threshold)) cor_threshold <- 0.5

  is_aneuploid <- cnv_score_obs > score_threshold & cnvcor_on_obs > cor_threshold
  is_aneuploid[is.na(is_aneuploid)] <- FALSE
  list(
    is_aneuploid = is_aneuploid,
    cnv_score = cnv_score_obs,
    cnv_cor = cnvcor_on_obs,
    cnv_score_ref = cnv_score_ref,
    cnv_cor_ref = cor_ref_obs,
    cnv_score_threshold = score_threshold,
    cnv_cor_threshold = cor_threshold
  )
}

mat_cpkt <- as.matrix(cpkt$CNAmat[, 4:ncol(cpkt$CNAmat), drop = FALSE])
prediction_in_cna <- make.names(rownames(prediction)) %in% colnames(mat_cpkt)
if (any(!prediction_in_cna)) {
  message(sprintf(
    "[R] Dropping %d prediction rows absent from CNAmat before Tirosh scoring.",
    sum(!prediction_in_cna)
  ))
}
idx_obs <- match(make.names(obs_bc_v), colnames(mat_cpkt))
idx_ref <- grep("^hbca_ref_", colnames(mat_cpkt))
keep_obs <- which(!is.na(idx_obs))
if (length(idx_ref) == 0) stop("No HBCA reference columns found in CopyKAT CNAmat.")
if (length(keep_obs) == 0) stop("No observed barcodes matched CopyKAT CNAmat.")

tirosh <- decide_aneuploid(mat_cpkt, idx_ref, idx_obs[keep_obs])
saveRDS(tirosh, file.path(outdir, "copykat_tirosh_intermediates.rds"))

tirosh_pred <- rep("Unknown", length(obs_bc_v))
names(tirosh_pred) <- obs_bc_v
tirosh_pred[keep_obs] <- ifelse(tirosh$is_aneuploid, "aneuploid", "diploid")

report <- data.frame(
  barcode = obs_bc_v,
  copykat_pred_default = unname(default),
  copykat_pred_tirosh = unname(tirosh_pred),
  cnv_score = NA_real_,
  cnv_cor = NA_real_,
  stringsAsFactors = FALSE
)
report$cnv_score[keep_obs] <- unname(tirosh$cnv_score)
report$cnv_cor[keep_obs] <- unname(tirosh$cnv_cor)
write.csv(report, gzfile(file.path(outdir, "copykat_pred_report.csv.gz")), row.names = FALSE, quote = FALSE)

summary <- list(
  n_observed = ncol(obs),
  n_hbca_ref = ncol(ref),
  n_genes = nrow(obs),
  copykat_pred_default = as.list(table(report$copykat_pred_default, useNA = "ifany")),
  copykat_pred_tirosh = as.list(table(report$copykat_pred_tirosh, useNA = "ifany")),
  cnv_score_threshold = tirosh$cnv_score_threshold,
  cnv_cor_threshold = tirosh$cnv_cor_threshold,
  params = params
)
write_json(summary, file.path(outdir, "copykat_summary.json"), pretty = TRUE, auto_unbox = TRUE)
message("[R] DONE.")
