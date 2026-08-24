#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(spacexr)
  library(jsonlite)
})

read_lines_gz <- function(path) {
  con <- gzfile(path, open = "rt")
  on.exit(close(con))
  readLines(con)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 9) {
  stop("Usage: run_rctd.R <ref_mtx_gz> <ref_genes_gz> <ref_barcodes_gz> <ref_celltypes_gz> <st_mtx_gz> <st_genes_gz> <st_barcodes_gz> <st_coords_gz> <outdir> [params_json]")
}

ref_mtx       <- args[[1]]
ref_genes     <- args[[2]]
ref_barcodes  <- args[[3]]
ref_celltypes <- args[[4]]
st_mtx        <- args[[5]]
st_genes      <- args[[6]]
st_barcodes   <- args[[7]]
st_coords     <- args[[8]]
outdir        <- args[[9]]
params_json   <- if (length(args) >= 10) args[[10]] else "{}"

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

message("[R] Loading params...")
params <- fromJSON(params_json)

# Make downsampling & other stochastic steps reproducible
set.seed(1)

quiet <- if (!is.null(params$quiet)) isTRUE(params$quiet) else FALSE

# If quiet==TRUE, force single-core to avoid parallel worker "spacexr loaded" spam.
# (This is the only robust way without changing spacexr internals.)
max_cores   <- if (quiet) 1 else if (!is.null(params$max_cores)) params$max_cores else 4
gene_cutoff <- if (!is.null(params$gene_cutoff)) params$gene_cutoff else 0
fc_cutoff   <- if (!is.null(params$fc_cutoff)) params$fc_cutoff else 0
doublets    <- if (!is.null(params$doublets)) params$doublets else TRUE

max_cells_per_type <- if (!is.null(params$max_cells_per_type)) as.integer(params$max_cells_per_type) else 10000
if (is.na(max_cells_per_type) || max_cells_per_type <= 0) max_cells_per_type <- 10000

message("[R] Reading reference matrix...")
ref_counts  <- readMM(gzfile(ref_mtx))
ref_counts  <- as(ref_counts, "CsparseMatrix")  # IMPORTANT: keep sparse in stable format

ref_genes_v <- read_lines_gz(ref_genes)
ref_bcs_v   <- read_lines_gz(ref_barcodes)
ref_cts_v   <- read_lines_gz(ref_celltypes)

if (nrow(ref_counts) != length(ref_genes_v)) stop("ref genes length mismatch with ref matrix nrow")
if (ncol(ref_counts) != length(ref_bcs_v))   stop("ref barcodes length mismatch with ref matrix ncol")
if (length(ref_cts_v) != length(ref_bcs_v))  stop("ref celltypes length mismatch with ref barcodes length")

# set dimnames in one shot (less chance of coercion)
dimnames(ref_counts) <- list(ref_genes_v, ref_bcs_v)

cell_types <- as.factor(ref_cts_v)
names(cell_types) <- ref_bcs_v

# Hard check: cell_types names must match reference matrix columns
stopifnot(all(names(cell_types) == colnames(ref_counts)))
              # IMPORTANT: required by spacexr::Reference()

# Optional: cap reference cells per cell type (common values: 10000 or 5000)
# This avoids the built-in Reference() downsampling warning and gives you a simple knob.
if (max_cells_per_type <= 10000) {
  ct_levels <- levels(cell_types)
  keep_mask <- rep(FALSE, length(cell_types))
  names(keep_mask) <- names(cell_types)

  for (ct in ct_levels) {
    idx <- which(cell_types == ct)
    if (length(idx) <= max_cells_per_type) {
      keep_mask[idx] <- TRUE
    } else {
      keep_mask[sample(idx, max_cells_per_type)] <- TRUE
    }
  }

  keep_idx <- which(keep_mask)
  keep_idx <- sort(keep_idx)  # preserve original column order

  ref_counts <- ref_counts[, keep_idx, drop = FALSE]
  cell_types <- droplevels(cell_types[keep_idx])
  names(cell_types) <- colnames(ref_counts)

  stopifnot(all(names(cell_types) == colnames(ref_counts)))
  message(sprintf("[R] Reference downsampling applied: max_cells_per_type=%d; kept %d cells", max_cells_per_type, ncol(ref_counts)))
}

message("[R] Reading spatial matrix...")
st_counts  <- readMM(gzfile(st_mtx))
st_counts  <- as(st_counts, "CsparseMatrix")    # IMPORTANT

st_genes_v <- read_lines_gz(st_genes)
st_bcs_v   <- read_lines_gz(st_barcodes)

if (nrow(st_counts) != length(st_genes_v)) stop("st genes length mismatch with st matrix nrow")
if (ncol(st_counts) != length(st_bcs_v))   stop("st barcodes length mismatch with st matrix ncol")

dimnames(st_counts) <- list(st_genes_v, st_bcs_v)

# Hard check: gene lists must match exactly (content + order)
if (!identical(ref_genes_v, st_genes_v)) {
  stop('Gene lists differ between reference and spatial input (content/order). This indicates an upstream alignment problem.')
}


message("[R] Reading coords...")
coords_df <- read.table(gzfile(st_coords), header = TRUE, sep = "\t",
                        stringsAsFactors = FALSE, check.names = FALSE)
req_cols <- c("barcode","x","y")
if (!all(req_cols %in% colnames(coords_df))) {
  stop(paste("coords file must contain columns:", paste(req_cols, collapse=", ")))
}

coords_df <- coords_df[match(st_bcs_v, coords_df$barcode), , drop = FALSE]
if (any(is.na(coords_df$barcode))) stop("Some spatial barcodes missing in coords file.")

coords_use <- coords_df[, c("x","y"), drop = FALSE]
coords_use$x <- as.numeric(coords_use$x)
coords_use$y <- as.numeric(coords_use$y)
rownames(coords_use) <- coords_df$barcode

# Hard checks: coords must align 1:1 with spatial matrix columns
stopifnot(all(rownames(coords_use) == colnames(st_counts)))
stopifnot(nrow(coords_use) == ncol(st_counts))


message("[R] Building Reference + SpatialRNA...")
reference  <- Reference(ref_counts, cell_types)
spatialRNA <- SpatialRNA(coords_use, st_counts)

message("[R] Creating RCTD object...")
rctd <- create.RCTD(
  spatialRNA,
  reference,
  max_cores   = max_cores,
  gene_cutoff = gene_cutoff,
  fc_cutoff   = fc_cutoff
)

message("[R] Running RCTD...")
if (isTRUE(doublets)) {
  rctd <- run.RCTD(rctd, doublet_mode = "full")
} else {
  rctd <- run.RCTD(rctd, doublet_mode = "none")
}

w <- rctd@results$weights
if (is.null(w)) stop("rctd@results$weights is NULL; check spacexr output/logs.")

# weights is typically spots x celltypes and not huge; safe to densify here
weights_spot_ct <- as.matrix(w)

# normalize to proportions (use base rowSums on dense small matrix is fine)
row_sums <- rowSums(weights_spot_ct)
row_sums[row_sums == 0] <- 1
props_spot_ct <- weights_spot_ct / row_sums

weights_path <- file.path(outdir, "rctd_weights.csv.gz")
props_path   <- file.path(outdir, "rctd_props.csv.gz")
rds_path     <- file.path(outdir, "rctd_obj.rds")
meta_path    <- file.path(outdir, "rctd_metadata.json")

write.csv(weights_spot_ct, gzfile(weights_path), quote = FALSE)
write.csv(props_spot_ct,   gzfile(props_path),   quote = FALSE)
saveRDS(rctd, rds_path)

meta <- list(
  n_spots = nrow(weights_spot_ct),
  n_celltypes = ncol(weights_spot_ct),
  celltypes = colnames(weights_spot_ct),
  params = params
)
write_json(meta, meta_path, pretty = TRUE, auto_unbox = TRUE)

message("[R] DONE.")