#!/usr/bin/env Rscript

# Convert HDL .rda reference panel files into TSVs consumable by the Rust HDL loader.
# Usage:
#   Rscript convert_hdl_rda_to_tsv.R /path/to/LD.path /path/to/output_dir

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: convert_hdl_rda_to_tsv.R <LD.path> <output_dir>")
}

ld_path <- args[[1]]
out_dir <- args[[2]]

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ld_files <- list.files(ld_path)

snp_counter_file <- ld_files[grep(x = ld_files, pattern = "UKB_snp_counter.*")]
snp_list_file <- ld_files[grep(x = ld_files, pattern = "UKB_snp_list.*")]

if (length(snp_counter_file) == 0 || length(snp_list_file) == 0) {
  stop("Missing UKB_snp_counter.* or UKB_snp_list.* in LD.path")
}

load(file = file.path(ld_path, snp_counter_file))
load(file = file.path(ld_path, snp_list_file))

if (!exists("snps.list.imputed.vector") || !exists("nsnps.list.imputed")) {
  stop("Expected snps.list.imputed.vector and nsnps.list.imputed in the .rda files")
}

snps_name_list <- snps.list.imputed.vector
nsnps_list <- nsnps.list.imputed

write.table(
  data.frame(SNP = snps_name_list),
  file = file.path(out_dir, "snps.list.imputed.vector.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

nsnps_df <- data.frame(
  chr = rep(seq_along(nsnps_list), times = vapply(nsnps_list, length, integer(1))),
  piece = unlist(lapply(nsnps_list, seq_along)),
  nsnps = unlist(nsnps_list)
)

write.table(
  nsnps_df,
  file = file.path(out_dir, "nsnps.list.imputed.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

for (chr in 1:22) {
  k <- length(nsnps_list[[chr]])
  for (piece in 1:k) {
    rda_file <- ld_files[grep(x = ld_files, pattern = paste0("chr", chr, "\.", piece, ".*rda"))]
    bim_file <- ld_files[grep(x = ld_files, pattern = paste0("chr", chr, "\.", piece, ".*bim"))]
    if (length(rda_file) == 0 || length(bim_file) == 0) {
      warning(sprintf("Missing rda/bim for chr %d piece %d", chr, piece))
      next
    }

    load(file = file.path(ld_path, rda_file))

    if (!exists("LDsc") || !exists("lam") || !exists("V")) {
      stop(sprintf("Missing LDsc/lam/V in %s", rda_file))
    }

    prefix <- sprintf("chr%d.%d", chr, piece)

    write.table(
      data.frame(LDsc = LDsc),
      file = file.path(out_dir, paste0(prefix, ".ldsc.tsv")),
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )

    write.table(
      data.frame(lam = lam),
      file = file.path(out_dir, paste0(prefix, ".lam.tsv")),
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )

    write.table(
      V,
      file = file.path(out_dir, paste0(prefix, ".v.tsv")),
      sep = "\t",
      row.names = FALSE,
      col.names = FALSE,
      quote = FALSE
    )

    snps_ref <- read.table(file.path(ld_path, bim_file), stringsAsFactors = FALSE)
    colnames(snps_ref) <- c("chr", "id", "non", "pos", "A1", "A2")
    write.table(
      snps_ref,
      file = file.path(out_dir, paste0(prefix, ".bim.tsv")),
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )
  }
}

cat("Conversion complete. Files written to", out_dir, "\n")
