# GenomicSEM Rust Rewrite Progress

Date: 2026-02-27

## Completed
- Created `genomicsem` crate and module skeleton.
- Flattened Rust module layout to one `src/` directory with per-module files (no `mod.rs` folders).
- Implemented schema mapping and sanity checks mirroring R utils.
- Implemented `munge` core pipeline with logs, P-value sanity checks, allele filters, and `.sumstats.gz` output.
- Added compressed input support (`.gz`, `.bz2`) with streaming decompression to temp files.
- Added quoted-field parsing for whitespace-delimited inputs.
- Implemented `sumstats` core pipeline with transformations for OLS / linprob / logistic, DIRECTION filtering, duplicate SNP removal, MAF handling, INFO/MAF filtering, and merge across traits.
- Added sumstats output file writing (`*_sumstats.tsv`).
- Added CLI with `munge` and `sumstats` subcommands using clap.
- Tightened sumstats parity: ref-only MAF handling, linprob standardization, OLS SE computation, and NA/zero filtering.
- Added parallel sumstats execution with per-trait logs and optional thread count.
- Added NA parsing for CSV and whitespace readers.
- Implemented initial LDSC input parsing (LD scores, weights, M files, trait merging, chi-square filtering).
- Implemented LDSC regression + block jackknife with liability scaling and standardized outputs (when `stand`).
- Added LDSC JSON serialization (`*_ldsc.json` or `--output`).
- Added LDSC trait name handling and warnings (n.blocks > 1000, trait name hyphens).
- Added CSV string trimming after read to match R whitespace handling.
- Added commonfactor/usermodel SEM estimation for non‑SNP models (modelfit + results, standardized fit).
- Added commonfactorGWAS data preparation (checks, beta/se extraction, SNP variance).
- Added per-SNP S/V expansion with GC adjustments and a parallel scaffold (model fit still TODO).
- Added userGWAS scaffold with per-SNP expansion and lavaan TODO placeholders.
- Added finite intercept coordinate handling and cores capping warnings for GWAS scaffolds.
- Added SEM engine interface (SemEngine) with placeholder implementation.
- Implemented stratified LD-score regression (`s_ldsc`) pipeline with annotation parsing, weight construction, block jackknife, and output assembly.
- Added `s-ldsc` CLI subcommand and stratified LDSC JSON IO.
- Implemented `read_fusion` (T-SEM input prep) with binary/continuous handling and permutation support.
- Implemented HDL piecewise pipeline (optimizer stubbed; jackknife mode stubbed).
- Implemented GTACCC utilities: `indexS`, `subSV`, `summaryGLS`, `summaryGLSbands` (Plotly HTML plotting).
- Implemented `paLDSC` (non-FA) with MVN sampling and diagonal option (Plotly HTML scree plots).
- Implemented `localSRMD`.
- Implemented `simLDSC` (matrix input only; model-string input stubbed), with robust MVN fallback.
- Added `qtrait` stub (SEM + plotting).
- Added CLI parity for wiki workflows:
  - `read-fusion`, `hdl`, `pa-ldsc`, `index-s`, `sub-sv`, `summary-gls`, `summary-gls-bands`, `local-srmd`, `sim-ldsc`, `qtrait` (stub).
- Added `post_ldsc` and `sim_ldsc` modules to crate exports.
- Updated `README.md` (Rust) to mirror original README and list implemented vs stubbed functionality.
- Refactored common DataFrame, logging, matrix, and IO utilities into shared modules (`df_utils.rs`, `logging.rs`, `matrix.rs`, `io.rs`) and updated call sites.
- Implemented paLDSC FA mode (minres-style FA with NLopt L-BFGS + numerical gradients), including diagonalized FA PA path.
- Implemented HDL optimizer (llfun + llfun.gcov.part.2) using NLopt L-BFGS with bounds and numerical gradients.
- Implemented HDL jackknife mode and parallel leave-one-piece-out runs via rayon.
- Implemented TWAS mode in GWAS pipelines (uses `HSQ` variance, `Gene/Panel` metadata, and read_fusion inputs).
- Added GWAS output parity tables (prep metadata + results) for commonfactorGWAS and userGWAS.
- Added CLI parity for `commonfactorGWAS` and `userGWAS`, including covstruc JSON reader and model file support.
- Implemented userGWAS `sub` filtering with multi-parameter outputs and per-sub TSVs (CLI list-style parity).
- Converted `genomicsem` into a workspace and added internal `lavaan` crate (Rust SEM engine).
- Implemented SEM engine in `lavaan` (DWLS/ML objectives, LISREL implied covariance, sandwich SEs, standardization, fit stats).
- Implemented chumsky Pratt parser for lavaan-style syntax and defined parameters.
- Implemented parameter table output and ordering aligned with lavaan; defined parameters supported.
- Implemented H1 expected information for ML and lavaan-compatible vech ordering.
- Implemented DWLS sandwich weighting (diagonal WLS.V) and Q-based fit stats (GenomicSEM parity).
- Implemented `fix_measurement`/Q_SNP support paths via ParTable freeze in GWAS pipelines.
- Implemented `rgmodel` (returns genetic correlation matrix + sampling covariance via SEM fit).
- Implemented `QTrait` core logic (CPM/IPM/FUM fits, heterogeneity tests, lSRMR, outlier loop) and CLI output.
- Added SEM parity harness (R vs Rust) and aligned it to GenomicSEM’s usage.
- Added tracing instrumentation for SEM fit stats debugging (optional).
- Implemented `enrich` (Stratified Genomic SEM step 3) with baseline + per-annotation fits and CLI integration.

## Pending / Known Gaps (Stubs)
- Not planned: `QFactor` in the Rust rewrite.
- Not planned: `simLDSC` model-string input.
- Not planned: ggplot/ggpubr PDF parity; plotting is Plotly HTML (QTrait, summaryGLSbands, paLDSC).
- Not planned: HDL `.rda` reference panels; use `genomicsem/scripts/convert_hdl_rda_to_tsv.R` to convert to TSV inputs.
- Not planned: legacy helpers not used in current workflows (`addSNPs`, `addGenes`, `multiSNP`, `multiGene`, `write.model`); `addSNPs` is deprecated in R.
- summaryGLSbands fit line now spans user axis range (matching ggplot stat_function + coord_cartesian behavior).

## Important Files
- `genomicsem/src/munge.rs`
- `genomicsem/src/sumstats.rs`
- `genomicsem/src/ldsc.rs`
- `genomicsem/src/stratified.rs`
- `genomicsem/src/twas.rs`
- `genomicsem/src/hdl.rs`
- `genomicsem/src/post_ldsc.rs`
- `genomicsem/src/sim_ldsc.rs`
- `genomicsem/src/qtrait.rs`
- `genomicsem/src/bin/genomicsem.rs`
- `genomicsem/README.md`
- `genomicsem/lavaan/src/*`

## Notes
- Stubs are hard `todo!()` per user request.
- SEM optimizer uses NLopt `Slsqp` (bound-aware); lavaan R uses `nlminb` (PORT routines).
- SEM chisq uses Q-based statistic for DWLS/ML to match GenomicSEM outputs (differs from lavaan R ML).
- Identifier parsing matches lavaan (no `-` or `:` in names; start with `[_.A-Za-z]`).
- `paLDSC` eigenvalues sorted descending to match R.
- `simLDSC` filenames match R (`iter{r}GWAS{i}.sumstats`).
- Random sampling uses `rand` + `rand_distr`; MVN fallback uses eigen-decomposition if Cholesky fails.
- paLDSC FA uses NLopt L-BFGS with numerical gradients and minres objective (off-diagonal residuals); smoothing uses eigenvalue clipping on correlation matrices.
- HDL optimizer uses NLopt L-BFGS with bounds; gcov jackknife includes fallback starting values (0, ±0.5*sqrt(h11*h22)).

## Potential Refactor: Third‑Party Crate Shell‑Out Plan (Not Implemented)
- Linear algebra helpers (WLS/Gamma/H1): ~180–280 LoC reduction; higher parity risk (exact eigen/threshold behavior).
- SEM parser: ~200–350 LoC reduction; medium parity risk (grammar edge cases).
- Matrix conversion helpers: ~40–80 LoC reduction; low parity risk if logic unchanged.
- Polars column casting helpers: ~40–70 LoC reduction; medium parity risk (lazy/eager and NA coercion).
- Plot helpers (min/max/path): ~10–25 LoC reduction; low parity risk.
- Thread pool helpers: ~20–40 LoC reduction; low parity risk (per‑call pool control).
- Model file reader helper: ~15–25 LoC reduction; low parity risk.

## Adversarial Parity Checklist (R vs Rust)

### LDSC
- [ ] Not planned: Log output parity (R logs liability-scale results + genetic correlation results; Rust logs are abbreviated).
- [ ] Not planned: Output objects: R optionally returns `S_Stand/V_Stand` only when `stand=TRUE`; Rust always computes `s_stand/v_stand` when `stand` and diag positive (matches behavior but JSON output differs from R list layout).
- [ ] Not planned: R uses merged$weights in covariance block (but is undefined in R); Rust explicitly uses trait-1 weights to keep deterministic behavior. Documented in code.

### Stratified LDSC (s_ldsc)
- [ ] Not planned: Lower-triangle ordering: Rust uses row-major lower triangle for `s_ldsc` V scaling; R uses gdata::lowerTriangle column-major. Rust notes ordering assumption; needs explicit parity check with R if V reshaping differs.
- [x] Single-annotation handling now errors to match R behavior.
- [ ] Not planned: Log output parity vs R (abbreviated in Rust).

### HDL
- [x] Optimizer and jackknife mode implemented via NLopt L-BFGS with bounds; R uses `optim` (L-BFGS-B).
- [ ] Not planned: LD reference input format mismatch: R expects HDL `.rda` + `UKB_snp_counter` files; Rust expects `.bim`, `.ldsc`, `.lam`, `.v` per piece.

### read_fusion (T-SEM Step 3)
- [x] Rust warns when `binary` not provided (assumes all binary).
- [ ] Not planned: Output columns: Rust matches `Panel/Gene/beta.* / se.*` layout; verify `HSQ` placement parity.

### post_ldsc utilities (GTACCC)
- [x] `summaryGLSbands` Plotly HTML plotting implemented (CLI helper).
- [x] `paLDSC` FA mode implemented; uses minres-style FA with smoothing and NLopt.
- [ ] Not planned: `indexS`/`subSV` rely on column-major vech; Rust uses column-major in `post_ldsc`, but s_ldsc uses row-major—verify consistency across workflows.
- [x] `subSV` TYPE=R now accepted but uses S_Stand/V_Stand proxy (R/V_R not implemented); CLI warns when used.
- [x] `summaryGLSbands` now keeps additional predictors unchanged when shifting the first predictor (closer to R behavior with CONTROLVARS/QUAD).

### simLDSC
- [ ] Not planned: Model-string `covmat` input (lavaan) stubbed in Rust.
- [x] MHC region filtering implemented when `CHR`/`BP` are present (chr6: 25–34Mb).
- [ ] Not planned: Parallel behavior: R uses `doParallel`; Rust uses rayon and per-iteration RNG seeding (deterministic).
- [ ] Not planned: If LD scores omit `CHR`/`BP`, Rust skips MHC filtering; R may filter earlier in pipeline.
- [ ] Not planned: Rust assumes `L2` column name in LD scores; R uses `L2` but some datasets may differ in casing.

### CLI Parity
- [ ] Not planned: CLI file inputs assume TSV/CSV with headers; R accepts raw matrices in memory. Need doc on required headers for matrix/vector inputs.
- [ ] Not planned: Some R workflows are function-only (e.g., QTrait/QFactor); Rust CLI stubs these to hard-fail.

### IO / Schema / Utils / Logging
- [ ] Not planned: `io.rs` uses Polars parsing with `ignore_errors=true`; R `read.table` does not silently ignore parse errors. Potential row drops/NA differences.
- [ ] Not planned: `io.rs` decompresses `.gz/.bz2` to temp file (not true streaming into Polars); R reads via read.table/readr directly.
- [ ] Not planned: `io.rs` whitespace parser supports quotes; still may diverge from R `read.table(..., quote="\"")` on edge cases.
- [ ] Not planned: `schema.rs` mirrors `.get_renamed_colnames`, but R logs and warnings sometimes both warn + log; Rust logs only (no warnings).
- [ ] Not planned: `schema.rs` warning behavior for `Z` column when `warn_z_as_effect` differs from R (R logs + warning).
- [ ] Not planned: `utils.rs` contains only strand ambiguity check; many R helpers (e.g., `.rearrange`, `.tryCatch.W.E`) are not in Rust because SEM is stubbed.
- [ ] Not planned: `logging.rs` uses `tracing` vs R `.LOG` (log text parity intentionally not exact).

### munge.rs
- [ ] Not planned: Parallel branch disabled in Rust (only serial); R supports parallel + per‑file logs.
- [ ] Not planned: R logs overwrite behavior when files exist; Rust only warns on overwrite flag at write time.

### sumstats.rs
- [ ] Not planned: R supports list input (deprecated) and logs more warnings; Rust requires vec and logs are shorter.
- [ ] Not planned: R uses `inner_join` (dplyr) semantics; Rust uses Polars join (should match but check NA handling).

### gwas.rs (commonfactorGWAS/userGWAS)
- [x] SEM estimation implemented via internal `lavaan` crate.
- [x] `fix_measurement` path (measurement model fixing) implemented via ParTable freeze.
- [x] `sub` argument handling for userGWAS implemented (filtering + per-sub outputs).
- [ ] Not planned: Model/SEM rearrange ordering (`.rearrange`) not implemented.
- [x] TWAS mode implemented (HSQ variance + Gene/Panel metadata).
- [x] Output parity tables for TWAS and non-TWAS (SNPs[,1:6] vs HSQ/Panel/Gene) implemented.

### sem.rs / rgmodel.rs / enrich
- [x] SEM engine implemented (DWLS/ML, sandwich SEs, std.all, fit stats).
- [x] `rgmodel` implemented (genetic correlation model + V_R).
- [ ] Not planned: `enrich` does not apply `toler` to the sandwich bread inversion (ndarray-linalg lacks tol).
- [ ] Not planned: `enrich` fallback model is simplified (residual bounds only; no R-style write.Model1).

### SEM / lavaan parity
- [x] DWLS/ML estimation and sandwich SEs implemented; validated via parity harness.
- [x] Fit stats are GenomicSEM Q-based (differs from lavaan ML chisq).
- [x] Gamma inverse uses raw eigen inverse (no threshold) to match R (can produce Inf if singular).
- [ ] Not planned: Constraint handling differs from lavaan (SLSQP vs nlminb); bounds enforced, but performance may differ.

### twas.rs (read_fusion)
- [ ] Not planned: R prints warnings about assumed binary when `binary=NULL`; Rust requires explicit or defaults to true without warning.
- [ ] Not planned: `Panel` extraction uses path split; verify exact regex parity with R `sub(".*//([^/]+/[^/]+)$|.*/([^/]+/[^/]+)$", ...)`.

### hdl.rs
- [ ] Not planned: Optimizer/jackknife stubbed; only piecewise pipeline works up to WLS start values.
- [ ] Not planned: LD reference input format differs from R HDL (`.rda`/UKB_snp_counter` vs `.bim/.ldsc/.lam/.v`).
- [x] Rust now detects `.rda` reference panels and hard-fails with TODO (documented).

### ldsc.rs
- [x] Rust logs now include liability-scale and genetic correlation sections (text may differ).
- [ ] Not planned: R uses `warning()` side effects; Rust only logs warnings via tracing.
- [x] Rust preserves input order (no CHR/BP sort) to match `merge(..., sort=FALSE)`.
- [ ] Not planned: Trait list in log includes file paths in R; Rust logs only counts/traits.
- [ ] Not planned: R uses `S_Stand`/`V_Stand` list output; Rust writes JSON (CLI) and returns struct (library).

### post_ldsc.rs
- [x] `summaryGLSbands` Plotly HTML plotting implemented (CLI helper).
- [x] `paLDSC` Plotly HTML scree plots implemented (CLI helper).

### sim_ldsc.rs
- [ ] Not planned: `covmat` model-string input (lavaan) stubbed.
- [x] MHC region filtering implemented (chr6: 25–34Mb) when `CHR`/`BP` present.
- [ ] Not planned: Output files are always written to CWD (R uses `gzip` and can be controlled by `R.utils` behavior).
- [ ] Not planned: R writes `.sumstats` then gzips; Rust writes `.sumstats.gz` directly (no uncompressed file).
- [ ] Not planned: R uses `readr` and preserves column types; Rust uses Polars and may coerce types differently when writing output.

### qtrait.rs
- [x] Core QTrait logic implemented (CPM/IPM/FUM + outlier loop).
- [x] Plotting is implemented via Plotly HTML; ggplot/ggpubr PDF parity not available.

### bin/genomicsem.rs
- [ ] Not planned: CLI requires TSV/CSV inputs with headers for matrices/vectors; R accepts in-memory matrices.
- [ ] Not planned: CLI writes TSV/JSON instead of in-memory list return values; docs need to describe outputs.
- [x] `sub-sv` CLI now uses `--ty` to warn about expected matrix type (S_Stand/R not implemented).

### types.rs / error.rs / logging.rs / lib.rs
- [ ] Not planned: Rust-only infrastructure (no direct R analog).
