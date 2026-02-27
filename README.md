# GenomicSEM (Rust Rewrite)

This repository is a Rust rewrite of the GenomicSEM R package. It mirrors the R API through a CLI and internal library modules, and it includes a Rust `lavaan`-style SEM engine scoped to GenomicSEM’s needs.

The rewrite is built as a Cargo workspace:
- `genomicsem/` — the GenomicSEM Rust crate (CLI + library)
- `genomicsem/lavaan/` — internal SEM engine crate (DWLS/ML, parser, fit stats, SEs)

## Installation

### Prerequisites
- Rust toolchain (stable)
- BLAS/LAPACK (default: system OpenBLAS via pkg-config). On macOS, Accelerate works; on Linux, OpenBLAS is typical.
- `gfortran` (required by `ndarray-linalg` / BLAS linkage)

### Build
```bash
cd genomicsem
cargo build --release
```

To build with OpenBLAS from source instead:
```bash
cargo build --release --no-default-features --features blas-openblas-static
```

### Run
```bash
./target/release/genomicsem --help
```

## User Guide (For GenomicSEM R Users)

The Rust rewrite exposes the same workflows as the R package via CLI subcommands. Inputs are files, and outputs are TSV/JSON. The goal is functional parity for the workflows GenomicSEM users actually run.

### R function to Rust CLI mapping

| R function | Rust CLI | Output |
| --- | --- | --- |
| `munge()` | `genomicsem munge` | `.sumstats.gz` per trait |
| `sumstats()` | `genomicsem sumstats` | merged `*_sumstats.tsv` |
| `ldsc()` | `genomicsem ldsc` | JSON + log |
| `s_ldsc()` | `genomicsem s-ldsc` | JSON + log |
| `read_fusion()` | `genomicsem read-fusion` | merged TWAS table |
| `commonfactorGWAS()` | `genomicsem commonfactor-gwas` | `commonfactorGWAS.tsv` |
| `userGWAS()` | `genomicsem user-gwas` | `userGWAS.tsv` or per‑sub TSVs |
| `commonfactor()` | `genomicsem commonfactor` | `*_modelfit.tsv`, `*_results.tsv` |
| `usermodel()` | `genomicsem usermodel` | `*_modelfit.tsv`, `*_results.tsv` |
| `rgmodel()` | `genomicsem rgmodel` | `rgmodel.tsv` |
| `enrich()` | `genomicsem enrich` | `enrich*.tsv` |
| `paLDSC()` | `genomicsem pa-ldsc` | scree plots + tables |
| `summaryGLS()` | `genomicsem summary-gls` | GLS table |
| `summaryGLSbands()` | `genomicsem summary-gls-bands` | GLS bands + plots |
| `localSRMD()` | `genomicsem local-srmd` | residual diagnostics |
| `simLDSC()` | `genomicsem sim-ldsc` | simulated sumstats |
| `QTrait()` | `genomicsem qtrait` | QTrait results + plots |

### Input conventions
- Matrix inputs: TSV/CSV with headers, numeric columns.
- Vector inputs: first column used.
- List args: comma‑separated (`--trait-names A,B,C`).
- Model inputs: `--model "..."` or `--model-file path`.

### Output conventions
- Tables are TSV (`.tsv`) with R‑like column names.
- Logs are emitted via `tracing` (set `RUST_LOG=info` to see progress).

## Wiki Workflow Summary (R vs Rust)

This mirrors the GenomicSEM wiki workflow with CLI equivalents. The typical end‑to‑end flow is:

1. **Munge**: `genomicsem munge`
2. **Sumstats merge**: `genomicsem sumstats`
3. **LDSC**: `genomicsem ldsc`
4. **Stratified LDSC (optional)**: `genomicsem s-ldsc`
5. **SEM model fit (non‑SNP)**: `genomicsem commonfactor` or `genomicsem usermodel`
6. **GWAS SEM**: `genomicsem commonfactor-gwas` / `genomicsem user-gwas`
7. **T‑SEM prep**: `genomicsem read-fusion`
8. **Post‑LDSC utilities**: `summary-gls`, `summary-gls-bands`, `pa-ldsc`, `local-srmd`, `sim-ldsc`
9. **QTrait**: `genomicsem qtrait`

### Changes vs R
- The Rust rewrite is file‑first: CLI inputs are tables on disk, not in‑memory R objects.
- Outputs are TSV/JSON rather than R lists.
- Plotting uses Plotly HTML output (no ggplot/ggpubr PDF parity).
- SEM estimation is internal (`lavaan` crate) and follows lavaan semantics needed by GenomicSEM.

## Not‑Planned Parity Gaps

These are intentionally not implemented in the Rust rewrite (see `docs/rewrite_progress.md` for the exhaustive list):
- `QFactor` workflow is not planned.
- `simLDSC` model‑string (lavaan) input is not planned.
- ggplot/ggpubr PDF plotting parity is not planned (Plotly HTML instead).
- HDL `.rda` reference panels are not supported (TSV conversion script provided).
- Exact log text parity and R list‑layout outputs are not planned.
- `.rearrange` ordering logic is a no‑op; lavaan engine preserves input order.
- Legacy helpers not ported: `addSNPs`, `addGenes`, `multiSNP`, `multiGene`, and `write.model` (not part of the current wiki workflow; `addSNPs` is deprecated in R).

## Resources
- Wiki: [GenomicSEM Wiki](https://github.com/GenomicSEM/GenomicSEM/wiki)
- Google Group: [genomic-sem-users](https://groups.google.com/forum/#!forum/genomic-sem-users)
- Paper: [Grotzinger et al., 2019](https://www.nature.com/articles/s41562-019-0566-x)

## Credits
Original GenomicSEM authors: Andrew Grotzinger, Matthijs van der Zee, Mijke Rhemtulla, Hill Ip, Michel Nivard, Elliot Tucker-Drob.

## Development Notes

### Workspace Layout
- `genomicsem/` is the workspace root.
- `genomicsem/lavaan/` is the SEM engine crate used by GenomicSEM.

### Testing
- There is no R‑style test suite in Rust. Validation is done via:
  - parity harness (`genomicsem/lavaan/src/bin/sem_parity.rs`)
  - workflow smoke‑tests via CLI

### Performance
- Parallel execution uses `rayon` in selected pipelines.
- For Linux performance parity with R, consider limiting BLAS/OpenMP threads (e.g., `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`).

## License
GPL-3.0-only (matches LICENSE).
