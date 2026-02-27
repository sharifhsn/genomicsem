# Performance Improvement Review (Rust GenomicSEM)

Goal: identify performance improvements that preserve **identical functionality**. Ordered by **estimated impact** (highest first). Each item includes feasibility and complexity.

## Very High Impact (Largest expected runtime wins)

1. **Precompute gamma inverse / WLS weights when invariant**
   - **Where**: `lavaan` fit stats and Q computation; `genomicsem` GWAS loops.
   - **Impact**: **Very high** (removes repeated eigen decompositions across SNPs).
   - **Feasibility**: Medium
   - **Complexity**: Medium
   - **Pros**: Huge savings when the same `gamma`/`wls_v` is reused many times.
   - **Cons**: Must prove invariance per workflow; otherwise correctness risk.

2. **Reuse scratch buffers for S/V assembly per SNP**
   - **Where**: `genomicsem/src/gwas.rs` (`build_s_full`, `build_v_full`, `build_v_snp`).
   - **Impact**: **Very high** (allocations per SNP dominate in large GWAS).
   - **Feasibility**: High
   - **Complexity**: Low‑Medium
   - **Pros**: Cuts allocation/GC pressure without changing math.
   - **Cons**: Requires careful full overwrite of buffers.

3. **Avoid repeated `Vec<Vec<f64>>` → `Array2` conversions for fixed matrices**
   - **Where**: `lavaan` and `gwas` call sites.
   - **Impact**: **High–Very high** in tight loops (removes repeated allocations and conversions).
   - **Feasibility**: High
   - **Complexity**: Low‑Medium
   - **Pros**: Less conversion overhead; enables caching in hot paths.
   - **Cons**: Requires structs to carry cached arrays safely.

4. **Flatten matrix storage (single `Vec<f64>` + shape)**
   - **Where**: `genomicsem` + `lavaan` matrix types.
   - **Impact**: **High** (cache locality in all matrix ops).
   - **Feasibility**: Low
   - **Complexity**: High
   - **Pros**: Faster linear algebra; fewer allocations.
   - **Cons**: Large refactor, higher bug risk.

5. **End‑to‑end `ndarray` in SEM engine**
   - **Where**: `lavaan` modules.
   - **Impact**: **High** (removes repeated conversions; enables optimized ops).
   - **Feasibility**: Low
   - **Complexity**: High
   - **Pros**: Cleaner math pipeline; fewer conversions.
   - **Cons**: Large refactor; parity risks.

## High Impact (Significant but smaller wins)

6. **Polars lazy pipeline + streaming for munge/sumstats**
   - **Where**: `munge.rs`, `sumstats.rs`.
   - **Impact**: **High** for large files (IO + memory reduction).
   - **Feasibility**: Medium
   - **Complexity**: Medium
   - **Pros**: Faster IO, lower memory footprint.
   - **Cons**: Risk of subtle parsing/NA behavior changes.

7. **Parallel chunking for GWAS and SEM**
   - **Where**: `gwas.rs`.
   - **Impact**: **High** on large SNP sets (reduces scheduling overhead).
   - **Feasibility**: Medium
   - **Complexity**: Medium
   - **Pros**: Better CPU utilization.
   - **Cons**: Harder to preserve warning ordering; outputs remain identical.

8. **Cholesky for SPD matrices with eigen fallback**
   - **Where**: `lavaan/src/stats.rs`, `lavaan/src/model.rs` (inverse/logdet paths).
   - **Impact**: **High** where matrices are SPD.
   - **Feasibility**: Medium
   - **Complexity**: Medium
   - **Pros**: Faster inverse/logdet on SPD matrices.
   - **Cons**: Must guarantee identical numeric behavior; fallback required.

## Medium Impact (Noticeable but smaller wins)

9. **Precompute per‑run constants in GWAS loops**
   - **Where**: `genomicsem/src/gwas.rs`.
   - **Impact**: **Medium** (reduces per‑SNP allocation/formatting overhead).
   - **Feasibility**: High
   - **Complexity**: Low
   - **Pros**: Low‑risk speedup.
   - **Cons**: Minor refactor, mostly organizational.

10. **Minimize DataFrame clones in hot paths**
    - **Where**: `munge`, `sumstats`, `ldsc`.
    - **Impact**: **Medium** (memory + CPU reduction in IO heavy stages).
    - **Feasibility**: High
    - **Complexity**: Low
    - **Pros**: Lower peak memory; modest speedups.
    - **Cons**: Ownership management must stay correct.

## Low / Variable Impact

11. **Batch log writing**
    - **Where**: `logging.rs`, `sumstats`, `munge`.
    - **Impact**: **Low** (IO overhead is minor vs computation).
    - **Feasibility**: High
    - **Complexity**: Low
    - **Pros**: Slightly cleaner IO pattern.
    - **Cons**: Minimal overall runtime impact.

12. **SIMD or BLAS tuning**
    - **Where**: overall runtime; linear algebra heavy modules.
    - **Impact**: **Variable** (can be large on specific machines).
    - **Feasibility**: Low (environment dependent)
    - **Complexity**: Low (build config + env variables)
    - **Pros**: Potentially large speedups.
    - **Cons**: Not portable and may conflict with user defaults.

## Notes
- All suggestions preserve **identical functionality**.
- Any change to numerical routines (inverse/logdet) must be validated with parity harness and regression runs.
