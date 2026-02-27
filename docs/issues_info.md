# Open Issues Review (Most Discussion)

Source: `gh issue list -S "is:issue is:open sort:comments-desc" --limit 15` (GenomicSEM/GenomicSEM).

## Summary Themes
- **Model scaling & implied sample size**: user confusion about N_hat for factor GWAS depending on identification (unit loading vs unit variance) and reference indicator h2.
- **DWLS/commonfactor/userGWAS failures tied to lavaan versions**: errors like `Model1_Results`/`ReorderModel` not found; DWLS path failing in certain versions.
- **Matrix singularity / tolerance**: `system is computationally singular` errors in usermodel/commonfactorGWAS.
- **Sumstats transformations**: effect size standardization (STDY vs STDXY) and logistic/OLS transformations.
- **P-value underflow**: very small p-values read as zero then dropped; discussion on clamping/log10 output.
- **User errors in model specification**: missing SNP regression or typos in variable names cause opaque errors.

## Issues with Highest Discussion (Top 15 by comments)

1. **#100 – Question about implied sample size**
   - Theme: N_hat depends on factor scaling; unit loading on low-h2 indicator leads to very large N_hat.
   - Lesson: Add explicit guidance in Rust docs about factor scaling and implied sample size interpretation.

2. **#129 – `commonfactor()`/`commonfactorGWAS()` fail: `Model1_Results`/`ReorderModel` not found**
   - Theme: DWLS path failing; version issues (GenomicSEM 0.0.5 + lavaan 0.6.20 works).
   - Lesson: Rust should provide clearer error path when model fit fails and avoid hidden state; also warn about version compatibility in R but Rust is insulated.

3. **#61 – SNP effect size standardization in `sumstats_main.R`**
   - Theme: STDY vs STDXY distinction; GenomicSEM uses STDY for SNP effects.
   - Lesson: Keep Rust standardization aligned (STDY), document explicitly.

4. **#110 – Very small p-values set to zero by `read.table`, SNPs dropped**
   - Theme: Underflow on read/write; discussion of clamping or writing `-log10(p)`.
   - Lesson: Rust should handle p-value underflow robustly on read and write, and avoid dropping rows when p→0.

5. **#57 – `usermodel` computationally singular (`solve.default(V_LD)`)**
   - Theme: Singular sampling covariance matrix.
   - Lesson: Improve diagnostics around near-singular V; consider tolerance knobs and explicit warnings in Rust.

6. **#83 – `system is computationally singular` in commonfactorGWAS**
   - Theme: Similar singular V issue as #57.
   - Lesson: Same as above; detect and report early.

7. **#108 – userGWAS error (`subscript out of bounds`)**
   - Theme: Model missing SNP regression; fix_measurement defaults; mismatch between SNP inputs and model.
   - Lesson: Add explicit validation: ensure SNP predictor is in model and that S/V dimensions match expected structure.

8. **#53 – commonfactorGWAS error**
   - Theme: Likely model variable typos.
   - Lesson: Better parser error messages and name validation help avoid this class of error.

9. **#67 – errors running common factor GWAS (debugging advice)**
   - Theme: Suggest debug on small chromosome.
   - Lesson: Add a “debug small input” suggestion to Rust README or error logs.

10. **#75 – userGWAS error: non‑conformable arguments**
    - Theme: Likely mismatch of dimensions or model/data misalignment.
    - Lesson: Add shape checks with clearer messages.

11. **#80 – usermodel error (`ReorderModel` not found)**
    - Theme: Usually a typo in model variable name.
    - Lesson: Rust should keep strict name validation and clear error messages.

## Actionable Lessons for Rust Rewrite
- **Docs**: Add guidance on implied sample size and factor scaling (unit loading vs unit variance).
- **Validation**: Fail fast with clear messages for model typos, missing SNP regression, and S/V dimension mismatches.
- **Numerics**: Add explicit handling/logging for near‑singular matrices and optional tolerances.
- **P‑values**: Implement underflow‑safe parsing and output (e.g., clamp, or allow `-log10(p)` output in a new flag) while preserving current outputs by default.
- **Debugging UX**: Suggest “small‑input” debugging in CLI logs.

