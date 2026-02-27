use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Eigh, Inverse, UPLO};
use nlopt::{Algorithm, Nlopt, Target, approximate_gradient};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use statrs::distribution::{ContinuousCDF, Normal};
use tracing::info;

use crate::matrix::{ensure_square, to_array2};
use crate::types::{LdscOutput, Matrix};

#[derive(Debug, Clone, Copy)]
pub enum SubSvType {
    S,
    SStand,
    R,
}

#[derive(Debug, Clone)]
pub struct SubSvOutput {
    pub sub_v: Matrix,
    pub sub_s: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SummaryGlsOutput {
    pub betas: Vec<f64>,
    pub pvals: Vec<f64>,
    pub se: Vec<f64>,
    pub z: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SummaryGlsBandsOutput {
    pub gls: SummaryGlsOutput,
    pub band_predictors: Option<Vec<f64>>,
    pub band_upper: Option<Vec<f64>>,
    pub band_lower: Option<Vec<f64>>,
    pub band_se: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct PaLdscOutput {
    pub observed: Vec<f64>,
    pub parallel: Vec<f64>,
    pub n_factors: Option<usize>,
    pub observed_diag: Option<Vec<f64>>,
    pub parallel_diag: Option<Vec<f64>>,
    pub n_factors_diag: Option<usize>,
    pub observed_fa: Option<Vec<f64>>,
    pub parallel_fa: Option<Vec<f64>>,
    pub n_factors_fa: Option<usize>,
    pub observed_fa_diag: Option<Vec<f64>>,
    pub parallel_fa_diag: Option<Vec<f64>>,
    pub n_factors_fa_diag: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct LocalSrmdOutput {
    pub value: f64,
    pub localdiff: Vec<f64>,
}

type PaFaResult = (
    Vec<f64>,
    Vec<f64>,
    Option<usize>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<usize>,
);

#[allow(clippy::needless_range_loop)]
pub fn index_s_from_matrix(matrix: &Matrix, include_diag: bool) -> Vec<Vec<usize>> {
    let k = matrix.len();
    let mut out = vec![vec![0usize; k]; k];
    let mut idx = 1usize;
    for j in 0..k {
        for i in j..k {
            if include_diag || i != j {
                out[i][j] = idx;
                idx += 1;
            }
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            out[i][j] = out[j][i];
        }
    }
    out
}

pub fn index_s_from_ldsc(ldsc: &LdscOutput, include_diag: bool) -> Vec<Vec<usize>> {
    index_s_from_matrix(&ldsc.s, include_diag)
}

pub fn sub_sv_from_ldsc(
    ldsc: &LdscOutput,
    index_vals: &[usize],
    ty: SubSvType,
) -> Result<SubSvOutput> {
    match ty {
        SubSvType::S => sub_sv_from_matrices(&ldsc.s, &ldsc.v, index_vals),
        SubSvType::SStand => {
            let s = ldsc
                .s_stand
                .as_ref()
                .context("S_Stand missing from LDSC output")?;
            let v = ldsc
                .v_stand
                .as_ref()
                .context("V_Stand missing from LDSC output")?;
            sub_sv_from_matrices(s, v, index_vals)
        }
        SubSvType::R => {
            let s = ldsc
                .s_stand
                .as_ref()
                .context("S_Stand missing from LDSC output")?;
            let v = ldsc
                .v_stand
                .as_ref()
                .context("V_Stand missing from LDSC output")?;
            // R and V_R are not yet implemented; use S_Stand/V_Stand as a proxy.
            sub_sv_from_matrices(s, v, index_vals)
        }
    }
}

pub fn sub_sv_from_matrices(s: &Matrix, v: &Matrix, index_vals: &[usize]) -> Result<SubSvOutput> {
    ensure_square(s, "S")?;
    ensure_square(v, "V")?;
    if index_vals.is_empty() {
        return Err(anyhow::anyhow!("INDEXVALS must not be empty"));
    }

    let vech = vech_lower_triangle(s);
    let max_index = *index_vals.iter().max().unwrap_or(&0);
    if max_index == 0 {
        return Err(anyhow::anyhow!("INDEXVALS must be 1-based"));
    }
    if max_index > vech.len() {
        return Err(anyhow::anyhow!(
            "INDEXVALS includes {max_index} but only {} elements exist",
            vech.len()
        ));
    }
    if max_index > v.len() {
        return Err(anyhow::anyhow!(
            "INDEXVALS includes {max_index} but V is {}x{}",
            v.len(),
            v.first().map(|row| row.len()).unwrap_or(0)
        ));
    }

    let indices: Vec<usize> = index_vals.iter().map(|v| v.saturating_sub(1)).collect();
    let sub_s = indices.iter().map(|i| vech[*i]).collect::<Vec<_>>();

    let mut sub_v = vec![vec![0.0; indices.len()]; indices.len()];
    for (i_out, &i) in indices.iter().enumerate() {
        for (j_out, &j) in indices.iter().enumerate() {
            sub_v[i_out][j_out] = v[i][j];
        }
    }

    Ok(SubSvOutput { sub_v, sub_s })
}

pub fn summary_gls(
    y: &[f64],
    v_y: &Matrix,
    predictors: &[Vec<f64>],
    intercept: bool,
) -> Result<SummaryGlsOutput> {
    ensure_square(v_y, "V_Y")?;
    if y.is_empty() {
        return Err(anyhow::anyhow!("Y must not be empty"));
    }
    if predictors.len() != y.len() {
        return Err(anyhow::anyhow!(
            "Predictors rows ({}) must match Y length ({})",
            predictors.len(),
            y.len()
        ));
    }

    let n = y.len();
    let p = predictors.first().map(|row| row.len()).unwrap_or(0);
    if p == 0 {
        return Err(anyhow::anyhow!("Predictors must have at least one column"));
    }
    for (i, row) in predictors.iter().enumerate() {
        if row.len() != p {
            return Err(anyhow::anyhow!(
                "Predictors row {i} length {} does not match {p}",
                row.len()
            ));
        }
    }

    let x_cols = if intercept { p + 1 } else { p };
    let mut x_data = Vec::with_capacity(n * x_cols);
    for row in predictors {
        if intercept {
            x_data.push(1.0);
        }
        x_data.extend_from_slice(row);
    }
    let x = Array2::from_shape_vec((n, x_cols), x_data).context("X shape")?;
    let y = Array1::from_vec(y.to_vec());

    let v = to_array2(v_y)?;
    if v.dim().0 != n {
        return Err(anyhow::anyhow!(
            "V_Y dimension {} does not match Y length {}",
            v.dim().0,
            n
        ));
    }

    let v_inv = v.inv().context("invert V_Y")?;
    let xt = x.t().to_owned();
    let xt_v_inv = xt.dot(&v_inv);
    let xt_v_inv_x = xt_v_inv.dot(&x);
    let xt_v_inv_y = xt_v_inv.dot(&y);

    let cov = xt_v_inv_x.inv().context("covariance")?;
    let betas = cov.dot(&xt_v_inv_y);

    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    let mut out_betas = Vec::with_capacity(x_cols);
    let mut out_se = Vec::with_capacity(x_cols);
    let mut out_z = Vec::with_capacity(x_cols);
    let mut out_p = Vec::with_capacity(x_cols);

    for i in 0..x_cols {
        let beta = betas[i];
        let se = cov[(i, i)].abs().sqrt();
        let z = if se != 0.0 { beta / se } else { f64::NAN };
        let p = if z.is_finite() {
            2.0 * (1.0 - normal.cdf(z.abs()))
        } else {
            f64::NAN
        };
        out_betas.push(beta);
        out_se.push(se);
        out_z.push(z);
        out_p.push(p);
    }

    Ok(SummaryGlsOutput {
        betas: out_betas,
        pvals: out_p,
        se: out_se,
        z: out_z,
    })
}

pub fn summary_gls_from_subsv(
    sub: &SubSvOutput,
    predictors: &[Vec<f64>],
    intercept: bool,
) -> Result<SummaryGlsOutput> {
    summary_gls(&sub.sub_s, &sub.sub_v, predictors, intercept)
}

#[allow(clippy::too_many_arguments)]
pub fn summary_gls_bands(
    y: &[f64],
    v_y: &Matrix,
    predictors: &[Vec<f64>],
    intervals: usize,
    controlvars: Option<&[Vec<f64>]>,
    intercept: bool,
    quad: bool,
    bands: bool,
    band_size: f64,
) -> Result<SummaryGlsBandsOutput> {
    let gls = summary_gls(y, v_y, predictors, intercept)?;
    if !bands {
        return Ok(SummaryGlsBandsOutput {
            gls,
            band_predictors: None,
            band_upper: None,
            band_lower: None,
            band_se: None,
        });
    }
    if predictors.is_empty() {
        return Err(anyhow::anyhow!("Predictors must not be empty"));
    }
    let n = predictors.len();
    let p = predictors[0].len();
    if p == 0 {
        return Err(anyhow::anyhow!("Predictors must have at least one column"));
    }
    if intervals == 0 {
        return Err(anyhow::anyhow!("Intervals must be > 0"));
    }

    let mut base = Vec::with_capacity(n);
    for row in predictors {
        base.push(row[0]);
    }
    let min_x = base.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = base.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_x - min_x;

    let predictors_seq: Vec<f64> = (0..intervals)
        .map(|i| min_x + (i as f64) * range / (intervals as f64))
        .collect();
    let mut band_se = Vec::with_capacity(intervals);
    for shift in &predictors_seq {
        let mut shifted = Vec::with_capacity(n);
        for row in predictors {
            let val = row[0] - shift;
            let mut row_out = Vec::new();
            if quad {
                row_out.push(val);
                row_out.push(val * val);
            } else {
                row_out.push(val);
            }
            // Append remaining predictors unchanged (matches R behavior).
            if row.len() > 1 {
                row_out.extend_from_slice(&row[1..]);
            }
            shifted.push(row_out);
        }
        if let Some(ctrl) = controlvars {
            if ctrl.len() != n {
                return Err(anyhow::anyhow!(
                    "Controlvars rows ({}) must match Y length ({})",
                    ctrl.len(),
                    n
                ));
            }
            for (idx, row) in ctrl.iter().enumerate() {
                shifted[idx].extend_from_slice(row);
            }
        }
        let out = summary_gls(y, v_y, &shifted, intercept)?;
        band_se.push(out.se.first().copied().unwrap_or(f64::NAN));
    }

    let base_idx = if intercept { 1 } else { 0 };
    let b0 = if intercept {
        gls.betas.first().copied().unwrap_or(f64::NAN)
    } else {
        0.0
    };
    let b1 = gls.betas.get(base_idx).copied().unwrap_or(f64::NAN);
    let b2 = if quad {
        gls.betas.get(base_idx + 1).copied().unwrap_or(f64::NAN)
    } else {
        0.0
    };

    let mut band_upper = Vec::with_capacity(intervals);
    let mut band_lower = Vec::with_capacity(intervals);
    for (x, se) in predictors_seq.iter().zip(&band_se) {
        let eq = if quad {
            b0 + b1 * x + b2 * x * x
        } else {
            b0 + b1 * x
        };
        band_upper.push(eq + band_size * se);
        band_lower.push(eq - band_size * se);
    }

    // Plotting is handled by CLI helpers (Plotly HTML) to mirror R usage.
    Ok(SummaryGlsBandsOutput {
        gls,
        band_predictors: Some(predictors_seq),
        band_upper: Some(band_upper),
        band_lower: Some(band_lower),
        band_se: Some(band_se),
    })
}

pub fn pa_ldsc(
    s: &Matrix,
    v: &Matrix,
    r: usize,
    p: f64,
    diag: bool,
    fa: bool,
    nfactors: usize,
) -> Result<PaLdscOutput> {
    ensure_square(s, "S")?;
    ensure_square(v, "V")?;
    if r == 0 {
        return Err(anyhow::anyhow!("r must be > 0"));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(anyhow::anyhow!("p must be in [0,1]"));
    }

    let k = s.len();
    let kstar = k * (k + 1) / 2;
    if v.len() != kstar {
        return Err(anyhow::anyhow!("V must be {kstar}x{kstar} for S {k}x{k}"));
    }

    let s_null = diag_only_matrix(s);
    let s_null_vec = vech_lower_triangle(&s_null);

    let mut rng = rand::rng();
    let mut eigs = Vec::with_capacity(r);
    for i in 0..r {
        let sample = mvn_sample(&s_null_vec, v, &mut rng)?;
        let mat = vech_to_symmetric(&sample, k);
        eigs.push(eigenvalues(&mat)?);
        info!("Running parallel analysis. Replication number: {}", i + 1);
    }
    let parallel = row_quantile(&eigs, p);
    let observed = eigenvalues(s)?;
    let n_factors = count_parallel(&observed, &parallel);

    let (observed_diag, parallel_diag, n_factors_diag) = if diag {
        let mut eigs_diag = Vec::with_capacity(r);
        let diag_var = diag_only(v);
        for i in 0..r {
            let sample = mvn_sample_diag(&s_null_vec, &diag_var, &mut rng);
            let mat = vech_to_symmetric(&sample, k);
            eigs_diag.push(eigenvalues(&mat)?);
            info!(
                "Running diagonalized parallel analysis. Replication number: {}",
                i + 1
            );
        }
        let parallel_diag = row_quantile(&eigs_diag, p);
        let n_factors_diag = count_parallel(&observed, &parallel_diag);
        (Some(observed.clone()), Some(parallel_diag), n_factors_diag)
    } else {
        (None, None, None)
    };

    let (
        observed_fa,
        parallel_fa,
        n_factors_fa,
        observed_fa_diag,
        parallel_fa_diag,
        n_factors_fa_diag,
    ) = if fa {
        let (obs, par, nf, obs_diag, par_diag, nf_diag) = pa_ldsc_fa(s, v, r, p, diag, nfactors)?;
        (Some(obs), Some(par), nf, obs_diag, par_diag, nf_diag)
    } else {
        (None, None, None, None, None, None)
    };

    // Plotting is handled by CLI helpers (Plotly HTML) to mirror R usage.
    Ok(PaLdscOutput {
        observed,
        parallel,
        n_factors,
        observed_diag,
        parallel_diag,
        n_factors_diag,
        observed_fa,
        parallel_fa,
        n_factors_fa,
        observed_fa_diag,
        parallel_fa_diag,
        n_factors_fa_diag,
    })
}

fn pa_ldsc_fa(
    s: &Matrix,
    v: &Matrix,
    r: usize,
    p: f64,
    diag: bool,
    nfactors: usize,
) -> Result<PaFaResult> {
    let k = s.len();
    let kstar = k * (k + 1) / 2;
    if v.len() != kstar {
        return Err(anyhow::anyhow!("V must be {kstar}x{kstar} for S {k}x{k}"));
    }

    let s_corr = smooth_corr_matrix(s)?;
    let observed = fa_minres_eigenvalues(&s_corr, nfactors)?;

    let s_null = diag_only_matrix(s);
    let s_null_vec = vech_lower_triangle(&s_null);

    let mut rng = rand::rng();
    let mut eigs = Vec::with_capacity(r);
    for i in 0..r {
        let sample = mvn_sample(&s_null_vec, v, &mut rng)?;
        let mat = vech_to_symmetric(&sample, k);
        let mat = smooth_corr_matrix(&mat)?;
        eigs.push(fa_minres_eigenvalues(&mat, nfactors)?);
        info!(
            "Running FA parallel analysis. Replication number: {}",
            i + 1
        );
    }
    let parallel = row_quantile(&eigs, p);
    let n_factors = count_parallel(&observed, &parallel).or(Some(1));

    if diag {
        let diag_var = diag_only(v);
        let mut eigs_diag = Vec::with_capacity(r);
        for i in 0..r {
            let sample = mvn_sample_diag(&s_null_vec, &diag_var, &mut rng);
            let mat = vech_to_symmetric(&sample, k);
            let mat = smooth_corr_matrix(&mat)?;
            eigs_diag.push(fa_minres_eigenvalues(&mat, nfactors)?);
            info!(
                "Running FA diagonalized parallel analysis. Replication number: {}",
                i + 1
            );
        }
        let parallel_diag = row_quantile(&eigs_diag, p);
        let n_factors_diag = count_parallel(&observed, &parallel_diag).or(Some(1));
        return Ok((
            observed.clone(),
            parallel,
            n_factors,
            Some(observed),
            Some(parallel_diag),
            n_factors_diag,
        ));
    }

    Ok((observed, parallel, n_factors, None, None, None))
}

fn smooth_corr_matrix(matrix: &Matrix) -> Result<Matrix> {
    ensure_square(matrix, "S")?;
    let k = matrix.len();
    let mut corr = vec![vec![0.0; k]; k];
    let mut diag = vec![0.0; k];
    for i in 0..k {
        let v = matrix[i][i];
        diag[i] = if v > 0.0 { v.sqrt() } else { 1.0 };
    }
    for i in 0..k {
        for j in 0..k {
            let denom = diag[i] * diag[j];
            corr[i][j] = if denom != 0.0 {
                matrix[i][j] / denom
            } else {
                0.0
            };
        }
    }

    near_pd_corr(&corr, 1e-8, 1e-7, 100)
}

fn fa_minres_eigenvalues(r: &Matrix, nfactors: usize) -> Result<Vec<f64>> {
    let k = r.len();
    let nf = nfactors.max(1).min(k);
    let a = to_array2(r)?;
    let smc_vals = smc_from_corr(&a)?;
    let mut start = Vec::with_capacity(k);
    for i in 0..k {
        start.push(r[i][i] - smc_vals[i]);
    }

    let upper = smc_vals.iter().copied().fold(1.0_f64, |acc, v| acc.max(v));

    let data = FaMinresData { s: a, nf };
    let obj = |x: &[f64], grad: Option<&mut [f64]>, data: &mut FaMinresData| -> f64 {
        let f = minres_residual_objective(x, data);
        if let Some(g) = grad {
            approximate_gradient(x, |x| minres_residual_objective(x, data), g);
        }
        f
    };

    let mut opt = Nlopt::new(Algorithm::Lbfgs, k, obj, Target::Minimize, data);
    let _ = opt.set_lower_bounds(&vec![0.005; k]);
    let _ = opt.set_upper_bounds(&vec![upper; k]);
    let _ = opt.set_ftol_rel(1e-7);
    let _ = opt.set_maxeval(1000);
    let mut psi = start.clone();
    let _ = opt.optimize(&mut psi);

    let loadings = faout_wls(&psi, r, nf)?;
    let ll = loadings.dot(&loadings.t());
    let mut s_model = to_array2(r)?;
    for i in 0..k {
        s_model[(i, i)] = ll[(i, i)];
    }
    let (eigvals, _eigvecs) = s_model.eigh(UPLO::Lower).context("eigh")?;
    let mut vals = eigvals.to_vec();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(vals)
}

fn near_pd_corr(matrix: &Matrix, eps: f64, tol: f64, max_iter: usize) -> Result<Matrix> {
    let mut x = to_array2(matrix)?;
    let k = x.dim().0;
    let mut delta = Array2::<f64>::zeros((k, k));
    let mut last = x.clone();
    for _ in 0..max_iter {
        let mut r = &x - &delta;
        r = (&r + &r.t()) * 0.5;
        let (eigvals, eigvecs) = r.eigh(UPLO::Lower).context("eigh")?;
        let adj = eigvals.mapv(|v| if v.is_finite() && v > eps { v } else { eps });
        let diag = Array2::from_diag(&adj);
        x = eigvecs.dot(&diag).dot(&eigvecs.t());
        delta = &x - &r;
        for i in 0..k {
            x[(i, i)] = 1.0;
        }
        let diff = (&x - &last).mapv(|v| v * v).sum().sqrt();
        if diff < tol {
            break;
        }
        last = x.clone();
    }
    let mut out = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            out[i][j] = x[(i, j)];
        }
    }
    Ok(out)
}

struct FaMinresData {
    s: Array2<f64>,
    nf: usize,
}

fn minres_residual_objective(psi: &[f64], data: &FaMinresData) -> f64 {
    let k = data.s.dim().0;
    let s_data = &data.s;
    let mut s_star = s_data.clone();
    for i in 0..k {
        s_star[(i, i)] -= psi[i];
    }
    let (eigvals, eigvecs) = match s_star.eigh(UPLO::Lower) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let mut pairs: Vec<(f64, Vec<f64>)> = eigvals
        .iter()
        .zip(eigvecs.columns())
        .map(|(v, col)| (*v, col.to_vec()))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let nf = data.nf.min(k);
    let mut load = Array2::<f64>::zeros((k, nf));
    for (j, (eig, vec)) in pairs.iter().take(nf).enumerate() {
        let scale = eig.max(0.0).sqrt();
        for i in 0..k {
            load[(i, j)] = vec[i] * scale;
        }
    }
    let model = load.dot(&load.t());
    let mut sum = 0.0;
    for i in 0..k {
        for j in 0..i {
            if i != j {
                let diff = s_data[(i, j)] - model[(i, j)];
                sum += diff * diff;
            }
        }
    }
    sum
}

fn faout_wls(psi: &[f64], r: &Matrix, nfactors: usize) -> Result<Array2<f64>> {
    let k = r.len();
    let nf = nfactors.max(1).min(k);
    let mut s = to_array2(r)?;
    for i in 0..k {
        s[(i, i)] = r[i][i] - psi[i];
    }
    let (eigvals, eigvecs) = s.eigh(UPLO::Lower).context("eigh")?;
    let mut pairs: Vec<(f64, Vec<f64>)> = eigvals
        .iter()
        .zip(eigvecs.columns())
        .map(|(v, col)| (*v, col.to_vec()))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut load = Array2::<f64>::zeros((k, nf));
    for (j, (eig, vec)) in pairs.iter().take(nf).enumerate() {
        let scale = eig.max(0.0).sqrt();
        for i in 0..k {
            load[(i, j)] = vec[i] * scale;
        }
    }
    Ok(load)
}

fn smc_from_corr(matrix: &Array2<f64>) -> Result<Vec<f64>> {
    let k = matrix.dim().0;
    let inv = match matrix.clone().inv() {
        Ok(v) => v,
        Err(_) => {
            let (eigvals, eigvecs) = matrix.eigh(UPLO::Lower).context("eigh")?;
            let inv_vals = eigvals.mapv(|v| if v > 1e-12 { 1.0 / v } else { 0.0 });
            let diag = Array2::from_diag(&inv_vals);
            eigvecs.dot(&diag).dot(&eigvecs.t())
        }
    };
    let mut smc = Vec::with_capacity(k);
    for i in 0..k {
        let denom = inv[(i, i)];
        let val = if denom.is_finite() && denom != 0.0 {
            1.0 - 1.0 / denom
        } else {
            0.0
        };
        smc.push(val.clamp(0.0, 1.0));
    }
    Ok(smc)
}

pub fn local_srmd(
    unconstrained: &[f64],
    constrained: &[f64],
    lhsvar: &[Vec<f64>],
    rhsvar: &[Vec<f64>],
) -> Result<LocalSrmdOutput> {
    if unconstrained.len() != constrained.len() {
        return Err(anyhow::anyhow!(
            "Unconstrained length {} does not match constrained length {}",
            unconstrained.len(),
            constrained.len()
        ));
    }
    if lhsvar.is_empty() || rhsvar.is_empty() {
        return Err(anyhow::anyhow!("lhsvar and rhsvar must not be empty"));
    }
    if lhsvar.len() != rhsvar.len() {
        return Err(anyhow::anyhow!(
            "lhsvar length {} does not match rhsvar length {}",
            lhsvar.len(),
            rhsvar.len()
        ));
    }

    let lhs_pooled = pooled_sd(lhsvar)?;
    let rhs_pooled = pooled_sd(rhsvar)?;

    let mut localdiff = Vec::with_capacity(unconstrained.len());
    for i in 0..unconstrained.len() {
        let lhs = lhs_pooled[i % lhs_pooled.len()];
        let rhs = rhs_pooled[i % rhs_pooled.len()];
        let denom = lhs * rhs;
        let diff = if denom == 0.0 {
            f64::NAN
        } else {
            (unconstrained[i] - constrained[i]) / denom
        };
        localdiff.push(diff * diff);
    }
    let mut sum = 0.0;
    let mut count = 0.0;
    for v in &localdiff {
        if v.is_finite() {
            sum += v;
            count += 1.0;
        }
    }
    let value = if count > 0.0 {
        (sum / count).sqrt()
    } else {
        f64::NAN
    };
    Ok(LocalSrmdOutput { value, localdiff })
}

#[allow(clippy::needless_range_loop)]
fn vech_lower_triangle(matrix: &Matrix) -> Vec<f64> {
    let n = matrix.len();
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    // Column-major lower-triangle ordering to match R's column-major indexing.
    for j in 0..n {
        for i in j..n {
            out.push(matrix[i][j]);
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn vech_to_symmetric(values: &[f64], k: usize) -> Matrix {
    let mut out = vec![vec![0.0; k]; k];
    let mut idx = 0usize;
    for j in 0..k {
        for i in j..k {
            let v = values.get(idx).copied().unwrap_or(f64::NAN);
            out[i][j] = v;
            out[j][i] = v;
            idx += 1;
        }
    }
    out
}

fn eigenvalues(matrix: &Matrix) -> Result<Vec<f64>> {
    let a = to_array2(matrix)?;
    let (eigvals, _eigvecs) = a.eigh(UPLO::Lower).context("eigh")?;
    let mut vals = eigvals.to_vec();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(vals)
}

#[allow(clippy::needless_range_loop)]
fn row_quantile(eigs: &[Vec<f64>], p: f64) -> Vec<f64> {
    if eigs.is_empty() {
        return vec![];
    }
    let k = eigs[0].len();
    let mut out = vec![f64::NAN; k];
    for i in 0..k {
        let mut vals = Vec::with_capacity(eigs.len());
        for row in eigs {
            if let Some(v) = row.get(i) {
                vals.push(*v);
            }
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out[i] = quantile_sorted(&vals, p);
    }
    out
}

fn quantile_sorted(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let n = values.len() as f64;
    let pos = p.clamp(0.0, 1.0) * (n - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let w = pos - lo as f64;
        values[lo] * (1.0 - w) + values[hi] * w
    }
}

fn count_parallel(observed: &[f64], parallel: &[f64]) -> Option<usize> {
    for (i, (obs, par)) in observed.iter().zip(parallel).enumerate() {
        if obs < par {
            return Some(i);
        }
    }
    None
}

fn diag_only(matrix: &Matrix) -> Vec<f64> {
    let mut out = Vec::with_capacity(matrix.len());
    for (i, row) in matrix.iter().enumerate() {
        out.push(row[i]);
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn diag_only_matrix(matrix: &Matrix) -> Matrix {
    let n = matrix.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        out[i][i] = matrix[i][i];
    }
    out
}

fn mvn_sample<R: Rng + ?Sized>(mean: &[f64], cov: &Matrix, rng: &mut R) -> Result<Vec<f64>> {
    let k = mean.len();
    let a = to_array2(cov)?;
    let mut z = Array1::zeros(k);
    for i in 0..k {
        z[i] = StandardNormal.sample(rng);
    }
    let sample: Array1<f64> = match a.cholesky(UPLO::Lower) {
        Ok(chol) => chol.dot(&z),
        Err(_) => {
            let (eigvals, eigvecs) = a.eigh(UPLO::Lower).context("eigh")?;
            let sqrt_vals = eigvals.mapv(|v| {
                if v.is_finite() && v > 0.0 {
                    v.sqrt()
                } else {
                    0.0
                }
            });
            let diag = Array2::from_diag(&sqrt_vals);
            eigvecs.dot(&diag).dot(&z)
        }
    };
    Ok(sample.iter().zip(mean).map(|(v, m)| v + m).collect())
}

fn mvn_sample_diag<R: Rng + ?Sized>(mean: &[f64], diag: &[f64], rng: &mut R) -> Vec<f64> {
    mean.iter()
        .zip(diag)
        .map(|(m, v)| {
            let draw: f64 = StandardNormal.sample(rng);
            *m + v.abs().sqrt() * draw
        })
        .collect()
}

fn pooled_sd(values: &[Vec<f64>]) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(values.len());
    for (idx, row) in values.iter().enumerate() {
        if row.is_empty() {
            return Err(anyhow::anyhow!("lhsvar/rhsvar row {idx} is empty"));
        }
        let mut sum = 0.0;
        let mut count = 0.0;
        for v in row {
            if v.is_finite() {
                sum += *v;
                count += 1.0;
            }
        }
        if count == 0.0 {
            out.push(f64::NAN);
        } else {
            out.push((sum / count).sqrt());
        }
    }
    Ok(out)
}

// ensure_square and to_array2 are provided by matrix.rs
