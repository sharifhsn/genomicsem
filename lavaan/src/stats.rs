use anyhow::Result;
use ndarray_linalg::Eigh;
use ndarray_linalg::UPLO;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use tracing::{debug, trace};

use crate::implied::{cov2cor, to_array2, vech_col_major};
use crate::linalg::inverse_from_eigh;
use crate::types::{Estimation, Matrix};

pub struct FitStatsInput<'a> {
    pub s_obs: &'a Matrix,
    pub sigma_obs: &'a Matrix,
    pub gamma: &'a Matrix,
    pub n_free: usize,
    pub estimation: Estimation,
    pub n_obs: Option<f64>,
    pub fx: f64,
}

pub fn compute_stats(input: FitStatsInput<'_>) -> Result<(f64, i64, f64, f64, f64, f64)> {
    let k = input.s_obs.len();
    let p_star = k * (k + 1) / 2;
    let df = (p_star as i64) - (input.n_free as i64);

    debug!(
        "stats: k={}, p_star={}, n_free={}, df={}, estimation={:?}, n_obs={:?}, fx={}",
        k, p_star, input.n_free, df, input.estimation, input.n_obs, input.fx
    );

    let q = compute_q(input.s_obs, input.sigma_obs, input.gamma)?;
    debug!("stats: q={}", q);

    let aic = q + 2.0 * (input.n_free as f64);

    // GenomicSEM uses the residual-based Q statistic as chisq for both DWLS and ML.
    // This differs from lavaan R, which reports a likelihood-based chisq for ML.
    let chisq = q;
    debug!("stats: chisq={}, aic={}", chisq, aic);

    let p_chisq = if df > 0 {
        let chi = ChiSquared::new(df as f64).unwrap();
        1.0 - chi.cdf(chisq)
    } else {
        f64::NAN
    };
    debug!("stats: p_chisq={}", p_chisq);

    let srmr = compute_srmr(input.s_obs, input.sigma_obs);
    debug!("stats: srmr={}", srmr);

    let cfi = compute_cfi(input.s_obs, input.gamma, q, df as f64)?;
    debug!("stats: cfi={}", cfi);

    Ok((chisq, df, aic, cfi, srmr, p_chisq))
}

pub fn compute_q(s_obs: &Matrix, sigma_obs: &Matrix, gamma: &Matrix) -> Result<f64> {
    let resid = residual_matrix(s_obs, sigma_obs);
    let eta = vech_col_major(&resid);
    trace!(
        "compute_q: k={}, eta_len={}, gamma_dim=({},{})",
        s_obs.len(),
        eta.len(),
        gamma.len(),
        gamma.first().map(|r| r.len()).unwrap_or(0)
    );
    quad_form_vinv(&eta, gamma)
}

fn residual_matrix(s: &Matrix, sigma: &Matrix) -> Matrix {
    let k = s.len();
    let mut out = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            out[i][j] = s[i][j] - sigma[i][j];
        }
    }
    out
}

fn quad_form_vinv(vec: &[f64], v: &Matrix) -> Result<f64> {
    let v_arr = to_array2(v)?;
    trace!(
        "quad_form_vinv: vec_len={}, v_dim=({},{})",
        vec.len(),
        v_arr.nrows(),
        v_arr.ncols()
    );
    let (eigvals, eigvecs) = v_arr.eigh(UPLO::Lower)?;
    trace!("quad_form_vinv: eigvals={:?}", eigvals.to_vec());
    let v_inv = inverse_from_eigh(&eigvals, &eigvecs, None);

    let mut tmp = vec![0.0; vec.len()];
    for i in 0..vec.len() {
        let mut sum = 0.0;
        for j in 0..vec.len() {
            sum += v_inv[(i, j)] * vec[j];
        }
        tmp[i] = sum;
    }
    let mut out = 0.0;
    for i in 0..vec.len() {
        out += vec[i] * tmp[i];
    }
    Ok(out)
}

fn compute_cfi(s: &Matrix, v: &Matrix, q: f64, df: f64) -> Result<f64> {
    let mut resid_cfi = s.clone();
    for (i, row) in resid_cfi.iter_mut().enumerate() {
        row[i] = 0.0;
    }
    let eta_cfi = vech_col_major(&resid_cfi);
    let q_cfi = quad_form_vinv(&eta_cfi, v)?;
    let k = s.len();
    let df_cfi = (k * (k + 1) / 2 - k) as f64;
    let denom = q_cfi - df_cfi;
    debug!(
        "compute_cfi: q_cfi={}, df_cfi={}, denom={}, q={}, df={}",
        q_cfi, df_cfi, denom, q, df
    );
    if !denom.is_finite() || denom == 0.0 {
        return Ok(f64::NAN);
    }
    let mut cfi = ((q_cfi - df_cfi) - (q - df)) / denom;
    if cfi > 1.0 {
        cfi = 1.0;
    }
    Ok(cfi)
}

fn compute_srmr(s: &Matrix, sigma: &Matrix) -> f64 {
    let r_obs = cov2cor(s);
    let r_hat = cov2cor(sigma);
    let k = r_obs.len();
    let mut sum = 0.0;
    let mut count = 0.0;
    for i in 0..k {
        for j in 0..=i {
            let diff = r_obs[i][j] - r_hat[i][j];
            sum += diff * diff;
            count += 1.0;
        }
    }
    let srmr = if count == 0.0 {
        f64::NAN
    } else {
        (sum / count).sqrt()
    };
    debug!("compute_srmr: count={}, sum={}, srmr={}", count, sum, srmr);
    srmr
}
