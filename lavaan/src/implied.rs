use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::{Eigh, Inverse, UPLO};

use crate::model::SemModel;
use crate::types::Matrix;

pub fn implied_covariance(model: &SemModel, theta: &[f64]) -> Result<Matrix> {
    let (a, s) = model.build_matrices(theta)?;
    let a = to_array2(&a)?;
    let s = to_array2(&s)?;
    let n = a.dim().0;
    let i = Array2::<f64>::eye(n);
    let inv = (i - a).inv()?;
    let sigma = inv.dot(&s).dot(&inv.t());
    Ok(from_array2(&sigma))
}

pub fn implied_observed(model: &SemModel, theta: &[f64]) -> Result<Matrix> {
    let sigma_all = implied_covariance(model, theta)?;
    let k = model.obs_names.len();
    let mut out = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            out[i][j] = sigma_all[i][j];
        }
    }
    Ok(out)
}

pub fn vech_col_major(matrix: &Matrix) -> Vec<f64> {
    let n = matrix.len();
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    for (j, _) in matrix.iter().enumerate().take(n) {
        for row in matrix.iter().skip(j).take(n - j) {
            out.push(row[j]);
        }
    }
    out
}

pub fn cov2cor(matrix: &Matrix) -> Matrix {
    let n = matrix.len();
    let mut out = vec![vec![0.0; n]; n];
    let mut sd = vec![0.0; n];
    for i in 0..n {
        let v = matrix[i][i];
        sd[i] = if v > 0.0 { v.sqrt() } else { 0.0 };
    }
    for i in 0..n {
        for j in 0..n {
            let denom = sd[i] * sd[j];
            out[i][j] = if denom != 0.0 {
                matrix[i][j] / denom
            } else {
                0.0
            };
        }
    }
    out
}

pub fn logdet_sym(matrix: &Matrix) -> f64 {
    if let Ok(a) = to_array2(matrix)
        && let Ok((eigvals, _)) = a.eigh(UPLO::Lower)
    {
        let mut sum = 0.0;
        for v in eigvals.iter() {
            if *v <= 0.0 || !v.is_finite() {
                return f64::NAN;
            }
            sum += v.ln();
        }
        return sum;
    }
    f64::NAN
}

pub fn to_array2(matrix: &Matrix) -> Result<Array2<f64>> {
    let n = matrix.len();
    let m = matrix.first().map(|r| r.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(n * m);
    for row in matrix {
        if row.len() != m {
            return Err(anyhow::anyhow!("Matrix not rectangular"));
        }
        data.extend_from_slice(row);
    }
    Ok(Array2::from_shape_vec((n, m), data)?)
}

pub fn from_array2(matrix: &Array2<f64>) -> Matrix {
    let (n, m) = matrix.dim();
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            out[i][j] = matrix[(i, j)];
        }
    }
    out
}
