use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::Inverse;

use crate::implied::{implied_observed, to_array2, vech_col_major};
use crate::model::SemModel;

pub fn jacobian(model: &SemModel, theta: &[f64]) -> Result<Array2<f64>> {
    let base = vech_col_major(&implied_observed(model, theta)?);
    let p = base.len();
    let q = theta.len();
    let mut jac = Array2::<f64>::zeros((p, q));
    for j in 0..q {
        let mut t_plus = theta.to_vec();
        let mut t_minus = theta.to_vec();
        let eps = 1e-6 * theta[j].abs().max(1.0);
        t_plus[j] += eps;
        t_minus[j] -= eps;
        let plus = vech_col_major(&implied_observed(model, &t_plus)?);
        let minus = vech_col_major(&implied_observed(model, &t_minus)?);
        for i in 0..p {
            jac[(i, j)] = (plus[i] - minus[i]) / (2.0 * eps);
        }
    }
    Ok(jac)
}

pub fn sandwich_covariance(model: &SemModel, theta: &[f64]) -> Result<Array2<f64>> {
    let delta = jacobian(model, theta)?;
    let _p = delta.dim().0;
    let q = delta.dim().1;

    let w = to_array2(&model.wls_v)?;
    let v = to_array2(&model.gamma)?;

    let (a, b) = if matches!(model.estimation, crate::types::Estimation::Dwls) {
        let p = delta.dim().0;
        let d = w.diag().to_owned();
        let mut wd = delta.clone();
        for i in 0..p {
            let scale = d[i];
            for j in 0..q {
                wd[(i, j)] *= scale;
            }
        }
        let a = delta.t().dot(&wd);
        let b = wd.t().dot(&v.dot(&wd));
        (a, b)
    } else {
        let a = delta.t().dot(&w).dot(&delta);
        let b = delta.t().dot(&w).dot(&v).dot(&w).dot(&delta);
        (a, b)
    };
    let a_inv = a.inv()?;
    let cov = a_inv.dot(&b).dot(&a_inv);
    if cov.dim().0 != q {
        return Err(anyhow::anyhow!("cov dimension mismatch"));
    }
    Ok(cov)
}
