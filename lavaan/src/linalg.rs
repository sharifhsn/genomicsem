use anyhow::Result;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

pub fn inverse_from_eigh(
    eigvals: &Array1<f64>,
    eigvecs: &Array2<f64>,
    threshold: Option<f64>,
) -> Array2<f64> {
    let mut inv_vals = eigvals.to_vec();
    for v in &mut inv_vals {
        *v = match threshold {
            Some(t) if *v <= t => 0.0,
            _ => 1.0 / *v,
        };
    }
    let inv_diag = Array2::from_diag(&Array1::from_vec(inv_vals));
    eigvecs.dot(&inv_diag).dot(&eigvecs.t())
}

pub fn inverse_from_matrix(matrix: &Array2<f64>, threshold: Option<f64>) -> Result<Array2<f64>> {
    let (eigvals, eigvecs) = matrix.eigh(UPLO::Lower)?;
    Ok(inverse_from_eigh(&eigvals, &eigvecs, threshold))
}
