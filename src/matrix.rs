use anyhow::Result;
use ndarray::Array2;

use crate::types::Matrix;

pub fn ensure_square(matrix: &Matrix, name: &str) -> Result<()> {
    let n = matrix.len();
    if n == 0 {
        return Err(anyhow::anyhow!("{name} must not be empty"));
    }
    for (i, row) in matrix.iter().enumerate() {
        if row.len() != n {
            return Err(anyhow::anyhow!(
                "{name} row {i} length {} does not match {n}",
                row.len()
            ));
        }
    }
    Ok(())
}

pub fn to_array2(matrix: &Matrix) -> Result<Array2<f64>> {
    let n = matrix.len();
    let m = matrix.first().map(|row| row.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(n * m);
    for row in matrix {
        if row.len() != m {
            return Err(anyhow::anyhow!("Matrix is not rectangular"));
        }
        data.extend_from_slice(row);
    }
    Array2::from_shape_vec((n, m), data).map_err(|e| anyhow::anyhow!(e.to_string()))
}
