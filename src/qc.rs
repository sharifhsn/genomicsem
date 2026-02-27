use std::path::Path;

use crate::error::{GenomicSemError, Result};

pub fn check_equal_length(
    left_len: usize,
    right_len: usize,
    left_name: &str,
    right_name: &str,
) -> Result<()> {
    if left_len != right_len {
        return Err(GenomicSemError::InvalidArgument(format!(
            "Length of {left_name} and {right_name} should be equal"
        )));
    }
    Ok(())
}

pub fn check_range_f64(value: f64, min: f64, max: f64, inclusive: bool, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(GenomicSemError::InvalidArgument(format!(
            "Value of {name} should be finite"
        )));
    }
    if inclusive {
        if value <= min {
            return Err(GenomicSemError::InvalidArgument(format!(
                "Value of {name} should be above {min}"
            )));
        }
        if value >= max {
            return Err(GenomicSemError::InvalidArgument(format!(
                "Value of {name} should be below {max}"
            )));
        }
    } else {
        if value < min {
            return Err(GenomicSemError::InvalidArgument(format!(
                "Value of {name} should be above {min}"
            )));
        }
        if value > max {
            return Err(GenomicSemError::InvalidArgument(format!(
                "Value of {name} should be below {max}"
            )));
        }
    }
    Ok(())
}

pub fn check_file_exists(path: &Path, name: &str) -> Result<()> {
    if !path.exists() {
        return Err(GenomicSemError::InvalidArgument(format!(
            "File {path:?} passed to {name} does not exist"
        )));
    }
    Ok(())
}
