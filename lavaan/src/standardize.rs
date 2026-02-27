use std::collections::HashMap;

use anyhow::Result;

use crate::implied::implied_covariance;
use crate::model::{MatKind, SemModel};

pub fn std_all_map(model: &SemModel, theta: &[f64]) -> Result<HashMap<String, f64>> {
    let sigma = implied_covariance(model, theta)?;
    let n = sigma.len();
    let mut sd = vec![0.0; n];
    for i in 0..n {
        sd[i] = if sigma[i][i] > 0.0 {
            sigma[i][i].sqrt()
        } else {
            0.0
        };
    }

    let mut out = HashMap::new();
    for slot in &model.slots {
        let value = if slot.free_idx > 0 {
            theta.get(slot.free_idx - 1).copied().unwrap_or(f64::NAN)
        } else {
            slot.fixed.unwrap_or(f64::NAN)
        };
        let key = format!("{}{}{}", slot.lhs, slot.op, slot.rhs);
        let std_val = match slot.kind {
            MatKind::A => {
                let denom = sd[slot.row];
                let numer = sd[slot.col];
                if denom != 0.0 {
                    value * numer / denom
                } else {
                    f64::NAN
                }
            }
            MatKind::S => {
                if slot.row == slot.col {
                    let denom = sd[slot.row] * sd[slot.row];
                    if denom != 0.0 {
                        value / denom
                    } else {
                        f64::NAN
                    }
                } else {
                    let denom = sd[slot.row] * sd[slot.col];
                    if denom != 0.0 {
                        value / denom
                    } else {
                        f64::NAN
                    }
                }
            }
        };
        out.insert(key, std_val);
    }

    Ok(out)
}
