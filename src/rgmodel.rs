use anyhow::Result;

use crate::sem::{self, SemFit};
use crate::types::{Estimation, LdscOutput, Matrix};

#[derive(Debug, Clone)]
pub struct RgModelOutput {
    pub ldsc: LdscOutput,
    pub r: Matrix,
    pub v_r: Matrix,
    pub model_fit: SemFit,
}

pub fn rgmodel(ldsc_output: &LdscOutput, _model: &str) -> Result<RgModelOutput> {
    let trait_names = ldsc_output.trait_names.clone();
    let k = trait_names.len();
    if k == 0 {
        return Err(anyhow::anyhow!("rgmodel requires at least one trait"));
    }

    let syntax = build_rgmodel_syntax(&trait_names);

    let (s_used, _, _) = sem::smooth_if_needed(&ldsc_output.s)?;
    let (v_used, _, _) = sem::smooth_if_needed(&ldsc_output.v)?;

    let mut input = ldsc_output.clone();
    input.s = s_used;
    input.v = v_used;

    let fit = sem::usermodel_fit(&input, &syntax, Estimation::Dwls)?;

    let mut r = vec![vec![0.0; k]; k];
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        r[i][i] = 1.0;
    }

    let mut est_map = std::collections::HashMap::new();
    let mut free_map = std::collections::HashMap::new();
    for p in &fit.params {
        let key = format!("{}{}{}", p.lhs, p.op, p.rhs);
        est_map.insert(key.clone(), p.est);
        if p.free > 0 {
            free_map.insert(key, p.free);
        }
    }

    #[allow(clippy::needless_range_loop)]
    for j in 0..k {
        for i in (j + 1)..k {
            let lhs = format!("var{}", j + 1);
            let rhs = format!("var{}", i + 1);
            let key = format!("{lhs}~~{rhs}");
            if let Some(est) = est_map.get(&key) {
                r[i][j] = *est;
                r[j][i] = *est;
            } else {
                r[i][j] = f64::NAN;
                r[j][i] = f64::NAN;
            }
        }
    }

    let pairs = rg_lower_tri_pairs(k);
    let m = pairs.len();
    let mut v_r = vec![vec![f64::NAN; m]; m];
    for (a_idx, (i_a, j_a)) in pairs.iter().enumerate() {
        let lhs_a = format!("var{}", j_a + 1);
        let rhs_a = format!("var{}", i_a + 1);
        let key_a = format!("{lhs_a}~~{rhs_a}");
        let free_a = free_map.get(&key_a).copied();
        for (b_idx, (i_b, j_b)) in pairs.iter().enumerate() {
            let lhs_b = format!("var{}", j_b + 1);
            let rhs_b = format!("var{}", i_b + 1);
            let key_b = format!("{lhs_b}~~{rhs_b}");
            let free_b = free_map.get(&key_b).copied();
            if let (Some(fa), Some(fb)) = (free_a, free_b) {
                let ia = fa - 1;
                let ib = fb - 1;
                v_r[a_idx][b_idx] = fit
                    .vcov
                    .get(ia)
                    .and_then(|row| row.get(ib))
                    .copied()
                    .unwrap_or(f64::NAN);
            }
        }
    }

    Ok(RgModelOutput {
        ldsc: ldsc_output.clone(),
        r,
        v_r,
        model_fit: fit,
    })
}

fn build_rgmodel_syntax(trait_names: &[String]) -> String {
    let mut lines = Vec::new();
    for (i, name) in trait_names.iter().enumerate() {
        lines.push(format!("var{} =~ start(0.1)*{}", i + 1, name));
    }
    for i in 0..trait_names.len() {
        lines.push(format!("var{} ~~ 1*var{}", i + 1, i + 1));
    }
    for name in trait_names {
        lines.push(format!("{name} ~~ 0*{name}"));
    }
    for i in 0..trait_names.len() {
        for j in (i + 1)..trait_names.len() {
            lines.push(format!("var{} ~~ var{}", i + 1, j + 1));
        }
    }
    lines.join("\n")
}

fn rg_lower_tri_pairs(k: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for j in 0..k {
        for i in (j + 1)..k {
            out.push((i, j));
        }
    }
    out
}
