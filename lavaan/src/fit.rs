use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::Inverse;
use nlopt::{Algorithm, Nlopt, Target, approximate_gradient};
use std::cell::Cell;

use crate::implied::{
    cov2cor, implied_covariance, implied_observed, logdet_sym, to_array2, vech_col_major,
};
use crate::model::{ParamSlot, SemModel};
use crate::parser::{eval_expr, parse_model};
use crate::se::{jacobian, sandwich_covariance};
use crate::standardize::std_all_map;
use crate::stats::{FitStatsInput, compute_stats};
use crate::types::{
    DefinedEstimate, Estimation, ParTableRow, ParamEstimate, SemFit, SemFitStats, SemInput,
};

pub trait SemEngine {
    fn fit(&self, input: &SemInput) -> Result<SemFit>;
}

#[derive(Debug, Clone)]
pub struct SemEngineImpl;

impl SemEngine for SemEngineImpl {
    fn fit(&self, input: &SemInput) -> Result<SemFit> {
        fit_internal(input)
    }
}

fn fit_internal(input: &SemInput) -> Result<SemFit> {
    let mut input = input.clone();
    if input.sample_cov_rescale
        && matches!(input.estimation, Estimation::Ml)
        && let Some(n_obs) = input.n_obs
        && n_obs > 1.0
    {
        let factor = (n_obs - 1.0) / n_obs;
        scale_matrix(&mut input.s, factor);
    }

    let (mut model, spec) = if let Some(table) = input.model_table.clone() {
        SemModel::build_model_from_table(&input, &table)?
    } else {
        let spec = parse_model(&input.model)?;
        let model = SemModel::build_model(&input, &spec)?;
        (model, spec)
    };
    model.apply_constraints();

    let errors = Vec::new();
    let mut warnings = Vec::new();

    let mut theta = model.theta_start();

    let toler = input.toler.unwrap_or(1e-7);

    let (lower, upper) = model.bounds();

    let obj_data = ObjData {
        model: &model,
        s_vec: model.s_vec.clone(),
        estimation: input.estimation,
        s_obs: model.s_obs.clone(),
        logdet_s: logdet_sym(&model.s_obs),
        evals: Cell::new(0),
    };

    let obj = |x: &[f64], grad: Option<&mut [f64]>, data: &mut ObjData| -> f64 {
        data.evals.set(data.evals.get() + 1);
        let f = objective(x, data);
        if let Some(g) = grad {
            approximate_gradient(x, |x| objective(x, data), g);
        }
        f
    };

    // Use a bound-aware optimizer so simple inequality constraints are respected.
    // Note: lavaan R uses nlminb (PORT routines). Switching to nlminb-equivalent
    // methods may be a performance opportunity in the future.
    let mut opt = Nlopt::new(
        Algorithm::Slsqp,
        theta.len(),
        obj,
        Target::Minimize,
        obj_data,
    );
    let _ = opt.set_ftol_rel(toler);
    let _ = opt.set_maxeval(input.iter_max.unwrap_or(2000) as u32);
    if let Some(lb) = lower {
        let _ = opt.set_lower_bounds(&lb);
    }
    if let Some(ub) = upper {
        let _ = opt.set_upper_bounds(&ub);
    }

    let result = opt.optimize(&mut theta);
    if let Err((fail, _)) = result {
        warnings.push(format!("Optimizer failed: {:?}", fail));
    }

    let obj_data = opt.recover_user_data();
    let iterations = obj_data.evals.get();
    let fx = objective(&theta, &obj_data);

    let sigma_full = implied_covariance(&model, &theta)?;
    let sigma_obs = implied_observed(&model, &theta)?;
    let residual = residual_matrix(&model.s_obs, &sigma_obs);

    let delta = jacobian(&model, &theta)?;
    let cov = sandwich_covariance(&model, &theta)?;
    let se = diag_sqrt(&cov);

    let (chisq, df, aic, cfi, srmr, p_chisq) = compute_stats(FitStatsInput {
        s_obs: &model.s_obs,
        sigma_obs: &sigma_obs,
        gamma: &model.gamma,
        n_free: model.free.len(),
        estimation: input.estimation,
        n_obs: input.n_obs,
        fx,
    })?;

    let stats = SemFitStats {
        chisq,
        df,
        aic,
        cfi,
        srmr,
        p_chisq,
    };

    let std_map = std_all_map(&model, &theta).ok();

    let slots = sorted_slots(&model);
    let params = build_param_estimates(&slots, &theta, &se, std_map.as_ref());

    let mut defined = build_defined_estimates(&model, &spec, &theta, &cov)?;
    if matches!(input.estimation, Estimation::Ml) {
        for def in &mut defined {
            def.se = f64::NAN;
        }
    }
    let par_table = build_par_table(&slots, &theta, &se, &defined);

    let mut converged = result.is_ok();
    if !converged && let Some(dx_tol) = input.optim_dx_tol {
        let mut grad = vec![0.0; theta.len()];
        approximate_gradient(&theta, |x| objective(x, &obj_data), &mut grad);
        let max_grad = grad.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if max_grad.is_finite() && max_grad <= dx_tol {
            converged = true;
        }
    }
    if input.optim_force_converged {
        converged = true;
    }
    if !converged {
        warnings.push("lavaan WARNING: model has NOT converged!".to_string());
    }

    let cor_lv = build_cor_lv(&model, &sigma_full);

    Ok(SemFit {
        params,
        defined,
        stats,
        converged,
        iterations,
        npar: model.free.len(),
        par_table,
        delta: crate::implied::from_array2(&delta),
        wls_v: model.wls_v.clone(),
        vcov: crate::implied::from_array2(&cov),
        cor_lv,
        warnings,
        errors,
        implied: sigma_obs,
        residual,
        std_all: std_map,
    })
}

struct ObjData<'a> {
    model: &'a SemModel,
    s_vec: Vec<f64>,
    estimation: Estimation,
    s_obs: Vec<Vec<f64>>,
    logdet_s: f64,
    evals: Cell<usize>,
}

fn objective(theta: &[f64], data: &ObjData<'_>) -> f64 {
    match data.estimation {
        Estimation::Dwls => dwls_obj(theta, data),
        Estimation::Ml => ml_obj(theta, data),
    }
}

fn dwls_obj(theta: &[f64], data: &ObjData<'_>) -> f64 {
    let sigma_obs = match implied_observed(data.model, theta) {
        Ok(s) => s,
        Err(_) => return f64::INFINITY,
    };
    let sigma_vec = vech_col_major(&sigma_obs);
    let mut sum = 0.0;
    for (i, sigma_val) in sigma_vec.iter().enumerate() {
        let diff = data.s_vec[i] - sigma_val;
        let w = data.model.w_diag.get(i).copied().unwrap_or(0.0);
        sum += diff * diff * w;
    }
    sum
}

fn ml_obj(theta: &[f64], data: &ObjData<'_>) -> f64 {
    let sigma_obs = match implied_observed(data.model, theta) {
        Ok(s) => s,
        Err(_) => return f64::INFINITY,
    };
    let logdet = logdet_sym(&sigma_obs);
    if !logdet.is_finite() {
        return f64::INFINITY;
    }
    let inv = match to_array2(&sigma_obs)
        .and_then(|a| a.inv().map_err(|e| anyhow::anyhow!(e.to_string())))
    {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let s_arr = match to_array2(&data.s_obs) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let trace = (s_arr.dot(&inv)).diag().sum();
    let p = data.s_obs.len() as f64;
    let mut f = logdet + trace - data.logdet_s - p;
    if !f.is_finite() {
        f = f64::INFINITY;
    }
    f
}

fn residual_matrix(s: &[Vec<f64>], sigma: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = s.len();
    let mut out = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            out[i][j] = s[i][j] - sigma[i][j];
        }
    }
    out
}

fn diag_sqrt(cov: &Array2<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(cov.dim().0);
    for i in 0..cov.dim().0 {
        let v = cov[(i, i)];
        out.push(if v.is_finite() && v >= 0.0 {
            v.sqrt()
        } else {
            f64::NAN
        });
    }
    out
}

fn build_param_estimates(
    slots: &[&ParamSlot],
    theta: &[f64],
    se: &[f64],
    std_map: Option<&std::collections::HashMap<String, f64>>,
) -> Vec<ParamEstimate> {
    let mut out = Vec::new();
    for slot in slots {
        let est = if slot.free_idx > 0 {
            theta.get(slot.free_idx - 1).copied().unwrap_or(f64::NAN)
        } else {
            slot.fixed.unwrap_or(f64::NAN)
        };
        let se_val = if slot.free_idx > 0 {
            se.get(slot.free_idx - 1).copied().unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        let key = format!("{}{}{}", slot.lhs, slot.op, slot.rhs);
        let std_all = std_map.and_then(|m| m.get(&key).copied());
        out.push(ParamEstimate {
            lhs: slot.lhs.clone(),
            op: slot.op.clone(),
            rhs: slot.rhs.clone(),
            label: slot.label.clone(),
            free: slot.free_idx,
            est,
            se: se_val,
            est_std: None,
            se_std: None,
            est_std_all: std_all,
        });
    }
    out
}

fn build_defined_estimates(
    model: &SemModel,
    spec: &crate::parser::ModelSpec,
    theta: &[f64],
    cov: &Array2<f64>,
) -> Result<Vec<DefinedEstimate>> {
    let mut out = Vec::new();
    if spec.defines.is_empty() {
        return Ok(out);
    }
    let mut label_vals = std::collections::HashMap::new();
    for slot in &model.slots {
        if let Some(label) = &slot.label {
            if slot.free_idx > 0 {
                label_vals.insert(label.clone(), theta[slot.free_idx - 1]);
            } else if let Some(fixed) = slot.fixed {
                label_vals.insert(label.clone(), fixed);
            }
        }
    }

    for def in &spec.defines {
        let est = eval_expr(&def.expr, &label_vals)?;
        let grad = defined_gradient(&def.expr, &label_vals, model, theta)?;
        let mut var = 0.0;
        for i in 0..grad.len() {
            for j in 0..grad.len() {
                var += grad[i] * cov[(i, j)] * grad[j];
            }
        }
        let se_val = if var.is_finite() && var >= 0.0 {
            var.sqrt()
        } else {
            f64::NAN
        };
        out.push(DefinedEstimate {
            name: def.name.clone(),
            expr: def.expr_raw.clone(),
            est,
            se: se_val,
        });
    }
    Ok(out)
}

fn build_par_table(
    slots: &[&ParamSlot],
    theta: &[f64],
    se: &[f64],
    defined: &[DefinedEstimate],
) -> Vec<ParTableRow> {
    let mut out = Vec::new();
    for slot in slots {
        let est = if slot.free_idx > 0 {
            theta.get(slot.free_idx - 1).copied().unwrap_or(f64::NAN)
        } else {
            slot.fixed.unwrap_or(f64::NAN)
        };
        let se_val = if slot.free_idx > 0 {
            se.get(slot.free_idx - 1).copied().unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        let ustart = slot.start.unwrap_or_else(|| slot.fixed.unwrap_or(est));
        out.push(ParTableRow {
            lhs: slot.lhs.clone(),
            op: slot.op.clone(),
            rhs: slot.rhs.clone(),
            free: slot.free_idx,
            label: slot.label.clone(),
            ustart,
            est,
            se: se_val,
        });
    }

    for def in defined {
        out.push(ParTableRow {
            lhs: def.name.clone(),
            op: ":=".to_string(),
            rhs: def.expr.clone(),
            free: 0,
            label: None,
            ustart: def.est,
            est: def.est,
            se: def.se,
        });
    }

    out
}

fn sorted_slots(model: &SemModel) -> Vec<&ParamSlot> {
    let mut slots: Vec<&ParamSlot> = model.slots.iter().collect();
    let latent_set: std::collections::HashSet<&str> =
        model.latent_names.iter().map(|s| s.as_str()).collect();
    let obs_set: std::collections::HashSet<&str> =
        model.obs_names.iter().map(|s| s.as_str()).collect();

    slots.sort_by_key(|slot| {
        let op_rank = match slot.op.as_str() {
            "=~" => 0,
            "~~" => 1,
            "~" => 2,
            _ => 3,
        };
        let (row_key, col_key) = match slot.op.as_str() {
            "=~" => (slot.col, slot.row),
            "~" => (slot.row, slot.col),
            "~~" => {
                let min = slot.row.min(slot.col);
                let max = slot.row.max(slot.col);
                (min, max)
            }
            _ => (slot.row, slot.col),
        };
        let group = if slot.op == "~~" {
            let lhs_latent = latent_set.contains(slot.lhs.as_str());
            let rhs_latent = latent_set.contains(slot.rhs.as_str());
            let lhs_obs = obs_set.contains(slot.lhs.as_str());
            let rhs_obs = obs_set.contains(slot.rhs.as_str());
            if lhs_latent && rhs_latent {
                0
            } else if lhs_obs && rhs_obs {
                1
            } else {
                2
            }
        } else {
            0
        };
        (op_rank, group, row_key, col_key)
    });
    slots
}

fn build_cor_lv(
    model: &SemModel,
    sigma_full: &crate::types::Matrix,
) -> Option<crate::types::Matrix> {
    if model.latent_names.is_empty() {
        return None;
    }
    let mut idx = Vec::with_capacity(model.latent_names.len());
    for name in &model.latent_names {
        if let Some(pos) = model.var_names.iter().position(|v| v == name) {
            idx.push(pos);
        }
    }
    if idx.is_empty() {
        return None;
    }
    let mut cov = vec![vec![0.0; idx.len()]; idx.len()];
    for (i, &ri) in idx.iter().enumerate() {
        for (j, &cj) in idx.iter().enumerate() {
            cov[i][j] = sigma_full[ri][cj];
        }
    }
    Some(cov2cor(&cov))
}

fn scale_matrix(matrix: &mut crate::types::Matrix, factor: f64) {
    for row in matrix.iter_mut() {
        for val in row.iter_mut() {
            *val *= factor;
        }
    }
}

fn defined_gradient(
    expr: &crate::parser::Expr,
    label_vals: &std::collections::HashMap<String, f64>,
    model: &SemModel,
    theta: &[f64],
) -> Result<Vec<f64>> {
    let mut grad = vec![0.0; theta.len()];
    for i in 0..theta.len() {
        let eps = 1e-6 * theta[i].abs().max(1.0);
        let mut map_plus = label_vals.clone();
        let mut map_minus = label_vals.clone();
        // update all labels tied to this free parameter
        for slot in &model.slots {
            if slot.free_idx == i + 1
                && let Some(label) = &slot.label
            {
                map_plus.insert(label.clone(), theta[i] + eps);
                map_minus.insert(label.clone(), theta[i] - eps);
            }
        }
        let f_plus = eval_expr(expr, &map_plus)?;
        let f_minus = eval_expr(expr, &map_minus)?;
        grad[i] = (f_plus - f_minus) / (2.0 * eps);
    }
    Ok(grad)
}
