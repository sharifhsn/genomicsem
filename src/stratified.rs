use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use polars::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::io::read_table;
use crate::logging::{log_line, warn_line};
use crate::sem::smooth_if_needed;
use crate::types::{Matrix, StratifiedLdscOutput};
use lavaan as lavaan_crate;
use lavaan_crate::SemEngine as LavaanSemEngineTrait;

#[derive(Debug, Clone)]
pub struct SLdscConfig {
    pub traits: Vec<PathBuf>,
    pub sample_prev: Option<Vec<Option<f64>>>,
    pub population_prev: Option<Vec<Option<f64>>>,
    pub ld: Vec<PathBuf>,
    pub wld: PathBuf,
    pub frq: PathBuf,
    pub trait_names: Option<Vec<String>>,
    pub n_blocks: usize,
    pub ldsc_log: Option<PathBuf>,
    pub exclude_cont: bool,
}

pub fn s_ldsc(config: &SLdscConfig) -> Result<StratifiedLdscOutput> {
    let mut log = open_log_file(config)?;
    let begin = std::time::SystemTime::now();
    log_line(&mut log, &format!("Analysis started at {:?}", begin), true)?;

    let n_traits = config.traits.len();
    if n_traits == 0 {
        return Err(anyhow::anyhow!("No traits supplied"));
    }

    let mut n_blocks = config.n_blocks;
    if n_traits > 18 {
        n_blocks = ((n_traits + 1) * (n_traits + 2)) / 2 + 1;
        log_line(
            &mut log,
            &format!("Setting n.blocks to {n_blocks} based on number of traits"),
            true,
        )?;
        if n_blocks > 1000 {
            warn_line(
                &mut log,
                "WARNING: The number of blocks needed to estimate V is > 1000, which may bias results.",
            )?;
        }
    }

    let sample_prev = normalize_optional_vec(config.sample_prev.clone(), n_traits)?;
    let population_prev = normalize_optional_vec(config.population_prev.clone(), n_traits)?;
    let trait_names = resolve_trait_names(config, n_traits)?;

    log_line(
        &mut log,
        &format!(
            "The following traits are being analyzed: {}",
            trait_names.join(", ")
        ),
        true,
    )?;
    log_line(
        &mut log,
        &format!(
            "The following annotations were added to the model: {}",
            config
                .ld
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        true,
    )?;

    let (ld1, ld2) = split_ld_dirs(&config.ld)?;
    log_line(
        &mut log,
        &format!("Reading in LD scores from {}", ld1.display()),
        true,
    )?;
    let mut x = read_ld_score_dir(&ld1)?;
    drop_cols_if_present(&mut x, &["CM", "MAF"])?;

    for extra in &ld2 {
        log_line(
            &mut log,
            &format!("Reading in LD scores from {}", extra.display()),
            true,
        )?;
        let mut extra_x = read_ld_score_dir(extra)?;
        drop_cols_if_present(&mut extra_x, &["CM", "MAF", "CHR", "BP"])?;
        x = x.join(&extra_x, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    }

    let mut m = read_m_dir(&ld1)?;
    for extra in &ld2 {
        let mut extra_m = read_m_dir(extra)?;
        m.append(&mut extra_m);
    }

    let annotation_cols = annotation_columns(&x);
    if m.len() != annotation_cols.len() {
        return Err(anyhow::anyhow!(
            "Number of annotations in M files ({}) does not match LD score columns ({})",
            m.len(),
            annotation_cols.len()
        ));
    }

    let (annot_matrix, annotation_cols, m, select) = build_annotation_matrix(
        &ld1,
        &ld2,
        &config.frq,
        &annotation_cols,
        m,
        config.exclude_cont,
        &mut x,
        &mut log,
    )?;

    let n_annot = annotation_cols.len();
    if n_annot <= 1 {
        return Err(anyhow::anyhow!(
            "The function cannot handle single annotation LDSR yet."
        ));
    }
    let m_tot: f64 = m.iter().sum();
    let mut overlap = vec![vec![0.0; n_annot]; n_annot];
    for i in 0..n_annot {
        let denom = m[i];
        for j in 0..n_annot {
            overlap[i][j] = if denom != 0.0 {
                annot_matrix[i][j] / denom
            } else {
                0.0
            };
        }
    }

    log_line(
        &mut log,
        &format!(
            "LD scores contain {} SNPs and {} annotations",
            x.height(),
            n_annot
        ),
        true,
    )?;

    log_line(
        &mut log,
        &format!("Reading weighted LD scores from {}", config.wld.display()),
        true,
    )?;
    let mut w = read_ld_score_dir(&config.wld)?;
    drop_cols_if_present(&mut w, &["CM", "MAF", "CHR", "BP"])?;
    rename_last_column(&mut w, "wLD")?;

    let mut all_y = Vec::with_capacity(n_traits);
    for (idx, trait_path) in config.traits.iter().enumerate() {
        let df = read_trait_for_sldsc(
            trait_path,
            &w,
            &x,
            &annotation_cols,
            &mut log,
            idx + 1,
            n_traits,
        )?;
        all_y.push(df);
    }

    let n_v = n_traits * (n_traits + 1) / 2;
    let mut total_pseudo = vec![vec![0.0; n_v * n_annot]; n_blocks];
    let mut s_list = vec![vec![vec![f64::NAN; n_traits]; n_traits]; n_annot];
    let mut tau_list = vec![vec![vec![f64::NAN; n_traits]; n_traits]; n_annot];
    let mut n_vec = vec![f64::NAN; n_v];
    let mut i_mat = vec![vec![f64::NAN; n_traits]; n_traits];

    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;

    let mut s_index = 0usize;
    for j in 0..n_traits {
        let y1 = &all_y[j];
        let n = extract_f64_column(y1, "N")?;
        let z = extract_f64_column(y1, "Z")?;
        let wld = extract_f64_column(y1, "wLD")?;
        let chi: Vec<f64> = z.iter().map(|v| v * v).collect();
        let ld_cols = extract_ld_columns(y1, &annotation_cols)?;
        let x_tot = row_sums(&ld_cols);

        let (init_w, _tot_agg) = compute_initial_weights(&chi, &n, &x_tot, &wld, m_tot);
        let weights = normalize_weights(&init_w);
        let n_bar = mean(&n);

        let (reg, delete_values) = weighted_regression_blocks(
            &ld_cols,
            &chi,
            &weights,
            n_blocks,
            Some((n.as_slice(), n_bar)),
        )?;

        let intercept = reg[n_annot];
        i_mat[j][j] = intercept;

        let coefs: Vec<f64> = reg[..n_annot].iter().map(|v| v / n_bar).collect();
        let cats: Vec<f64> = coefs.iter().zip(&m).map(|(c, m)| c * m).collect();
        let reg_tot: f64 = cats.iter().sum();

        let pseudo_tau = jackknife_pseudo(&reg, &delete_values, n_blocks);
        let jack_cov = covariance(&pseudo_tau);
        let intercept_se = jack_cov[n_annot][n_annot].sqrt();
        let coef_cov = scale_covariance(&jack_cov, n_annot, n_bar);
        let tot_se = total_se(&coef_cov, &m);

        log_line(
            &mut log,
            &format!("Heritability results for trait {}", trait_names[j]),
            true,
        )?;
        log_line(&mut log, &format!("Mean Chi^2: {:.4}", mean(&chi)), true)?;
        if let Some(lambda) = lambda_gc(&chi) {
            log_line(&mut log, &format!("Lambda GC: {:.4}", lambda), true)?;
        }
        log_line(
            &mut log,
            &format!("Intercept: {:.4} ({:.4})", intercept, intercept_se),
            true,
        )?;
        let ratio = safe_div(intercept - 1.0, mean(&chi) - 1.0);
        let ratio_se = safe_div(intercept_se, mean(&chi) - 1.0);
        if ratio.is_finite() && ratio_se.is_finite() {
            log_line(
                &mut log,
                &format!("Ratio: {:.4} ({:.4})", ratio, ratio_se),
                true,
            )?;
        }
        log_line(
            &mut log,
            &format!("Total Observed Scale h2: {:.4} ({:.4})", reg_tot, tot_se),
            true,
        )?;

        let hsq_tot = mat_vec_mul(&overlap, &cats);
        for f in 0..n_annot {
            s_list[f][j][j] = hsq_tot[f];
            tau_list[f][j][j] = cats[f];
        }

        let pseudo = jackknife_pseudo_cats(&cats, &delete_values, n_blocks, n_bar, &m);
        let offset = s_index * n_annot;
        for b in 0..n_blocks {
            total_pseudo[b][offset..offset + n_annot].copy_from_slice(&pseudo[b]);
        }
        n_vec[s_index] = n_bar;
        s_index += 1;

        for k in (j + 1)..n_traits {
            let y2 = &all_y[k];
            let merged = merge_trait_pair(y1, y2, &annotation_cols)?;
            let n_x = extract_f64_column(&merged, "N")?;
            let n_y = extract_f64_column(&merged, "N_y")?;
            let z_x = extract_f64_column(&merged, "Z")?;
            let z_y = extract_f64_column(&merged, "Z_y")?;
            let a1_x = extract_string_column(&merged, "A1")?;
            let a1_y = extract_string_column(&merged, "A1_y")?;
            let wld = extract_f64_column(&merged, "wLD")?;
            let ld_cols = extract_ld_columns(&merged, &annotation_cols)?;

            let mut z_x_aligned = vec![0.0; z_x.len()];
            for i in 0..z_x.len() {
                z_x_aligned[i] = if a1_x[i] == a1_y[i] { z_x[i] } else { -z_x[i] };
            }
            let chi1: Vec<f64> = z_x_aligned.iter().map(|v| v * v).collect();
            let chi2: Vec<f64> = z_y.iter().map(|v| v * v).collect();
            let zz: Vec<f64> = z_x_aligned.iter().zip(&z_y).map(|(a, b)| a * b).collect();

            let x_tot = row_sums(&ld_cols);
            let (init_w1, _tot1) = compute_initial_weights(&chi1, &n_x, &x_tot, &wld, m_tot);
            let (init_w2, _tot2) = compute_initial_weights(&chi2, &n_y, &x_tot, &wld, m_tot);
            let weights_cov = combine_weights(&init_w1, &init_w2);

            let n_bar = (mean(&n_x) * mean(&n_y)).sqrt();
            let (reg, delete_values) =
                weighted_regression_blocks(&ld_cols, &zz, &weights_cov, n_blocks, None)?;

            let intercept = reg[n_annot];
            i_mat[j][k] = intercept;
            i_mat[k][j] = intercept;

            let coefs: Vec<f64> = reg[..n_annot].iter().map(|v| v / n_bar).collect();
            let cats: Vec<f64> = coefs.iter().zip(&m).map(|(c, m)| c * m).collect();
            let reg_tot: f64 = cats.iter().sum();

            let pseudo_tau = jackknife_pseudo(&reg, &delete_values, n_blocks);
            let jack_cov = covariance(&pseudo_tau);
            let coef_cov = scale_covariance(&jack_cov, n_annot, n_bar);
            let tot_se = total_se(&coef_cov, &m);

            let mean_zz = mean(&zz);
            log_line(
                &mut log,
                &format!(
                    "Results for covariance between {} and {}",
                    trait_names[j], trait_names[k]
                ),
                true,
            )?;
            log_line(&mut log, &format!("Mean Z*Z: {:.4}", mean_zz), true)?;
            log_line(
                &mut log,
                &format!(
                    "Cross trait Intercept: {:.4} ({:.4})",
                    intercept,
                    jack_cov[n_annot][n_annot].sqrt()
                ),
                true,
            )?;
            log_line(
                &mut log,
                &format!("cov_g: {:.4} ({:.4})", reg_tot, tot_se),
                true,
            )?;

            let cov_tot = mat_vec_mul(&overlap, &cats);
            for f in 0..n_annot {
                s_list[f][j][k] = cov_tot[f];
                s_list[f][k][j] = cov_tot[f];
                tau_list[f][j][k] = cats[f];
                tau_list[f][k][j] = cats[f];
            }

            let pseudo = jackknife_pseudo_cats(&cats, &delete_values, n_blocks, n_bar, &m);
            let offset = s_index * n_annot;
            for b in 0..n_blocks {
                total_pseudo[b][offset..offset + n_annot].copy_from_slice(&pseudo[b]);
            }
            n_vec[s_index] = n_bar;
            s_index += 1;

            let gcov_z = safe_div(reg_tot, tot_se);
            if gcov_z.is_finite() {
                let gcov_p = 2.0 * (1.0 - normal.cdf(gcov_z.abs()));
                log_line(
                    &mut log,
                    &format!("gcov Z: {:.3}, p={:.3e}", gcov_z, gcov_p),
                    true,
                )?;
            }
        }
    }

    let total_pseudo_cov = covariance(&total_pseudo);
    let total_pseudo_cov = scale_matrix(&total_pseudo_cov, 1.0 / n_blocks as f64);

    let mut v_out = vec![vec![vec![f64::NAN; n_v]; n_v]; n_annot];
    let mut v_out_tau = vec![vec![vec![f64::NAN; n_v]; n_v]; n_annot];

    for u in 0..n_v {
        for p in 0..n_v {
            let block = block_from_cov(&total_pseudo_cov, u, p, n_annot);
            let sample_var = sample_var_from_block(&block, &overlap);
            for annot in 0..n_annot {
                v_out[annot][u][p] = sample_var[annot];
                v_out_tau[annot][u][p] = block[annot][annot];
            }
        }
    }

    let liab = liability_vector(&sample_prev, &population_prev)?;
    let mut s = Vec::with_capacity(n_annot);
    let mut s_tau = Vec::with_capacity(n_annot);
    let mut v = Vec::with_capacity(n_annot);
    let mut v_tau = Vec::with_capacity(n_annot);

    for annot in 0..n_annot {
        let s_scaled = scale_liability(&s_list[annot], &liab);
        let s_tau_scaled = scale_liability(&tau_list[annot], &liab);

        let scale_o = lower_triangle_ratio(&s_scaled, &s_list[annot]);
        let scale_o_tau = lower_triangle_ratio(&s_tau_scaled, &tau_list[annot]);

        let v_scaled = scale_v(&v_out[annot], &scale_o);
        let v_tau_scaled = scale_v(&v_out_tau[annot], &scale_o_tau);

        s.push(s_scaled);
        s_tau.push(s_tau_scaled);
        v.push(v_scaled);
        v_tau.push(v_tau_scaled);
    }

    let prop = if !m.is_empty() {
        let base = m[0];
        m.iter().map(|v| safe_div(*v, base)).collect()
    } else {
        vec![]
    };

    let end = std::time::SystemTime::now();
    log_line(&mut log, &format!("Analysis ended at {:?}", end), true)?;

    Ok(StratifiedLdscOutput {
        s,
        v,
        s_tau,
        v_tau,
        i: i_mat,
        n: n_vec,
        m,
        prop,
        select,
        annotation_names: annotation_cols,
        trait_names,
    })
}

#[derive(Debug, Clone)]
pub struct EnrichConfig {
    pub model: String,
    pub params: Vec<String>,
    pub fix: String,
    pub std_lv: bool,
    pub rm_flank: bool,
    pub tau: bool,
    pub base: bool,
    pub toler: Option<f64>,
    pub fixparam: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct EnrichOutput {
    pub results: Vec<DataFrame>,
    pub base_results: Option<DataFrame>,
}

pub fn enrich(s_covstruc: &StratifiedLdscOutput, config: &EnrichConfig) -> Result<EnrichOutput> {
    let model_has_ops =
        config.model.contains('~') || config.model.contains('=') || config.model.contains('+');
    if !model_has_ops {
        tracing::warn!(
            "Model string may be quoted; remove surrounding quotes if model parsing fails."
        );
    }
    // NOTE: lavaan's enrich uses `toler` in solve(..., tol=...) for sandwich "bread".
    // We currently do not apply `toler` because ndarray-linalg doesn't expose a tolerance
    // knob on inversion. A faithful port would require a custom pseudo-inverse or SVD.

    let mut params: Vec<String> = config.params.iter().map(|s| s.replace(' ', "")).collect();
    let mut fixparam: Vec<String> = config
        .fixparam
        .clone()
        .unwrap_or_default()
        .iter()
        .map(|s| s.replace(' ', ""))
        .collect();
    params.retain(|p| !p.is_empty());
    fixparam.retain(|p| !p.is_empty());

    let (s_list, v_list) = if config.tau {
        (&s_covstruc.s_tau, &s_covstruc.v_tau)
    } else {
        (&s_covstruc.s, &s_covstruc.v)
    };
    if s_list.is_empty() {
        return Err(anyhow::anyhow!("enrich requires stratified LDSC output"));
    }
    let (annotation_names, select) = enrich_annotation_meta(s_covstruc);
    let (s_base, v_base, trait_names, keep_idx) = subset_by_model(
        &s_list[0],
        &v_list[0],
        &s_covstruc.trait_names,
        &config.model,
    )?;

    let s_base_orig = s_base.clone();
    let v_base_orig = v_base.clone();
    let (s_base, _smoothed_s, _ld_sdiff) = smooth_if_needed(&s_base_orig)?;
    let (v_base, _smoothed_v, _ld_sdiff2) = smooth_if_needed(&v_base_orig)?;

    let wls_v = wls_v_diag_inverse(&v_base);
    let mut model = config.model.clone();
    let mut fit = fit_sem_model(
        &s_base,
        &v_base,
        Some(wls_v.clone()),
        &model,
        &trait_names,
        config.std_lv,
    )?;
    if !fit.converged || !fit.errors.is_empty() {
        // NOTE: R's enrich builds a more complex fallback model (write.Model1).
        // We only add residual bounds to keep the fallback simple.
        model = add_resid_bounds(&model, &trait_names);
        fit = fit_sem_model(
            &s_base,
            &v_base,
            Some(wls_v.clone()),
            &model,
            &trait_names,
            config.std_lv,
        )?;
    }
    if !fit.converged || !fit.errors.is_empty() {
        return Err(anyhow::anyhow!(
            "Baseline annotation model failed to converge"
        ));
    }

    let mut model_table = fit.par_table.clone();
    for row in &mut model_table {
        row.free = 0;
    }
    let mut free_idx = 1usize;
    for row in &mut model_table {
        let key = format!("{}{}{}", row.lhs, row.op, row.rhs);
        let is_param = params.iter().any(|p| p == &key);
        let is_fixed = fixparam.iter().any(|p| p == &key);
        let allow = match config.fix.as_str() {
            "regressions" => is_param || row.op == "~~",
            "covariances" => is_param || row.op == "~" || row.op == "=~" || row.lhs == row.rhs,
            "variances" => {
                is_param
                    || row.op == "~"
                    || row.op == "=~"
                    || (row.op == "~~" && row.lhs != row.rhs)
            }
            _ => is_param || row.op == "~~",
        };
        if allow && !is_fixed {
            row.free = free_idx;
            free_idx += 1;
        }
        row.ustart = row.est;
        if row.free > 0 {
            row.ustart = 1.0;
        }
    }
    model_table.retain(|row| row.op != "==");

    if model_table.iter().all(|r| r.free == 0) {
        return Err(anyhow::anyhow!(
            "All parameters are fixed from the baseline model"
        ));
    }
    if model_table.iter().all(|r| r.free > 0) {
        tracing::warn!(
            "All parameters are freely estimated from the baseline model; enrichment results may not be trustworthy"
        );
    }

    let baseline_param_map = param_estimates_from_table(&fit.par_table, &params);
    let base_results = if config.base {
        Some(build_base_results(&fit, &model_table)?)
    } else {
        None
    };

    let check_fit = fit_sem_table(
        &s_base,
        &v_base,
        Some(wls_v.clone()),
        model_table.clone(),
        &trait_names,
        config.std_lv,
    );
    match check_fit {
        Ok(check_fit) => {
            let check_map = param_estimates_from_table(&check_fit.par_table, &params);
            let mut max_round = f64::NEG_INFINITY;
            for (key, base_est) in baseline_param_map.iter() {
                if let Some(check_est) = check_map.get(key) {
                    let diff = base_est - check_est;
                    max_round = max_round.max(diff.round());
                }
            }
            if max_round.is_finite() && max_round != 0.0 {
                tracing::warn!(
                    "Fixed model re-estimated in baseline annotation differs from freely estimated model; model may be poorly specified."
                );
            }
        }
        Err(err) => {
            tracing::warn!("Baseline fixed-model check failed: {err}. Results may be unstable.");
        }
    }

    let param_specs = collect_param_specs(&model_table, &params);
    if param_specs.is_empty() {
        return Err(anyhow::anyhow!(
            "None of the requested params were found in the model"
        ));
    }
    let mut per_param_rows: Vec<Vec<EnrichRow>> =
        vec![Vec::with_capacity(s_list.len()); param_specs.len()];

    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    for (idx, s_mat) in s_list.iter().enumerate() {
        let annotation = annotation_names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("Annot{idx}"));
        let select_flag = select.get(idx).copied().unwrap_or(1);
        if select_flag != 1 {
            for (p_idx, spec) in param_specs.iter().enumerate() {
                per_param_rows[p_idx].push(EnrichRow {
                    annotation: annotation.clone(),
                    lhs: spec.lhs.clone(),
                    op: spec.op.clone(),
                    rhs: spec.rhs.clone(),
                    cov_smooth: f64::NAN,
                    z_smooth: f64::NAN,
                    enrichment: f64::NAN,
                    enrichment_se: f64::NAN,
                    enrichment_p: f64::NAN,
                    error: "0".to_string(),
                    warning: "This annotation was not analyzed as it is either a continuous or flanking annotation.".to_string(),
                });
            }
            continue;
        }

        let (s_sub, v_sub) = subset_s_v(s_mat, &v_list[idx], &keep_idx)?;
        if diag_all_negative(&s_sub) {
            for (p_idx, spec) in param_specs.iter().enumerate() {
                per_param_rows[p_idx].push(EnrichRow {
                    annotation: annotation.clone(),
                    lhs: spec.lhs.clone(),
                    op: spec.op.clone(),
                    rhs: spec.rhs.clone(),
                    cov_smooth: f64::NAN,
                    z_smooth: f64::NAN,
                    enrichment: f64::NAN,
                    enrichment_se: f64::NAN,
                    enrichment_p: f64::NAN,
                    error: "0".to_string(),
                    warning: "This annotation was not analyzed as all heritability estimates were below 0.".to_string(),
                });
            }
            continue;
        }

        let s_sub_orig = s_sub.clone();
        let v_sub_orig = v_sub.clone();
        let (s_sub, _smoothed_s, ld_sdiff) = smooth_if_needed(&s_sub_orig)?;
        let (v_sub, _smoothed_v, _ld_sdiff2) = smooth_if_needed(&v_sub_orig)?;
        let z_diff = z_diff_metric_prepost(&s_sub_orig, &s_sub, &v_sub_orig, &v_sub);
        let wls_v = wls_v_diag_inverse(&v_sub);

        let fit_part = fit_sem_table(
            &s_sub,
            &v_sub,
            Some(wls_v),
            model_table.clone(),
            &trait_names,
            config.std_lv,
        );

        let (param_map, err, warn) = match fit_part {
            Ok(fit_part) => {
                let map = param_estimates_from_fit(&fit_part);
                let err = if fit_part.errors.is_empty() {
                    "0".to_string()
                } else {
                    fit_part.errors.join("; ")
                };
                let warn = if fit_part.warnings.is_empty() {
                    "0".to_string()
                } else {
                    fit_part.warnings.join("; ")
                };
                (Some(map), err, warn)
            }
            Err(e) => (None, e.to_string(), "0".to_string()),
        };

        for (p_idx, spec) in param_specs.iter().enumerate() {
            let base_est = baseline_param_map
                .get(&spec.key)
                .copied()
                .unwrap_or(f64::NAN);
            let (est, se) = param_map
                .as_ref()
                .and_then(|m| m.get(&spec.key).copied())
                .unwrap_or((f64::NAN, f64::NAN));
            let prop = s_covstruc.prop.get(idx).copied().unwrap_or(1.0);
            let enrichment = if base_est.is_finite() && prop != 0.0 {
                (est / base_est) / prop
            } else {
                f64::NAN
            };
            let enrichment_se = if base_est.is_finite() && prop != 0.0 {
                (se / base_est.abs()) / prop
            } else {
                f64::NAN
            };
            let enrichment_p = if enrichment_se.is_finite() && enrichment_se != 0.0 {
                let z = (enrichment - 1.0) / enrichment_se;
                1.0 - normal.cdf(z)
            } else {
                f64::NAN
            };
            per_param_rows[p_idx].push(EnrichRow {
                annotation: annotation.clone(),
                lhs: spec.lhs.clone(),
                op: spec.op.clone(),
                rhs: spec.rhs.clone(),
                cov_smooth: ld_sdiff,
                z_smooth: z_diff,
                enrichment,
                enrichment_se,
                enrichment_p,
                error: err.clone(),
                warning: warn.clone(),
            });
        }
    }

    let mut results = Vec::new();
    for rows in per_param_rows {
        let mut df = enrich_rows_to_df(&rows)?;
        if config.rm_flank && df.column("Warning").is_ok() {
            let warning_col = df.column("Warning")?.str()?;
            let mask: BooleanChunked = warning_col
                    .into_iter()
                    .map(|v| match v {
                        Some(val) => val != "This annotation was not analyzed as it is either a continuous or flanking annotation.",
                        None => true,
                    })
                    .collect();
            df = df.filter(&mask)?;
        }
        results.push(df);
    }

    Ok(EnrichOutput {
        results,
        base_results,
    })
}

#[derive(Clone)]
struct ParamSpec {
    key: String,
    lhs: String,
    op: String,
    rhs: String,
}

#[derive(Clone)]
struct EnrichRow {
    annotation: String,
    lhs: String,
    op: String,
    rhs: String,
    cov_smooth: f64,
    z_smooth: f64,
    enrichment: f64,
    enrichment_se: f64,
    enrichment_p: f64,
    error: String,
    warning: String,
}

fn enrich_rows_to_df(rows: &[EnrichRow]) -> Result<DataFrame> {
    let mut annotation = Vec::with_capacity(rows.len());
    let mut lhs = Vec::with_capacity(rows.len());
    let mut op = Vec::with_capacity(rows.len());
    let mut rhs = Vec::with_capacity(rows.len());
    let mut cov_smooth = Vec::with_capacity(rows.len());
    let mut z_smooth = Vec::with_capacity(rows.len());
    let mut enrichment = Vec::with_capacity(rows.len());
    let mut enrichment_se = Vec::with_capacity(rows.len());
    let mut enrichment_p = Vec::with_capacity(rows.len());
    let mut error = Vec::with_capacity(rows.len());
    let mut warning = Vec::with_capacity(rows.len());

    for row in rows {
        annotation.push(row.annotation.clone());
        lhs.push(row.lhs.clone());
        op.push(row.op.clone());
        rhs.push(row.rhs.clone());
        cov_smooth.push(row.cov_smooth);
        z_smooth.push(row.z_smooth);
        enrichment.push(row.enrichment);
        enrichment_se.push(row.enrichment_se);
        enrichment_p.push(row.enrichment_p);
        error.push(row.error.clone());
        warning.push(row.warning.clone());
    }

    let df = DataFrame::new(
        rows.len(),
        vec![
            Series::new("Annotation".into(), annotation).into(),
            Series::new("lhs".into(), lhs).into(),
            Series::new("op".into(), op).into(),
            Series::new("rhs".into(), rhs).into(),
            Series::new("Cov_Smooth".into(), cov_smooth).into(),
            Series::new("Z_smooth".into(), z_smooth).into(),
            Series::new("Enrichment".into(), enrichment).into(),
            Series::new("Enrichment_SE".into(), enrichment_se).into(),
            Series::new("Enrichment_p_value".into(), enrichment_p).into(),
            Series::new("Error".into(), error).into(),
            Series::new("Warning".into(), warning).into(),
        ],
    )?;
    Ok(df)
}

fn enrich_annotation_meta(s_covstruc: &StratifiedLdscOutput) -> (Vec<String>, Vec<i32>) {
    let mut names = s_covstruc.annotation_names.clone();
    let mut select = s_covstruc.select.clone();
    if names.len() + 1 == s_covstruc.s.len() {
        names.insert(0, "Base".to_string());
    }
    if select.len() + 1 == s_covstruc.s.len() {
        select.insert(0, 1);
    }
    if names.len() != s_covstruc.s.len() {
        names = (0..s_covstruc.s.len())
            .map(|i| format!("Annot{i}"))
            .collect();
    }
    if select.len() != s_covstruc.s.len() {
        select = vec![1; s_covstruc.s.len()];
    }
    (names, select)
}

fn subset_by_model(
    s: &Matrix,
    v: &Matrix,
    trait_names: &[String],
    model: &str,
) -> Result<(Matrix, Matrix, Vec<String>, Vec<usize>)> {
    if trait_names.is_empty() {
        return Err(anyhow::anyhow!("trait names required for enrich"));
    }
    let mut keep = Vec::new();
    for (idx, name) in trait_names.iter().enumerate() {
        if model_contains_name(model, name) {
            keep.push(idx);
        }
    }
    if keep.is_empty() {
        return Err(anyhow::anyhow!(
            "None of the trait names in the LDSC output match names in the model"
        ));
    }
    let s_sub = subset_square(s, &keep);
    let v_sub = subset_sampling_covariance(v, &keep, trait_names.len())?;
    let names = keep.iter().map(|&i| trait_names[i].clone()).collect();
    Ok((s_sub, v_sub, names, keep))
}

fn subset_s_v(s: &Matrix, v: &Matrix, keep: &[usize]) -> Result<(Matrix, Matrix)> {
    let s_sub = subset_square(s, keep);
    let v_sub = subset_sampling_covariance(v, keep, s.len())?;
    Ok((s_sub, v_sub))
}

fn subset_square(matrix: &Matrix, keep: &[usize]) -> Matrix {
    let mut out = vec![vec![0.0; keep.len()]; keep.len()];
    for (i_out, &i) in keep.iter().enumerate() {
        for (j_out, &j) in keep.iter().enumerate() {
            out[i_out][j_out] = matrix[i][j];
        }
    }
    out
}

fn subset_sampling_covariance(v: &Matrix, keep: &[usize], k: usize) -> Result<Matrix> {
    let z = k * (k + 1) / 2;
    if v.len() != z {
        return Err(anyhow::anyhow!(
            "V matrix must be {z}x{z} for {k} traits; found {}x{}",
            v.len(),
            v.first().map(|row| row.len()).unwrap_or(0)
        ));
    }
    let mut indices = Vec::new();
    for (pos_j, &j) in keep.iter().enumerate() {
        for &i in keep.iter().skip(pos_j) {
            indices.push(vech_index(i, j, k));
        }
    }
    let m = indices.len();
    let mut out = vec![vec![0.0; m]; m];
    for (i_out, &i_idx) in indices.iter().enumerate() {
        for (j_out, &j_idx) in indices.iter().enumerate() {
            out[i_out][j_out] = v[i_idx][j_idx];
        }
    }
    Ok(out)
}

fn vech_index(i: usize, j: usize, k: usize) -> usize {
    let (row, col) = if i >= j { (i, j) } else { (j, i) };
    col * k - (col * (col.saturating_sub(1))) / 2 + (row - col)
}

fn model_contains_name(model: &str, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.';
    let mut start = 0usize;
    while let Some(pos) = model[start..].find(name) {
        let idx = start + pos;
        let end = idx + name.len();
        let before = model[..idx].chars().last();
        let after = model[end..].chars().next();
        let before_ok = before.is_none_or(|c| !is_word(c));
        let after_ok = after.is_none_or(|c| !is_word(c));
        if before_ok && after_ok {
            return true;
        }
        start = end;
    }
    false
}

fn z_diff_metric_prepost(
    s_orig: &Matrix,
    s_smooth: &Matrix,
    v_orig: &Matrix,
    v_smooth: &Matrix,
) -> f64 {
    let k = s_orig.len();
    let se_pre = lower_triangle_se(v_orig, k);
    let se_post = lower_triangle_se(v_smooth, k);
    if se_pre.is_empty() || se_post.is_empty() {
        return 0.0;
    }
    let z_pre = elementwise_div(s_orig, &se_pre);
    let z_post = elementwise_div(s_smooth, &se_post);
    max_abs_diff(&z_pre, &z_post)
}

fn max_abs_diff(a: &Matrix, b: &Matrix) -> f64 {
    let mut max = 0.0;
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            let diff = (a[i][j] - b[i][j]).abs();
            if diff > max {
                max = diff;
            }
        }
    }
    max
}

#[allow(clippy::needless_range_loop)]
fn lower_triangle_se(v: &Matrix, k: usize) -> Matrix {
    if k == 0 {
        return vec![];
    }
    let mut se = vec![vec![0.0; k]; k];
    let mut idx = 0usize;
    for j in 0..k {
        for i in j..k {
            let val = v
                .get(idx)
                .and_then(|row| row.get(idx))
                .copied()
                .unwrap_or(0.0);
            if val.is_finite() && val >= 0.0 {
                se[i][j] = val.sqrt();
            }
            idx += 1;
        }
    }
    se
}

fn elementwise_div(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = vec![vec![0.0; a[0].len()]; a.len()];
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            let denom = b[i][j];
            out[i][j] = if denom != 0.0 { a[i][j] / denom } else { 0.0 };
        }
    }
    out
}

fn wls_v_diag_inverse(v: &Matrix) -> Matrix {
    let n = v.len();
    let mut out = vec![vec![0.0; n]; n];
    for (i, row) in v.iter().enumerate() {
        let val = row[i];
        if val.is_finite() && val != 0.0 {
            out[i][i] = 1.0 / val;
        }
    }
    out
}

fn fit_sem_model(
    s: &Matrix,
    v: &Matrix,
    wls_v: Option<Matrix>,
    model: &str,
    names: &[String],
    std_lv: bool,
) -> Result<lavaan_crate::SemFit> {
    let input = lavaan_crate::SemInput {
        s: s.clone(),
        v: v.clone(),
        wls_v,
        model: model.to_string(),
        model_table: None,
        estimation: lavaan_crate::Estimation::Dwls,
        toler: Some(f64::EPSILON),
        std_lv,
        fix_measurement: false,
        q_snp: false,
        names: names.to_vec(),
        n_obs: Some(2.0),
        optim_dx_tol: Some(0.01),
        optim_force_converged: false,
        iter_max: None,
        sample_cov_rescale: false,
    };
    let engine = lavaan_crate::SemEngineImpl;
    engine.fit(&input)
}

fn fit_sem_table(
    s: &Matrix,
    v: &Matrix,
    wls_v: Option<Matrix>,
    model_table: Vec<lavaan_crate::ParTableRow>,
    names: &[String],
    std_lv: bool,
) -> Result<lavaan_crate::SemFit> {
    let input = lavaan_crate::SemInput {
        s: s.clone(),
        v: v.clone(),
        wls_v,
        model: String::new(),
        model_table: Some(model_table),
        estimation: lavaan_crate::Estimation::Dwls,
        toler: Some(f64::EPSILON),
        std_lv,
        fix_measurement: false,
        q_snp: false,
        names: names.to_vec(),
        n_obs: Some(2.0),
        optim_dx_tol: Some(0.01),
        optim_force_converged: false,
        iter_max: None,
        sample_cov_rescale: false,
    };
    let engine = lavaan_crate::SemEngineImpl;
    engine.fit(&input)
}

fn add_resid_bounds(model: &str, trait_names: &[String]) -> String {
    let mut out = model.to_string();
    let mut counter = 0usize;
    for name in trait_names {
        loop {
            let candidate = label_from_index(counter);
            counter += 1;
            if !model.contains(&candidate) {
                out.push_str(&format!(
                    "\n{name} ~~ {candidate}*{name}\n{candidate} > .0001"
                ));
                break;
            }
        }
    }
    out
}

fn label_from_index(mut idx: usize) -> String {
    let letters = b"abcdefghijklmnopqrstuvwxyz";
    let mut out = String::with_capacity(4);
    for _ in 0..4 {
        let c = letters[idx % 26] as char;
        out.push(c);
        idx /= 26;
    }
    out
}

fn param_estimates_from_table(
    par_table: &[lavaan_crate::ParTableRow],
    params: &[String],
) -> std::collections::HashMap<String, f64> {
    let mut out = std::collections::HashMap::new();
    for row in par_table {
        let key = format!("{}{}{}", row.lhs, row.op, row.rhs);
        if params.iter().any(|p| p == &key) {
            out.insert(key, row.est);
        }
    }
    out
}

fn collect_param_specs(
    model_table: &[lavaan_crate::ParTableRow],
    params: &[String],
) -> Vec<ParamSpec> {
    let mut out = Vec::new();
    for row in model_table.iter().filter(|r| r.free > 0) {
        let key = format!("{}{}{}", row.lhs, row.op, row.rhs);
        if params.iter().any(|p| p == &key) {
            out.push(ParamSpec {
                key,
                lhs: row.lhs.clone(),
                op: row.op.clone(),
                rhs: row.rhs.clone(),
            });
        }
    }
    out
}

fn param_estimates_from_fit(
    fit: &lavaan_crate::SemFit,
) -> std::collections::HashMap<String, (f64, f64)> {
    let mut out = std::collections::HashMap::new();
    for p in &fit.params {
        if p.free > 0 {
            let key = format!("{}{}{}", p.lhs, p.op, p.rhs);
            out.insert(key, (p.est, p.se));
        }
    }
    out
}

fn build_base_results(
    fit: &lavaan_crate::SemFit,
    model_table: &[lavaan_crate::ParTableRow],
) -> Result<DataFrame> {
    let mut lhs = Vec::new();
    let mut op = Vec::new();
    let mut rhs = Vec::new();
    let mut est = Vec::new();
    let mut se = Vec::new();
    let mut fixed = Vec::new();

    let mut free_map = std::collections::HashMap::new();
    for row in model_table {
        let key = format!("{}{}{}", row.lhs, row.op, row.rhs);
        free_map.insert(key, row.free > 0);
    }

    for row in &fit.par_table {
        if row.op != "=~" && row.op != "~~" && row.op != "~" {
            continue;
        }
        let key = format!("{}{}{}", row.lhs, row.op, row.rhs);
        lhs.push(row.lhs.clone());
        op.push(row.op.clone());
        rhs.push(row.rhs.clone());
        est.push(row.est);
        se.push(row.se);
        fixed.push(if *free_map.get(&key).unwrap_or(&false) {
            "No".to_string()
        } else {
            "Yes".to_string()
        });
    }

    let mut df = DataFrame::new(
        lhs.len(),
        vec![
            Series::new("lhs".into(), lhs).into(),
            Series::new("op".into(), op).into(),
            Series::new("rhs".into(), rhs).into(),
            Series::new("est".into(), est).into(),
            Series::new("SE".into(), se).into(),
            Series::new("Fixed_Enrich".into(), fixed).into(),
        ],
    )?;
    if df.column("Fixed_Enrich").is_ok() {
        df = df.sort(["Fixed_Enrich"], Default::default())?;
    }
    Ok(df)
}

fn diag_all_negative(s: &Matrix) -> bool {
    s.iter()
        .enumerate()
        .all(|(i, row)| row[i].is_finite() && row[i] < 0.0)
}

fn open_log_file(config: &SLdscConfig) -> Result<File> {
    if let Some(name) = &config.ldsc_log {
        return File::create(format!("{}_Partitioned.log", name.display()))
            .context("create log file");
    }
    let logtraits = config
        .traits
        .iter()
        .map(|p| p.file_name().and_then(|s| s.to_str()).unwrap_or("trait"))
        .collect::<Vec<_>>()
        .join("_");
    let mut log2 = logtraits;
    if log2.len() > 200 {
        log2.truncate(80);
    }
    File::create(format!("{log2}_Partitioned.log")).context("create log file")
}

// log_line and warn_line are provided by logging.rs

fn normalize_optional_vec(input: Option<Vec<Option<f64>>>, len: usize) -> Result<Vec<Option<f64>>> {
    match input {
        Some(v) => {
            if v.len() != len {
                return Err(anyhow::anyhow!(
                    "Expected {} prevalence values; got {}",
                    len,
                    v.len()
                ));
            }
            Ok(v)
        }
        None => Ok(vec![None; len]),
    }
}

fn resolve_trait_names(config: &SLdscConfig, n: usize) -> Result<Vec<String>> {
    if let Some(names) = &config.trait_names {
        if names.len() != n {
            return Err(anyhow::anyhow!(
                "trait.names length {} does not match traits length {}",
                names.len(),
                n
            ));
        }
        return Ok(names.clone());
    }
    Ok((1..=n).map(|i| format!("V{i}")).collect())
}

fn split_ld_dirs(ld: &[PathBuf]) -> Result<(PathBuf, Vec<PathBuf>)> {
    if ld.is_empty() {
        return Err(anyhow::anyhow!("ld must include at least one directory"));
    }
    let mut ld1 = ld[0].clone();
    let mut extras = Vec::new();
    for path in ld {
        let name = path.to_string_lossy().to_lowercase();
        if name.contains("baseline") {
            ld1 = path.clone();
        } else if path != &ld1 {
            extras.push(path.clone());
        }
    }
    Ok((ld1, extras))
}

fn list_files_with_any(dir: &Path, patterns: &[&str]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|s| s.to_str())
            && patterns.iter().any(|pat| name.contains(pat))
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn read_ld_score_dir(dir: &Path) -> Result<DataFrame> {
    let files = list_files_with_any(dir, &["l2.ldscore", "l2.ldscor"])?;
    if files.is_empty() {
        return Err(anyhow::anyhow!(
            "No LD score files found in {}",
            dir.display()
        ));
    }
    let mut frames = Vec::new();
    for file in files {
        let df = read_table(&file)?;
        frames.push(df);
    }
    let mut out = frames
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No LD score data"))?;
    for df in frames.iter().skip(1) {
        out.vstack_mut(df)?;
    }
    Ok(out)
}

fn read_m_dir(dir: &Path) -> Result<Vec<f64>> {
    let files = list_files_with_any(dir, &["M_5_50"])?;
    if files.is_empty() {
        return Err(anyhow::anyhow!(
            "No M_5_50 files found in {}",
            dir.display()
        ));
    }
    let mut sums: Vec<f64> = Vec::new();
    for file in files {
        let mut cols = read_numeric_matrix(&file)?;
        if sums.is_empty() {
            sums = std::mem::take(&mut cols);
        } else {
            if cols.len() != sums.len() {
                return Err(anyhow::anyhow!(
                    "M_5_50 column count mismatch in {}",
                    file.display()
                ));
            }
            for (i, val) in cols.iter().enumerate() {
                sums[i] += val;
            }
        }
    }
    Ok(sums)
}

fn read_numeric_matrix(path: &Path) -> Result<Vec<f64>> {
    let reader = open_maybe_compressed(path)?;
    let mut sums = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vals: Vec<f64> = line
            .split_whitespace()
            .filter_map(|v| v.parse::<f64>().ok())
            .collect();
        if sums.is_empty() {
            sums = vec![0.0; vals.len()];
        }
        if vals.len() != sums.len() {
            return Err(anyhow::anyhow!(
                "Inconsistent column count in {}",
                path.display()
            ));
        }
        for (i, v) in vals.iter().enumerate() {
            sums[i] += v;
        }
    }
    Ok(sums)
}

fn open_maybe_compressed(path: &Path) -> Result<Box<dyn BufRead>> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    if ext == "gz" {
        let reader = GzDecoder::new(file);
        return Ok(Box::new(BufReader::new(reader)));
    }
    if ext == "bz2" {
        let reader = BzDecoder::new(file);
        return Ok(Box::new(BufReader::new(reader)));
    }
    Ok(Box::new(BufReader::new(file)))
}

fn drop_cols_if_present(df: &mut DataFrame, names: &[&str]) -> Result<()> {
    for name in names {
        if df.column(name).is_ok() {
            df.drop_in_place(name)?;
        }
    }
    Ok(())
}

fn rename_last_column(df: &mut DataFrame, name: &str) -> Result<()> {
    let cols = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    if let Some(last) = cols.last() {
        df.rename(last, name.into())?;
    }
    Ok(())
}

fn annotation_columns(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .filter(|name| !matches!(name.as_ref(), "CHR" | "BP" | "SNP"))
        .map(|s| s.to_string())
        .collect()
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn build_annotation_matrix(
    ld1: &Path,
    ld2: &[PathBuf],
    frq_dir: &Path,
    annotation_cols: &[String],
    mut m: Vec<f64>,
    exclude_cont: bool,
    x: &mut DataFrame,
    log: &mut File,
) -> Result<(Matrix, Vec<String>, Vec<f64>, Vec<i32>)> {
    let annot_files = list_files_with_any(ld1, &["annot.gz", "annot"])?;
    if annot_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No annotation files found in {}",
            ld1.display()
        ));
    }
    let frq_files = list_files_with_any(frq_dir, &[".frq"])?;
    if frq_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No frq files found in {}",
            frq_dir.display()
        ));
    }
    let n_files = annot_files.len().min(frq_files.len());

    let mut continuous = Vec::new();
    let mut annot_matrix = vec![vec![0.0; annotation_cols.len()]; annotation_cols.len()];

    for idx in 0..n_files {
        let mut annot = read_table(&annot_files[idx])?;
        drop_cols_if_present(&mut annot, &["CM"])?;

        for extra in ld2 {
            let extra_files = list_files_with_any(extra, &["annot.gz", "annot"])?;
            if idx >= extra_files.len() {
                continue;
            }
            let mut extra_annot = read_table(&extra_files[idx])?;
            drop_cols_if_present(&mut extra_annot, &["CHR", "BP", "CM"])?;
            annot = annot.join(&extra_annot, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
        }

        let frq = read_table(&frq_files[idx])?;
        let maf = frq.column("MAF")?.f64().context("MAF")?;
        let mask = maf.gt(0.05) & maf.lt(0.95);
        let mut frq = frq.filter(&mask)?;
        frq = frq.select(["SNP", "MAF"])?;

        let mut selected = annot.join(&frq, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
        drop_cols_if_present(&mut selected, &["MAF", "CHR", "BP"])?;

        if idx == 0 {
            for name in annotation_cols {
                let series = selected.column(name)?.f64().context("annotation")?;
                let mut is_binary = true;
                for v in series {
                    let v = v.unwrap_or(0.0);
                    if !(v == 0.0 || v == 1.0) {
                        is_binary = false;
                        break;
                    }
                }
                if !is_binary {
                    continuous.push(name.clone());
                }
            }
            if exclude_cont {
                log_line(
                    log,
                    &format!("Excluding {} continuous annotations", continuous.len()),
                    true,
                )?;
            }
        }

        if exclude_cont && !continuous.is_empty() {
            for name in &continuous {
                if selected.column(name).is_ok() {
                    selected.drop_in_place(name)?;
                }
            }
        }

        let cols = annotation_columns(&selected);
        let cols = cols
            .iter()
            .filter(|name| *name != "SNP")
            .cloned()
            .collect::<Vec<_>>();
        let matrix = dataframe_to_columns(&selected, &cols)?;
        add_cross_product(&mut annot_matrix, &matrix);
    }

    let mut annotation_cols = annotation_cols.to_vec();
    if exclude_cont && !continuous.is_empty() {
        let mut keep = Vec::new();
        let mut keep_idx = Vec::new();
        for (idx, name) in annotation_cols.iter().enumerate() {
            let base = name.trim_end_matches("L2");
            if !continuous.iter().any(|c| c == base || c == name) {
                keep.push(name.clone());
                keep_idx.push(idx);
            }
        }
        annotation_cols = keep;
        m = keep_idx.iter().map(|idx| m[*idx]).collect();
        let mut keep_cols = vec!["CHR".to_string(), "BP".to_string(), "SNP".to_string()];
        keep_cols.extend(annotation_cols.iter().cloned());
        *x = x.select(keep_cols)?;
        annot_matrix = shrink_matrix(&annot_matrix, &keep_idx);
    }

    let select = annotation_cols
        .iter()
        .map(|name| {
            if name.to_lowercase().contains("flanking") {
                2
            } else {
                1
            }
        })
        .collect::<Vec<_>>();

    Ok((annot_matrix, annotation_cols, m, select))
}

fn shrink_matrix(matrix: &Matrix, keep_idx: &[usize]) -> Matrix {
    let mut out = vec![vec![0.0; keep_idx.len()]; keep_idx.len()];
    for (i_out, &i) in keep_idx.iter().enumerate() {
        for (j_out, &j) in keep_idx.iter().enumerate() {
            out[i_out][j_out] = matrix[i][j];
        }
    }
    out
}

fn dataframe_to_columns(df: &DataFrame, cols: &[String]) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for name in cols {
        let series = df.column(name)?.f64().context("annotation col")?;
        out.push(series.into_iter().map(|v| v.unwrap_or(0.0)).collect());
    }
    Ok(out)
}

#[allow(clippy::needless_range_loop)]
fn add_cross_product(annot_matrix: &mut Matrix, matrix: &[Vec<f64>]) {
    let n = matrix.len();
    if n == 0 {
        return;
    }
    let rows = matrix[0].len();
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for r in 0..rows {
                sum += matrix[i][r] * matrix[j][r];
            }
            annot_matrix[i][j] += sum;
        }
    }
}

fn read_trait_for_sldsc(
    path: &Path,
    w: &DataFrame,
    x: &DataFrame,
    annotation_cols: &[String],
    log: &mut File,
    idx: usize,
    total: usize,
) -> Result<DataFrame> {
    let mut y = read_table(path)?;
    y = y.select(["SNP", "N", "Z", "A1"])?;
    y = y.drop_nulls::<&str>(None)?;

    log_line(
        log,
        &format!(
            "Read in summary statistics [{idx}/{total}] from {}",
            path.display()
        ),
        true,
    )?;

    let merged = y.join(w, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    let merged = merged.join(x, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;

    let merged = merged.sort(["CHR", "BP"], Default::default())?;
    log_line(
        log,
        &format!(
            "Out of {} SNPs, {} remain after merging with LD-score files",
            y.height(),
            merged.height()
        ),
        true,
    )?;

    let n = extract_f64_column(&merged, "N")?;
    let z = extract_f64_column(&merged, "Z")?;
    let chi: Vec<f64> = z.iter().map(|v| v * v).collect();
    let chisq_max = (0.001 * n.iter().cloned().fold(0.0, f64::max)).max(80.0);
    let mask_values: Vec<bool> = chi
        .iter()
        .map(|v| v.is_finite() && *v <= chisq_max)
        .collect();
    let mask = BooleanChunked::new("mask".into(), mask_values);
    let removed = mask.iter().filter(|v| matches!(v, Some(false))).count();
    let merged = merged.filter(&mask)?;

    log_line(
        log,
        &format!(
            "Removing {} SNPs with Chi^2 > {}; {} remain",
            removed,
            chisq_max,
            merged.height()
        ),
        true,
    )?;

    let mut keep_cols = vec![
        "SNP".to_string(),
        "N".to_string(),
        "Z".to_string(),
        "A1".to_string(),
        "wLD".to_string(),
        "CHR".to_string(),
        "BP".to_string(),
    ];
    keep_cols.extend(annotation_cols.iter().cloned());
    Ok(merged.select(keep_cols)?)
}

fn merge_trait_pair(
    y1: &DataFrame,
    y2: &DataFrame,
    annotation_cols: &[String],
) -> Result<DataFrame> {
    let mut y2 = y2.clone();
    let names = y2
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    for name in names {
        if name != "SNP" {
            y2.rename(&name, format!("{name}_y").into())?;
        }
    }
    let merged = y1.join(&y2, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    let mut keep_cols = vec![
        "SNP".to_string(),
        "N".to_string(),
        "Z".to_string(),
        "A1".to_string(),
        "wLD".to_string(),
        "N_y".to_string(),
        "Z_y".to_string(),
        "A1_y".to_string(),
    ];
    keep_cols.extend(annotation_cols.iter().cloned());
    Ok(merged.select(keep_cols)?)
}

fn extract_f64_column(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let series = df.column(name)?.f64().context("f64 column")?;
    Ok(series.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

fn extract_string_column(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    let series = df.column(name)?.str().context("string column")?;
    Ok(series
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect())
}

fn extract_ld_columns(df: &DataFrame, cols: &[String]) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for name in cols {
        let series = df.column(name)?.f64().context("ld col")?;
        out.push(series.into_iter().map(|v| v.unwrap_or(0.0)).collect());
    }
    Ok(out)
}

fn row_sums(cols: &[Vec<f64>]) -> Vec<f64> {
    if cols.is_empty() {
        return vec![];
    }
    let rows = cols[0].len();
    let mut sums = vec![0.0; rows];
    for col in cols {
        for (i, v) in col.iter().enumerate() {
            sums[i] += v;
        }
    }
    sums
}

fn compute_initial_weights(
    chi: &[f64],
    n: &[f64],
    x_tot: &[f64],
    wld: &[f64],
    m_tot: f64,
) -> (Vec<f64>, f64) {
    let mean_chi = mean(chi);
    let mean_xn = mean_product(x_tot, n);
    let mut tot_agg = if mean_xn != 0.0 {
        m_tot * (mean_chi - 1.0) / mean_xn
    } else {
        0.0
    };
    tot_agg = tot_agg.clamp(0.0, 1.0);

    let mut init = vec![0.0; chi.len()];
    for i in 0..chi.len() {
        let ld = x_tot[i].max(1.0);
        let w_ld = wld[i].max(1.0);
        let c = tot_agg * n[i] / m_tot;
        let het_w = 1.0 / (2.0 * (1.0 + c * ld).powi(2));
        let oc_w = 1.0 / w_ld;
        let w = het_w * oc_w;
        init[i] = w.sqrt();
    }
    (init, tot_agg)
}

fn normalize_weights(init: &[f64]) -> Vec<f64> {
    let sum: f64 = init.iter().sum();
    if sum == 0.0 {
        return vec![0.0; init.len()];
    }
    init.iter().map(|v| v / sum).collect()
}

fn combine_weights(w1: &[f64], w2: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; w1.len()];
    let mut sum = 0.0;
    for i in 0..w1.len() {
        let val = w1[i] + w2[i];
        out[i] = val;
        sum += val;
    }
    if sum != 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

fn weighted_regression_blocks(
    ld_cols: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
    n_blocks: usize,
    scale: Option<(&[f64], f64)>,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = y.len();
    let p = ld_cols.len() + 1;
    let mut xty_blocks = vec![vec![0.0; p]; n_blocks];
    let mut xtx_blocks = vec![vec![vec![0.0; p]; p]; n_blocks];

    for row in 0..n {
        let block = row * n_blocks / n.max(1);
        let weight_sq = weights[row] * weights[row];
        let mut x_vals = vec![0.0; p];
        for (j, col) in ld_cols.iter().enumerate() {
            let mut v = col[row];
            if let Some((n_vec, n_bar)) = scale {
                v *= n_vec[row] / n_bar;
            }
            x_vals[j] = v;
        }
        x_vals[p - 1] = 1.0;

        for i in 0..p {
            xty_blocks[block][i] += x_vals[i] * y[row] * weight_sq;
        }
        for i in 0..p {
            for j in 0..p {
                xtx_blocks[block][i][j] += x_vals[i] * x_vals[j] * weight_sq;
            }
        }
    }

    let mut xty = vec![0.0; p];
    let mut xtx = vec![vec![0.0; p]; p];
    for b in 0..n_blocks {
        for i in 0..p {
            xty[i] += xty_blocks[b][i];
        }
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] += xtx_blocks[b][i][j];
            }
        }
    }

    let reg = solve_linear(&xtx, &xty)?;

    let mut delete_values = vec![vec![0.0; p]; n_blocks];
    for b in 0..n_blocks {
        let mut xty_del = xty.clone();
        let mut xtx_del = xtx.clone();
        for i in 0..p {
            xty_del[i] -= xty_blocks[b][i];
        }
        for i in 0..p {
            for j in 0..p {
                xtx_del[i][j] -= xtx_blocks[b][i][j];
            }
        }
        delete_values[b] = solve_linear(&xtx_del, &xty_del)?;
    }

    Ok((reg, delete_values))
}

fn solve_linear(xtx: &[Vec<f64>], xty: &[f64]) -> Result<Vec<f64>> {
    let p = xty.len();
    let mut data = Vec::with_capacity(p * p);
    for row in xtx {
        data.extend_from_slice(row);
    }
    let a = Array2::from_shape_vec((p, p), data).context("xtx shape")?;
    let b = Array1::from_vec(xty.to_vec());
    let x = a.solve_into(b).context("solve linear system")?;
    Ok(x.to_vec())
}

fn jackknife_pseudo(reg: &[f64], delete_values: &[Vec<f64>], n_blocks: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; reg.len()]; n_blocks];
    let nb = n_blocks as f64;
    for b in 0..n_blocks {
        for i in 0..reg.len() {
            out[b][i] = nb * reg[i] - (nb - 1.0) * delete_values[b][i];
        }
    }
    out
}

fn jackknife_pseudo_cats(
    cats: &[f64],
    delete_values: &[Vec<f64>],
    n_blocks: usize,
    n_bar: f64,
    m: &[f64],
) -> Vec<Vec<f64>> {
    let nb = n_blocks as f64;
    let mut out = vec![vec![0.0; cats.len()]; n_blocks];
    for b in 0..n_blocks {
        for i in 0..cats.len() {
            let delete = delete_values[b][i] / n_bar * m[i];
            out[b][i] = nb * cats[i] - (nb - 1.0) * delete;
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn covariance(matrix: &[Vec<f64>]) -> Matrix {
    if matrix.is_empty() {
        return vec![];
    }
    let n = matrix.len();
    let p = matrix[0].len();
    let mut means = vec![0.0; p];
    for row in matrix {
        for i in 0..p {
            means[i] += row[i];
        }
    }
    for i in 0..p {
        means[i] /= n as f64;
    }
    let mut cov = vec![vec![0.0; p]; p];
    for row in matrix {
        for i in 0..p {
            let di = row[i] - means[i];
            for j in 0..p {
                cov[i][j] += di * (row[j] - means[j]);
            }
        }
    }
    let denom = (n.saturating_sub(1)) as f64;
    if denom > 0.0 {
        for i in 0..p {
            for j in 0..p {
                cov[i][j] /= denom;
            }
        }
    }
    cov
}

fn scale_covariance(cov: &Matrix, n_annot: usize, n_bar: f64) -> Matrix {
    let mut out = vec![vec![0.0; n_annot]; n_annot];
    for i in 0..n_annot {
        for j in 0..n_annot {
            out[i][j] = cov[i][j] / (n_bar * n_bar);
        }
    }
    out
}

fn total_se(coef_cov: &Matrix, m: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..m.len() {
        for j in 0..m.len() {
            total += coef_cov[i][j] * m[i] * m[j];
        }
    }
    total.max(0.0).sqrt()
}

fn mat_vec_mul(mat: &Matrix, vec: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; mat.len()];
    for (i, row) in mat.iter().enumerate() {
        let mut sum = 0.0;
        for (j, v) in row.iter().enumerate() {
            sum += v * vec[j];
        }
        out[i] = sum;
    }
    out
}

fn block_from_cov(cov: &Matrix, u: usize, p: usize, n_annot: usize) -> Matrix {
    let mut out = vec![vec![0.0; n_annot]; n_annot];
    let row_start = u * n_annot;
    let col_start = p * n_annot;
    for i in 0..n_annot {
        for j in 0..n_annot {
            out[i][j] = cov[row_start + i][col_start + j];
        }
    }
    out
}

fn sample_var_from_block(block: &Matrix, overlap: &Matrix) -> Vec<f64> {
    let n = overlap.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let aij = overlap[i][j];
            if aij == 0.0 {
                continue;
            }
            let mut inner = 0.0;
            for k in 0..n {
                inner += block[j][k] * overlap[i][k];
            }
            sum += aij * inner;
        }
        out[i] = sum;
    }
    out
}

fn liability_vector(sample_prev: &[Option<f64>], pop_prev: &[Option<f64>]) -> Result<Vec<f64>> {
    let mut out = vec![1.0; sample_prev.len()];
    for i in 0..sample_prev.len() {
        if let (Some(pop), Some(samp)) = (pop_prev[i], sample_prev[i]) {
            out[i] = liability_conversion(pop, samp)?;
        }
    }
    Ok(out)
}

fn liability_conversion(pop: f64, samp: f64) -> Result<f64> {
    if pop <= 0.0 || pop >= 1.0 || samp <= 0.0 || samp >= 1.0 {
        return Err(anyhow::anyhow!(
            "Population/sample prevalence must be in (0,1)"
        ));
    }
    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    let t = normal.inverse_cdf(1.0 - pop);
    let z = normal.pdf(t);
    Ok((pop * (1.0 - pop)).powi(2) / (samp * (1.0 - samp) * z * z))
}

fn scale_liability(s: &Matrix, liab: &[f64]) -> Matrix {
    let k = s.len();
    let mut out = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            out[i][j] = s[i][j] * liab[i].sqrt() * liab[j].sqrt();
        }
    }
    out
}

fn lower_triangle_ratio(a: &Matrix, b: &Matrix) -> Vec<f64> {
    let k = a.len();
    let mut out = Vec::new();
    // Note: lower-triangle ordering follows row-major (i >= j) traversal.
    // R's gdata::lowerTriangle uses column-major ordering. If V is reshaped
    // using column-major vech, this can be a parity mismatch.
    for i in 0..k {
        for j in 0..=i {
            if b[i][j] != 0.0 {
                out.push(a[i][j] / b[i][j]);
            } else {
                out.push(0.0);
            }
        }
    }
    out
}

fn scale_v(v: &Matrix, scale: &[f64]) -> Matrix {
    let n = v.len();
    let mut diag = vec![0.0; n];
    for i in 0..n {
        diag[i] = v[i][i].abs().sqrt();
    }
    let mut scaled_diag = vec![0.0; n];
    for i in 0..n {
        scaled_diag[i] = diag[i] * scale.get(i).copied().unwrap_or(1.0);
    }
    let vcor = cov2cor(v, &diag);
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = vcor[i][j] * scaled_diag[i] * scaled_diag[j];
        }
    }
    out
}

fn cov2cor(v: &Matrix, diag: &[f64]) -> Matrix {
    let n = v.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if diag[i] != 0.0 && diag[j] != 0.0 {
                out[i][j] = v[i][j] / (diag[i] * diag[j]);
            }
        }
    }
    out
}

fn scale_matrix(mat: &Matrix, factor: f64) -> Matrix {
    let mut out = mat.clone();
    for row in &mut out {
        for v in row {
            *v *= factor;
        }
    }
    out
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_product(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0.0;
    for (x, y) in a.iter().zip(b) {
        sum += x * y;
        count += 1.0;
    }
    if count == 0.0 { 0.0 } else { sum / count }
}

fn lambda_gc(chi: &[f64]) -> Option<f64> {
    if chi.is_empty() {
        return None;
    }
    let mut sorted = chi.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    Some(median / 0.4549)
}

fn safe_div(a: f64, b: f64) -> f64 {
    if b == 0.0 { f64::NAN } else { a / b }
}
