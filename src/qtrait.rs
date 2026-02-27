use anyhow::{Context, Result};
use polars::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
use std::collections::HashMap;

use crate::plot_utils::ensure_plots_dir;
use crate::sem::smooth_if_needed;
use crate::types::LdscOutput;
use crate::types::Matrix;
use lavaan as lavaan_crate;
use lavaan_crate::SemEngine as LavaanSemEngineTrait;
use lavaan_crate::implied::cov2cor;
use plotly::common::color::NamedColor;
use plotly::common::{Line, Marker, Mode};
use plotly::layout::Axis;
use plotly::{HeatMap, Layout, Plot, Scatter};

#[derive(Debug, Clone)]
pub struct QTraitConfig {
    pub indicators: Vec<String>,
    pub traits: Vec<String>,
    pub mresid: f64,
    pub mresid_threshold: f64,
    pub lsrmr: f64,
    pub lsrmr_threshold: f64,
    pub save_plots: bool,
    pub stdout: bool,
}

pub fn qtrait(ldsc: &LdscOutput, config: &QTraitConfig) -> Result<DataFrame> {
    if ldsc.s_stand.is_none() || ldsc.v_stand.is_none() {
        return Err(anyhow::anyhow!(
            "QTrait requires S_Stand/V_Stand; rerun ldsc with stand=TRUE"
        ));
    }

    let trait_names = resolve_trait_names(ldsc)?;
    let name_index = build_index_map(&trait_names);

    for indicator in &config.indicators {
        if !name_index.contains_key(indicator) {
            return Err(anyhow::anyhow!(
                "Indicator {indicator} not found in LDSC trait names"
            ));
        }
    }
    for trait_name in &config.traits {
        if !name_index.contains_key(trait_name) {
            return Err(anyhow::anyhow!(
                "Trait {trait_name} not found in LDSC trait names"
            ));
        }
    }

    let cf_model = build_common_factor_model(&config.indicators);
    let (s_cfm, v_cfm, names_cfm) =
        subset_covstruc(&ldsc.s, &ldsc.v, &trait_names, &config.indicators)?;
    let fit_cfm = fit_sem(&s_cfm, &v_cfm, &cf_model, &names_cfm, true)?;

    let mut unstd_lambdas = Vec::new();
    for indicator in &config.indicators {
        let key = ("F1", "=~", indicator.as_str());
        let est = find_param_est(&fit_cfm, key)
            .with_context(|| format!("Missing loading for {indicator} in CFM"))?;
        unstd_lambdas.push((indicator.clone(), est));
    }
    let cf_fixed = build_common_factor_fixed(&unstd_lambdas);

    let s_stand = ldsc.s_stand.as_ref().expect("checked above");
    let v_stand = ldsc.v_stand.as_ref().expect("checked above");

    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    let bfpvalt = 0.05 / config.traits.len() as f64;

    let mut out_rows: Vec<QTraitRow> = Vec::with_capacity(config.traits.len());

    for trait_name in &config.traits {
        tracing::info!("Fitting QTrait models for external trait {trait_name}");
        let ext_latent = format!("{trait_name}F");
        let vars = collect_vars(&config.indicators, trait_name);
        let (s_sub, v_sub, names_sub) = subset_covstruc(&ldsc.s, &ldsc.v, &trait_names, &vars)?;

        let cpm_model = build_cpm_model(&cf_fixed, trait_name, &ext_latent);
        let cpm_fit = fit_sem(&s_sub, &v_sub, &cpm_model, &names_sub, false)?;
        let cpchi = cpm_fit.stats.chisq;
        let cpdf = cpm_fit.stats.df;

        let (beta_cpm, se_cpm, _std_all_cpm) = extract_beta(&cpm_fit, "F1", "~", &ext_latent)?;
        let pval_beta_cpm = z_to_p(&normal, beta_cpm, se_cpm);

        let residuals = residual_variances(&cpm_fit, &config.indicators);
        let std_lambda_map = compute_std_lambdas(&unstd_lambdas, &residuals);

        let ipm_model = build_ipm_model(&cf_fixed, trait_name, &ext_latent, &config.indicators);
        let ipm_fit = fit_sem(&s_sub, &v_sub, &ipm_model, &names_sub, false)?;
        let mut ipchi = ipm_fit.stats.chisq;
        let ipdf = ipm_fit.stats.df;
        if !ipchi.is_finite() {
            ipchi = 0.0;
        }
        let nested_chi = cpchi - ipchi;
        let nested_df = (cpdf - ipdf) as f64;
        let p_value_cpm = pchisq(nested_chi, nested_df);

        let (observed, _implied, residual) = build_rgs(
            s_stand,
            &trait_names,
            trait_name,
            &config.indicators,
            &cpm_fit,
            &names_sub,
        )?;
        let se_rgs = se_rgs_from_vstand(v_stand, &trait_names, trait_name, &config.indicators)?;
        let (beta_ind, beta_ind_se) =
            extract_indicator_betas(&ipm_fit, &config.indicators, &ext_latent);

        let lsrmr_cpm = rms(&residual);
        let mr_g = rms(&observed);
        let beta_sig_cpm = sig_from_pval(pval_beta_cpm, bfpvalt);
        let q_sig_cpm = sig_from_pval(p_value_cpm, bfpvalt);
        let lsrmr_above = lsrmr_flag(lsrmr_cpm, config.lsrmr_threshold, config.lsrmr, mr_g);
        let sig_het_cpm = beta_sig_cpm == "*" && q_sig_cpm == "*" && lsrmr_above == "Yes";

        let mut row = QTraitRow::new(trait_name.clone());
        row.rgf1trait_cpm = beta_cpm;
        row.sergf1trait_cpm = se_cpm;
        row.pvalrgf1trait_cpm = pval_beta_cpm;
        row.rgf1trait_significant_cpm = beta_sig_cpm.clone();
        row.qtrait_cpm = nested_chi;
        row.df_cpm = nested_df;
        row.p_value_cpm = p_value_cpm;
        row.qsignificant_cpm = q_sig_cpm.clone();
        row.lsrmr_cpm = lsrmr_cpm;
        row.lsrmr_above_threshold_cpm = lsrmr_above.clone();
        row.heterogeneity_cpm = if sig_het_cpm { "Yes" } else { "No" }.to_string();

        let mut outlier_fum_label = "None".to_string();
        let mut fum_slope_label: Option<String> = None;
        let mut fum_slope_value: Option<f64> = None;

        if sig_het_cpm {
            let (outliers, _inside) = identify_outliers(
                &residual,
                &observed,
                &config.indicators,
                config.mresid,
                config.mresid_threshold,
            );
            if outliers != "None" {
                let mut outlier_fum = most_extreme_outlier(&residual, &config.indicators);
                let mut unconstrained_paths = outlier_fum.clone();
                let mut fum_fit = fit_fum(
                    &cf_fixed,
                    trait_name,
                    &ext_latent,
                    &s_sub,
                    &v_sub,
                    &names_sub,
                    &outlier_fum,
                )?;
                let mut fumchi = fum_fit.stats.chisq;
                let mut fumdf = fum_fit.stats.df;
                if !fumchi.is_finite() {
                    fumchi = 0.0;
                }
                let mut nested_chi_fum = fumchi - ipchi;
                let mut nested_df_fum = (fumdf - ipdf) as f64;

                let mut rgs_fum = build_rgs(
                    s_stand,
                    &trait_names,
                    trait_name,
                    &config.indicators,
                    &fum_fit,
                    &names_sub,
                )?;
                let mut lsrmr_fum = rms(&rgs_fum.2);
                let mut p_value_fum = pchisq(nested_chi_fum, nested_df_fum);
                let mut q_sig_fum = sig_from_pval(p_value_fum, bfpvalt);
                let mut lsrmr_above_fum =
                    lsrmr_flag(lsrmr_fum, config.lsrmr_threshold, config.lsrmr, mr_g);

                let (mut beta_fum, mut se_fum, _std_all_fum) =
                    extract_beta(&fum_fit, "F1", "~", &ext_latent)?;
                let mut pval_beta_fum = z_to_p(&normal, beta_fum, se_fum);
                let mut beta_sig_fum = sig_from_pval(pval_beta_fum, bfpvalt);
                let mut sig_het_fum =
                    beta_sig_fum == "*" && q_sig_fum == "*" && lsrmr_above_fum == "Yes";

                let ordered_names = order_by_residual(&residual, &config.indicators);
                while sig_het_fum
                    && nested_df_fum != 0.0
                    && !all_residuals_within(
                        &rgs_fum.2,
                        &observed,
                        config.mresid,
                        config.mresid_threshold,
                    )
                {
                    let next = next_outlier(&ordered_names, &unconstrained_paths);
                    if let Some(next) = next {
                        unconstrained_paths = merge_outliers(&unconstrained_paths, &next);
                    } else {
                        break;
                    }

                    fum_fit = fit_fum(
                        &cf_fixed,
                        trait_name,
                        &ext_latent,
                        &s_sub,
                        &v_sub,
                        &names_sub,
                        &unconstrained_paths,
                    )?;
                    fumchi = fum_fit.stats.chisq;
                    fumdf = fum_fit.stats.df;
                    if fumdf == 0 {
                        nested_chi_fum = 0.0;
                        nested_df_fum = 0.0;
                    } else {
                        nested_chi_fum = fumchi - ipchi;
                        nested_df_fum = (fumdf - ipdf) as f64;
                    }
                    rgs_fum = build_rgs(
                        s_stand,
                        &trait_names,
                        trait_name,
                        &config.indicators,
                        &fum_fit,
                        &names_sub,
                    )?;
                    lsrmr_fum = rms(&rgs_fum.2);
                    p_value_fum = pchisq(nested_chi_fum, nested_df_fum);
                    q_sig_fum = sig_from_pval(p_value_fum, bfpvalt);
                    lsrmr_above_fum =
                        lsrmr_flag(lsrmr_fum, config.lsrmr_threshold, config.lsrmr, mr_g);
                    let beta_out = extract_beta(&fum_fit, "F1", "~", &ext_latent)?;
                    beta_fum = beta_out.0;
                    se_fum = beta_out.1;
                    pval_beta_fum = z_to_p(&normal, beta_fum, se_fum);
                    beta_sig_fum = sig_from_pval(pval_beta_fum, bfpvalt);
                    sig_het_fum =
                        beta_sig_fum == "*" && q_sig_fum == "*" && lsrmr_above_fum == "Yes";
                    outlier_fum = unconstrained_paths.clone();
                }

                row.rgf1trait_fum = Some(beta_fum);
                row.sergf1trait_fum = Some(se_fum);
                row.pvalrgf1trait_fum = Some(pval_beta_fum);
                row.rgf1trait_significant_fum = Some(beta_sig_fum);
                row.qtrait_fum = Some(nested_chi_fum);
                row.df_fum = Some(nested_df_fum);
                row.p_value_fum = Some(p_value_fum);
                row.qsignificant_fum = Some(q_sig_fum);
                row.lsrmr_fum = Some(lsrmr_fum);
                row.lsrmr_above_threshold_fum = Some(lsrmr_above_fum);
                row.reduction_lsrmr = Some(reduction_percent(lsrmr_cpm, lsrmr_fum));
                row.heterogeneity_fum = Some(if sig_het_fum { "Yes" } else { "No" }.to_string());
                row.unconstrained_paths = Some(outlier_fum.clone());
                outlier_fum_label = outlier_fum.clone();
                fum_slope_label = Some(format!("Unconstrained paths: {outlier_fum}"));
                fum_slope_value = Some(beta_fum);

                let n_outliers = outlier_fum.split(',').filter(|s| !s.is_empty()).count();
                if n_outliers as f64 > 0.5 * config.indicators.len() as f64 {
                    tracing::warn!(
                        "Majority of indicators identified as outlying; common factor model may be inadequate!"
                    );
                }
            } else {
                // Match R behavior: when heterogeneity is detected but no outliers are identified,
                // the result row is left unpopulated.
                row.mark_all_na();
            }
        }

        if config.save_plots {
            let lambda_unstd: Vec<f64> = unstd_lambdas.iter().map(|(_, v)| *v).collect();
            let lambda_std_map = &std_lambda_map;
            let mut lambda_std = Vec::with_capacity(config.indicators.len());
            for ind in &config.indicators {
                lambda_std.push(*lambda_std_map.get(ind).unwrap_or(&f64::NAN));
            }
            let plot_data = QTraitPlotData {
                indicators: &config.indicators,
                trait_name,
                observed: &observed,
                implied: &_implied,
                residual: &residual,
                lambda_unstd: &lambda_unstd,
                lambda_std: &lambda_std,
                beta_ind: &beta_ind,
                beta_ind_se: &beta_ind_se,
                se_rgs: &se_rgs,
                outliers: &outlier_fum_label,
                slope_cpm: beta_cpm,
                slope_fum_label: fum_slope_label.as_deref(),
                slope_fum: fum_slope_value,
                stdout: config.stdout,
            };
            write_qtrait_plots(plot_data)?;
        }

        out_rows.push(row);
    }

    build_qtrait_df(&out_rows)
}

#[derive(Debug, Clone)]
struct QTraitRow {
    trait_name: String,
    rgf1trait_cpm: f64,
    sergf1trait_cpm: f64,
    pvalrgf1trait_cpm: f64,
    rgf1trait_significant_cpm: String,
    qtrait_cpm: f64,
    df_cpm: f64,
    p_value_cpm: f64,
    qsignificant_cpm: String,
    lsrmr_cpm: f64,
    lsrmr_above_threshold_cpm: String,
    heterogeneity_cpm: String,
    rgf1trait_fum: Option<f64>,
    sergf1trait_fum: Option<f64>,
    pvalrgf1trait_fum: Option<f64>,
    rgf1trait_significant_fum: Option<String>,
    qtrait_fum: Option<f64>,
    df_fum: Option<f64>,
    p_value_fum: Option<f64>,
    qsignificant_fum: Option<String>,
    lsrmr_fum: Option<f64>,
    lsrmr_above_threshold_fum: Option<String>,
    reduction_lsrmr: Option<String>,
    heterogeneity_fum: Option<String>,
    unconstrained_paths: Option<String>,
}

impl QTraitRow {
    fn new(trait_name: String) -> Self {
        Self {
            trait_name,
            rgf1trait_cpm: f64::NAN,
            sergf1trait_cpm: f64::NAN,
            pvalrgf1trait_cpm: f64::NAN,
            rgf1trait_significant_cpm: "NS".to_string(),
            qtrait_cpm: f64::NAN,
            df_cpm: f64::NAN,
            p_value_cpm: f64::NAN,
            qsignificant_cpm: "NS".to_string(),
            lsrmr_cpm: f64::NAN,
            lsrmr_above_threshold_cpm: "No".to_string(),
            heterogeneity_cpm: "No".to_string(),
            rgf1trait_fum: None,
            sergf1trait_fum: None,
            pvalrgf1trait_fum: None,
            rgf1trait_significant_fum: None,
            qtrait_fum: None,
            df_fum: None,
            p_value_fum: None,
            qsignificant_fum: None,
            lsrmr_fum: None,
            lsrmr_above_threshold_fum: None,
            reduction_lsrmr: None,
            heterogeneity_fum: None,
            unconstrained_paths: None,
        }
    }

    fn mark_all_na(&mut self) {
        self.rgf1trait_cpm = f64::NAN;
        self.sergf1trait_cpm = f64::NAN;
        self.pvalrgf1trait_cpm = f64::NAN;
        self.rgf1trait_significant_cpm = "NA".to_string();
        self.qtrait_cpm = f64::NAN;
        self.df_cpm = f64::NAN;
        self.p_value_cpm = f64::NAN;
        self.qsignificant_cpm = "NA".to_string();
        self.lsrmr_cpm = f64::NAN;
        self.lsrmr_above_threshold_cpm = "NA".to_string();
        self.heterogeneity_cpm = "NA".to_string();
        self.rgf1trait_fum = None;
        self.sergf1trait_fum = None;
        self.pvalrgf1trait_fum = None;
        self.rgf1trait_significant_fum = Some("NA".to_string());
        self.qtrait_fum = None;
        self.df_fum = None;
        self.p_value_fum = None;
        self.qsignificant_fum = Some("NA".to_string());
        self.lsrmr_fum = None;
        self.lsrmr_above_threshold_fum = Some("NA".to_string());
        self.reduction_lsrmr = Some("NA".to_string());
        self.heterogeneity_fum = Some("NA".to_string());
        self.unconstrained_paths = Some("None".to_string());
    }
}

fn build_qtrait_df(rows: &[QTraitRow]) -> Result<DataFrame> {
    let mut trait_name = Vec::new();
    let mut rgf1trait_cpm = Vec::new();
    let mut sergf1trait_cpm = Vec::new();
    let mut pvalrgf1trait_cpm = Vec::new();
    let mut rgf1trait_significant_cpm = Vec::new();
    let mut qtrait_cpm = Vec::new();
    let mut df_cpm = Vec::new();
    let mut p_value_cpm = Vec::new();
    let mut qsignificant_cpm = Vec::new();
    let mut lsrmr_cpm = Vec::new();
    let mut lsrmr_above_threshold_cpm = Vec::new();
    let mut heterogeneity_cpm = Vec::new();
    let mut rgf1trait_fum = Vec::new();
    let mut sergf1trait_fum = Vec::new();
    let mut pvalrgf1trait_fum = Vec::new();
    let mut rgf1trait_significant_fum = Vec::new();
    let mut qtrait_fum = Vec::new();
    let mut df_fum = Vec::new();
    let mut p_value_fum = Vec::new();
    let mut qsignificant_fum = Vec::new();
    let mut lsrmr_fum = Vec::new();
    let mut lsrmr_above_threshold_fum = Vec::new();
    let mut reduction_lsrmr = Vec::new();
    let mut heterogeneity_fum = Vec::new();
    let mut unconstrained_paths = Vec::new();

    for row in rows {
        trait_name.push(row.trait_name.clone());
        rgf1trait_cpm.push(row.rgf1trait_cpm);
        sergf1trait_cpm.push(row.sergf1trait_cpm);
        pvalrgf1trait_cpm.push(row.pvalrgf1trait_cpm);
        rgf1trait_significant_cpm.push(row.rgf1trait_significant_cpm.clone());
        qtrait_cpm.push(row.qtrait_cpm);
        df_cpm.push(row.df_cpm);
        p_value_cpm.push(row.p_value_cpm);
        qsignificant_cpm.push(row.qsignificant_cpm.clone());
        lsrmr_cpm.push(row.lsrmr_cpm);
        lsrmr_above_threshold_cpm.push(row.lsrmr_above_threshold_cpm.clone());
        heterogeneity_cpm.push(row.heterogeneity_cpm.clone());
        rgf1trait_fum.push(row.rgf1trait_fum.unwrap_or(f64::NAN));
        sergf1trait_fum.push(row.sergf1trait_fum.unwrap_or(f64::NAN));
        pvalrgf1trait_fum.push(row.pvalrgf1trait_fum.unwrap_or(f64::NAN));
        rgf1trait_significant_fum.push(
            row.rgf1trait_significant_fum
                .clone()
                .unwrap_or_else(|| "-".to_string()),
        );
        qtrait_fum.push(row.qtrait_fum.unwrap_or(f64::NAN));
        df_fum.push(row.df_fum.unwrap_or(f64::NAN));
        p_value_fum.push(row.p_value_fum.unwrap_or(f64::NAN));
        qsignificant_fum.push(
            row.qsignificant_fum
                .clone()
                .unwrap_or_else(|| "-".to_string()),
        );
        lsrmr_fum.push(row.lsrmr_fum.unwrap_or(f64::NAN));
        lsrmr_above_threshold_fum.push(
            row.lsrmr_above_threshold_fum
                .clone()
                .unwrap_or_else(|| "-".to_string()),
        );
        reduction_lsrmr.push(
            row.reduction_lsrmr
                .clone()
                .unwrap_or_else(|| "-".to_string()),
        );
        heterogeneity_fum.push(
            row.heterogeneity_fum
                .clone()
                .unwrap_or_else(|| "-".to_string()),
        );
        unconstrained_paths.push(
            row.unconstrained_paths
                .clone()
                .unwrap_or_else(|| "None".to_string()),
        );
    }

    let df = DataFrame::new(
        rows.len(),
        vec![
            Series::new("".into(), trait_name).into(),
            Series::new("rGF1Trait_CPM".into(), rgf1trait_cpm).into(),
            Series::new("SErGF1Trait_CPM".into(), sergf1trait_cpm).into(),
            Series::new("pvalrGF1Trait_CPM".into(), pvalrgf1trait_cpm).into(),
            Series::new("rGF1Trait_significat_CPM".into(), rgf1trait_significant_cpm).into(),
            Series::new("QTrait_CPM".into(), qtrait_cpm).into(),
            Series::new("df_CPM".into(), df_cpm).into(),
            Series::new("p_value_CPM".into(), p_value_cpm).into(),
            Series::new("Qsignificant_CPM".into(), qsignificant_cpm).into(),
            Series::new("lSRMR_CPM".into(), lsrmr_cpm).into(),
            Series::new(
                "lSRMR_above_threshold_CPM".into(),
                lsrmr_above_threshold_cpm,
            )
            .into(),
            Series::new("heterogeneity_CPM".into(), heterogeneity_cpm).into(),
            Series::new("rGF1Trait_FUM".into(), rgf1trait_fum).into(),
            Series::new("SErGF1Trait_FUM".into(), sergf1trait_fum).into(),
            Series::new("pvalrGF1Trait_FUM".into(), pvalrgf1trait_fum).into(),
            Series::new("rGF1Trait_significat_FUM".into(), rgf1trait_significant_fum).into(),
            Series::new("QTrait_FUM".into(), qtrait_fum).into(),
            Series::new("df_FUM".into(), df_fum).into(),
            Series::new("p_value_FUM".into(), p_value_fum).into(),
            Series::new("Qsignificant_FUM".into(), qsignificant_fum).into(),
            Series::new("lSRMR_FUM".into(), lsrmr_fum).into(),
            Series::new(
                "lSRMR_above_threshold_FUM".into(),
                lsrmr_above_threshold_fum,
            )
            .into(),
            Series::new("reduction_lSRMR".into(), reduction_lsrmr).into(),
            Series::new("heterogeneity_FUM".into(), heterogeneity_fum).into(),
            Series::new("Unconstrained_paths".into(), unconstrained_paths).into(),
        ],
    )?;
    Ok(df)
}

fn resolve_trait_names(ldsc: &LdscOutput) -> Result<Vec<String>> {
    if !ldsc.trait_names.is_empty() {
        return Ok(ldsc.trait_names.clone());
    }
    let k = ldsc.s.len();
    if k == 0 {
        return Err(anyhow::anyhow!("LDSC output is empty"));
    }
    Ok((1..=k).map(|i| format!("V{i}")).collect())
}

fn build_index_map(names: &[String]) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for (idx, name) in names.iter().enumerate() {
        out.insert(name.clone(), idx);
    }
    out
}

fn collect_vars(indicators: &[String], trait_name: &str) -> Vec<String> {
    let mut out = indicators.to_vec();
    if !out.iter().any(|v| v == trait_name) {
        out.push(trait_name.to_string());
    }
    out
}

fn build_common_factor_model(indicators: &[String]) -> String {
    let mut model = String::new();
    if indicators.is_empty() {
        return model;
    }
    model.push_str("F1 =~ ");
    model.push_str(&indicators.join(" + "));
    for (idx, ind) in indicators.iter().enumerate() {
        let rv = format!("rv{}", idx + 1);
        model.push_str(&format!("\n{ind} ~~ {rv}*{ind}\n{rv} >0.001"));
    }
    model
}

fn build_common_factor_fixed(lambdas: &[(String, f64)]) -> String {
    let mut model = String::new();
    if lambdas.is_empty() {
        return model;
    }
    model.push_str("F1 =~ ");
    let terms = lambdas
        .iter()
        .map(|(name, est)| format!("{:.15}*{name}", est))
        .collect::<Vec<_>>()
        .join(" + ");
    model.push_str(&terms);
    for (idx, (name, _)) in lambdas.iter().enumerate() {
        let rv = format!("rv{}", idx + 1);
        model.push_str(&format!("\n{name} ~~ {rv}*{name}\n{rv} >0.001"));
    }
    model
}

fn build_cpm_model(cf_fixed: &str, trait_name: &str, ext_latent: &str) -> String {
    format!(
        "{cf_fixed}\n{ext_latent} =~ NA*{trait_name}\n{trait_name} ~~ 0*{trait_name}\nF1 ~ {ext_latent}\n{ext_latent} ~~ 1*{ext_latent}"
    )
}

fn build_ipm_model(
    cf_fixed: &str,
    trait_name: &str,
    ext_latent: &str,
    indicators: &[String],
) -> String {
    let regress = indicators.join(" + ");
    format!(
        "{cf_fixed}\n{ext_latent} =~ NA*{trait_name}\n{trait_name} ~~ 0*{trait_name}\n{regress} ~ {ext_latent}\n{ext_latent} ~~ 1*{ext_latent}\nF1~~0*{ext_latent}\nF1~~rvl*F1\nrvl>0.001"
    )
}

fn build_fum_model(
    cf_fixed: &str,
    trait_name: &str,
    ext_latent: &str,
    unconstrained: &str,
) -> String {
    format!(
        "{cf_fixed}\n{ext_latent} =~ NA*{trait_name}\n{trait_name} ~~ 0*{trait_name}\nF1 ~ {ext_latent}\n{ext_latent} ~~ 1*{ext_latent}\n{unconstrained} ~ {ext_latent}\nF1~~rvl*F1\nrvl>0.001"
    )
}

fn fit_sem(
    s: &Matrix,
    v: &Matrix,
    model: &str,
    names: &[String],
    std_lv: bool,
) -> Result<lavaan_crate::SemFit> {
    let (s_smoothed, _smoothed_s, _ld_sdiff) = smooth_if_needed(s)?;
    let (v_smoothed, _smoothed_v, _ld_sdiff2) = smooth_if_needed(v)?;
    let input = lavaan_crate::SemInput {
        s: s_smoothed,
        v: v_smoothed,
        wls_v: None,
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

fn subset_covstruc(
    s: &Matrix,
    v: &Matrix,
    all_names: &[String],
    keep_names: &[String],
) -> Result<(Matrix, Matrix, Vec<String>)> {
    let mut keep_idx = Vec::new();
    for name in keep_names {
        if let Some(pos) = all_names.iter().position(|n| n == name) {
            keep_idx.push(pos);
        }
    }
    if keep_idx.is_empty() {
        return Err(anyhow::anyhow!("No matching traits for model"));
    }
    let mut s_sub = vec![vec![0.0; keep_idx.len()]; keep_idx.len()];
    for (i_out, &i) in keep_idx.iter().enumerate() {
        for (j_out, &j) in keep_idx.iter().enumerate() {
            s_sub[i_out][j_out] = s[i][j];
        }
    }
    let v_sub = subset_sampling_covariance(v, &keep_idx, s.len())?;
    let names_sub = keep_idx.iter().map(|&i| all_names[i].clone()).collect();
    Ok((s_sub, v_sub, names_sub))
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

fn find_param_est(fit: &lavaan_crate::SemFit, key: (&str, &str, &str)) -> Option<f64> {
    fit.params
        .iter()
        .find(|p| p.lhs == key.0 && p.op == key.1 && p.rhs == key.2)
        .map(|p| p.est)
}

fn extract_beta(
    fit: &lavaan_crate::SemFit,
    lhs: &str,
    op: &str,
    rhs: &str,
) -> Result<(f64, f64, f64)> {
    let param = fit
        .params
        .iter()
        .find(|p| p.lhs == lhs && p.op == op && p.rhs == rhs)
        .ok_or_else(|| anyhow::anyhow!("Missing parameter {lhs} {op} {rhs}"))?;
    Ok((param.est, param.se, param.est_std_all.unwrap_or(f64::NAN)))
}

fn residual_variances(fit: &lavaan_crate::SemFit, indicators: &[String]) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for p in &fit.params {
        if p.op == "~~" && p.lhs == p.rhs && indicators.iter().any(|i| i == &p.lhs) {
            out.insert(p.lhs.clone(), p.est);
        }
    }
    out
}

fn extract_indicator_betas(
    fit: &lavaan_crate::SemFit,
    indicators: &[String],
    rhs: &str,
) -> (Vec<f64>, Vec<f64>) {
    let mut betas = Vec::with_capacity(indicators.len());
    let mut ses = Vec::with_capacity(indicators.len());
    for ind in indicators {
        if let Some(p) = fit
            .params
            .iter()
            .find(|p| p.lhs == *ind && p.op == "~" && p.rhs == rhs)
        {
            betas.push(p.est);
            ses.push(p.se);
        } else {
            betas.push(f64::NAN);
            ses.push(f64::NAN);
        }
    }
    (betas, ses)
}

fn compute_std_lambdas(
    lambdas: &[(String, f64)],
    residuals: &HashMap<String, f64>,
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for (name, lambda) in lambdas {
        let resid = residuals.get(name).copied().unwrap_or(f64::NAN);
        let denom = (lambda * lambda + resid).sqrt();
        let std = if denom.is_finite() && denom != 0.0 {
            *lambda / denom
        } else {
            f64::NAN
        };
        out.insert(name.clone(), std);
    }
    out
}

fn build_rgs(
    s_stand: &Matrix,
    all_names: &[String],
    trait_name: &str,
    indicators: &[String],
    fit: &lavaan_crate::SemFit,
    sub_names: &[String],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let t_idx = all_names
        .iter()
        .position(|n| n == trait_name)
        .ok_or_else(|| anyhow::anyhow!("Trait {trait_name} not found"))?;
    let mut observed = Vec::with_capacity(indicators.len());
    for ind in indicators {
        let idx = all_names
            .iter()
            .position(|n| n == ind)
            .ok_or_else(|| anyhow::anyhow!("Indicator {ind} not found"))?;
        observed.push(s_stand[t_idx][idx]);
    }

    let implied_cor = cov2cor(&fit.implied);
    let t_sub_idx = sub_names
        .iter()
        .position(|n| n == trait_name)
        .ok_or_else(|| anyhow::anyhow!("Trait {trait_name} not in subset"))?;
    let mut implied = Vec::with_capacity(indicators.len());
    for ind in indicators {
        let idx = sub_names
            .iter()
            .position(|n| n == ind)
            .ok_or_else(|| anyhow::anyhow!("Indicator {ind} not in subset"))?;
        implied.push(implied_cor[t_sub_idx][idx]);
    }

    let residual = observed.iter().zip(&implied).map(|(o, m)| o - m).collect();
    Ok((observed, implied, residual))
}

fn se_rgs_from_vstand(
    v_stand: &Matrix,
    all_names: &[String],
    trait_name: &str,
    indicators: &[String],
) -> Result<Vec<f64>> {
    let k = all_names.len();
    let t_idx = all_names
        .iter()
        .position(|n| n == trait_name)
        .ok_or_else(|| anyhow::anyhow!("Trait {trait_name} not found"))?;
    let mut out = Vec::with_capacity(indicators.len());
    for ind in indicators {
        let idx = all_names
            .iter()
            .position(|n| n == ind)
            .ok_or_else(|| anyhow::anyhow!("Indicator {ind} not found"))?;
        let (row, col) = if t_idx >= idx {
            (t_idx, idx)
        } else {
            (idx, t_idx)
        };
        let vech = vech_index(row, col, k);
        let val = v_stand
            .get(vech)
            .and_then(|r| r.get(vech))
            .copied()
            .unwrap_or(f64::NAN);
        if val.is_finite() && val >= 0.0 {
            out.push(val.sqrt());
        } else {
            out.push(f64::NAN);
        }
    }
    Ok(out)
}

fn z_to_p(normal: &Normal, est: f64, se: f64) -> f64 {
    if !est.is_finite() || !se.is_finite() || se == 0.0 {
        return f64::NAN;
    }
    let z = est / se;
    let mut p = 2.0 * (1.0 - normal.cdf(z.abs()));
    if p.is_finite() && p < 5e-300 {
        p = 5e-300;
    }
    p
}

fn pchisq(chisq: f64, df: f64) -> f64 {
    if !chisq.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    let dist = ChiSquared::new(df).ok();
    if let Some(dist) = dist {
        1.0 - dist.cdf(chisq.max(0.0))
    } else {
        f64::NAN
    }
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mean = values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64;
    mean.sqrt()
}

fn sig_from_pval(pval: f64, threshold: f64) -> String {
    if !pval.is_finite() {
        "NA".to_string()
    } else if pval < threshold {
        "*".to_string()
    } else {
        "NS".to_string()
    }
}

fn lsrmr_flag(lsrmr: f64, threshold: f64, lsrmr_prop: f64, mr_g: f64) -> String {
    if !lsrmr.is_finite() || !mr_g.is_finite() {
        return "NA".to_string();
    }
    if lsrmr > threshold && lsrmr > lsrmr_prop * mr_g {
        "Yes".to_string()
    } else {
        "No".to_string()
    }
}

fn identify_outliers(
    residual: &[f64],
    observed: &[f64],
    indicators: &[String],
    mresid: f64,
    mresid_threshold: f64,
) -> (String, Vec<String>) {
    let mr_g = rms(observed);
    let mut inside = Vec::with_capacity(indicators.len());
    for (idx, ind) in indicators.iter().enumerate() {
        let resid = residual.get(idx).copied().unwrap_or(0.0);
        if resid.abs() < mresid_threshold || resid.abs() < mresid * mr_g {
            inside.push(String::new());
        } else {
            inside.push(ind.clone());
        }
    }
    let outliers: Vec<String> = inside.iter().filter(|s| !s.is_empty()).cloned().collect();
    if outliers.is_empty() {
        ("None".to_string(), inside)
    } else {
        (outliers.join(","), inside)
    }
}

fn most_extreme_outlier(residual: &[f64], indicators: &[String]) -> String {
    let mut max_idx = 0usize;
    let mut max_val = 0.0;
    for (idx, val) in residual.iter().enumerate() {
        let abs = val.abs();
        if abs > max_val {
            max_val = abs;
            max_idx = idx;
        }
    }
    indicators
        .get(max_idx)
        .cloned()
        .unwrap_or_else(|| "".to_string())
}

fn order_by_residual(residual: &[f64], indicators: &[String]) -> Vec<String> {
    let mut pairs: Vec<(usize, f64)> = residual
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.abs()))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs
        .into_iter()
        .filter_map(|(idx, _)| indicators.get(idx).cloned())
        .collect()
}

fn next_outlier(ordered: &[String], existing: &str) -> Option<String> {
    let existing_set: std::collections::HashSet<&str> =
        existing.split(',').filter(|s| !s.is_empty()).collect();
    for name in ordered {
        if !existing_set.contains(name.as_str()) {
            return Some(name.clone());
        }
    }
    None
}

fn merge_outliers(existing: &str, next: &str) -> String {
    let mut all: Vec<String> = existing
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if !all.iter().any(|s| s == next) {
        all.push(next.to_string());
    }
    all.join(",")
}

fn fit_fum(
    cf_fixed: &str,
    trait_name: &str,
    ext_latent: &str,
    s_sub: &Matrix,
    v_sub: &Matrix,
    names_sub: &[String],
    unconstrained_paths: &str,
) -> Result<lavaan_crate::SemFit> {
    let unconstrained = unconstrained_paths
        .split(',')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" + ");
    let model = build_fum_model(cf_fixed, trait_name, ext_latent, &unconstrained);
    fit_sem(s_sub, v_sub, &model, names_sub, false)
}

fn all_residuals_within(
    residual: &[f64],
    observed: &[f64],
    mresid: f64,
    mresid_threshold: f64,
) -> bool {
    let mr_g = rms(observed);
    residual
        .iter()
        .all(|val| val.abs() < mresid_threshold || val.abs() < mresid * mr_g)
}

fn reduction_percent(base: f64, new: f64) -> String {
    if !base.is_finite() || base == 0.0 || !new.is_finite() {
        return "-".to_string();
    }
    let pct = ((base - new) / base) * 100.0;
    format!("{:.2}%", pct)
}

struct QTraitPlotData<'a> {
    indicators: &'a [String],
    trait_name: &'a str,
    observed: &'a [f64],
    implied: &'a [f64],
    residual: &'a [f64],
    lambda_unstd: &'a [f64],
    lambda_std: &'a [f64],
    beta_ind: &'a [f64],
    beta_ind_se: &'a [f64],
    se_rgs: &'a [f64],
    outliers: &'a str,
    slope_cpm: f64,
    slope_fum_label: Option<&'a str>,
    slope_fum: Option<f64>,
    stdout: bool,
}

fn write_qtrait_plots(data: QTraitPlotData<'_>) -> Result<()> {
    ensure_plots_dir()?;
    let suffix = data.indicators.join("_");

    let outlier_set: std::collections::HashSet<&str> =
        data.outliers.split(',').filter(|s| !s.is_empty()).collect();
    let colors: Vec<NamedColor> = data
        .indicators
        .iter()
        .map(|ind| {
            if outlier_set.contains(ind.as_str()) {
                NamedColor::Red
            } else {
                NamedColor::Black
            }
        })
        .collect();

    let (x_vals, y_vals, weight_vals, x_label, y_label, x_limits) = if data.stdout {
        let weights = weights_from_se(data.se_rgs);
        (
            data.lambda_std.to_vec(),
            data.observed.to_vec(),
            weights,
            "Lambda (Standardized)".to_string(),
            "Genetic correlation (rG)".to_string(),
            Some((0.0, 1.0)),
        )
    } else {
        let weights = weights_from_se(data.beta_ind_se);
        let min_x = data
            .lambda_unstd
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        let max_x = data
            .lambda_unstd
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let limits = if min_x.is_finite() && max_x.is_finite() {
            Some((min_x, max_x))
        } else {
            None
        };
        (
            data.lambda_unstd.to_vec(),
            data.beta_ind.to_vec(),
            weights,
            "Lambda (Unstandardized)".to_string(),
            "Beta (Unstandardized)".to_string(),
            limits,
        )
    };

    let sizes = scale_sizes(&weight_vals);
    let marker = Marker::default().size_array(sizes).color_array(colors);
    let text = data
        .indicators
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let scatter = Scatter::new(x_vals.clone(), y_vals.clone())
        .mode(Mode::MarkersText)
        .text_array(text)
        .marker(marker)
        .name("Indicators");

    let mut plot = Plot::new();
    plot.add_trace(scatter);

    let mut line_traces: Vec<(String, f64, NamedColor)> = vec![(
        format!("Common pathway model (slope {:.3})", data.slope_cpm),
        data.slope_cpm,
        NamedColor::Black,
    )];
    if let (Some(label), Some(slope)) = (data.slope_fum_label, data.slope_fum) {
        line_traces.push((
            format!("{label} (slope {:.3})", slope),
            slope,
            NamedColor::DarkGray,
        ));
    }

    for (label, slope, color) in line_traces {
        let (x_line, y_line) = if data.stdout {
            (vec![0.0, 1.0], vec![0.0, slope])
        } else if let Some((min_x, max_x)) = x_limits {
            (vec![min_x, max_x], vec![slope * min_x, slope * max_x])
        } else {
            (vec![0.0, 1.0], vec![0.0, slope])
        };
        let line = Line::default().color(color);
        let trace = Scatter::new(x_line, y_line)
            .mode(Mode::Lines)
            .name(label)
            .line(line);
        plot.add_trace(trace);
    }

    let mut x_axis = Axis::new().title(x_label.clone());
    let mut y_axis = Axis::new().title(y_label.clone());
    if data.stdout {
        x_axis = x_axis.range(vec![0.0_f64, 1.0_f64]);
        let (y_min, y_max, y_ticks) = standardized_y_axis(data.observed);
        y_axis = y_axis.range(vec![y_min, y_max]).tick_values(y_ticks);
        x_axis = x_axis.tick_values(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    }
    let layout = Layout::new()
        .title(format!("QTrait {}", data.trait_name))
        .x_axis(x_axis)
        .y_axis(y_axis);
    plot.set_layout(layout);
    let scatter_path = format!(
        "Plots/QTrait_Statistics_{suffix}_{}_scatter.html",
        data.trait_name
    );
    plot.write_html(scatter_path);

    let x_labels = ["Observed", "Model implied", "Residual"];
    let mut z = Vec::with_capacity(data.indicators.len());
    for idx in 0..data.indicators.len() {
        let row = vec![
            data.observed.get(idx).copied().unwrap_or(f64::NAN),
            data.implied.get(idx).copied().unwrap_or(f64::NAN),
            data.residual.get(idx).copied().unwrap_or(f64::NAN),
        ];
        z.push(row);
    }
    let heat_text = build_heat_text(
        data.observed,
        data.implied,
        data.residual,
        data.indicators.len(),
    );
    let heatmap = HeatMap::new(
        x_labels.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        data.indicators.to_vec(),
        z,
    )
    .name("Correlations")
    .text_matrix(heat_text);
    let mut heat_plot = Plot::new();
    heat_plot.add_trace(heatmap);
    heat_plot.set_layout(Layout::new().title(format!(
        "Observed, model-implied, and residual correlations ({})",
        data.trait_name
    )));
    let heat_path = format!(
        "Plots/QTrait_Statistics_{suffix}_{}_heatmap.html",
        data.trait_name
    );
    heat_plot.write_html(heat_path);

    Ok(())
}

fn weights_from_se(se: &[f64]) -> Vec<f64> {
    se.iter()
        .map(|v| {
            if v.is_finite() && *v > 0.0 {
                1.0 / (v * v)
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn scale_sizes(weights: &[f64]) -> Vec<usize> {
    let finite: Vec<f64> = weights.iter().copied().filter(|v| v.is_finite()).collect();
    let (min_w, max_w) = if finite.is_empty() {
        (0.0, 1.0)
    } else {
        (
            finite.iter().cloned().fold(f64::INFINITY, f64::min),
            finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let min_size = 8.0;
    let max_size = 18.0;
    weights
        .iter()
        .map(|w| {
            if !w.is_finite() || (max_w - min_w).abs() < f64::EPSILON {
                min_size as usize
            } else {
                let t = (*w - min_w) / (max_w - min_w);
                (min_size + t * (max_size - min_size))
                    .round()
                    .clamp(min_size, max_size) as usize
            }
        })
        .collect()
}

fn standardized_y_axis(observed: &[f64]) -> (f64, f64, Vec<f64>) {
    if observed.iter().all(|v| v.is_finite() && *v < 0.0) {
        return (-1.0, 0.0, vec![-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]);
    }
    if observed.iter().all(|v| v.is_finite() && *v > 0.0) {
        return (0.0, 1.0, vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    }
    (-1.0, 1.0, vec![-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
}

fn build_heat_text(
    observed: &[f64],
    implied: &[f64],
    residual: &[f64],
    rows: usize,
) -> Vec<Vec<String>> {
    let mut out = Vec::with_capacity(rows);
    for idx in 0..rows {
        let row = vec![
            format!("{:.2}", observed.get(idx).copied().unwrap_or(f64::NAN)),
            format!("{:.2}", implied.get(idx).copied().unwrap_or(f64::NAN)),
            format!("{:.2}", residual.get(idx).copied().unwrap_or(f64::NAN)),
        ];
        out.push(row);
    }
    out
}
