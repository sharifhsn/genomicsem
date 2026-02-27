use std::path::PathBuf;

use anyhow::{Context, Result};
use polars::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use tracing::info;

use crate::io::read_table;
use crate::types::SumstatsTable;

#[derive(Debug, Clone)]
pub struct ReadFusionConfig {
    pub files: Vec<PathBuf>,
    pub trait_names: Option<Vec<String>>,
    pub binary: Option<Vec<bool>>,
    pub n: Option<Vec<f64>>,
    pub perm: bool,
}

pub fn read_fusion(_config: &ReadFusionConfig) -> Result<SumstatsTable> {
    let config = _config;
    let n_files = config.files.len();
    if n_files == 0 {
        return Err(anyhow::anyhow!("No TWAS files provided"));
    }

    let trait_names = if let Some(names) = &config.trait_names {
        if names.len() != n_files {
            return Err(anyhow::anyhow!(
                "trait.names length {} does not match files length {}",
                names.len(),
                n_files
            ));
        }
        names.clone()
    } else {
        (1..=n_files).map(|i| format!("{i}")).collect()
    };

    let binary = if let Some(b) = config.binary.clone() {
        b
    } else {
        tracing::warn!(
            "Running read_fusion assuming all traits are binary; set --binary to override."
        );
        vec![true; n_files]
    };
    if binary.len() != n_files {
        return Err(anyhow::anyhow!(
            "binary length {} does not match files length {}",
            binary.len(),
            n_files
        ));
    }

    let n = config.n.clone().unwrap_or_default();
    if (!n.is_empty() && n.len() != n_files) || (n.is_empty() && binary.iter().any(|b| *b)) {
        return Err(anyhow::anyhow!(
            "N must be provided for each file when computing TWAS effects"
        ));
    }

    info!("Please note that the TWAS files should be in the same order as ldsc.");

    let chi = ChiSquared::new(1.0).context("chi-squared")?;

    let mut data_out: Option<DataFrame> = None;
    for (idx, path) in config.files.iter().enumerate() {
        let mut df = read_table(path)?;
        df = df.select(["FILE", "ID", "TWAS.Z", "HSQ"])?;
        if config.perm {
            df = df.hstack(&[df.column("PERM.PV")?.clone(), df.column("PERM.N")?.clone()])?;
        }

        let hsq_vals: Vec<f64> = df
            .column("HSQ")?
            .f64()
            .context("HSQ")?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        let twas_vals: Vec<f64> = df
            .column("TWAS.Z")?
            .f64()
            .context("TWAS.Z")?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();

        let mut effect = Vec::with_capacity(df.height());
        let mut se = Vec::with_capacity(df.height());

        let n_i = n.get(idx).copied().unwrap_or(f64::NAN);

        let (perm_pv, perm_n) = if config.perm {
            (
                Some(df.column("PERM.PV")?.f64().context("PERM.PV")?),
                Some(df.column("PERM.N")?.f64().context("PERM.N")?),
            )
        } else {
            (None, None)
        };

        for row in 0..df.height() {
            let h = hsq_vals.get(row).copied().unwrap_or(f64::NAN);
            let z = twas_vals.get(row).copied().unwrap_or(f64::NAN);
            let mut z_use = z;
            if let (Some(pv), Some(n_perm)) = (perm_pv.as_ref(), perm_n.as_ref()) {
                let mut p = pv.get(row).unwrap_or(1.0);
                let nperm = n_perm.get(row).unwrap_or(1.0);
                if p == 1.0 && nperm > 0.0 {
                    p = (nperm - 1.0) / nperm;
                }
                if p == 0.0 && nperm > 0.0 {
                    p = 1.0 / nperm;
                }
                let z_perm = (chi.inverse_cdf(1.0 - p)).sqrt();
                z_use = z.signum() * z_perm;
            }

            if binary[idx] {
                let denom = (n_i / 4.0 * h).sqrt();
                let eff = if denom != 0.0 {
                    z_use / denom
                } else {
                    f64::NAN
                };
                let se_i = if config.perm {
                    (eff / z_use).abs()
                } else if denom != 0.0 {
                    1.0 / denom
                } else {
                    f64::NAN
                };
                effect.push(eff);
                se.push(se_i);
            } else {
                let denom = (n_i * h).sqrt();
                let eff = if denom != 0.0 {
                    z_use / denom
                } else {
                    f64::NAN
                };
                let se_i = (eff / z_use).abs();
                effect.push(eff);
                se.push(se_i);
            }
        }

        let effect = Series::new("effect".into(), effect);
        let se = Series::new("se".into(), se);
        df.with_column(effect.into())?;
        df.with_column(se.into())?;

        let beta_name = format!("beta.{}", trait_names[idx]);
        let se_name = format!("se.{}", trait_names[idx]);

        let panel = df.column("FILE")?.str().context("FILE")?;
        let panel_vals: Vec<String> = panel
            .into_iter()
            .map(|v| strip_panel_path(v.unwrap_or("")))
            .collect();

        let panel_series = Series::new("Panel".into(), panel_vals);
        let mut gene_series = df.column("ID")?.clone();
        gene_series.rename("Gene".into());

        let hsq_series = Series::new("HSQ".into(), hsq_vals.clone());

        let mut beta = Vec::with_capacity(df.height());
        let mut se_out = Vec::with_capacity(df.height());
        let effect_col = df.column("effect")?.f64().context("effect")?;
        let se_col = df.column("se")?.f64().context("se")?;
        for (row, hsq) in hsq_vals.iter().enumerate().take(df.height()) {
            let eff = effect_col.get(row).unwrap_or(f64::NAN);
            let se_i = se_col.get(row).unwrap_or(f64::NAN);
            if binary[idx] {
                let denom = (eff * eff * hsq + std::f64::consts::PI.powi(2) / 3.0).sqrt();
                beta.push(if denom != 0.0 { eff / denom } else { f64::NAN });
                se_out.push(if denom != 0.0 { se_i / denom } else { f64::NAN });
            } else {
                beta.push(eff);
                se_out.push(se_i);
            }
        }
        let beta_series = Series::new(beta_name.as_str().into(), beta);
        let se_series = Series::new(se_name.as_str().into(), se_out);

        let mut out = DataFrame::new(
            df.height(),
            vec![
                panel_series.into(),
                gene_series,
                beta_series.into(),
                se_series.into(),
            ],
        )?;

        if idx == 0 {
            out.with_column(hsq_series.into())?;
            out = out.select(["HSQ", "Panel", "Gene", &beta_name, &se_name])?;
            out = out.drop_nulls::<&str>(None)?;
            data_out = Some(out);
        } else {
            let mut merged = data_out.take().context("Missing prior TWAS merge")?.join(
                &out,
                ["Gene", "Panel"],
                ["Gene", "Panel"],
                JoinType::Inner.into(),
                None,
            )?;
            merged = merged.drop_nulls::<&str>(None)?;
            data_out = Some(merged);
        }
    }

    let mut final_df = data_out.context("No TWAS output")?;
    final_df = final_df.unique::<&[&str], &str>(None, UniqueKeepStrategy::First, None)?;
    Ok(SumstatsTable { df: final_df })
}

fn strip_panel_path(value: &str) -> String {
    // Matches R regex sub(".*//([^/]+/[^/]+)$|.*/([^/]+/[^/]+)$", ...):
    // keep the last two non-empty path segments.
    let parts: Vec<&str> = value.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() >= 2 {
        format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1])
    } else {
        value.to_string()
    }
}
