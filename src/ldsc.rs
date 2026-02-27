use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use polars::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::io::{read_table, write_ldsc_json};
use crate::logging::{log_line, warn_line};
use crate::types::{LdscOutput, Matrix};

type WeightedRegressionData = (Vec<f64>, Vec<[f64; 2]>, Vec<f64>);

#[derive(Debug, Clone)]
pub enum ChromosomeSelect {
    All,
    Odd,
    Even,
    List(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct LdscConfig {
    pub traits: Vec<PathBuf>,
    pub sample_prev: Vec<Option<f64>>,
    pub population_prev: Vec<Option<f64>>,
    pub ld: PathBuf,
    pub wld: PathBuf,
    pub trait_names: Option<Vec<String>>,
    pub sep_weights: bool,
    pub chr: u8,
    pub n_blocks: usize,
    pub ldsc_log: Option<PathBuf>,
    pub stand: bool,
    pub select: ChromosomeSelect,
    pub chisq_max: Option<f64>,
    pub output: Option<PathBuf>,
}

pub fn ldsc(config: &LdscConfig) -> Result<LdscOutput> {
    let mut log = open_log_file(config)?;
    log_line(
        &mut log,
        &format!(
            "Multivariate ld-score regression of {} traits began",
            config.traits.len()
        ),
        true,
    )?;

    let n_traits = config.traits.len();
    if n_traits == 1 {
        warn_line(
            &mut log,
            "WARNING: Our version of ldsc requires 2 or more traits. Please include an additional trait.",
        )?;
    }

    let mut n_blocks = config.n_blocks;
    if n_traits > 18 {
        n_blocks = ((n_traits + 1) * (n_traits + 2)) / 2 + 1;
        log_line(
            &mut log,
            &format!("Setting n_blocks to {n_blocks} based on the number of traits."),
            true,
        )?;
        if n_blocks > 1000 {
            warn_line(
                &mut log,
                "WARNING: The number of blocks needed to estimate V is > 1000, which may bias results.",
            )?;
        }
    }

    let sample_prev = normalize_optional_vec(&config.sample_prev, n_traits)?;
    let population_prev = normalize_optional_vec(&config.population_prev, n_traits)?;
    let trait_names = resolve_trait_names(config, n_traits, &mut log)?;

    let chromosomes = resolve_chromosomes(config);
    log_line(
        &mut log,
        &format!("Reading LD scores for {} chromosomes", chromosomes.len()),
        true,
    )?;

    let ld_scores = read_ld_scores(&config.ld, &chromosomes)?;
    let weights = read_weights(&config.wld, config.sep_weights, &chromosomes, &ld_scores)?;
    let m_tot = read_m_values(&config.ld, &chromosomes)?;

    let mut merged_traits = Vec::new();
    let mut chisq_max = config.chisq_max;
    for (idx, trait_path) in config.traits.iter().enumerate() {
        let trait_df = read_trait_sumstats(trait_path, &mut log, idx + 1, config.traits.len())?;
        let merged = merge_with_ld_scores(&trait_df, &weights, &ld_scores, &mut log, trait_path)?;
        let filtered = filter_chisq(merged, &mut chisq_max, &mut log, trait_path)?;
        merged_traits.push(filtered);
    }

    let n_pairs = n_traits * (n_traits + 1) / 2;
    let chi_sq = statrs::distribution::ChiSquared::new(1.0).context("chi-square")?;
    let chi_sq_median = chi_sq.inverse_cdf(0.5);
    let normal = Normal::new(0.0, 1.0).context("normal distribution")?;
    let mut cov = vec![vec![f64::NAN; n_traits]; n_traits];
    let mut intercepts = vec![vec![f64::NAN; n_traits]; n_traits];
    let mut v_hold = vec![vec![0.0; n_pairs]; n_blocks];
    let mut n_vec = vec![f64::NAN; n_pairs];
    let mut liab_s = vec![1.0; n_traits];

    let mut s_index = 0usize;
    for j in 0..n_traits {
        let y1 = &merged_traits[j];
        let y1_arrays = trait_arrays(y1)?;

        let chi1: Vec<f64> = y1_arrays.z.iter().map(|v| v * v).collect();
        let mean_chi = mean(&chi1);
        let mean_l2n = mean_product(&y1_arrays.l2, &y1_arrays.n);
        let n_bar = mean(&y1_arrays.n);

        let (_weights, weighted_ld, weighted_chi) =
            weights_for_h2(&y1_arrays, &chi1, m_tot, mean_chi, mean_l2n)?;

        let (reg, pseudo_values) =
            jackknife_regression(&weighted_ld, &weighted_chi, n_blocks, y1_arrays.len())?;

        let intercept = reg[1];
        let coef = reg[0] / n_bar;
        let reg_tot = coef * m_tot;

        let (intercept_se, tot_se) = jackknife_se(&pseudo_values, n_bar, m_tot)?;
        let lambda_gc = median(&chi1).map(|med| med / chi_sq_median);
        let ratio = safe_div(intercept - 1.0, mean_chi - 1.0);
        let ratio_se = safe_div(intercept_se, mean_chi - 1.0);

        for (b, pv) in pseudo_values.iter().enumerate() {
            v_hold[b][s_index] = pv[0];
        }
        n_vec[s_index] = n_bar;
        cov[j][j] = reg_tot;
        intercepts[j][j] = intercept;

        if let (Some(pop), Some(samp)) = (population_prev[j], sample_prev[j]) {
            let conv = liability_conversion(pop, samp)?;
            liab_s[j] = conv;
            log_line(
                &mut log,
                &format!("Using liability conversion for trait {}", j + 1),
                true,
            )?;
        }

        log_line(
            &mut log,
            &format!("Heritability results for trait {}", trait_names[j]),
            true,
        )?;
        log_line(&mut log, &format!("Mean Chi^2: {:.4}", mean_chi), true)?;
        if let Some(lambda) = lambda_gc {
            log_line(&mut log, &format!("Lambda GC: {:.4}", lambda), true)?;
        }
        log_line(
            &mut log,
            &format!("Intercept: {:.4} ({:.4})", intercept, intercept_se),
            true,
        )?;
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
        let h2_z = safe_div(reg_tot, tot_se);
        if h2_z.is_finite() {
            log_line(&mut log, &format!("h2 Z: {:.3}", h2_z), true)?;
        }

        s_index += 1;

        for k in (j + 1)..n_traits {
            let y2 = &merged_traits[k];
            let merged = merge_trait_pairs(y1, y2)?;
            let pair_arrays = pair_arrays(&merged)?;

            let chi1_pair: Vec<f64> = pair_arrays.z_x.iter().map(|v| v * v).collect();
            let chi2_pair: Vec<f64> = pair_arrays.z_y.iter().map(|v| v * v).collect();
            let zz: Vec<f64> = pair_arrays
                .z_x_aligned
                .iter()
                .zip(&pair_arrays.z_y)
                .map(|(a, b)| a * b)
                .collect();

            let (weights_trait1, weights_cov) =
                weights_for_cov(&pair_arrays, &chi1_pair, &chi2_pair, m_tot)?;

            // R uses merged$weights here, but in the covariance block it is never defined.
            // We use trait-1 weights to keep behavior deterministic and aligned with the
            // heritability weighting scheme.
            let weighted_ld: Vec<[f64; 2]> = pair_arrays
                .l2
                .iter()
                .zip(&weights_trait1)
                .map(|(l2, w)| [l2 * w, *w])
                .collect();
            let weighted_chi: Vec<f64> = zz.iter().zip(&weights_cov).map(|(v, w)| v * w).collect();

            let n_bar = (mean(&pair_arrays.n_x) * mean(&pair_arrays.n_y)).sqrt();

            let (reg, pseudo_values) =
                jackknife_regression(&weighted_ld, &weighted_chi, n_blocks, pair_arrays.len())?;

            let intercept = reg[1];
            let coef = reg[0] / n_bar;
            let reg_tot = coef * m_tot;

            let (intercept_se, tot_se) = jackknife_se(&pseudo_values, n_bar, m_tot)?;
            let mean_zz = mean(&zz);
            let gcov_z = safe_div(reg_tot, tot_se);
            let gcov_p = if gcov_z.is_finite() {
                2.0 * (1.0 - normal.cdf(gcov_z.abs()))
            } else {
                f64::NAN
            };

            for (b, pv) in pseudo_values.iter().enumerate() {
                v_hold[b][s_index] = pv[0];
            }
            n_vec[s_index] = n_bar;

            cov[j][k] = reg_tot;
            cov[k][j] = reg_tot;
            intercepts[j][k] = intercept;
            intercepts[k][j] = intercept;

            log_line(
                &mut log,
                &format!(
                    "Genetic covariance between {} and {}",
                    trait_names[j], trait_names[k]
                ),
                true,
            )?;
            log_line(&mut log, &format!("Mean Z*Z: {:.4}", mean_zz), true)?;
            log_line(
                &mut log,
                &format!(
                    "Cross trait Intercept: {:.4} ({:.4})",
                    intercept, intercept_se
                ),
                true,
            )?;
            log_line(
                &mut log,
                &format!(
                    "Total Observed Scale Genetic Covariance: {:.4} ({:.4})",
                    reg_tot, tot_se
                ),
                true,
            )?;
            if gcov_z.is_finite() {
                log_line(&mut log, &format!("g_cov Z: {:.3}", gcov_z), true)?;
            }
            if gcov_p.is_finite() {
                log_line(&mut log, &format!("g_cov P-value: {:.5}", gcov_p), true)?;
            }

            s_index += 1;
        }
    }

    let v_out = sampling_covariance(&v_hold, &n_vec, n_blocks, m_tot)?;
    let ratio = outer_sqrt(&liab_s);
    let s_matrix = elementwise_mul(&cov, &ratio);
    // Assumes column-major lower-triangle ordering to match gdata::lowerTriangle.
    let scale_o = lower_triangle_values(&ratio);
    let v_matrix = scale_sampling_covariance(&v_out, &scale_o);

    if liab_s.iter().any(|v| (*v - 1.0).abs() > f64::EPSILON) {
        log_line(&mut log, "Liability Scale Results", true)?;
        let se = diag_se_from_v(&v_matrix, n_traits);
        for j in 0..n_traits {
            for k in j..n_traits {
                if j == k {
                    log_line(
                        &mut log,
                        &format!(
                            "Total Liability Scale h2 for {}: {:.4} ({:.4})",
                            trait_names[j], s_matrix[j][j], se[j][j]
                        ),
                        true,
                    )?;
                } else {
                    log_line(
                        &mut log,
                        &format!(
                            "Total Liability Scale Genetic Covariance between {} and {}: {:.4} ({:.4})",
                            trait_names[j], trait_names[k], s_matrix[k][j], se[k][j]
                        ),
                        true,
                    )?;
                }
            }
        }
    }

    let (s_stand, v_stand) = if config.stand && diag_positive(&s_matrix) {
        let inv_sqrt = s_matrix
            .iter()
            .enumerate()
            .map(|(i, row)| 1.0 / row[i].sqrt())
            .collect::<Vec<_>>();
        let ratio = outer_from_vec(&inv_sqrt);
        let s_stand = elementwise_mul(&s_matrix, &ratio);
        let scale_o = lower_triangle_values(&ratio);
        let v_stand = scale_sampling_covariance(&v_matrix, &scale_o);
        log_line(&mut log, "Genetic Correlation Results", true)?;
        let se = diag_se_from_v(&v_stand, n_traits);
        for j in 0..n_traits {
            for k in (j + 1)..n_traits {
                log_line(
                    &mut log,
                    &format!(
                        "Genetic Correlation between {} and {}: {:.4} ({:.4})",
                        trait_names[j], trait_names[k], s_stand[k][j], se[k][j]
                    ),
                    true,
                )?;
            }
        }
        (Some(s_stand), Some(v_stand))
    } else {
        if config.stand {
            warn_line(
                &mut log,
                "WARNING: Genetic correlation results could not be computed due to negative heritability estimates.",
            )?;
        }
        (None, None)
    };

    let output = LdscOutput {
        s: s_matrix,
        v: v_matrix,
        i: intercepts,
        n: n_vec,
        m: m_tot.round() as usize,
        s_stand,
        v_stand,
        trait_names,
    };
    write_ldsc_output(&output, config)?;
    Ok(output)
}

#[allow(clippy::needless_range_loop)]
fn diag_se_from_v(v: &Matrix, k: usize) -> Matrix {
    let mut out = vec![vec![0.0; k]; k];
    let mut idx = 0usize;
    for j in 0..k {
        for i in j..k {
            let val = v[idx][idx];
            out[i][j] = if val.is_finite() && val >= 0.0 {
                val.sqrt()
            } else {
                f64::NAN
            };
            out[j][i] = out[i][j];
            idx += 1;
        }
    }
    out
}

fn normalize_optional_vec(values: &[Option<f64>], n: usize) -> Result<Vec<Option<f64>>> {
    if values.is_empty() {
        return Ok(vec![None; n]);
    }
    if values.len() == 1 {
        return Ok(vec![values[0]; n]);
    }
    if values.len() != n {
        return Err(anyhow::anyhow!(
            "Expected {} values but got {}",
            n,
            values.len()
        ));
    }
    Ok(values.to_vec())
}

fn resolve_trait_names(
    config: &LdscConfig,
    n_traits: usize,
    log: &mut File,
) -> Result<Vec<String>> {
    if let Some(names) = &config.trait_names {
        if names.len() != n_traits {
            return Err(anyhow::anyhow!(
                "Expected {} trait names but got {}",
                n_traits,
                names.len()
            ));
        }
        if names.iter().any(|name| name.contains('-')) {
            warn_line(
                log,
                "WARNING: Trait names include '-' which may be misread by downstream lavaan.",
            )?;
        }
        return Ok(names.clone());
    }
    Ok((1..=n_traits).map(|i| format!("V{i}")).collect())
}

fn liability_conversion(pop: f64, samp: f64) -> Result<f64> {
    let norm = Normal::new(0.0, 1.0).context("normal distribution")?;
    let z = norm.inverse_cdf(1.0 - pop);
    let denom = samp * (1.0 - samp) * norm.pdf(z).powi(2);
    Ok((pop * pop * (1.0 - pop) * (1.0 - pop)) / denom)
}

fn open_log_file(config: &LdscConfig) -> Result<File> {
    let log_name = if let Some(path) = &config.ldsc_log {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("ldsc")
            .to_string()
    } else {
        let mut joined = config
            .traits
            .iter()
            .filter_map(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .collect::<Vec<_>>()
            .join("_");
        if joined.len() > 200 {
            joined.truncate(100);
        }
        if joined.is_empty() {
            "ldsc".to_string()
        } else {
            joined
        }
    };
    let path = format!("{log_name}_ldsc.log");
    Ok(File::create(path)?)
}

// log_line and warn_line are provided by logging.rs

fn resolve_chromosomes(config: &LdscConfig) -> Vec<u8> {
    match &config.select {
        ChromosomeSelect::All => (1..=config.chr).collect(),
        ChromosomeSelect::Odd => (1..=config.chr).filter(|v| v % 2 == 1).collect(),
        ChromosomeSelect::Even => (1..=config.chr).filter(|v| v % 2 == 0).collect(),
        ChromosomeSelect::List(list) => list.clone(),
    }
}

fn read_ld_scores(ld_dir: &Path, chromosomes: &[u8]) -> Result<DataFrame> {
    let mut frames = Vec::new();
    for chr in chromosomes {
        let path = ld_dir.join(format!("{chr}.l2.ldscore.gz"));
        let mut df = read_table(&path).with_context(|| format!("read {}", path.display()))?;
        if df.column("CM").is_ok() {
            df.drop_in_place("CM")?;
        }
        if df.column("MAF").is_ok() {
            df.drop_in_place("MAF")?;
        }
        frames.push(df);
    }
    concat_frames(frames).context("concat ld score frames")
}

fn read_weights(
    wld_dir: &Path,
    sep_weights: bool,
    chromosomes: &[u8],
    ld_scores: &DataFrame,
) -> Result<DataFrame> {
    let mut df = if sep_weights {
        let mut frames = Vec::new();
        for chr in chromosomes {
            let path = wld_dir.join(format!("{chr}.l2.ldscore.gz"));
            let mut df = read_table(&path).with_context(|| format!("read {}", path.display()))?;
            if df.column("CM").is_ok() {
                df.drop_in_place("CM")?;
            }
            if df.column("MAF").is_ok() {
                df.drop_in_place("MAF")?;
            }
            frames.push(df);
        }
        concat_frames(frames).context("concat weight frames")?
    } else {
        ld_scores.clone()
    };

    let last = df.get_column_names().iter().last().map(|s| s.to_string());
    if let Some(last) = last {
        df.rename(&last, "wLD".into())?;
    }
    Ok(df)
}

fn concat_frames(mut frames: Vec<DataFrame>) -> Result<DataFrame> {
    let mut iter = frames.drain(..);
    let mut base = iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("no frames to concatenate"))?;
    for frame in iter {
        base.vstack_mut(&frame)?;
    }
    Ok(base)
}

fn read_m_values(ld_dir: &Path, chromosomes: &[u8]) -> Result<f64> {
    let mut total = 0.0;
    for chr in chromosomes {
        let path = ld_dir.join(format!("{chr}.l2.M_5_50"));
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            for token in line.split(|c: char| c.is_whitespace() || c == ',') {
                if token.trim().is_empty() {
                    continue;
                }
                if let Ok(val) = token.trim().parse::<f64>() {
                    total += val;
                }
            }
            line.clear();
        }
    }
    Ok(total)
}

fn read_trait_sumstats(path: &Path, log: &mut File, idx: usize, total: usize) -> Result<DataFrame> {
    let mut df = read_table(path)?;
    let needed = ["SNP", "N", "Z", "A1"];
    for col in needed {
        if df.column(col).is_err() {
            return Err(anyhow::anyhow!(
                "Missing required column {col} in {}",
                path.display()
            ));
        }
    }
    df = df.select(needed)?;
    df = df.drop_nulls(Some(&needed))?;
    log_line(
        log,
        &format!(
            "Read in summary statistics [{idx}/{total}] from: {}",
            path.display()
        ),
        true,
    )?;
    Ok(df)
}

fn merge_with_ld_scores(
    trait_df: &DataFrame,
    weights: &DataFrame,
    ld_scores: &DataFrame,
    log: &mut File,
    path: &Path,
) -> Result<DataFrame> {
    let before = trait_df.height();
    let merged = trait_df.join(weights, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    let merged = merged.join(ld_scores, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    let merged = merged.drop_nulls(Some(&["SNP", "N", "Z", "A1", "L2", "wLD"]))?;
    log_line(
        log,
        &format!(
            "Out of {before} SNPs, {} remain after merging with LD-score files for {}",
            merged.height(),
            path.display()
        ),
        true,
    )?;

    // R merge(..., sort = FALSE) preserves input order; do not sort here to match parity.
    Ok(merged)
}

fn filter_chisq(
    mut df: DataFrame,
    chisq_max: &mut Option<f64>,
    log: &mut File,
    path: &Path,
) -> Result<DataFrame> {
    if df.column("Z").is_err() {
        return Ok(df);
    }
    let z = df.column("Z")?.as_series().context("Z")?.f64()?;
    let n = df
        .column("N")
        .ok()
        .and_then(|c| c.as_series())
        .and_then(|s| s.f64().ok());

    let max_n = n
        .as_ref()
        .map(|series| series.max().unwrap_or(0.0))
        .unwrap_or(0.0);
    if chisq_max.is_none() {
        *chisq_max = Some((0.001 * max_n).max(80.0));
    }
    let chisq_max = chisq_max.unwrap_or((0.001 * max_n).max(80.0));

    let mask: BooleanChunked = z
        .into_iter()
        .map(|v| v.map(|zv| (zv * zv) <= chisq_max))
        .collect();

    let before = df.height();
    df = df.filter(&mask)?;
    let removed = before.saturating_sub(df.height());
    log_line(
        log,
        &format!(
            "Removing {removed} SNPs with Chi^2 > {chisq_max}; {} remain for {}",
            df.height(),
            path.display()
        ),
        true,
    )?;
    Ok(df)
}

struct TraitArrays {
    l2: Vec<f64>,
    wld: Vec<f64>,
    n: Vec<f64>,
    z: Vec<f64>,
}

impl TraitArrays {
    fn len(&self) -> usize {
        self.l2.len()
    }
}

struct PairArrays {
    l2: Vec<f64>,
    wld: Vec<f64>,
    n_x: Vec<f64>,
    n_y: Vec<f64>,
    z_x: Vec<f64>,
    z_y: Vec<f64>,
    z_x_aligned: Vec<f64>,
}

impl PairArrays {
    fn len(&self) -> usize {
        self.l2.len()
    }
}

fn trait_arrays(df: &DataFrame) -> Result<TraitArrays> {
    let l2 = df.column("L2")?.as_series().context("L2")?.f64()?;
    let wld = df.column("wLD")?.as_series().context("wLD")?.f64()?;
    let n = df.column("N")?.as_series().context("N")?.f64()?;
    let z = df.column("Z")?.as_series().context("Z")?.f64()?;
    let mut out_l2 = Vec::new();
    let mut out_wld = Vec::new();
    let mut out_n = Vec::new();
    let mut out_z = Vec::new();

    for i in 0..df.height() {
        if let (Some(l2), Some(wld), Some(n), Some(z)) = (l2.get(i), wld.get(i), n.get(i), z.get(i))
            && l2.is_finite()
            && wld.is_finite()
            && n.is_finite()
            && z.is_finite()
        {
            out_l2.push(l2);
            out_wld.push(wld);
            out_n.push(n);
            out_z.push(z);
        }
    }

    Ok(TraitArrays {
        l2: out_l2,
        wld: out_wld,
        n: out_n,
        z: out_z,
    })
}

fn merge_trait_pairs(y1: &DataFrame, y2: &DataFrame) -> Result<DataFrame> {
    let mut y2_sub = y2.select(["SNP", "N", "Z", "A1"])?;
    y2_sub.rename("N", "N_Y".into())?;
    y2_sub.rename("Z", "Z_Y".into())?;
    y2_sub.rename("A1", "A1_Y".into())?;
    let merged = y1.join(&y2_sub, ["SNP"], ["SNP"], JoinType::Inner.into(), None)?;
    let merged = merged.drop_nulls(Some(&[
        "SNP", "N", "Z", "A1", "L2", "wLD", "N_Y", "Z_Y", "A1_Y",
    ]))?;
    Ok(merged)
}

fn pair_arrays(df: &DataFrame) -> Result<PairArrays> {
    let l2 = df.column("L2")?.as_series().context("L2")?.f64()?;
    let wld = df.column("wLD")?.as_series().context("wLD")?.f64()?;
    let n_x = df.column("N")?.as_series().context("N")?.f64()?;
    let z_x = df.column("Z")?.as_series().context("Z")?.f64()?;
    let a1_x = df.column("A1")?.as_series().context("A1")?.str()?;
    let n_y = df.column("N_Y")?.as_series().context("N_Y")?.f64()?;
    let z_y = df.column("Z_Y")?.as_series().context("Z_Y")?.f64()?;
    let a1_y = df.column("A1_Y")?.as_series().context("A1_Y")?.str()?;

    let mut out_l2 = Vec::new();
    let mut out_wld = Vec::new();
    let mut out_nx = Vec::new();
    let mut out_ny = Vec::new();
    let mut out_zx = Vec::new();
    let mut out_zy = Vec::new();
    let mut out_zx_aligned = Vec::new();

    for i in 0..df.height() {
        if let (Some(l2), Some(wld), Some(nx), Some(ny), Some(zx), Some(zy), Some(a1x), Some(a1y)) = (
            l2.get(i),
            wld.get(i),
            n_x.get(i),
            n_y.get(i),
            z_x.get(i),
            z_y.get(i),
            a1_x.get(i),
            a1_y.get(i),
        ) && l2.is_finite()
            && wld.is_finite()
            && nx.is_finite()
            && ny.is_finite()
        {
            out_l2.push(l2);
            out_wld.push(wld);
            out_nx.push(nx);
            out_ny.push(ny);
            out_zx.push(zx);
            out_zy.push(zy);
            let aligned = if a1x == a1y { zx } else { -zx };
            out_zx_aligned.push(aligned);
        }
    }

    Ok(PairArrays {
        l2: out_l2,
        wld: out_wld,
        n_x: out_nx,
        n_y: out_ny,
        z_x: out_zx,
        z_y: out_zy,
        z_x_aligned: out_zx_aligned,
    })
}

fn weights_for_h2(
    arrays: &TraitArrays,
    chi1: &[f64],
    m_tot: f64,
    mean_chi: f64,
    mean_l2n: f64,
) -> Result<WeightedRegressionData> {
    let mut tot_agg = (m_tot * (mean_chi - 1.0)) / mean_l2n;
    if !tot_agg.is_finite() {
        tot_agg = 0.0;
    }
    tot_agg = tot_agg.clamp(0.0, 1.0);

    let mut weights = Vec::with_capacity(arrays.len());
    let mut sum_w = 0.0;
    for i in 0..arrays.len() {
        let ld = arrays.l2[i].max(1.0);
        let wld = arrays.wld[i].max(1.0);
        let c = tot_agg * arrays.n[i] / m_tot;
        let het_w = 1.0 / (2.0 * (1.0 + c * ld).powi(2));
        let oc_w = 1.0 / wld;
        let w = (het_w * oc_w).sqrt();
        sum_w += w;
        weights.push(w);
    }
    if sum_w == 0.0 {
        return Err(anyhow::anyhow!("sum of weights is zero"));
    }
    for w in weights.iter_mut() {
        *w /= sum_w;
    }

    let weighted_ld: Vec<[f64; 2]> = arrays
        .l2
        .iter()
        .zip(&weights)
        .map(|(l2, w)| [l2 * w, *w])
        .collect();
    let weighted_chi: Vec<f64> = chi1.iter().zip(&weights).map(|(c, w)| c * w).collect();

    Ok((weights, weighted_ld, weighted_chi))
}

fn weights_for_cov(
    arrays: &PairArrays,
    chi1: &[f64],
    chi2: &[f64],
    m_tot: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mean_chi1 = mean(chi1);
    let mean_chi2 = mean(chi2);
    let mean_l2n1 = mean_product(&arrays.l2, &arrays.n_x);
    let mean_l2n2 = mean_product(&arrays.l2, &arrays.n_y);

    let mut tot_agg = (m_tot * (mean_chi1 - 1.0)) / mean_l2n1;
    if !tot_agg.is_finite() {
        tot_agg = 0.0;
    }
    tot_agg = tot_agg.clamp(0.0, 1.0);

    let mut tot_agg2 = (m_tot * (mean_chi2 - 1.0)) / mean_l2n2;
    if !tot_agg2.is_finite() {
        tot_agg2 = 0.0;
    }
    tot_agg2 = tot_agg2.clamp(0.0, 1.0);

    let mut initial_w1 = Vec::with_capacity(arrays.len());
    let mut initial_w2 = Vec::with_capacity(arrays.len());
    let mut sum1 = 0.0;

    for i in 0..arrays.len() {
        let ld = arrays.l2[i].max(1.0);
        let wld = arrays.wld[i].max(1.0);

        let c1 = tot_agg * arrays.n_x[i] / m_tot;
        let het_w1 = 1.0 / (2.0 * (1.0 + c1 * ld).powi(2));
        let oc_w1 = 1.0 / wld;
        let w1 = (het_w1 * oc_w1).sqrt();
        sum1 += w1;
        initial_w1.push(w1);

        let c2 = tot_agg2 * arrays.n_y[i] / m_tot;
        let het_w2 = 1.0 / (2.0 * (1.0 + c2 * ld).powi(2));
        let oc_w2 = 1.0 / wld;
        let w2 = (het_w2 * oc_w2).sqrt();
        initial_w2.push(w2);
    }

    let mut weights = initial_w1.clone();
    for w in weights.iter_mut() {
        *w /= sum1.max(1.0);
    }

    let mut weights_cov = Vec::with_capacity(arrays.len());
    let sum_cov: f64 = initial_w1.iter().zip(&initial_w2).map(|(a, b)| a + b).sum();
    for i in 0..arrays.len() {
        weights_cov.push((initial_w1[i] + initial_w2[i]) / sum_cov.max(1.0));
    }

    Ok((weights, weights_cov))
}

fn jackknife_regression(
    weighted_ld: &[[f64; 2]],
    weighted_chi: &[f64],
    n_blocks: usize,
    n_snps: usize,
) -> Result<([f64; 2], Vec<[f64; 2]>)> {
    let bounds = block_bounds(n_snps, n_blocks);
    let mut xty_blocks = vec![[0.0; 2]; n_blocks];
    let mut xtx_blocks = vec![[[0.0; 2]; 2]; n_blocks];

    for (b, (start, end)) in bounds.iter().enumerate() {
        for i in *start..*end {
            let ld = weighted_ld[i];
            let chi = weighted_chi[i];
            xty_blocks[b][0] += ld[0] * chi;
            xty_blocks[b][1] += ld[1] * chi;
            xtx_blocks[b][0][0] += ld[0] * ld[0];
            xtx_blocks[b][0][1] += ld[0] * ld[1];
            xtx_blocks[b][1][0] += ld[1] * ld[0];
            xtx_blocks[b][1][1] += ld[1] * ld[1];
        }
    }

    let mut xty = [0.0; 2];
    let mut xtx = [[0.0; 2]; 2];
    for b in 0..n_blocks {
        xty[0] += xty_blocks[b][0];
        xty[1] += xty_blocks[b][1];
        xtx[0][0] += xtx_blocks[b][0][0];
        xtx[0][1] += xtx_blocks[b][0][1];
        xtx[1][0] += xtx_blocks[b][1][0];
        xtx[1][1] += xtx_blocks[b][1][1];
    }

    let reg = solve_2x2(xtx, xty)?;

    let mut pseudo_values = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let xty_del = [xty[0] - xty_blocks[b][0], xty[1] - xty_blocks[b][1]];
        let xtx_del = [
            [
                xtx[0][0] - xtx_blocks[b][0][0],
                xtx[0][1] - xtx_blocks[b][0][1],
            ],
            [
                xtx[1][0] - xtx_blocks[b][1][0],
                xtx[1][1] - xtx_blocks[b][1][1],
            ],
        ];
        let del = solve_2x2(xtx_del, xty_del)?;
        let pseudo = [
            (n_blocks as f64) * reg[0] - (n_blocks as f64 - 1.0) * del[0],
            (n_blocks as f64) * reg[1] - (n_blocks as f64 - 1.0) * del[1],
        ];
        pseudo_values.push(pseudo);
    }

    Ok((reg, pseudo_values))
}

fn jackknife_se(pseudo_values: &[[f64; 2]], n_bar: f64, m_tot: f64) -> Result<(f64, f64)> {
    let n = pseudo_values.len() as f64;
    let mean0 = pseudo_values.iter().map(|v| v[0]).sum::<f64>() / n;
    let mean1 = pseudo_values.iter().map(|v| v[1]).sum::<f64>() / n;

    let mut cov00 = 0.0;
    let mut cov11 = 0.0;
    for v in pseudo_values {
        cov00 += (v[0] - mean0).powi(2);
        cov11 += (v[1] - mean1).powi(2);
    }
    let denom = (n - 1.0).max(1.0);
    cov00 = cov00 / denom / n;
    cov11 = cov11 / denom / n;

    let intercept_se = cov11.sqrt();
    let coef_cov = cov00 / (n_bar * n_bar);
    let tot_cov = coef_cov * m_tot * m_tot;
    let tot_se = tot_cov.sqrt();
    Ok((intercept_se, tot_se))
}

fn sampling_covariance(
    v_hold: &[Vec<f64>],
    n_vec: &[f64],
    n_blocks: usize,
    m_tot: f64,
) -> Result<Vec<Vec<f64>>> {
    let cov = cov_matrix(v_hold);
    let scale: Vec<f64> = n_vec
        .iter()
        .map(|v| v * (n_blocks as f64).sqrt() / m_tot)
        .collect();

    let mut out = vec![vec![0.0; n_vec.len()]; n_vec.len()];
    for i in 0..n_vec.len() {
        for j in 0..n_vec.len() {
            let denom = scale[i] * scale[j];
            out[i][j] = if denom != 0.0 {
                cov[i][j] / denom
            } else {
                f64::NAN
            };
        }
    }
    Ok(out)
}

fn cov_matrix(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_rows = rows.len();
    let n_cols = rows.first().map(|r| r.len()).unwrap_or(0);
    if n_rows == 0 || n_cols == 0 {
        return vec![vec![0.0; n_cols]; n_cols];
    }
    let mut means = vec![0.0; n_cols];
    for row in rows {
        for (i, v) in row.iter().enumerate() {
            means[i] += v;
        }
    }
    for m in means.iter_mut() {
        *m /= n_rows as f64;
    }
    let mut cov = vec![vec![0.0; n_cols]; n_cols];
    for row in rows {
        for i in 0..n_cols {
            for j in 0..n_cols {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    let denom = (n_rows as f64 - 1.0).max(1.0);
    for row in cov.iter_mut() {
        for val in row.iter_mut() {
            *val /= denom;
        }
    }
    cov
}

fn block_bounds(n_snps: usize, n_blocks: usize) -> Vec<(usize, usize)> {
    let mut bounds = Vec::with_capacity(n_blocks);
    let n = n_snps as f64;
    let blocks = n_blocks as f64;
    let mut starts = Vec::with_capacity(n_blocks + 1);
    for i in 0..=n_blocks {
        let val = 1.0 + (i as f64) * (n - 1.0) / blocks;
        starts.push(val.floor() as isize);
    }
    for i in 0..n_blocks {
        let start = (starts[i] - 1).max(0) as usize;
        let end = if i == n_blocks - 1 {
            n_snps
        } else {
            (starts[i + 1] - 1).max(0) as usize
        };
        bounds.push((start.min(n_snps), end.min(n_snps)));
    }
    bounds
}

fn solve_2x2(xtx: [[f64; 2]; 2], xty: [f64; 2]) -> Result<[f64; 2]> {
    let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
    if det == 0.0 || !det.is_finite() {
        return Err(anyhow::anyhow!("Singular matrix in regression"));
    }
    let inv = [
        [xtx[1][1] / det, -xtx[0][1] / det],
        [-xtx[1][0] / det, xtx[0][0] / det],
    ];
    let b0 = inv[0][0] * xty[0] + inv[0][1] * xty[1];
    let b1 = inv[1][0] * xty[0] + inv[1][1] * xty[1];
    Ok([b0, b1])
}

fn mean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for v in values {
        if v.is_finite() {
            sum += v;
            count += 1.0;
        }
    }
    if count == 0.0 { f64::NAN } else { sum / count }
}

fn median(values: &[f64]) -> Option<f64> {
    let mut vals: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        Some(vals[mid])
    } else {
        Some((vals[mid - 1] + vals[mid]) / 2.0)
    }
}

fn mean_product(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for (x, y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            sum += x * y;
            count += 1.0;
        }
    }
    if count == 0.0 { f64::NAN } else { sum / count }
}

fn safe_div(num: f64, den: f64) -> f64 {
    if den == 0.0 || !den.is_finite() {
        f64::NAN
    } else {
        num / den
    }
}

fn outer_sqrt(values: &[f64]) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; values.len()]; values.len()];
    for i in 0..values.len() {
        for j in 0..values.len() {
            out[i][j] = values[i].sqrt() * values[j].sqrt();
        }
    }
    out
}

fn outer_from_vec(values: &[f64]) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; values.len()]; values.len()];
    for i in 0..values.len() {
        for j in 0..values.len() {
            out[i][j] = values[i] * values[j];
        }
    }
    out
}

fn elementwise_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; a[0].len()]; a.len()];
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            out[i][j] = a[i][j] * b[i][j];
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn lower_triangle_values(matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = matrix.len();
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        for i in j..n {
            out.push(matrix[i][j]);
        }
    }
    out
}

fn scale_sampling_covariance(v_out: &[Vec<f64>], scale: &[f64]) -> Vec<Vec<f64>> {
    let n = scale.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = v_out[i][j] * scale[i] * scale[j];
        }
    }
    out
}

fn diag_positive(matrix: &[Vec<f64>]) -> bool {
    matrix
        .iter()
        .enumerate()
        .all(|(i, row)| row[i].is_finite() && row[i] > 0.0)
}

fn write_ldsc_output(output: &LdscOutput, config: &LdscConfig) -> Result<()> {
    let path = if let Some(out) = &config.output {
        out.clone()
    } else if let Some(log) = &config.ldsc_log {
        let stem = log.file_stem().and_then(|s| s.to_str()).unwrap_or("ldsc");
        PathBuf::from(format!("{stem}_ldsc.json"))
    } else {
        let mut joined = config
            .traits
            .iter()
            .filter_map(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .collect::<Vec<_>>()
            .join("_");
        if joined.len() > 200 {
            joined.truncate(100);
        }
        if joined.is_empty() {
            joined = "ldsc".to_string();
        }
        PathBuf::from(format!("{joined}_ldsc.json"))
    };

    write_ldsc_json(output, &path)
}
