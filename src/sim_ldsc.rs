use std::fs::File;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use flate2::Compression;
use flate2::write::GzEncoder;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Eigh, UPLO};
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use tracing::info;

use crate::io::read_table;
use crate::matrix::{ensure_square, to_array2};
use crate::parallel::{collect_results, run_in_pool};
use crate::types::Matrix;

#[derive(Debug, Clone)]
pub enum CovInput {
    Matrix(Matrix),
    Model(String),
}

#[derive(Debug, Clone)]
pub enum NInput {
    Scalar(f64),
    Matrix(Matrix),
}

#[derive(Debug, Clone)]
pub enum MatrixOrScalar {
    Scalar(f64),
    Matrix(Matrix),
}

#[derive(Debug, Clone)]
pub enum VecOrScalar {
    Scalar(f64),
    Vec(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct SimLdscConfig {
    pub covmat: CovInput,
    pub n: NInput,
    pub seed: u64,
    pub ld: PathBuf,
    pub r_pheno: Option<MatrixOrScalar>,
    pub intercepts: Option<VecOrScalar>,
    pub n_overlap: f64,
    pub r: usize,
    pub gzip_output: bool,
    pub parallel: bool,
    pub cores: Option<usize>,
}

pub fn sim_ldsc(config: &SimLdscConfig) -> Result<()> {
    let cov = match &config.covmat {
        CovInput::Matrix(m) => m.clone(),
        CovInput::Model(_model) => {
            todo!("simLDSC model-string input requires SEM/lavaan-style simulation")
        }
    };
    ensure_square(&cov, "covmat")?;

    let k = cov.len();
    if k == 0 {
        return Err(anyhow::anyhow!("covmat must not be empty"));
    }

    let n_mat = build_n_matrix(&config.n, k, config.n_overlap)?;
    let r_g = cov2cor(&cov)?;
    let r_pheno = match &config.r_pheno {
        None => r_g.clone(),
        Some(MatrixOrScalar::Scalar(val)) => scalar_matrix(*val, k, 1.0),
        Some(MatrixOrScalar::Matrix(m)) => m.clone(),
    };

    let intercepts = match &config.intercepts {
        None => vec![1.0; k],
        Some(VecOrScalar::Scalar(val)) => vec![*val; k],
        Some(VecOrScalar::Vec(v)) => {
            if v.len() != k {
                return Err(anyhow::anyhow!(
                    "intercepts length {} does not match covmat size {}",
                    v.len(),
                    k
                ));
            }
            v.clone()
        }
    };

    let mut ld_scores = read_ld_scores(&config.ld)?;
    ld_scores = filter_mhc(ld_scores)?;
    let l2 = ld_scores
        .column("L2")
        .with_context(|| "L2 column missing in LD scores")?
        .f64()
        .context("L2 cast")?
        .clone();

    let m_tot = read_m_values(&config.ld)?;
    if m_tot == 0.0 {
        return Err(anyhow::anyhow!("M_5_50 sum is zero"));
    }

    let seeds: Vec<(usize, u64)> = (0..config.r)
        .map(|i| (i + 1, config.seed + i as u64))
        .collect();

    if config.parallel {
        let run = || {
            seeds
                .par_iter()
                .map(|(iter, seed)| {
                    simulate_iteration(
                        *iter,
                        *seed,
                        &ld_scores,
                        &l2,
                        &cov,
                        &r_pheno,
                        &intercepts,
                        &n_mat,
                        m_tot,
                        config.gzip_output,
                    )
                })
                .collect::<Vec<Result<()>>>()
        };
        let results = run_in_pool(config.cores, "build simLDSC thread pool", run)?;
        collect_results(results)?;
    } else {
        for (iter, seed) in seeds {
            simulate_iteration(
                iter,
                seed,
                &ld_scores,
                &l2,
                &cov,
                &r_pheno,
                &intercepts,
                &n_mat,
                m_tot,
                config.gzip_output,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn simulate_iteration(
    iter: usize,
    seed: u64,
    ld_scores: &DataFrame,
    l2: &Float64Chunked,
    cov: &Matrix,
    r_pheno: &Matrix,
    intercepts: &[f64],
    n_mat: &Matrix,
    m_tot: f64,
    gzip_output: bool,
) -> Result<()> {
    let k = cov.len();
    let n_snps = ld_scores.height();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut z_cols = vec![Vec::with_capacity(n_snps); k];
    for row in 0..n_snps {
        let l2_val = l2.get(row).unwrap_or(0.0);
        let sigma = build_sigma(cov, r_pheno, intercepts, n_mat, m_tot, l2_val);
        let z = mvn_sample_zero(&sigma, &mut rng)?;
        for t in 0..k {
            z_cols[t].push(z[t]);
        }
    }

    for t in 0..k {
        let mut df = ld_scores.clone();
        let z_series = Series::new("Z".into(), z_cols[t].clone());
        let n_series = Series::new("N".into(), vec![n_mat[t][t]; n_snps]);
        let a1_series = Series::new("A1".into(), vec!["A".to_string(); n_snps]);
        df.with_column(z_series.into())?;
        df.with_column(n_series.into())?;
        df.with_column(a1_series.into())?;

        let filename = format!("iter{}GWAS{}.sumstats", iter, t + 1);
        write_sumstats(&df, &filename, gzip_output)?;
        info!("Wrote {filename}");
    }

    Ok(())
}

fn build_sigma(
    cov: &Matrix,
    r_pheno: &Matrix,
    intercepts: &[f64],
    n_mat: &Matrix,
    m_tot: f64,
    l2_val: f64,
) -> Matrix {
    let k = cov.len();
    let mut sigma = vec![vec![0.0; k]; k];
    for i in 0..k {
        let n_i = n_mat[i][i];
        let var = (n_i * cov[i][i] / m_tot) * l2_val + intercepts[i];
        sigma[i][i] = var;
        for j in (i + 1)..k {
            let n_j = n_mat[j][j];
            let cov_z = (n_i * n_j).sqrt() * cov[i][j] / m_tot * l2_val
                + r_pheno[i][j] * n_mat[i][j] / (n_i * n_j).sqrt();
            sigma[i][j] = cov_z;
            sigma[j][i] = cov_z;
        }
    }
    sigma
}

fn mvn_sample_zero<R: Rng + ?Sized>(sigma: &Matrix, rng: &mut R) -> Result<Vec<f64>> {
    let k = sigma.len();
    let a = to_array2(sigma)?;
    let mut z = Array1::zeros(k);
    for i in 0..k {
        z[i] = StandardNormal.sample(rng);
    }
    let sample: Array1<f64> = match a.cholesky(UPLO::Lower) {
        Ok(chol) => chol.dot(&z),
        Err(_) => {
            let (eigvals, eigvecs) = a.eigh(UPLO::Lower).context("eigh")?;
            let sqrt_vals = eigvals.mapv(|v| {
                if v.is_finite() && v > 0.0 {
                    v.sqrt()
                } else {
                    0.0
                }
            });
            let diag = Array2::from_diag(&sqrt_vals);
            eigvecs.dot(&diag).dot(&z)
        }
    };
    Ok(sample.to_vec())
}

fn read_ld_scores(ld_dir: &Path) -> Result<DataFrame> {
    let mut frames = Vec::new();
    for chr in 1..=22u8 {
        let path = ld_dir.join(format!("{chr}.l2.ldscore.gz"));
        let df = read_table(&path).with_context(|| format!("read {}", path.display()))?;
        frames.push(df);
    }
    concat_frames(frames)
}

fn filter_mhc(mut df: DataFrame) -> Result<DataFrame> {
    if df.column("CHR").is_err() || df.column("BP").is_err() {
        return Ok(df);
    }
    let chr = df
        .column("CHR")?
        .as_series()
        .context("CHR")?
        .cast(&DataType::Int32)?;
    let bp = df
        .column("BP")?
        .as_series()
        .context("BP")?
        .cast(&DataType::Int64)?;
    let chr = chr.i32()?;
    let bp = bp.i64()?;
    // Standard MHC region (chr6: 25Mb–34Mb) as used in LDSC pipelines.
    let mask: BooleanChunked = chr
        .into_iter()
        .zip(bp)
        .map(|(c, b)| match (c, b) {
            (Some(6), Some(bp)) => !(25_000_000..=34_000_000).contains(&bp),
            _ => true,
        })
        .collect();
    df = df.filter(&mask)?;
    Ok(df)
}

fn read_m_values(ld_dir: &Path) -> Result<f64> {
    let mut total = 0.0;
    for chr in 1..=22u8 {
        let path = ld_dir.join(format!("{chr}.l2.M_5_50"));
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = std::io::BufReader::new(file);
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

fn build_n_matrix(n_input: &NInput, k: usize, n_overlap: f64) -> Result<Matrix> {
    match n_input {
        NInput::Scalar(n) => {
            if *n <= 0.0 {
                return Err(anyhow::anyhow!("N must be > 0"));
            }
            if !(0.0..=1.0).contains(&n_overlap) {
                return Err(anyhow::anyhow!("N_overlap must be in [0,1]"));
            }
            Ok(scalar_matrix(*n * n_overlap, k, *n))
        }
        NInput::Matrix(m) => {
            ensure_square(m, "N")?;
            Ok(m.clone())
        }
    }
}

fn cov2cor(cov: &Matrix) -> Result<Matrix> {
    let k = cov.len();
    let mut out = vec![vec![0.0; k]; k];
    let mut diag = vec![0.0; k];
    for i in 0..k {
        if cov[i][i] <= 0.0 {
            return Err(anyhow::anyhow!("covmat diagonal must be > 0"));
        }
        diag[i] = cov[i][i].sqrt();
    }
    for i in 0..k {
        for j in 0..k {
            out[i][j] = cov[i][j] / (diag[i] * diag[j]);
        }
    }
    Ok(out)
}

#[allow(clippy::needless_range_loop)]
fn scalar_matrix(value: f64, k: usize, diag_value: f64) -> Matrix {
    let mut out = vec![vec![value; k]; k];
    for i in 0..k {
        out[i][i] = diag_value;
    }
    out
}

// ensure_square and to_array2 are provided by matrix.rs

fn write_sumstats(df: &DataFrame, base: &str, gzip_output: bool) -> Result<()> {
    if gzip_output {
        let path = format!("{base}.gz");
        let file = File::create(&path)?;
        let encoder = GzEncoder::new(file, Compression::default());
        let mut writer = std::io::BufWriter::new(encoder);
        let mut csv = CsvWriter::new(&mut writer).with_separator(b'\t');
        let mut df = df.clone();
        csv.finish(&mut df)?;
        writer.flush()?;
    } else {
        let mut file = File::create(base)?;
        let mut csv = CsvWriter::new(&mut file).with_separator(b'\t');
        let mut df = df.clone();
        csv.finish(&mut df)?;
    }
    Ok(())
}
