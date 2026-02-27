use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimation {
    Dwls,
    Ml,
}

pub type Matrix = Vec<Vec<f64>>;

#[derive(Debug, Clone)]
pub struct SemInput {
    pub s: Matrix,
    pub v: Matrix,
    pub wls_v: Option<Matrix>,
    pub model: String,
    pub model_table: Option<Vec<ParTableRow>>,
    pub estimation: Estimation,
    pub toler: Option<f64>,
    pub std_lv: bool,
    pub fix_measurement: bool,
    pub q_snp: bool,
    pub names: Vec<String>,
    pub n_obs: Option<f64>,
    pub optim_dx_tol: Option<f64>,
    pub optim_force_converged: bool,
    pub iter_max: Option<usize>,
    pub sample_cov_rescale: bool,
}

#[derive(Debug, Clone)]
pub struct SemFitStats {
    pub chisq: f64,
    pub df: i64,
    pub aic: f64,
    pub cfi: f64,
    pub srmr: f64,
    pub p_chisq: f64,
}

#[derive(Debug, Clone)]
pub struct ParTableRow {
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub free: usize,
    pub label: Option<String>,
    pub ustart: f64,
    pub est: f64,
    pub se: f64,
}

#[derive(Debug, Clone)]
pub struct ParamEstimate {
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub label: Option<String>,
    pub free: usize,
    pub est: f64,
    pub se: f64,
    pub est_std: Option<f64>,
    pub se_std: Option<f64>,
    pub est_std_all: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct DefinedEstimate {
    pub name: String,
    pub expr: String,
    pub est: f64,
    pub se: f64,
}

#[derive(Debug, Clone)]
pub struct SemFit {
    pub params: Vec<ParamEstimate>,
    pub defined: Vec<DefinedEstimate>,
    pub stats: SemFitStats,
    pub converged: bool,
    pub iterations: usize,
    pub npar: usize,
    pub par_table: Vec<ParTableRow>,
    pub delta: Matrix,
    pub wls_v: Matrix,
    pub vcov: Matrix,
    pub cor_lv: Option<Matrix>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub implied: Matrix,
    pub residual: Matrix,
    pub std_all: Option<HashMap<String, f64>>, // key: lhs~rhs/op string
}
