#[derive(Debug, Clone, Copy)]
pub enum Estimation {
    Dwls,
    Ml,
}

#[derive(Debug, Clone, Copy)]
pub enum GenomicControl {
    Standard,
    Conserv,
    None,
}

pub type Matrix = Vec<Vec<f64>>;

#[derive(Debug, Clone)]
pub struct LdscOutput {
    pub s: Matrix,
    pub v: Matrix,
    pub i: Matrix,
    pub n: Vec<f64>,
    pub m: usize,
    pub s_stand: Option<Matrix>,
    pub v_stand: Option<Matrix>,
    pub trait_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StratifiedLdscOutput {
    pub s: Vec<Matrix>,
    pub v: Vec<Matrix>,
    pub s_tau: Vec<Matrix>,
    pub v_tau: Vec<Matrix>,
    pub i: Matrix,
    pub n: Vec<f64>,
    pub m: Vec<f64>,
    pub prop: Vec<f64>,
    pub select: Vec<i32>,
    pub annotation_names: Vec<String>,
    pub trait_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SumstatsTable {
    pub df: polars::prelude::DataFrame,
}
