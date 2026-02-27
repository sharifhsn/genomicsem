use thiserror::Error;

#[derive(Debug, Error)]
pub enum GenomicSemError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("missing column: {0}")]
    MissingColumn(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, GenomicSemError>;
