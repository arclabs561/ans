//! Error types for Asymmetric Numeral Systems.

use thiserror::Error;

/// Error variants for ANS operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Provided probability is invalid (e.g., zero or non-finite).
    #[error("invalid probability: {0}")]
    InvalidProbability(f32),

    /// The ANS state has overflowed its underlying storage.
    #[error("state overflow")]
    StateOverflow,

    /// An I/O error occurred during encoding or decoding.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A specialized Result type for ANS operations.
pub type Result<T> = std::result::Result<T, Error>;
