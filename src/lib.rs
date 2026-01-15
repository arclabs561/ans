//! # Asymmetric Numeral Systems (ANS)
//!
//! *Near-optimal entropy coding with Huffman-like speed.*
//!
//! ## Intuition First
//!
//! Imagine you're stacking blocks of different sizes. In standard numeral systems (like base-10),
//! every block is the same size. To encode a message, you just stack them up.
//!
//! Asymmetric Numeral Systems (ANS) allows you to stack blocks of *different* sizes (representing
//! different symbol probabilities) while ensuring that the "stack" remains an integer. It
//! spreads symbol ranges uniformly across the number line, achieving the compression rate
//! of arithmetic coding but avoiding its slow bit-by-bit processing.
//!
//! ## The Problem
//!
//! Before ANS, we had a trade-off:
//! - **Huffman coding**: Fast but suboptimal (approximates probabilities as powers of 2).
//! - **Arithmetic coding**: Optimal but slow (requires many multiplications and divisions per bit).
//!
//! ## Historical Context
//!
//! ```text
//! 1948  Shannon     Entropy as the fundamental limit
//! 1952  Huffman     Huffman coding: fast, prefix-based
//! 1976  Rissanen    Arithmetic coding: optimal rate
//! 2007  Duda        ANS: the "missing link" (optimal + fast)
//! 2014  Facebook    zstd integrates ANS (tANS)
//! 2015  Apple       LZFSE integrates ANS (rANS)
//! 2019  Townsend    Bits Back with ANS (BB-ANS) for neural compression
//! 2025  Steiner     Optimal tables for ANS: discrepancy minimization
//! 2023  Lin et al.  Parallel rANS decoding with interleaving (Recoil)
//! ```
//!
//! Jarek Duda's key insight was to generalize the standard numeral system to allow
//! symbols with arbitrary probabilities $p_s$ by mapping state $x$ to $x' \approx x / p_s$.
//!
//! ## Mathematical Formulation
//!
//! Given a set of symbols $S$ with probabilities $P = \{p_s\}_{s \in S}$, ANS defines a state $x \in \mathbb{N}$.
//! The encoding function $C(x, s)$ maps the current state and a symbol to a new state:
//!
//! ```text
//! C(x, s) = \lceil (x+1)/p_s \rceil - 1
//! ```
//!
//! In practice, variants like **tANS** (table-based) and **rANS** (range-based) use
//! fixed-precision integer arithmetic to make this efficient.
//!
//! ## Complexity Analysis
//!
//! - **Time**: $O(1)$ per symbol (for tANS lookup or rANS arithmetic).
//! - **Space**: $O(2^L)$ for tANS tables, where $L$ is the state bit-length.
//!
//! ## Failure Modes
//!
//! 1. **Precision Loss**: Low-bit states can lead to suboptimal compression if probabilities are very small.
//! 2. **Table Size**: High-precision tANS requires large tables that may miss the CPU cache.
//!
//! ## Implementation Notes
//!
//! This crate provides:
//! - **tANS**: Optimized for static distributions; extremely fast table-based decoding.
//! - **rANS**: Optimized for dynamic or high-precision distributions.
//!
//! ## References
//!
//! - Duda, J. (2009). "Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding."
//! - Townsend, J., et al. (2019). "Practical Lossless Compression with Latent Variables using Bits Back Coding."

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod huffman;
pub mod rans;
pub mod tans;

pub use error::Error;
pub use huffman::{HuffmanDecoder, HuffmanEncoder};
pub use rans::{InterleavedRansDecoder, InterleavedRansEncoder, RansDecoder, RansEncoder};
pub use tans::{TansDecoder, TansEncoder};
