//! Asymmetric Numeral Systems (ANS) entropy coding primitives.
//!
//! This crate provides a small, dependency-light implementation of **rANS**
//! (range Asymmetric Numeral Systems), suitable as a building block for higher-level
//! compression schemes (e.g. “bits-back” constructions used in ROC / set coding).
//!
//! ## Design
//! - **Explicit model**: callers provide counts (frequencies) and the precision.
//! - **Batch and streaming**: [`encode`]/[`decode`] for one-shot use,
//!   [`RansEncoder`]/[`RansDecoder`] for symbol-at-a-time control.
//! - **No I/O**: this crate is pure in-memory coding.
//! - **`no_std`**: works without `std` (requires `alloc`).
//!
//! ## Batch example
//!
//! ```
//! use ans::{decode, encode, FrequencyTable};
//!
//! let counts = [10u32, 20, 70]; // A, B, C
//! let table = FrequencyTable::from_counts(&counts, 14)?;
//! let message = [0u32, 2, 1, 2, 2, 0];
//!
//! let bytes = encode(&message, &table)?;
//! let back = decode(&bytes, &table, message.len())?;
//! assert_eq!(back, message);
//!
//! # Ok::<(), ans::AnsError>(())
//! ```
//!
//! ## Streaming example
//!
//! ```
//! use ans::{RansEncoder, RansDecoder, FrequencyTable};
//!
//! let table = FrequencyTable::from_counts(&[3, 7], 12)?;
//! let message = [0u32, 1, 1, 0, 1];
//!
//! // Encode symbols in reverse order (rANS requirement).
//! let mut enc = RansEncoder::new();
//! for &sym in message.iter().rev() {
//!     enc.put(sym, &table)?;
//! }
//! let bytes = enc.finish();
//!
//! // Decode symbols in forward order.
//! let mut dec = RansDecoder::new(&bytes)?;
//! let mut decoded = Vec::new();
//! for _ in 0..message.len() {
//!     decoded.push(dec.get(&table)?);
//! }
//! assert_eq!(decoded, message);
//!
//! # Ok::<(), ans::AnsError>(())
//! ```
//!
//! ## Notes
//! - This is not tuned for maximum speed; it is meant to be correct and easy to integrate.
//! - Encoding returns a byte vector in a **stack format**: the decoder consumes bytes from
//!   the end (LIFO). This avoids reversing buffers.

#![no_std]
#![warn(missing_docs)]
extern crate alloc;

use alloc::{format, string::String, string::ToString, vec, vec::Vec};
use core::fmt;

/// Lower bound for the rANS state. Encoding emits bytes to keep the state
/// below `RANS_L << 8`; decoding pulls bytes to bring the state back above
/// `RANS_L`.
const RANS_L: u32 = 1 << 23;

/// Errors for rANS operations.
#[derive(Debug, PartialEq, Eq)]
pub enum AnsError {
    /// `precision_bits` was outside the valid range `1..=20`.
    InvalidPrecision {
        /// The invalid precision value.
        precision_bits: u32,
    },

    /// The counts slice passed to [`FrequencyTable::from_counts`] was empty.
    EmptyAlphabet,

    /// A symbol index exceeded the alphabet size during encoding.
    InvalidSymbol {
        /// The out-of-range symbol.
        symbol: u32,
        /// The alphabet size of the table.
        alphabet_size: usize,
    },

    /// A symbol with zero frequency was encountered during encoding.
    ZeroFrequency {
        /// The zero-frequency symbol.
        symbol: u32,
    },

    /// The frequency normalization step could not produce a valid table.
    InvalidTable(String),

    /// The rANS state read from the byte stream was below `RANS_L`.
    InvalidState {
        /// The invalid state value.
        state: u32,
        /// The minimum valid state (`RANS_L`).
        min_state: u32,
    },

    /// The byte stream was shorter than expected during decoding.
    TruncatedInput {
        /// Bytes available.
        available: usize,
        /// Minimum bytes needed.
        needed: usize,
    },
}

impl fmt::Display for AnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrecision { precision_bits } => {
                write!(
                    f,
                    "invalid precision_bits={precision_bits} (must be in 1..=20)"
                )
            }
            Self::EmptyAlphabet => f.write_str("empty frequency table"),
            Self::InvalidSymbol {
                symbol,
                alphabet_size,
            } => write!(
                f,
                "invalid symbol {symbol} for alphabet size {alphabet_size}"
            ),
            Self::ZeroFrequency { symbol } => {
                write!(f, "frequency for symbol {symbol} is zero")
            }
            Self::InvalidTable(msg) => write!(f, "frequency table normalization failed: {msg}"),
            Self::InvalidState { state, min_state } => {
                write!(f, "invalid rANS state {state} (expected >= {min_state})")
            }
            Self::TruncatedInput { available, needed } => {
                write!(
                    f,
                    "truncated input ({available} bytes available, need at least {needed})"
                )
            }
        }
    }
}

impl core::error::Error for AnsError {}

/// A frequency model for rANS with total \(T = 2^{precision\_bits}\).
#[derive(Debug, Clone)]
pub struct FrequencyTable {
    precision_bits: u32,
    total: u32,
    freqs: Vec<u32>,
    cdf: Vec<u32>, // inclusive prefix sums: len = freqs.len() + 1, cdf[0]=0, cdf[last]=total
    sym_by_slot: Vec<u32>, // length=total, maps slot -> symbol
}

impl FrequencyTable {
    /// Build a normalized frequency table from raw counts.
    ///
    /// `precision_bits` sets \(T = 2^{precision\_bits}\), the total frequency mass.
    /// Counts are scaled to sum to \(T\), with a minimal correction pass to preserve
    /// nonzero symbols where possible.
    pub fn from_counts(counts: &[u32], precision_bits: u32) -> Result<Self, AnsError> {
        if !(1..=20).contains(&precision_bits) {
            return Err(AnsError::InvalidPrecision { precision_bits });
        }
        if counts.is_empty() {
            return Err(AnsError::EmptyAlphabet);
        }

        let total = 1u32 << precision_bits;
        let sum: u64 = counts.iter().map(|&c| c as u64).sum();
        if sum == 0 {
            return Err(AnsError::InvalidTable("all counts are zero".to_string()));
        }

        // Initial scaling.
        let mut freqs = vec![0u32; counts.len()];
        for (i, &c) in counts.iter().enumerate() {
            // floor(count * total / sum)
            let f = ((c as u128) * (total as u128) / (sum as u128)) as u32;
            freqs[i] = f;
        }

        // Ensure that any symbol with nonzero count gets at least 1 if possible.
        for (i, &c) in counts.iter().enumerate() {
            if c > 0 && freqs[i] == 0 {
                freqs[i] = 1;
            }
        }

        // Fix sum to exactly total by adjusting the largest frequencies.
        let mut cur_sum: i64 = freqs.iter().map(|&f| f as i64).sum();
        let target: i64 = total as i64;
        // Safety: sum > 0 was checked above, so at least one symbol has count > 0
        // and therefore freq >= 1 after the zero-floor correction.
        debug_assert!(
            cur_sum > 0,
            "cur_sum should be > 0 after zero-floor correction"
        );

        // We adjust greedily; correctness > optimality.
        while cur_sum != target {
            if cur_sum < target {
                // add one to the max-count symbol
                // unwrap: counts is non-empty (checked at function entry)
                let (idx, _) = counts.iter().enumerate().max_by_key(|&(_, &c)| c).unwrap();
                freqs[idx] += 1;
                cur_sum += 1;
            } else {
                // subtract one from a symbol with freq > 1 (prefer largest freq)
                let mut best: Option<(usize, u32)> = None;
                for (i, &f) in freqs.iter().enumerate() {
                    if f > 1 && best.map(|(_, bf)| f > bf).unwrap_or(true) {
                        best = Some((i, f));
                    }
                }
                let Some((idx, _)) = best else {
                    return Err(AnsError::InvalidTable(format!(
                        "cannot reduce total (cur_sum={cur_sum}, target={target}): \
                         all {} symbols have freq<=1",
                        freqs.len()
                    )));
                };
                freqs[idx] -= 1;
                cur_sum -= 1;
            }
        }

        // Build CDF and slot -> symbol lookup.
        let mut cdf = vec![0u32; freqs.len() + 1];
        for i in 0..freqs.len() {
            cdf[i + 1] = cdf[i] + freqs[i];
        }
        // The correction loop guarantees cur_sum == target, so the CDF must match.
        debug_assert_eq!(
            cdf.last().copied().unwrap_or(0),
            total,
            "cdf total mismatch after correction loop"
        );

        let mut sym_by_slot = vec![0u32; total as usize];
        for sym in 0..freqs.len() {
            let start = cdf[sym] as usize;
            let end = cdf[sym + 1] as usize;
            for slot in sym_by_slot.iter_mut().take(end).skip(start) {
                *slot = sym as u32;
            }
        }

        Ok(Self {
            precision_bits,
            total,
            freqs,
            cdf,
            sym_by_slot,
        })
    }

    /// Build a table from already-normalized frequencies that sum to `2^precision_bits`.
    ///
    /// Unlike [`from_counts`](Self::from_counts), this skips the scaling/normalization
    /// step. The caller is responsible for ensuring the frequencies sum to exactly
    /// `2^precision_bits` and that every entry is non-negative.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `precision_bits` is outside `1..=20`
    /// - `freqs` is empty
    /// - the frequencies do not sum to `2^precision_bits`
    pub fn from_normalized(freqs: &[u32], precision_bits: u32) -> Result<Self, AnsError> {
        if !(1..=20).contains(&precision_bits) {
            return Err(AnsError::InvalidPrecision { precision_bits });
        }
        if freqs.is_empty() {
            return Err(AnsError::EmptyAlphabet);
        }
        let total = 1u32 << precision_bits;
        let sum: u32 = freqs.iter().sum();
        if sum != total {
            return Err(AnsError::InvalidTable(format!(
                "frequencies sum to {sum}, expected {total}"
            )));
        }

        let freqs = freqs.to_vec();
        let mut cdf = vec![0u32; freqs.len() + 1];
        for i in 0..freqs.len() {
            cdf[i + 1] = cdf[i] + freqs[i];
        }

        let mut sym_by_slot = vec![0u32; total as usize];
        for sym in 0..freqs.len() {
            let start = cdf[sym] as usize;
            let end = cdf[sym + 1] as usize;
            for slot in sym_by_slot.iter_mut().take(end).skip(start) {
                *slot = sym as u32;
            }
        }

        Ok(Self {
            precision_bits,
            total,
            freqs,
            cdf,
            sym_by_slot,
        })
    }

    /// Build a table from floating-point probabilities.
    ///
    /// Each entry in `probs` should be non-negative. The values are normalized
    /// to sum to 1, then quantized to integer frequencies summing to
    /// `2^precision_bits`. Every symbol with a positive probability is guaranteed
    /// at least frequency 1.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `precision_bits` is outside `1..=20`
    /// - `probs` is empty
    /// - all probabilities are zero or negative
    pub fn from_float_probs(probs: &[f32], precision_bits: u32) -> Result<Self, AnsError> {
        if !(1..=20).contains(&precision_bits) {
            return Err(AnsError::InvalidPrecision { precision_bits });
        }
        if probs.is_empty() {
            return Err(AnsError::EmptyAlphabet);
        }

        let sum: f64 = probs.iter().map(|&p| (p.max(0.0)) as f64).sum();
        if sum == 0.0 {
            return Err(AnsError::InvalidTable(
                "all probabilities are zero or negative".to_string(),
            ));
        }

        // Convert to integer counts scaled to a large range, then delegate
        // to from_counts which handles the normalization and correction.
        let scale = (1u64 << 30) as f64;
        let counts: Vec<u32> = probs
            .iter()
            .map(|&p| {
                let p = (p.max(0.0)) as f64 / sum;
                // Use (x + 0.5) as u32 instead of round() for no_std compatibility.
                let v = p * scale;
                (if v > 0.0 { v + 0.5 } else { 0.0 }) as u32
            })
            .collect();

        Self::from_counts(&counts, precision_bits)
    }

    /// The number of precision bits, i.e. `log2(total)`.
    #[inline]
    #[must_use]
    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }

    /// Total frequency mass: `2^precision_bits`.
    #[inline]
    #[must_use]
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Number of symbols in the alphabet.
    #[inline]
    #[must_use]
    pub fn alphabet_size(&self) -> usize {
        self.freqs.len()
    }

    /// Normalized frequency for `sym`, or `None` if out of range.
    #[inline]
    #[must_use]
    pub fn freq(&self, sym: u32) -> Option<u32> {
        self.freqs.get(sym as usize).copied()
    }

    /// Cumulative frequency (CDF value) for `sym`, or `None` if out of range.
    #[inline]
    #[must_use]
    pub fn cum_freq(&self, sym: u32) -> Option<u32> {
        self.cdf.get(sym as usize).copied()
    }

    /// Look up the symbol that owns a cumulative-frequency slot.
    ///
    /// `slot` must be in `0..total`. Returns `None` if out of range.
    /// This is the inverse of the CDF: given a slot in `[0, T)`, it returns
    /// the symbol whose frequency interval contains that slot.
    #[inline]
    #[must_use]
    pub fn symbol_at_slot(&self, slot: u32) -> Option<u32> {
        self.sym_by_slot.get(slot as usize).copied()
    }

    /// The normalized frequency for every symbol as a slice (length = alphabet size).
    #[inline]
    #[must_use]
    pub fn freqs(&self) -> &[u32] {
        &self.freqs
    }

    /// The cumulative distribution function as a slice (length = alphabet size + 1).
    ///
    /// `cdf[0]` is always 0 and `cdf[alphabet_size]` equals [`total()`](Self::total).
    /// The frequency interval for symbol `s` is `cdf[s]..cdf[s+1]`.
    #[inline]
    #[must_use]
    pub fn cdf(&self) -> &[u32] {
        &self.cdf
    }
}

// ---------------------------------------------------------------------------
// Streaming encoder
// ---------------------------------------------------------------------------

/// A symbol-at-a-time rANS encoder.
///
/// Feed symbols with [`put`](RansEncoder::put) (in **reverse** message order -- this is
/// inherent to rANS), then call [`finish`](RansEncoder::finish) to obtain the byte stream.
///
/// The same byte stream is decodable by [`RansDecoder`] or [`decode`].
#[derive(Debug, Clone)]
pub struct RansEncoder {
    state: u32,
    buf: Vec<u8>,
}

impl RansEncoder {
    /// Create a new encoder with default buffer capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: RANS_L,
            buf: Vec::new(),
        }
    }

    /// Create a new encoder with pre-allocated buffer capacity.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            state: RANS_L,
            buf: Vec::with_capacity(cap),
        }
    }

    /// Encode a single symbol into the rANS state.
    ///
    /// **Symbols must be fed in reverse message order.** If the original message is
    /// `[A, B, C]`, call `put(C)`, `put(B)`, `put(A)`.
    pub fn put(&mut self, sym: u32, table: &FrequencyTable) -> Result<(), AnsError> {
        let sym_us = sym as usize;
        if sym_us >= table.freqs.len() {
            return Err(AnsError::InvalidSymbol {
                symbol: sym,
                alphabet_size: table.freqs.len(),
            });
        }
        let freq = table.freqs[sym_us];
        if freq == 0 {
            return Err(AnsError::ZeroFrequency { symbol: sym });
        }
        let start = table.cdf[sym_us];

        // Renormalize: emit bytes to keep state small enough.
        let x_max = ((RANS_L >> table.precision_bits) << 8) * freq;
        while self.state >= x_max {
            self.buf.push((self.state & 0xFF) as u8);
            self.state >>= 8;
        }

        let q = self.state / freq;
        let r = self.state - q * freq;
        self.state = (q << table.precision_bits) + r + start;
        Ok(())
    }

    /// Finalize the encoder, writing the final state and returning the byte stream.
    ///
    /// The returned bytes can be decoded by [`RansDecoder::new`] or [`decode`].
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.buf.extend_from_slice(&self.state.to_le_bytes());
        self.buf
    }

    /// Current rANS state (for bits-back interleaving).
    #[inline]
    #[must_use]
    pub fn state(&self) -> u32 {
        self.state
    }
}

impl Default for RansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Streaming decoder
// ---------------------------------------------------------------------------

/// A symbol-at-a-time rANS decoder.
///
/// Constructed from a byte stream produced by [`RansEncoder::finish`] or [`encode`].
/// Decode symbols with [`get`](RansDecoder::get), or use the bits-back primitives
/// [`peek`](RansDecoder::peek) + [`advance`](RansDecoder::advance).
#[derive(Debug)]
pub struct RansDecoder<'a> {
    state: u32,
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> RansDecoder<'a> {
    /// Initialize a decoder from an encoded byte stream.
    ///
    /// Returns an error if the stream is too short (< 4 bytes) or the
    /// initial state is below `RANS_L`.
    pub fn new(bytes: &'a [u8]) -> Result<Self, AnsError> {
        if bytes.len() < 4 {
            return Err(AnsError::TruncatedInput {
                available: bytes.len(),
                needed: 4,
            });
        }
        let cursor = bytes.len() - 4;
        // unwrap: length check above guarantees exactly 4 bytes available
        let state_bytes: [u8; 4] = bytes[cursor..cursor + 4].try_into().unwrap();
        let state = u32::from_le_bytes(state_bytes);
        if state < RANS_L {
            return Err(AnsError::InvalidState {
                state,
                min_state: RANS_L,
            });
        }
        Ok(Self {
            state,
            bytes,
            cursor,
        })
    }

    /// Decode a single symbol from the rANS state.
    ///
    /// Equivalent to calling [`peek`](RansDecoder::peek) followed by
    /// [`advance`](RansDecoder::advance).
    pub fn get(&mut self, table: &FrequencyTable) -> Result<u32, AnsError> {
        let sym = self.peek(table);
        self.advance(sym, table)?;
        Ok(sym)
    }

    /// Peek at the next symbol without advancing the state.
    ///
    /// Returns the symbol whose frequency interval contains the current slot.
    /// Use with [`advance`](RansDecoder::advance) for bits-back coding, where the
    /// caller needs the slot value before deciding how to advance.
    ///
    /// # Validity
    ///
    /// The returned symbol is guaranteed to be in-range only when the decoder
    /// state is valid, i.e. the decoder was just constructed via [`new`](Self::new)
    /// or the most recent [`advance`](Self::advance) call succeeded. If `advance`
    /// returned an error (truncated input), the state may be partially updated
    /// and `peek` results are undefined.
    #[inline]
    #[must_use]
    pub fn peek(&self, table: &FrequencyTable) -> u32 {
        debug_assert!(
            self.state >= RANS_L,
            "peek called with invalid state {} (expected >= {RANS_L})",
            self.state
        );
        let slot = (self.state & (table.total - 1)) as usize;
        debug_assert!(slot < table.sym_by_slot.len(), "slot {slot} out of range");
        table.sym_by_slot[slot]
    }

    /// Advance the decoder state after a [`peek`](RansDecoder::peek).
    ///
    /// `sym` must be the symbol returned by `peek` (or a valid symbol whose
    /// frequency interval contains the current slot). Passing the wrong symbol
    /// will silently corrupt the state.
    pub fn advance(&mut self, sym: u32, table: &FrequencyTable) -> Result<(), AnsError> {
        let mask = table.total - 1;
        let slot = self.state & mask;
        let sym_us = sym as usize;
        if sym_us >= table.freqs.len() {
            return Err(AnsError::InvalidSymbol {
                symbol: sym,
                alphabet_size: table.freqs.len(),
            });
        }
        let freq = table.freqs[sym_us];
        let start = table.cdf[sym_us];

        // Arithmetic safety: freq <= total <= 2^20, and (state >> precision_bits)
        // < 2^(32 - precision_bits) <= 2^31, so freq * (state >> precision_bits)
        // fits in u32 (max ~2^20 * 2^11 = 2^31 before renorm pushes state down).
        // (slot - start) < freq <= total, so the addition cannot overflow.
        self.state = freq * (self.state >> table.precision_bits) + (slot - start);

        // Renormalize: pull bytes while state < RANS_L.
        while self.state < RANS_L {
            if self.cursor == 0 {
                return Err(AnsError::TruncatedInput {
                    available: 0,
                    needed: 1,
                });
            }
            self.cursor -= 1;
            self.state = (self.state << 8) | (self.bytes[self.cursor] as u32);
        }
        Ok(())
    }

    /// Current rANS state (for bits-back interleaving).
    #[inline]
    #[must_use]
    pub fn state(&self) -> u32 {
        self.state
    }

    /// Number of unread bytes remaining in the buffer (excluding the initial state).
    #[inline]
    #[must_use]
    pub fn remaining_bytes(&self) -> usize {
        self.cursor
    }
}

// ---------------------------------------------------------------------------
// Batch API (wrappers around streaming types)
// ---------------------------------------------------------------------------

/// Encode `symbols` into a byte stream using rANS with the given frequency `table`.
///
/// The output byte vector is in **stack format**: the final 4 bytes hold the rANS
/// state, and any earlier bytes were emitted during renormalization. The decoder
/// consumes these bytes LIFO (from the end toward the front).
///
/// # Example
///
/// ```
/// use ans::{encode, decode, FrequencyTable};
///
/// let table = FrequencyTable::from_counts(&[3, 7], 12)?;
/// let message = [0u32, 1, 1, 0];
/// let bytes = encode(&message, &table)?;
/// let recovered = decode(&bytes, &table, message.len())?;
/// assert_eq!(recovered, message);
/// # Ok::<(), ans::AnsError>(())
/// ```
pub fn encode(symbols: &[u32], table: &FrequencyTable) -> Result<Vec<u8>, AnsError> {
    let mut enc = RansEncoder::with_capacity(symbols.len());
    for &sym in symbols.iter().rev() {
        enc.put(sym, table)?;
    }
    Ok(enc.finish())
}

/// Decode `len` symbols from an rANS byte stream produced by [`encode`].
///
/// The caller must know the message length because rANS is a stream codec with no
/// built-in end-of-message marker. Passing `len = 0` is valid and returns an empty
/// vector (the 4-byte state is still read and validated).
///
/// # Example
///
/// ```
/// use ans::{encode, decode, FrequencyTable};
///
/// let table = FrequencyTable::from_counts(&[5, 3, 2], 14)?;
/// let message = [2u32, 0, 1, 0, 2];
/// let bytes = encode(&message, &table)?;
/// let recovered = decode(&bytes, &table, message.len())?;
/// assert_eq!(recovered, message);
/// # Ok::<(), ans::AnsError>(())
/// ```
pub fn decode(bytes: &[u8], table: &FrequencyTable, len: usize) -> Result<Vec<u32>, AnsError> {
    let mut dec = RansDecoder::new(bytes)?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(dec.get(table)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 64-bit rANS (emits u32 words instead of bytes, higher precision)
// ---------------------------------------------------------------------------

/// Lower bound for the 64-bit rANS state. Renormalization emits 32-bit words
/// to keep the state in `[RANS64_L, RANS64_L << 32)`.
const RANS64_L: u64 = 1 << 31;

/// A symbol-at-a-time 64-bit rANS encoder.
///
/// Same interface as [`RansEncoder`] but uses a 64-bit state and emits
/// 32-bit words during renormalization. Supports `precision_bits` up to 31,
/// giving finer frequency resolution than the 32-bit variant.
///
/// The output format is **not** compatible with [`RansDecoder`]; use
/// [`Rans64Decoder`] to decode.
#[derive(Debug, Clone)]
pub struct Rans64Encoder {
    state: u64,
    buf: Vec<u8>,
}

impl Rans64Encoder {
    /// Create a new 64-bit encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: RANS64_L,
            buf: Vec::new(),
        }
    }

    /// Create a new 64-bit encoder with pre-allocated buffer capacity.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            state: RANS64_L,
            buf: Vec::with_capacity(cap),
        }
    }

    /// Encode a single symbol into the 64-bit rANS state.
    ///
    /// **Symbols must be fed in reverse message order.**
    pub fn put(&mut self, sym: u32, table: &FrequencyTable) -> Result<(), AnsError> {
        let sym_us = sym as usize;
        if sym_us >= table.freqs.len() {
            return Err(AnsError::InvalidSymbol {
                symbol: sym,
                alphabet_size: table.freqs.len(),
            });
        }
        let freq = table.freqs[sym_us] as u64;
        if freq == 0 {
            return Err(AnsError::ZeroFrequency { symbol: sym });
        }
        let start = table.cdf[sym_us] as u64;

        // Renormalize: emit a u32 word to keep state small enough.
        let x_max = ((RANS64_L >> table.precision_bits) << 32) * freq;
        while self.state >= x_max {
            self.buf
                .extend_from_slice(&(self.state as u32).to_le_bytes());
            self.state >>= 32;
        }

        let q = self.state / freq;
        let r = self.state - q * freq;
        self.state = (q << table.precision_bits) + r + start;
        Ok(())
    }

    /// Finalize the encoder, writing the final state and returning the byte stream.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.buf.extend_from_slice(&self.state.to_le_bytes());
        self.buf
    }

    /// Current rANS state (for bits-back interleaving).
    #[inline]
    #[must_use]
    pub fn state(&self) -> u64 {
        self.state
    }
}

impl Default for Rans64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A symbol-at-a-time 64-bit rANS decoder.
///
/// Constructed from a byte stream produced by [`Rans64Encoder::finish`] or
/// [`encode64`]. Decode symbols with [`get`](Rans64Decoder::get), or use the
/// bits-back primitives [`peek`](Rans64Decoder::peek) +
/// [`advance`](Rans64Decoder::advance).
#[derive(Debug)]
pub struct Rans64Decoder<'a> {
    state: u64,
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Rans64Decoder<'a> {
    /// Initialize a 64-bit decoder from an encoded byte stream.
    ///
    /// Returns an error if the stream is too short (< 8 bytes) or the
    /// initial state is below `RANS64_L`.
    pub fn new(bytes: &'a [u8]) -> Result<Self, AnsError> {
        if bytes.len() < 8 {
            return Err(AnsError::TruncatedInput {
                available: bytes.len(),
                needed: 8,
            });
        }
        let cursor = bytes.len() - 8;
        // unwrap: length check above guarantees exactly 8 bytes available
        let state_bytes: [u8; 8] = bytes[cursor..cursor + 8].try_into().unwrap();
        let state = u64::from_le_bytes(state_bytes);
        if state < RANS64_L {
            return Err(AnsError::InvalidState {
                state: state as u32,
                min_state: RANS64_L as u32,
            });
        }
        Ok(Self {
            state,
            bytes,
            cursor,
        })
    }

    /// Decode a single symbol from the 64-bit rANS state.
    pub fn get(&mut self, table: &FrequencyTable) -> Result<u32, AnsError> {
        let sym = self.peek(table);
        self.advance(sym, table)?;
        Ok(sym)
    }

    /// Peek at the next symbol without advancing the state.
    #[inline]
    #[must_use]
    pub fn peek(&self, table: &FrequencyTable) -> u32 {
        debug_assert!(
            self.state >= RANS64_L,
            "peek called with invalid state {} (expected >= {RANS64_L})",
            self.state
        );
        let slot = (self.state & (table.total as u64 - 1)) as usize;
        debug_assert!(slot < table.sym_by_slot.len(), "slot {slot} out of range");
        table.sym_by_slot[slot]
    }

    /// Advance the decoder state after a [`peek`](Rans64Decoder::peek).
    pub fn advance(&mut self, sym: u32, table: &FrequencyTable) -> Result<(), AnsError> {
        let mask = table.total as u64 - 1;
        let slot = self.state & mask;
        let sym_us = sym as usize;
        if sym_us >= table.freqs.len() {
            return Err(AnsError::InvalidSymbol {
                symbol: sym,
                alphabet_size: table.freqs.len(),
            });
        }
        let freq = table.freqs[sym_us] as u64;
        let start = table.cdf[sym_us] as u64;

        self.state = freq * (self.state >> table.precision_bits) + (slot - start);

        // Renormalize: pull u32 words while state < RANS64_L.
        while self.state < RANS64_L {
            if self.cursor < 4 {
                return Err(AnsError::TruncatedInput {
                    available: self.cursor,
                    needed: 4,
                });
            }
            self.cursor -= 4;
            let word =
                u32::from_le_bytes(self.bytes[self.cursor..self.cursor + 4].try_into().unwrap());
            self.state = (self.state << 32) | (word as u64);
        }
        Ok(())
    }

    /// Current rANS state (for bits-back interleaving).
    #[inline]
    #[must_use]
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Number of unread bytes remaining in the buffer.
    #[inline]
    #[must_use]
    pub fn remaining_bytes(&self) -> usize {
        self.cursor
    }
}

/// Encode `symbols` using 64-bit rANS with the given frequency `table`.
///
/// # Example
///
/// ```
/// use ans::{encode64, decode64, FrequencyTable};
///
/// let table = FrequencyTable::from_counts(&[3, 7], 12)?;
/// let message = [0u32, 1, 1, 0];
/// let bytes = encode64(&message, &table)?;
/// let recovered = decode64(&bytes, &table, message.len())?;
/// assert_eq!(recovered, message);
/// # Ok::<(), ans::AnsError>(())
/// ```
pub fn encode64(symbols: &[u32], table: &FrequencyTable) -> Result<Vec<u8>, AnsError> {
    let mut enc = Rans64Encoder::with_capacity(symbols.len());
    for &sym in symbols.iter().rev() {
        enc.put(sym, table)?;
    }
    Ok(enc.finish())
}

/// Decode `len` symbols from a 64-bit rANS byte stream produced by [`encode64`].
pub fn decode64(bytes: &[u8], table: &FrequencyTable, len: usize) -> Result<Vec<u32>, AnsError> {
    let mut dec = Rans64Decoder::new(bytes)?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(dec.get(table)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn smoke_roundtrip_small_alphabet() {
        let counts = [1u32, 2, 3, 4];
        let table = FrequencyTable::from_counts(&counts, 12).unwrap();
        let symbols = vec![0u32, 1, 2, 3, 2, 2, 1, 0, 3];
        let enc = encode(&symbols, &table).unwrap();
        let dec = decode(&enc, &table, symbols.len()).unwrap();
        assert_eq!(symbols, dec);
    }

    #[test]
    fn decode_rejects_invalid_final_state() {
        let counts = [1u32, 2, 3, 4];
        let table = FrequencyTable::from_counts(&counts, 12).unwrap();
        let symbols = vec![0u32, 1, 2, 3, 2, 2, 1, 0, 3];
        let mut enc = encode(&symbols, &table).unwrap();

        // Corrupt the final state bytes to force a too-small state.
        let n = enc.len();
        enc[n - 4..n].copy_from_slice(&0u32.to_le_bytes());

        let err = decode(&enc, &table, symbols.len()).unwrap_err();
        assert!(matches!(err, AnsError::InvalidState { .. }));
    }

    #[test]
    fn roundtrip_single_symbol_alphabet() {
        let counts = [42u32];
        let table = FrequencyTable::from_counts(&counts, 10).unwrap();
        let symbols = vec![0u32; 50];
        let enc = encode(&symbols, &table).unwrap();
        let dec = decode(&enc, &table, symbols.len()).unwrap();
        assert_eq!(symbols, dec);
    }

    #[test]
    fn roundtrip_empty_message() {
        let counts = [5u32, 3, 2];
        let table = FrequencyTable::from_counts(&counts, 12).unwrap();
        let symbols: Vec<u32> = vec![];
        let enc = encode(&symbols, &table).unwrap();
        let dec = decode(&enc, &table, 0).unwrap();
        assert_eq!(symbols, dec);
    }

    #[test]
    fn roundtrip_precision_boundaries() {
        for prec in [1, 2, 19, 20] {
            let counts = [3u32, 7];
            let table = FrequencyTable::from_counts(&counts, prec).unwrap();
            let symbols = vec![0u32, 1, 1, 0, 1];
            let enc = encode(&symbols, &table).unwrap();
            let dec = decode(&enc, &table, symbols.len()).unwrap();
            assert_eq!(symbols, dec, "failed at precision_bits={prec}");
        }
    }

    #[test]
    fn errors_on_invalid_precision() {
        assert!(FrequencyTable::from_counts(&[1], 0).is_err());
        assert!(FrequencyTable::from_counts(&[1], 21).is_err());
    }

    #[test]
    fn errors_on_empty_counts() {
        assert!(FrequencyTable::from_counts(&[], 12).is_err());
    }

    #[test]
    fn errors_on_all_zero_counts() {
        assert!(FrequencyTable::from_counts(&[0, 0, 0], 12).is_err());
    }

    #[test]
    fn errors_when_precision_too_small_for_alphabet() {
        // precision_bits=1 -> total=2, but 3 nonzero symbols each need freq>=1.
        // The correction loop can't reduce cur_sum=3 to target=2 because
        // all symbols have freq=1 (none > 1 to decrement).
        let err = FrequencyTable::from_counts(&[1, 1, 1], 1).unwrap_err();
        assert!(matches!(err, AnsError::InvalidTable(_)));
    }

    proptest! {
        #[test]
        fn prop_rans_roundtrip(
            symbols in prop::collection::vec(0u32..256u32, 0..200),
            counts in prop::collection::vec(1u32..100u32, 1..16),
        ) {
            let alphabet = counts.len().max(1);
            // Ensure precision_bits is large enough for the alphabet.
            let min_bits = (alphabet as f64).log2().ceil().max(1.0) as u32;
            let precision_bits = min_bits.clamp(1, 20);
            let table = FrequencyTable::from_counts(&counts, precision_bits).unwrap();
            let symbols: Vec<u32> = symbols.into_iter().map(|s| s % (alphabet as u32)).collect();
            let enc = encode(&symbols, &table)?;
            let dec = decode(&enc, &table, symbols.len())?;
            prop_assert_eq!(symbols, dec);
        }
    }

    // --- Streaming API tests ---

    #[test]
    fn streaming_roundtrip() {
        let counts = [1u32, 2, 3, 4];
        let table = FrequencyTable::from_counts(&counts, 12).unwrap();
        let message = vec![0u32, 1, 2, 3, 2, 2, 1, 0, 3];

        let mut enc = RansEncoder::new();
        for &sym in message.iter().rev() {
            enc.put(sym, &table).unwrap();
        }
        let bytes = enc.finish();

        let mut dec = RansDecoder::new(&bytes).unwrap();
        let mut decoded = Vec::new();
        for _ in 0..message.len() {
            decoded.push(dec.get(&table).unwrap());
        }
        assert_eq!(message, decoded);
    }

    proptest! {
        #[test]
        fn prop_streaming_matches_batch(
            precision_bits in 1u32..21,
            symbols in prop::collection::vec(0u32..256u32, 0..200),
            counts in prop::collection::vec(1u32..100u32, 1..32),
        ) {
            let alphabet = counts.len().max(1);
            let table = match FrequencyTable::from_counts(&counts, precision_bits) {
                Ok(t) => t,
                Err(_) => { return Ok(()); }
            };
            let symbols: Vec<u32> = symbols.into_iter().map(|s| s % (alphabet as u32)).collect();

            // Batch encode
            let batch_bytes = encode(&symbols, &table)?;

            // Streaming encode (same reverse order)
            let mut enc = RansEncoder::with_capacity(symbols.len());
            for &sym in symbols.iter().rev() {
                enc.put(sym, &table)?;
            }
            let stream_bytes = enc.finish();

            // Must be bitwise identical.
            prop_assert_eq!(&batch_bytes, &stream_bytes);
        }
    }

    #[test]
    fn peek_and_advance() {
        let counts = [3u32, 7];
        let table = FrequencyTable::from_counts(&counts, 12).unwrap();
        let message = vec![0u32, 1, 1, 0, 1];

        let bytes = encode(&message, &table).unwrap();
        let mut dec = RansDecoder::new(&bytes).unwrap();

        for &expected in &message {
            let sym = dec.peek(&table);
            assert_eq!(sym, expected);
            // Verify symbol_at_slot agrees
            let slot = dec.state() & (table.total() - 1);
            assert_eq!(table.symbol_at_slot(slot), Some(sym));
            dec.advance(sym, &table).unwrap();
        }
    }

    #[test]
    fn streaming_empty_message() {
        let table = FrequencyTable::from_counts(&[5, 3, 2], 12).unwrap();

        let enc = RansEncoder::new();
        let bytes = enc.finish();
        // Should be exactly 4 bytes (the initial state).
        assert_eq!(bytes.len(), 4);

        let dec = RansDecoder::new(&bytes).unwrap();
        // Decoding 0 symbols should succeed.
        assert_eq!(dec.remaining_bytes(), 0);
        // State should be RANS_L.
        assert_eq!(dec.state(), RANS_L);
        let _ = &table; // silence unused
    }

    #[test]
    fn streaming_single_symbol() {
        let counts = [42u32];
        let table = FrequencyTable::from_counts(&counts, 10).unwrap();
        let message = vec![0u32; 50];

        let mut enc = RansEncoder::new();
        for &sym in message.iter().rev() {
            enc.put(sym, &table).unwrap();
        }
        let bytes = enc.finish();

        let mut dec = RansDecoder::new(&bytes).unwrap();
        let mut decoded = Vec::new();
        for _ in 0..message.len() {
            decoded.push(dec.get(&table).unwrap());
        }
        assert_eq!(message, decoded);
    }

    #[test]
    fn streaming_precision_boundaries() {
        for prec in [1, 2, 19, 20] {
            let counts = [3u32, 7];
            let table = FrequencyTable::from_counts(&counts, prec).unwrap();
            let message = vec![0u32, 1, 1, 0, 1];

            let mut enc = RansEncoder::new();
            for &sym in message.iter().rev() {
                enc.put(sym, &table).unwrap();
            }
            let bytes = enc.finish();

            let mut dec = RansDecoder::new(&bytes).unwrap();
            let mut decoded = Vec::new();
            for _ in 0..message.len() {
                decoded.push(dec.get(&table).unwrap());
            }
            assert_eq!(message, decoded, "failed at precision_bits={prec}");
        }
    }

    #[test]
    fn streaming_encoder_error_invalid_symbol() {
        let table = FrequencyTable::from_counts(&[5, 3, 2], 12).unwrap();
        let mut enc = RansEncoder::new();
        let err = enc.put(3, &table).unwrap_err(); // alphabet size is 3, symbol 3 is OOB
        assert!(matches!(err, AnsError::InvalidSymbol { symbol: 3, .. }));
    }

    #[test]
    fn encode_zero_frequency_symbol() {
        // Table with a zero-count symbol: [5, 0, 3] -> symbol 1 has freq=0.
        let table = FrequencyTable::from_counts(&[5, 0, 3], 12).unwrap();
        assert_eq!(table.freq(1).unwrap(), 0);
        let mut enc = RansEncoder::new();
        let err = enc.put(1, &table).unwrap_err();
        assert!(matches!(err, AnsError::ZeroFrequency { symbol: 1 }));
    }

    #[test]
    fn decode_beyond_message_length() {
        let table = FrequencyTable::from_counts(&[3, 7], 12).unwrap();
        let message = [0u32, 1];
        let bytes = encode(&message, &table).unwrap();
        // Decoding more symbols than encoded should eventually error.
        let err = decode(&bytes, &table, 100).unwrap_err();
        assert!(matches!(err, AnsError::TruncatedInput { .. }));
    }

    #[test]
    fn batch_decode_zero_symbols() {
        let table = FrequencyTable::from_counts(&[5, 3, 2], 12).unwrap();
        // A valid 4-byte state (RANS_L in little-endian).
        let bytes = RANS_L.to_le_bytes();
        let result = decode(&bytes, &table, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn advance_rejects_invalid_symbol() {
        let table = FrequencyTable::from_counts(&[3, 7], 12).unwrap();
        let message = [0u32, 1];
        let bytes = encode(&message, &table).unwrap();
        let mut dec = RansDecoder::new(&bytes).unwrap();
        // Symbol 2 is out of range for a 2-symbol alphabet.
        let err = dec.advance(2, &table).unwrap_err();
        assert!(matches!(
            err,
            AnsError::InvalidSymbol {
                symbol: 2,
                alphabet_size: 2
            }
        ));
    }

    #[test]
    fn streaming_decoder_truncated() {
        // Less than 4 bytes.
        let err = RansDecoder::new(&[0u8, 1, 2]).unwrap_err();
        assert!(matches!(
            err,
            AnsError::TruncatedInput {
                available: 3,
                needed: 4
            }
        ));
        let err = RansDecoder::new(&[]).unwrap_err();
        assert!(matches!(
            err,
            AnsError::TruncatedInput {
                available: 0,
                needed: 4
            }
        ));
    }

    #[test]
    fn streaming_decoder_corrupted_state() {
        // 4 bytes encoding state=0, which is < RANS_L.
        let bytes = 0u32.to_le_bytes();
        let err = RansDecoder::new(&bytes).unwrap_err();
        assert!(matches!(err, AnsError::InvalidState { .. }));
    }

    #[test]
    fn decoder_exhaustion_no_panic() {
        let table = FrequencyTable::from_counts(&[3, 7], 12).unwrap();
        let message = [0u32, 1];
        let bytes = encode(&message, &table).unwrap();

        let mut dec = RansDecoder::new(&bytes).unwrap();
        // Decode the 2 real symbols.
        for _ in 0..2 {
            dec.get(&table).unwrap();
        }
        // Decoding beyond the encoded length: should either return an error
        // or produce garbage, but must not panic.
        for _ in 0..100 {
            match dec.get(&table) {
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }

    #[test]
    fn stress_large_message() {
        // 10K symbols with skewed distribution
        let counts = [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let table = FrequencyTable::from_counts(&counts, 14).unwrap();
        let message: Vec<u32> = (0..10_000).map(|i| (i % 10) as u32).collect();
        let bytes = encode(&message, &table).unwrap();
        let decoded = decode(&bytes, &table, message.len()).unwrap();
        assert_eq!(message, decoded);
    }

    #[test]
    fn bits_back_cross_model() {
        // Bits-back: decode from a "prior" model to extract latents,
        // then re-encode under a "posterior" model. The posterior encoding
        // must round-trip back through a posterior decode.
        let prior = FrequencyTable::from_counts(&[1, 1], 12).unwrap();
        let posterior = FrequencyTable::from_counts(&[8, 2], 12).unwrap();

        // Seed the state with some symbols under an arbitrary model.
        let seed_model = FrequencyTable::from_counts(&[3, 7], 12).unwrap();
        let seed = [1u32, 0, 1, 1, 0, 1, 0, 0];
        let seed_bytes = encode(&seed, &seed_model).unwrap();

        // Decode latent samples from the prior (bits-back "free bits" step).
        let mut dec = RansDecoder::new(&seed_bytes).unwrap();
        let mut latents = Vec::new();
        for _ in 0..3 {
            let z = dec.peek(&prior);
            dec.advance(z, &prior).unwrap();
            latents.push(z);
            assert!(z < prior.alphabet_size() as u32);
        }

        // Encode the same latents under the posterior.
        let mut enc = RansEncoder::new();
        for &z in latents.iter().rev() {
            enc.put(z, &posterior).unwrap();
        }
        let posterior_bytes = enc.finish();

        // Decode from posterior must recover the same latents.
        let mut dec2 = RansDecoder::new(&posterior_bytes).unwrap();
        let mut recovered = Vec::new();
        for _ in 0..3 {
            let z = dec2.peek(&posterior);
            dec2.advance(z, &posterior).unwrap();
            recovered.push(z);
        }
        assert_eq!(
            latents, recovered,
            "posterior decode must recover prior-decoded latents"
        );
    }

    #[test]
    fn frequency_table_invariants() {
        let counts = [10u32, 20, 30, 40];
        let table = FrequencyTable::from_counts(&counts, 14).unwrap();

        // CDF is monotonically increasing
        for i in 0..table.alphabet_size() {
            let cdf_i = table.cum_freq(i as u32).unwrap();
            let cdf_next = table.cum_freq((i + 1) as u32).unwrap_or(table.total());
            assert!(cdf_next >= cdf_i, "CDF not monotone at symbol {i}");
        }

        // Frequencies sum to total
        let sum: u32 = (0..table.alphabet_size())
            .map(|i| table.freq(i as u32).unwrap())
            .sum();
        assert_eq!(sum, table.total());

        // symbol_at_slot covers all slots
        for slot in 0..table.total() {
            let sym = table.symbol_at_slot(slot).unwrap();
            assert!((sym as usize) < table.alphabet_size());
            // Slot falls within symbol's CDF range
            let cdf = table.cum_freq(sym).unwrap();
            let freq = table.freq(sym).unwrap();
            assert!(
                slot >= cdf && slot < cdf + freq,
                "slot {slot} not in range [{cdf}, {}) for sym {sym}",
                cdf + freq
            );
        }

        // Out-of-range returns None
        assert!(table.freq(table.alphabet_size() as u32).is_none());
        // cum_freq(alphabet_size) is the valid final CDF entry (== total)
        assert_eq!(
            table.cum_freq(table.alphabet_size() as u32),
            Some(table.total())
        );
        // One past that is out of range
        assert!(table.cum_freq((table.alphabet_size() + 1) as u32).is_none());
        assert!(table.symbol_at_slot(table.total()).is_none());
    }

    proptest! {
        #[test]
        fn prop_frequency_table_cdf_invariants(
            // Include zero-count symbols in the mix.
            counts in prop::collection::vec(0u32..100u32, 1..16),
        ) {
            // Need at least one nonzero count.
            if counts.iter().all(|&c| c == 0) { return Ok(()); }
            let min_bits = {
                let nonzero = counts.iter().filter(|&&c| c > 0).count();
                (nonzero as f64).log2().ceil().max(1.0) as u32
            };
            let precision_bits = min_bits.clamp(1, 20);
            let table = FrequencyTable::from_counts(&counts, precision_bits).unwrap();

            // Frequencies sum to total
            let sum: u32 = table.freqs().iter().sum();
            prop_assert_eq!(sum, table.total());

            // Every nonzero-count symbol has freq >= 1; zero-count has freq == 0
            for (i, &c) in counts.iter().enumerate() {
                let f = table.freq(i as u32).unwrap();
                if c > 0 {
                    prop_assert!(f >= 1, "symbol {} had count {} but freq 0", i, c);
                } else {
                    prop_assert_eq!(f, 0, "symbol {} had count 0 but freq {}", i, f);
                }
            }

            // CDF is consistent with freqs
            let cdf = table.cdf();
            for i in 0..table.alphabet_size() {
                prop_assert_eq!(cdf[i + 1] - cdf[i], table.freq(i as u32).unwrap());
            }
        }
    }

    // --- from_normalized tests ---

    #[test]
    fn from_normalized_roundtrip() {
        // Build via from_counts, extract freqs, rebuild via from_normalized.
        let table1 = FrequencyTable::from_counts(&[3, 7], 12).unwrap();
        let table2 =
            FrequencyTable::from_normalized(table1.freqs(), table1.precision_bits()).unwrap();
        assert_eq!(table1.freqs(), table2.freqs());
        assert_eq!(table1.cdf(), table2.cdf());

        // Roundtrip through both tables must produce identical bytes.
        let message = [0u32, 1, 1, 0, 1];
        let bytes1 = encode(&message, &table1).unwrap();
        let bytes2 = encode(&message, &table2).unwrap();
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn from_normalized_rejects_wrong_sum() {
        let err = FrequencyTable::from_normalized(&[100, 200], 12).unwrap_err();
        assert!(matches!(err, AnsError::InvalidTable(_)));
    }

    #[test]
    fn from_normalized_rejects_empty() {
        let err = FrequencyTable::from_normalized(&[], 12).unwrap_err();
        assert!(matches!(err, AnsError::EmptyAlphabet));
    }

    // --- from_float_probs tests ---

    #[test]
    fn from_float_probs_roundtrip() {
        let table = FrequencyTable::from_float_probs(&[0.3, 0.7], 12).unwrap();
        assert_eq!(table.alphabet_size(), 2);
        assert_eq!(table.freqs().iter().sum::<u32>(), table.total());

        let message = [0u32, 1, 1, 0, 1, 0, 1, 1];
        let bytes = encode(&message, &table).unwrap();
        let decoded = decode(&bytes, &table, message.len()).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn from_float_probs_uniform() {
        let table = FrequencyTable::from_float_probs(&[1.0, 1.0, 1.0, 1.0], 12).unwrap();
        // Each symbol should get ~1024 out of 4096.
        for i in 0..4 {
            let f = table.freq(i).unwrap();
            assert!(
                (1020..=1028).contains(&f),
                "freq[{i}] = {f}, expected ~1024"
            );
        }
    }

    #[test]
    fn from_float_probs_with_zeros() {
        // Zero-probability symbols should get freq 0 (via from_counts behavior).
        let table = FrequencyTable::from_float_probs(&[0.0, 0.5, 0.5], 12).unwrap();
        assert_eq!(table.freq(0).unwrap(), 0);
        assert!(table.freq(1).unwrap() > 0);
        assert!(table.freq(2).unwrap() > 0);
    }

    #[test]
    fn from_float_probs_rejects_all_zero() {
        let err = FrequencyTable::from_float_probs(&[0.0, 0.0], 12).unwrap_err();
        assert!(matches!(err, AnsError::InvalidTable(_)));
    }

    // --- slice accessor tests ---

    #[test]
    fn freqs_and_cdf_accessors() {
        let table = FrequencyTable::from_counts(&[10, 20, 30], 12).unwrap();
        let freqs = table.freqs();
        let cdf = table.cdf();

        assert_eq!(freqs.len(), 3);
        assert_eq!(cdf.len(), 4);
        assert_eq!(cdf[0], 0);
        assert_eq!(*cdf.last().unwrap(), table.total());
        assert_eq!(freqs.iter().sum::<u32>(), table.total());

        // CDF is prefix sum of freqs.
        for i in 0..freqs.len() {
            assert_eq!(cdf[i + 1], cdf[i] + freqs[i]);
        }
    }

    #[test]
    fn encoder_state_always_valid() {
        let table = FrequencyTable::from_counts(&[1, 2, 3, 4], 12).unwrap();
        let mut enc = RansEncoder::new();
        // After init, state == RANS_L
        assert!(enc.state() >= RANS_L);
        for sym in [0u32, 1, 2, 3, 2, 1, 0] {
            enc.put(sym, &table).unwrap();
            // State is always >= RANS_L after put (renormalization invariant)
            assert!(
                enc.state() >= RANS_L,
                "state {} < RANS_L after encoding sym {}",
                enc.state(),
                sym
            );
        }
    }

    #[test]
    fn compression_ratio_sanity() {
        // Highly skewed: symbol 0 has 99% probability.
        // H(X) ~ 0.10 bits/sym -> 1000 symbols ~ 12.5 bytes theoretical.
        let counts = [990u32, 5, 3, 1, 1];
        let table = FrequencyTable::from_counts(&counts, 14).unwrap();
        let message: Vec<u32> = (0..1000)
            .map(|i| if i % 100 == 0 { 1 } else { 0 })
            .collect();
        let bytes = encode(&message, &table).unwrap();
        // Should be well under 100 bytes (< 0.8 bits/sym).
        assert!(
            bytes.len() < 100,
            "compressed {} bytes, expected < 100 for 99%-skewed distribution",
            bytes.len()
        );
    }

    #[test]
    fn decoder_remaining_bytes_monotone() {
        let table = FrequencyTable::from_counts(&[3, 7], 12).unwrap();
        let message: Vec<u32> = (0..50).map(|i| (i % 2) as u32).collect();
        let bytes = encode(&message, &table).unwrap();
        let mut dec = RansDecoder::new(&bytes).unwrap();
        let mut prev_remaining = dec.remaining_bytes();
        for _ in 0..50 {
            dec.get(&table).unwrap();
            // remaining_bytes should be monotonically non-increasing
            assert!(
                dec.remaining_bytes() <= prev_remaining,
                "remaining_bytes increased: {} > {}",
                dec.remaining_bytes(),
                prev_remaining
            );
            prev_remaining = dec.remaining_bytes();
        }
    }

    // --- 64-bit rANS tests ---

    #[test]
    fn rans64_roundtrip() {
        let counts = [1u32, 2, 3, 4];
        let table = FrequencyTable::from_counts(&counts, 12).unwrap();
        let symbols = vec![0u32, 1, 2, 3, 2, 2, 1, 0, 3];
        let enc = encode64(&symbols, &table).unwrap();
        let dec = decode64(&enc, &table, symbols.len()).unwrap();
        assert_eq!(symbols, dec);
    }

    #[test]
    fn rans64_streaming_roundtrip() {
        let counts = [3u32, 7];
        let table = FrequencyTable::from_counts(&counts, 12).unwrap();
        let message = vec![0u32, 1, 1, 0, 1];

        let mut enc = Rans64Encoder::new();
        for &sym in message.iter().rev() {
            enc.put(sym, &table).unwrap();
        }
        let bytes = enc.finish();

        let mut dec = Rans64Decoder::new(&bytes).unwrap();
        let mut decoded = Vec::new();
        for _ in 0..message.len() {
            decoded.push(dec.get(&table).unwrap());
        }
        assert_eq!(message, decoded);
    }

    #[test]
    fn rans64_peek_advance() {
        let table = FrequencyTable::from_counts(&[3, 7], 12).unwrap();
        let message = vec![0u32, 1, 1, 0, 1];
        let bytes = encode64(&message, &table).unwrap();
        let mut dec = Rans64Decoder::new(&bytes).unwrap();

        for &expected in &message {
            let sym = dec.peek(&table);
            assert_eq!(sym, expected);
            dec.advance(sym, &table).unwrap();
        }
    }

    #[test]
    fn rans64_large_message() {
        let counts = [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let table = FrequencyTable::from_counts(&counts, 14).unwrap();
        let message: Vec<u32> = (0..10_000).map(|i| (i % 10) as u32).collect();
        let bytes = encode64(&message, &table).unwrap();
        let decoded = decode64(&bytes, &table, message.len()).unwrap();
        assert_eq!(message, decoded);
    }

    #[test]
    fn rans64_empty_message() {
        let table = FrequencyTable::from_counts(&[5, 3, 2], 12).unwrap();
        let bytes = encode64(&[], &table).unwrap();
        // Should be exactly 8 bytes (the initial 64-bit state).
        assert_eq!(bytes.len(), 8);
        let decoded = decode64(&bytes, &table, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn rans64_single_symbol_alphabet() {
        let table = FrequencyTable::from_counts(&[42], 10).unwrap();
        let message = vec![0u32; 50];
        let bytes = encode64(&message, &table).unwrap();
        let decoded = decode64(&bytes, &table, message.len()).unwrap();
        assert_eq!(message, decoded);
    }

    proptest! {
        #[test]
        fn prop_rans64_roundtrip(
            symbols in prop::collection::vec(0u32..256u32, 0..200),
            counts in prop::collection::vec(1u32..100u32, 1..16),
        ) {
            let alphabet = counts.len().max(1);
            let min_bits = (alphabet as f64).log2().ceil().max(1.0) as u32;
            let precision_bits = min_bits.clamp(1, 20);
            let table = FrequencyTable::from_counts(&counts, precision_bits).unwrap();
            let symbols: Vec<u32> = symbols.into_iter().map(|s| s % (alphabet as u32)).collect();
            let enc = encode64(&symbols, &table)?;
            let dec = decode64(&enc, &table, symbols.len())?;
            prop_assert_eq!(symbols, dec);
        }
    }
}
