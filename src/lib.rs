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
extern crate alloc;

use alloc::{format, string::String, string::ToString, vec, vec::Vec};
use thiserror::Error;

/// Lower bound for the rANS state. Encoding emits bytes to keep the state
/// below `RANS_L << 8`; decoding pulls bytes to bring the state back above
/// `RANS_L`.
const RANS_L: u32 = 1 << 23;

/// Errors for rANS operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AnsError {
    /// `precision_bits` was outside the valid range `1..=20`.
    #[error("invalid precision_bits={precision_bits} (must be in 1..=20)")]
    InvalidPrecision { precision_bits: u32 },

    /// The counts slice passed to [`FrequencyTable::from_counts`] was empty.
    #[error("empty frequency table")]
    EmptyAlphabet,

    /// A symbol index exceeded the alphabet size during encoding.
    #[error("invalid symbol {symbol} for alphabet size {alphabet_size}")]
    InvalidSymbol { symbol: u32, alphabet_size: usize },

    /// A symbol with zero frequency was encountered during encoding.
    #[error("frequency for symbol {symbol} is zero")]
    ZeroFrequency { symbol: u32 },

    /// The frequency normalization step could not produce a valid table.
    #[error("frequency table normalization failed: {0}")]
    InvalidTable(String),

    /// The rANS state read from the byte stream was below `RANS_L`.
    #[error("invalid rANS state {state} (expected >= {min_state})")]
    InvalidState { state: u32, min_state: u32 },

    /// The byte stream was shorter than expected during decoding.
    #[error("truncated input ({available} bytes available, need at least {needed})")]
    TruncatedInput {
        /// Bytes available.
        available: usize,
        /// Minimum bytes needed.
        needed: usize,
    },
}

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
        if cur_sum == 0 {
            return Err(AnsError::InvalidTable(format!(
                "normalization produced zero total for {} symbols (precision_bits={precision_bits})",
                counts.len()
            )));
        }

        // We adjust greedily; correctness > optimality.
        while cur_sum != target {
            if cur_sum < target {
                // add one to the max-count symbol
                let (idx, _) = counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &c)| c)
                    .ok_or_else(|| {
                        AnsError::InvalidTable(format!(
                            "no symbols available to increment (cur_sum={cur_sum}, target={target})"
                        ))
                    })?;
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
        if cdf.last().copied().unwrap_or(0) != total {
            return Err(AnsError::InvalidTable(format!(
                "cdf total mismatch: got {}, expected {}",
                cdf.last().copied().unwrap_or(0),
                total
            )));
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
        let state_bytes: [u8; 4] =
            bytes[cursor..cursor + 4]
                .try_into()
                .map_err(|_| AnsError::TruncatedInput {
                    available: bytes.len(),
                    needed: 4,
                })?;
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

    proptest! {
        #[test]
        fn prop_rans_roundtrip(
            precision_bits in 1u32..21,
            symbols in prop::collection::vec(0u32..256u32, 0..200),
            counts in prop::collection::vec(1u32..100u32, 1..32),
        ) {
            let alphabet = counts.len().max(1);
            // Skip when total < alphabet (e.g. precision_bits=1, 3 symbols).
            let table = match FrequencyTable::from_counts(&counts, precision_bits) {
                Ok(t) => t,
                Err(_) => { return Ok(()); }
            };
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
    fn bits_back_simulation() {
        // Simulate bits-back: encode with one model, peek+advance with another
        let model_a = FrequencyTable::from_counts(&[3, 7], 12).unwrap();
        let model_b = FrequencyTable::from_counts(&[5, 5], 12).unwrap();

        // Encode a message with model_a
        let message = [0u32, 1, 1, 0, 1, 0, 1, 1];
        let bytes = encode(&message, &model_a).unwrap();

        // Decode first 4 symbols with model_a, peek the 5th with model_b
        let mut dec = RansDecoder::new(&bytes).unwrap();
        for (i, &expected) in message.iter().take(4).enumerate() {
            let sym = dec.get(&model_a).unwrap();
            assert_eq!(sym, expected, "mismatch at position {i}");
        }

        // Peek with model_b -- different model, different symbol interpretation
        let slot_b = dec.state() & (model_b.total() - 1);
        let sym_b = model_b.symbol_at_slot(slot_b).unwrap();
        // Just verify it doesn't panic and returns a valid symbol
        assert!(sym_b < model_b.alphabet_size() as u32);

        // Continue decoding with model_a
        for (i, &expected) in message.iter().skip(4).enumerate() {
            let sym = dec.get(&model_a).unwrap();
            assert_eq!(sym, expected, "mismatch at position {}", i + 4);
        }
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
        assert!(table.cum_freq((table.alphabet_size() + 1) as u32).is_none());
        assert!(table.symbol_at_slot(table.total()).is_none());
    }

    proptest! {
        #[test]
        fn prop_frequency_table_cdf_invariants(
            precision_bits in 1u32..21,
            counts in prop::collection::vec(1u32..100u32, 1..32),
        ) {
            let table = match FrequencyTable::from_counts(&counts, precision_bits) {
                Ok(t) => t,
                Err(_) => { return Ok(()); }
            };
            // Frequencies sum to total
            let sum: u32 = (0..table.alphabet_size())
                .map(|i| table.freq(i as u32).unwrap())
                .sum();
            prop_assert_eq!(sum, table.total());
            // Every nonzero-count symbol has freq >= 1
            for (i, &c) in counts.iter().enumerate() {
                if c > 0 {
                    prop_assert!(table.freq(i as u32).unwrap() >= 1,
                        "symbol {} had count {} but freq 0", i, c);
                }
            }
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
        // Highly skewed: symbol 0 has 99% probability
        let counts = [990u32, 5, 3, 1, 1];
        let table = FrequencyTable::from_counts(&counts, 14).unwrap();
        // Message dominated by symbol 0 should compress well
        let message: Vec<u32> = (0..1000)
            .map(|i| if i % 100 == 0 { 1 } else { 0 })
            .collect();
        let bytes = encode(&message, &table).unwrap();
        // Should be much less than 1 byte per symbol
        assert!(
            bytes.len() < message.len(),
            "compressed {} bytes >= {} symbols",
            bytes.len(),
            message.len()
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
}
