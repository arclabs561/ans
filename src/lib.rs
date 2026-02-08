//! Asymmetric Numeral Systems (ANS) entropy coding primitives.
//!
//! This crate provides a small, dependency-light implementation of **rANS**
//! (range Asymmetric Numeral Systems), suitable as a building block for higher-level
//! compression schemes (e.g. “bits-back” constructions used in ROC / set coding).
//!
//! ## Design
//! - **Explicit model**: callers provide counts (frequencies) and the precision.
//! - **Small surface**: encode/decode with a [`FrequencyTable`].
//! - **No I/O**: this crate is pure in-memory coding.
//!
//! ## Notes
//! - This is not tuned for maximum speed; it is meant to be correct and easy to integrate.
//! - Encoding returns a byte vector in a **stack format**: the decoder consumes bytes from
//!   the end (LIFO). This avoids reversing buffers.

use thiserror::Error;

/// Errors for rANS operations.
#[derive(Debug, Error)]
pub enum AnsError {
    #[error("invalid precision_bits={precision_bits} (must be in 1..=20)")]
    InvalidPrecision { precision_bits: u32 },

    #[error("empty frequency table")]
    EmptyAlphabet,

    #[error("invalid symbol {symbol} for alphabet size {alphabet_size}")]
    InvalidSymbol { symbol: u32, alphabet_size: usize },

    #[error("frequency for symbol {symbol} is zero")]
    ZeroFrequency { symbol: u32 },

    #[error("frequency table normalization failed: {0}")]
    InvalidTable(String),

    #[error("invalid rANS state {state} (expected >= {min_state})")]
    InvalidState { state: u32, min_state: u32 },

    #[error("truncated input")]
    TruncatedInput,
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
            return Err(AnsError::InvalidTable(
                "normalization produced zero total".to_string(),
            ));
        }

        // We adjust greedily; correctness > optimality.
        while cur_sum != target {
            if cur_sum < target {
                // add one to the max-count symbol
                let (idx, _) = counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &c)| c)
                    .ok_or_else(|| AnsError::InvalidTable("empty counts".to_string()))?;
                freqs[idx] += 1;
                cur_sum += 1;
            } else {
                // subtract one from a symbol with freq > 1 (prefer largest freq)
                let mut best: Option<(usize, u32)> = None;
                for (i, &f) in freqs.iter().enumerate() {
                    if f > 1 {
                        if best.map(|(_, bf)| f > bf).unwrap_or(true) {
                            best = Some((i, f));
                        }
                    }
                }
                let Some((idx, _)) = best else {
                    return Err(AnsError::InvalidTable(
                        "cannot reduce total without dropping some symbol to zero".to_string(),
                    ));
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
        for (sym, &_f) in freqs.iter().enumerate() {
            let start = cdf[sym] as usize;
            let end = cdf[sym + 1] as usize;
            for slot in start..end {
                sym_by_slot[slot] = sym as u32;
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

    #[inline]
    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }

    #[inline]
    pub fn total(&self) -> u32 {
        self.total
    }

    #[inline]
    pub fn alphabet_size(&self) -> usize {
        self.freqs.len()
    }

    #[inline]
    pub fn freq(&self, sym: u32) -> Option<u32> {
        self.freqs.get(sym as usize).copied()
    }

    #[inline]
    pub fn cum_freq(&self, sym: u32) -> Option<u32> {
        self.cdf.get(sym as usize).copied()
    }
}

/// Encode `symbols` with rANS using `table`.
///
/// Output is a byte vector treated as a stack: decoding consumes bytes from the end.
pub fn encode(symbols: &[u32], table: &FrequencyTable) -> Result<Vec<u8>, AnsError> {
    // Standard byte-based rANS.
    const RANS_L: u32 = 1 << 23;
    let mask = table.total - 1;
    debug_assert!(table.total.is_power_of_two());

    let mut state: u32 = RANS_L;
    let mut out: Vec<u8> = Vec::new();

    // Encode in reverse.
    for &sym in symbols.iter().rev() {
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

        // Renormalize: ensure state is small enough.
        // Threshold derived from keeping (state / freq) within 32-bit while emitting bytes.
        let x_max = ((RANS_L >> table.precision_bits) << 8) * freq;
        while state >= x_max {
            out.push((state & 0xFF) as u8);
            state >>= 8;
        }

        let q = state / freq;
        let r = state - q * freq;
        state = (q << table.precision_bits) + r + start;
        debug_assert_eq!(state & mask, (r + start) & mask);
    }

    // Final state (little-endian) is pushed onto the stack.
    out.extend_from_slice(&state.to_le_bytes());
    Ok(out)
}

/// Decode an rANS stream produced by [`encode`].
pub fn decode(bytes: &[u8], table: &FrequencyTable, len: usize) -> Result<Vec<u32>, AnsError> {
    const RANS_L: u32 = 1 << 23;
    if bytes.len() < 4 {
        return Err(AnsError::TruncatedInput);
    }
    let mut cursor = bytes.len();
    // Pop final state
    cursor -= 4;
    let mut state = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
    if state < RANS_L {
        // For a valid stream produced by `encode`, the final state should always be >= RANS_L.
        // Treat smaller states as corruption (often truncation, but not necessarily).
        return Err(AnsError::InvalidState {
            state,
            min_state: RANS_L,
        });
    }

    let mut out = Vec::with_capacity(len);
    let mask = table.total - 1;

    for _ in 0..len {
        let slot = (state & mask) as usize;
        let sym = table.sym_by_slot[slot];
        out.push(sym);

        let sym_us = sym as usize;
        let freq = table.freqs[sym_us];
        let start = table.cdf[sym_us];

        // Advance state.
        state = freq * (state >> table.precision_bits) + ((slot as u32) - start);

        // Renormalize: pull bytes while state < RANS_L.
        while state < RANS_L {
            if cursor == 0 {
                return Err(AnsError::TruncatedInput);
            }
            cursor -= 1;
            state = (state << 8) | (bytes[cursor] as u32);
        }
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

    proptest! {
        #[test]
        fn prop_rans_roundtrip(
            precision_bits in 8u32..14,
            symbols in prop::collection::vec(0u32..256u32, 0..200),
            counts in prop::collection::vec(1u32..100u32, 1..32),
        ) {
            let alphabet = counts.len().max(1);
            let table = FrequencyTable::from_counts(&counts, precision_bits)?;
            let symbols: Vec<u32> = symbols.into_iter().map(|s| s % (alphabet as u32)).collect();
            let enc = encode(&symbols, &table)?;
            let dec = decode(&enc, &table, symbols.len())?;
            prop_assert_eq!(symbols, dec);
        }
    }
}
