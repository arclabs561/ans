//! Range Asymmetric Numeral Systems (rANS).
//!
//! rANS is a variant of ANS that uses multiplication and division
//! to update the state, making it more flexible than tANS for
//! large alphabets or dynamic distributions.

use crate::error::{Error, Result};

/// Minimal state for rANS.
pub const RANS_L: u64 = 1 << 31;

/// rANS encoder.
pub struct RansEncoder {
    state: u64,
    output: Vec<u32>,
}

impl RansEncoder {
    /// Create a new rANS encoder.
    pub fn new() -> Self {
        Self {
            state: RANS_L,
            output: Vec::new(),
        }
    }

    /// Encode a symbol with given cumulative frequency and frequency.
    ///
    /// # Arguments
    /// * `cum_freq` - Cumulative frequency of symbol
    /// * `freq` - Frequency of symbol
    /// * `total_bits` - Precision in bits (total frequency = 1 << total_bits)
    ///
    /// # Errors
    /// Returns `Error::InvalidProbability` if `freq` is 0.
    pub fn encode(&mut self, cum_freq: u32, freq: u32, total_bits: u32) -> Result<()> {
        if freq == 0 {
            return Err(Error::InvalidProbability(0.0));
        }

        let total = 1u64 << total_bits;

        // Renormalize
        // state must be in [L, L * freq / total * 2^32) before encoding
        let x_max = (RANS_L >> total_bits).wrapping_mul(freq as u64) << 32;

        while self.state >= x_max {
            self.output.push((self.state & 0xFFFFFFFF) as u32);
            self.state >>= 32;
        }

        // state = (x / freq) * total + (x % freq) + cum_freq
        self.state =
            (self.state / freq as u64) * total + (self.state % freq as u64) + cum_freq as u64;
        Ok(())
    }

    /// Finish encoding and return the compressed data.
    ///
    /// Returns `(final_state, stream)`.
    pub fn finish(mut self) -> (u64, Vec<u32>) {
        self.output.reverse();
        (self.state, self.output)
    }

    /// Return the current internal state.
    pub fn get_state(&self) -> u64 {
        self.state
    }
}

impl Default for RansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// rANS decoder.
pub struct RansDecoder {
    state: u64,
    stream: std::vec::IntoIter<u32>,
}

impl RansDecoder {
    /// Create a new rANS decoder from final state and stream.
    pub fn new(state: u64, stream: Vec<u32>) -> Self {
        Self {
            state,
            stream: stream.into_iter(),
        }
    }

    /// Return the current internal state.
    pub fn get_state(&self) -> u64 {
        self.state
    }

    /// Get current cumulative frequency from state.
    pub fn get_cum_freq(&self, total_bits: u32) -> u32 {
        let total = 1u64 << total_bits;
        (self.state % total) as u32
    }

    /// Decode a symbol and update state.
    ///
    /// # Arguments
    /// * `cum_freq` - Cumulative frequency of the symbol that was decoded
    /// * `freq` - Frequency of the symbol that was decoded
    /// * `total_bits` - Precision in bits
    pub fn decode(&mut self, cum_freq: u32, freq: u32, total_bits: u32) {
        let total = 1u64 << total_bits;

        // state = freq * (state / total) + (state % total) - cum_freq
        self.state = (freq as u64) * (self.state / total) + (self.state % total) - cum_freq as u64;

        // Renormalize
        while self.state < RANS_L {
            if let Some(val) = self.stream.next() {
                self.state = (self.state << 32) | (val as u64);
            } else {
                break;
            }
        }
    }
}

/// Interleaved rANS encoder (4-way parallel).
pub struct InterleavedRansEncoder {
    states: [u64; 4],
    outputs: [Vec<u32>; 4],
}

impl InterleavedRansEncoder {
    /// Create a new 4-way interleaved encoder.
    pub fn new() -> Self {
        Self {
            states: [RANS_L; 4],
            outputs: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// Encode a symbol into one of the interleaved streams.
    pub fn encode(&mut self, lane: usize, cum_freq: u32, freq: u32, total_bits: u32) -> Result<()> {
        let total = 1u64 << total_bits;
        let x_max = (RANS_L >> total_bits).wrapping_mul(freq as u64) << 32;

        while self.states[lane] >= x_max {
            self.outputs[lane].push((self.states[lane] & 0xFFFFFFFF) as u32);
            self.states[lane] >>= 32;
        }

        self.states[lane] = (self.states[lane] / freq as u64) * total
            + (self.states[lane] % freq as u64)
            + cum_freq as u64;
        Ok(())
    }

    /// Finish encoding.
    pub fn finish(mut self) -> ([u64; 4], [Vec<u32>; 4]) {
        for lane in 0..4 {
            self.outputs[lane].reverse();
        }
        (self.states, self.outputs)
    }
}

impl Default for InterleavedRansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Interleaved rANS decoder.
pub struct InterleavedRansDecoder {
    states: [u64; 4],
    streams: [std::vec::IntoIter<u32>; 4],
}

impl InterleavedRansDecoder {
    /// Create a new interleaved decoder.
    pub fn new(states: [u64; 4], streams: [Vec<u32>; 4]) -> Self {
        let streams = streams.map(|stream| stream.into_iter());
        Self { states, streams }
    }

    /// Get current cumulative frequency for a lane.
    pub fn get_cum_freq(&self, lane: usize, total_bits: u32) -> u32 {
        let total = 1u64 << total_bits;
        (self.states[lane] % total) as u32
    }

    /// Decode for a lane.
    pub fn decode(&mut self, lane: usize, cum_freq: u32, freq: u32, total_bits: u32) {
        let total = 1u64 << total_bits;
        self.states[lane] = (freq as u64) * (self.states[lane] / total)
            + (self.states[lane] % total)
            - cum_freq as u64;

        while self.states[lane] < RANS_L {
            if let Some(val) = self.streams[lane].next() {
                self.states[lane] = (self.states[lane] << 32) | (val as u64);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_rans_basic_roundtrip() {
        let mut encoder = RansEncoder::new();
        let total_bits = 8;
        let symbols = [(0, 0, 128), (1, 128, 64), (2, 192, 64)];
        let input: Vec<usize> = vec![0, 1, 2, 0, 0];
        for &idx in input.iter().rev() {
            let (_, cum_freq, freq) = symbols[idx];
            encoder.encode(cum_freq, freq, total_bits).unwrap();
        }
        let (final_state, stream) = encoder.finish();
        let mut decoder = RansDecoder::new(final_state, stream);
        let mut output = Vec::new();
        for _ in 0..input.len() {
            let cf = decoder.get_cum_freq(total_bits);
            let idx = if cf < 128 {
                0
            } else if cf < 192 {
                1
            } else {
                2
            };
            output.push(idx);
            let (_, cum_freq, freq) = symbols[idx];
            decoder.decode(cum_freq, freq, total_bits);
        }
        assert_eq!(input, output);
    }

    #[test]
    fn test_interleaved_rans_basic() {
        let mut encoder = InterleavedRansEncoder::new();
        let total_bits = 8;
        let symbols = [(0, 0, 128), (1, 128, 128)];
        let input_per_lane = [0usize, 1, 0];

        for lane in 0..4 {
            for &val in input_per_lane.iter().rev() {
                let (_, cf, f) = symbols[val];
                encoder.encode(lane, cf, f, total_bits).unwrap();
            }
        }

        let (states, streams) = encoder.finish();
        let mut decoder = InterleavedRansDecoder::new(states, streams);

        for &expected in &input_per_lane {
            for lane in 0..4 {
                let cf = decoder.get_cum_freq(lane, total_bits);
                let val = if cf < 128 { 0 } else { 1 };
                assert_eq!(val, expected);
                let (_, cf, f) = symbols[val];
                decoder.decode(lane, cf, f, total_bits);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_rans_roundtrip_small_alphabet(
            a in 1u32..200,
            b in 1u32..200,
            c in 1u32..200,
            input in prop::collection::vec(0usize..3, 1..60),
        ) {
            let total_bits = 8;
            let total = 1u32 << total_bits;
            let sum = a + b + c;
            prop_assume!(sum < total);

            let f0 = a;
            let f1 = b;
            let f2 = total - sum;
            prop_assume!(f2 > 0);

            let symbols = [(0, 0u32, f0), (1, f0, f1), (2, f0 + f1, f2)];

            let mut encoder = RansEncoder::new();
            for &idx in input.iter().rev() {
                let (_, cum_freq, freq) = symbols[idx];
                encoder.encode(cum_freq, freq, total_bits).unwrap();
            }
            let (final_state, stream) = encoder.finish();

            let mut decoder = RansDecoder::new(final_state, stream);
            let mut output = Vec::with_capacity(input.len());
            for _ in 0..input.len() {
                let cf = decoder.get_cum_freq(total_bits);
                let idx = if cf < f0 { 0 } else if cf < f0 + f1 { 1 } else { 2 };
                output.push(idx);
                let (_, cum_freq, freq) = symbols[idx];
                decoder.decode(cum_freq, freq, total_bits);
            }

            prop_assert_eq!(input, output);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_interleaved_rans_roundtrip(
            symbols in prop::collection::vec(0u8..2, 1..32),
        ) {
            let total_bits = 8;
            let symbols_table = [(0, 0, 128), (1, 128, 128)];

            let mut encoder = InterleavedRansEncoder::new();
            for lane in 0..4 {
                for &sym in symbols.iter().rev() {
                    let (_, cum, freq) = symbols_table[sym as usize];
                    encoder.encode(lane, cum, freq, total_bits).unwrap();
                }
            }

            let (states, streams) = encoder.finish();
            let mut decoder = InterleavedRansDecoder::new(states, streams);

            let mut outputs: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for _ in 0..symbols.len() {
                for (lane, out) in outputs.iter_mut().enumerate() {
                    let cum = decoder.get_cum_freq(lane, total_bits);
                    let sym = if cum < 128 { 0 } else { 1 };
                    out.push(sym);
                    let (_, cf, f) = symbols_table[sym as usize];
                    decoder.decode(lane, cf, f, total_bits);
                }
            }

            for out in &outputs {
                prop_assert_eq!(&symbols, out);
            }
        }
    }
}
