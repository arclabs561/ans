//! Table Asymmetric Numeral Systems (tANS).
//!
//! tANS is a variant of ANS that pre-computes transitions into
//! tables, enabling extremely fast decoding at the cost of
//! fixed distributions and memory for tables.

use crate::error::{Error, Result};

/// A tANS symbol state transition (Decoder entry).
#[derive(Clone, Copy, Debug, Default)]
pub struct SymbolState {
    /// Base value for next state
    pub(crate) offset: u16,
    /// Number of bits to read from stream
    pub(crate) nb_bits: u8,
    /// The decoded symbol
    pub(crate) symbol: u8,
}

#[derive(Clone, Copy, Debug)]
struct SymbolInfo {
    /// Max value in the state range for this symbol *before* emitting bits.
    /// Used to determine if we need to emit a bit.
    max_val: u32,
    /// Number of bits to shift state during encoding normalization.
    /// (Typically just 1 in binary tANS if we emit bit-by-bit).
    /// But for fast encoding we store `k`.
    /// Here we store `min_bits`?
    /// Let's use the FSE `delta_nbits` approach.
    delta_nbits: u32,
    /// Start of the range for this symbol in the table (L).
    start: u32,
}

/// tANS encoder.
pub struct TansEncoder {
    state: u32,
    symtab: Vec<SymbolInfo>,
    origin: Vec<u32>, // Transition table: (symbol, sub_state) -> global_state
    // But we need random access by `(symbol, sub_state)`.
    // The `origin` table in typical FSE maps `sub_state` to `global_state`.
    // It is effectively `table[sub_state + start[symbol]]`.
    // But since `sub_state` isn't contiguous per symbol in the *interleaved* table,
    // we need `symtab` to help us find the right slot.
    //
    // Actually, `origin` is just the spread table reversed?
    // Let's stick to the standard:
    // next_state = table[sub_state + (symbol_start)]
    // We store `origin` as flat array where `origin[symbol_start + sub_state]` gives the new state.
    #[allow(dead_code)]
    l_bits: u32,
}

impl TansEncoder {
    /// Create a new tANS encoder with a given distribution.
    pub fn new(counts: &[u32], l_bits: u32) -> Result<Self> {
        let (symtab, origin, _table, nstates) = build_tables(counts, l_bits)?;
        Ok(Self {
            state: nstates, // Start in state L
            symtab,
            origin,
            l_bits,
        })
    }

    /// Encode a symbol and update state.
    pub fn encode<W: FnMut(u8)>(&mut self, symbol: u8, mut write_bit: W) {
        let s = symbol as usize;
        let info = self.symtab[s];

        // Renormalization: output bits until state is in symbol's range
        // Range is [Is, 2*Is - 1] usually, but here we track `sub_state`.
        // The `max_val` in `info` tells us the upper bound.

        let mut x = self.state;
        // In FSE:
        // nbBits = (x + max_val) >> 16 (or similar shift)
        // Here we can loop for simplicity and correctness in this "safe" version.
        // It's O(1) effectively since it runs 0 or 1 times mostly.

        while x > info.max_val {
            write_bit((x & 1) as u8);
            x >>= 1;
        }

        // Transition
        // new_state = table[x + delta]
        // `x` is now the sub-state relative to the symbol (Is .. 2Is - 1) - Is?
        // No, x is in [Is, 2*Is - 1].
        // We map it to [L, 2L - 1] via the table.
        // The table index is `x + start - Is`?
        // Let's use `x` directly if we set up `origin` correctly.
        // `origin` needs to accept `sub_state` directly.
        // If `origin` is size `nstates`, it maps the spread `x` back to `symbol`.
        // That's decoding.

        // For ENCODING, we need `table[symbol][sub_state]`.
        // Since we flatten it, we need `table[info.start + (x - count)]`.

        let idx = info.start + (x - (info.delta_nbits)); // delta_nbits stores `count` here? No.
                                                         // Let's rely on `info.start` being the offset for this symbol's block in `origin`.
                                                         // And `origin` stores the mapping `sub_state -> global_state`.

        self.state = self.origin[idx as usize];
    }

    /// Current encoder state.
    pub fn state(&self) -> u32 {
        self.state
    }
}

/// tANS decoder.
pub struct TansDecoder {
    state: u32,
    table: Vec<SymbolState>,
    nstates: u32,
    #[allow(dead_code)]
    l_bits: u32,
}

impl TansDecoder {
    /// Create a new tANS decoder with a given distribution.
    pub fn new(counts: &[u32], l_bits: u32) -> Result<Self> {
        let (_symtab, _origin, table, nstates) = build_tables(counts, l_bits)?;
        Ok(Self {
            state: nstates, // Start in state L
            table,
            nstates,
            l_bits,
        })
    }

    /// Decode a symbol and update state from bit stream.
    pub fn decode<I: Iterator<Item = u8>>(&mut self, bits: &mut I) -> u8 {
        // State x is in [L, 2L - 1].
        // We index table by (x - L).
        let idx = (self.state - self.nstates) as usize;
        let entry = self.table[idx];

        let mut val = 0u32;
        for i in 0..entry.nb_bits {
            val |= (bits.next().unwrap_or(0) as u32) << i;
        }

        // new_state = offset + bits
        self.state = entry.offset as u32 + val;
        entry.symbol
    }

    /// Set decoder state.
    pub fn set_state(&mut self, state: u32) {
        self.state = state;
    }
}

type Tables = (Vec<SymbolInfo>, Vec<u32>, Vec<SymbolState>, u32);

fn build_tables(counts: &[u32], l_bits: u32) -> Result<Tables> {
    validate_counts(counts, l_bits)?;
    let nstates = 1u32 << l_bits;
    let mask = nstates - 1;

    // 1. Spread symbols (Canonical FSE spread)
    // We use a simple spread: stepping by 5/8 (golden ratio approx) or similar
    // to distribute symbols.
    // Here we use the "shuffled" spread which is robust.

    let mut spread = vec![None; nstates as usize];
    let step = (nstates >> 1) + (nstates >> 3) + 3;
    let mut pos = 0;

    // Sort symbols by count (descending) for better spread properties?
    // FSE sorts. We'll skip sort for simplicity but use the step.
    // Actually, we must assign slots to symbols based on their counts.

    for (s, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            // Find next empty slot
            while spread[pos as usize].is_some() {
                pos = (pos + 1) & mask;
            }
            spread[pos as usize] = Some(s as u8);
            pos = (pos + step) & mask;
        }
    }

    // Convert back to u8 for usage (we know all slots are filled because sum(counts) == nstates)
    let spread: Vec<u8> = spread.into_iter().map(|s| s.unwrap()).collect();

    // 2. Build Encoding Table
    // For each symbol, we identify the ranges [Is, 2*Is - 1].
    // And map them to the slots in the spread table where that symbol appears.

    let mut symtab = vec![
        SymbolInfo {
            max_val: 0,
            delta_nbits: 0,
            start: 0
        };
        counts.len()
    ];

    // Let's build the decoder table first, it's easier.
    let mut table = vec![SymbolState::default(); nstates as usize];
    let mut sym_seen = vec![0u32; counts.len()]; // How many times we've seen symbol s in spread

    for x in 0..nstates {
        let s = spread[x as usize] as usize;
        let k_s = sym_seen[s]; // current sub-state index (0 .. count-1)
        sym_seen[s] += 1;

        // Corresponding state in the "stream" view
        // x_next = count[s] + k_s
        // We need to determine how many bits we read to get back to `x`.
        // state = (x_next << nbBits) + bits
        // We want `state` to be `x`.
        // So `x` should be in range [x_next << nbBits, (x_next + 1) << nbBits - 1].
        // Ideally.

        // Correct FSE construction:
        // We want to map `x` (decoder state) to `(symbol, next_sub_state)`.
        // `symbol` is `spread[x]`.
        // `next_sub_state` is `count[s] + k_s`.
        // Wait, `next_sub_state` needs to be normalized.
        // It consumes `n` bits.

        let count = counts[s];
        // Calculate `nbBits` such that `next_sub_state << nbBits` is approx `nstates`.
        // Specifically, we want `next_state` to be in [L, 2L-1] after reading bits.
        // No, `next_state` is `x` (the table index).
        // We are going FROM `x` TO `next_state`.
        // `next_state` will be `sub_state` + bits.
        // `sub_state` = `count + k_s`.

        // We need `(count + k_s) << nb_bits` to fall into [L, 2L - 1] roughly?
        // Actually: `nBits = l_bits - high_bit(count + k_s)`.
        let sub_state = count + k_s;
        let high_bit = 31 - sub_state.leading_zeros();
        let nb_bits = l_bits - high_bit; // number of bits to read

        let offset = (sub_state << nb_bits) as u16; // base for next state

        table[x as usize] = SymbolState {
            offset,
            nb_bits: nb_bits as u8,
            symbol: s as u8,
        };

        // Build Encoding Info (inverse)
        // For symbol `s`, and sub_state `sub_state`, we map TO `x`.
        // We need to store this in `origin`.
        // `origin` needs to be indexed by `(s, sub_state)`.
        // Since `sub_state` ranges from `count` to `2*count - 1`, we can flatten.
        // But `origin` indices for `s` are stored in `symtab[s].start`.

        // We accumulate these valid `x` values into `origin`.
        // But we need them sorted by `sub_state`?
        // `sym_seen` increments monotonically, so `sub_state` increments monotonically for `s`.
        // So `x` is the global state that maps to `sub_state`.
        // So `origin[start[s] + k_s] = x + nstates`.
        // Wait, encoder state is in [L, 2L - 1].
        // `x` is in [0, L - 1].
        // So `x + L` is the state.
    }

    // Prepare `origin` and `symtab`
    // We need to allocate space in `origin` for each symbol.
    // Total size = sum(counts) = L = nstates.
    let mut origin = vec![0u32; nstates as usize];
    let mut starts = vec![0u32; counts.len()];
    let mut current = 0;
    for (s, &c) in counts.iter().enumerate() {
        starts[s] = current;
        // Also set symtab
        symtab[s].start = current;
        symtab[s].delta_nbits = c; // Using this to store 'count' temporarily?
                                   // Max val for renormalization
                                   // For a state `y`, we emit bits until `y < 2*count`. (Roughly)
                                   // More precisely: `y` must be in `[count, 2*count - 1]` before looking up `origin`.
        symtab[s].max_val = (2 * c) - 1;

        current += c;
    }

    // Fill `origin`
    // We iterate `spread` again or use `sym_seen` history?
    // Let's re-iterate `spread`.
    let mut sym_seen_enc = vec![0u32; counts.len()];
    for x in 0..nstates {
        let s = spread[x as usize] as usize;
        let k = sym_seen_enc[s];
        sym_seen_enc[s] += 1;

        // `sub_state` = count + k.
        // We map `sub_state` -> `x + nstates`.
        let idx = starts[s] + k;
        origin[idx as usize] = nstates + x;
    }

    // Fix up `symtab` `delta_nbits` (not used as count anymore)
    for (s, info) in symtab.iter_mut().enumerate() {
        // delta_nbits was set to `count` above.
        // We need `info.start` to index into `origin` using `y - count`.
        // So `idx = start + (y - count)`.
        // So `idx = start - count + y`.
        // We can store `start - count` in `delta_nbits` (renaming field would be better but struct is fixed above).
        // Let's keep `start` as `start` and use `delta_nbits` for `count`.
        info.delta_nbits = counts[s];
    }

    Ok((symtab, origin, table, nstates))
}

fn validate_counts(counts: &[u32], l_bits: u32) -> Result<()> {
    if counts.is_empty() || counts.len() > 256 {
        return Err(Error::InvalidProbability(0.0));
    }
    if l_bits > 16 {
        return Err(Error::StateOverflow);
    }
    let total = 1u32 << l_bits;
    let sum = counts.iter().sum::<u32>();
    if sum != total {
        return Err(Error::InvalidProbability(0.0));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_tans_roundtrip_small() {
        let counts = vec![4, 2, 1, 1];
        let l_bits = 3;
        let mut encoder = TansEncoder::new(&counts, l_bits).unwrap();
        let input = vec![0u8, 1, 2, 3, 0, 1];
        let mut bits = Vec::new();
        // Encode in reverse
        for &sym in input.iter().rev() {
            let mut symbol_bits = Vec::new();
            encoder.encode(sym, |b| symbol_bits.push(b));
            // Prepend bits to stream to simulate LIFO for symbols but FIFO for bits within symbol
            // (Decoder reads from start of stream)
            bits.splice(0..0, symbol_bits);
        }
        let state = encoder.state();

        let mut decoder = TansDecoder::new(&counts, l_bits).unwrap();
        decoder.set_state(state);
        let mut out = Vec::new();
        let mut bit_iter = bits.into_iter();
        for _ in 0..input.len() {
            out.push(decoder.decode(&mut bit_iter));
        }
        assert_eq!(input, out);
    }

    proptest! {
        #[test]
        fn prop_tans_roundtrip(
            counts in prop::collection::vec(1u32..100, 2..16),
        ) {
            // Normalize counts to power of 2
            let sum: u32 = counts.iter().sum();
            let l_bits = 32 - sum.leading_zeros(); // round up
            let l_bits = l_bits.clamp(4, 12); // Keep small for speed
            let target = 1 << l_bits;

            // Simple normalization
            let mut normalized = vec![0u32; counts.len()];
            let mut current_sum = 0;
            for (i, &c) in counts.iter().enumerate() {
                let n = (c as u64 * target as u64 / sum as u64) as u32;
                let n = n.max(1);
                normalized[i] = n;
                current_sum += n;
            }

            // Fixup
            while current_sum < target {
                normalized[0] += 1;
                current_sum += 1;
            }
            while current_sum > target {
                if normalized[0] > 1 {
                    normalized[0] -= 1;
                    current_sum -= 1;
                } else {
                    // Find largest
                    let (idx, _) = normalized.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();
                    normalized[idx] -= 1;
                    current_sum -= 1;
                }
            }

            let encoder_res = TansEncoder::new(&normalized, l_bits);
            prop_assume!(encoder_res.is_ok());
            let mut encoder = encoder_res.unwrap();

            let input_len = 100;
            let input: Vec<u8> = (0..input_len).map(|i| (i % counts.len()) as u8).collect();

            let mut bits = Vec::new();
            for &sym in input.iter().rev() {
                let mut symbol_bits = Vec::new();
                encoder.encode(sym, |b| symbol_bits.push(b));
                bits.splice(0..0, symbol_bits);
            }
            let state = encoder.state();

            let mut decoder = TansDecoder::new(&normalized, l_bits).unwrap();
            decoder.set_state(state);
            let mut out = Vec::new();
            let mut bit_iter = bits.into_iter();
            for _ in 0..input.len() {
                out.push(decoder.decode(&mut bit_iter));
            }
            prop_assert_eq!(input, out);
        }
    }
}
