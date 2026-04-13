# ans

[![crates.io](https://img.shields.io/crates/v/ans.svg)](https://crates.io/crates/ans)
[![Documentation](https://docs.rs/ans/badge.svg)](https://docs.rs/ans)
[![CI](https://github.com/arclabs561/ans/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/ans/actions/workflows/ci.yml)

Asymmetric numeral systems entropy coding.

Pure Rust, `no_std`-compatible, with streaming primitives (`peek`/`advance`)
for bits-back coding (BB-ANS, ROC).

## Batch API

```rust
use ans::{decode, encode, FrequencyTable};

let counts = [10u32, 20, 70]; // A, B, C
let table = FrequencyTable::from_counts(&counts, 14)?;
let message = [0u32, 2, 1, 2, 2, 0];

let bytes = encode(&message, &table)?;
let back = decode(&bytes, &table, message.len())?;
assert_eq!(back, message);

# Ok::<(), ans::AnsError>(())
```

## Streaming API

Symbol-at-a-time encoding/decoding. Required for bits-back coding (BB-ANS, ROC).

```rust
use ans::{RansEncoder, RansDecoder, FrequencyTable};

let table = FrequencyTable::from_counts(&[3, 7], 12)?;
let message = [0u32, 1, 1, 0, 1];

// Encode in reverse order (rANS requirement).
let mut enc = RansEncoder::new();
for &sym in message.iter().rev() {
    enc.put(sym, &table)?;
}
let bytes = enc.finish();

// Decode in forward order.
let mut dec = RansDecoder::new(&bytes)?;
let mut decoded = Vec::new();
for _ in 0..message.len() {
    decoded.push(dec.get(&table)?);
}
assert_eq!(decoded, message);

# Ok::<(), ans::AnsError>(())
```

### Bits-back primitives

`RansDecoder::peek` and `RansDecoder::advance` allow inspecting the decoded slot
before advancing state, which is the key operation for bits-back coding:

```rust
# use ans::{RansEncoder, RansDecoder, FrequencyTable};
# let table = FrequencyTable::from_counts(&[3, 7], 12)?;
# let bytes = ans::encode(&[0u32, 1], &table)?;
let mut dec = RansDecoder::new(&bytes)?;
let sym = dec.peek(&table);       // look at slot without advancing
dec.advance(sym, &table)?;        // advance after external logic
# Ok::<(), ans::AnsError>(())
```

## Building frequency tables

From integer counts (quantized internally):

```rust
use ans::FrequencyTable;

let table = FrequencyTable::from_counts(&[10, 20, 70], 14)?;
# Ok::<(), ans::AnsError>(())
```

From floating-point probabilities (e.g. neural network output):

```rust
use ans::FrequencyTable;

let table = FrequencyTable::from_float_probs(&[0.1, 0.2, 0.7], 14)?;
# Ok::<(), ans::AnsError>(())
```

From pre-normalized frequencies (skip internal quantization):

```rust
use ans::FrequencyTable;

// Frequencies must sum to exactly 2^precision_bits.
let table = FrequencyTable::from_normalized(&[1024, 1024, 2048], 12)?;
# Ok::<(), ans::AnsError>(())
```

## `no_std`

This crate has zero dependencies and is `no_std`-compatible (requires `alloc`).

## Notes

- Encoding returns a byte vector in a **stack format**: decoding consumes bytes from the end.
- This crate is focused on correctness and integration simplicity (not maximum throughput).

### Choosing `precision_bits`

`FrequencyTable::from_counts(counts, precision_bits)` builds a model with total mass
\(T = 2^{precision\_bits}\). Practical guidance:

- Larger `precision_bits` approximates the empirical distribution more closely (less quantization),
  but increases memory and can slow decoding.
- Typical ranges are ~12–16 for small alphabets.

### Memory footprint

The table stores `sym_by_slot` of length \(T\), mapping each slot to a symbol. This dominates:

- Approx size \(\approx 4 \cdot 2^{precision\_bits}\) bytes (u32 per slot), plus `cdf`/`freqs`.
- Example: `precision_bits = 14` → \(2^{14} = 16384\) slots → ~64 KiB for `sym_by_slot`.

### Security / robustness

This is an entropy coder, not encryption. Do not treat it as a cryptographic primitive.

## License

MIT OR Apache-2.0
