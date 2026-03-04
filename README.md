# ans

[![crates.io](https://img.shields.io/crates/v/ans.svg)](https://crates.io/crates/ans)
[![Documentation](https://docs.rs/ans/badge.svg)](https://docs.rs/ans)
[![CI](https://github.com/arclabs561/ans/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/ans/actions/workflows/ci.yml)

Asymmetric Numeral Systems (rANS) entropy coding primitives.

This crate provides a small, dependency-light implementation of byte-oriented rANS.

## Example

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
