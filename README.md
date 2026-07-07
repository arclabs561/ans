# ans

[![crates.io](https://img.shields.io/crates/v/ans.svg)](https://crates.io/crates/ans)
[![Documentation](https://docs.rs/ans/badge.svg)](https://docs.rs/ans)

rANS entropy coding with bits-back primitives.

```toml
[dependencies]
ans = "0.4"
```

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
before advancing state. `RansDecoder::into_encoder` preserves the residual stack
so the caller can encode the next model on the same ANS state:

```rust
# use ans::{RansEncoder, RansDecoder, FrequencyTable};
# let table = FrequencyTable::from_counts(&[3, 7], 12)?;
# let bytes = ans::encode(&[0u32, 1], &table)?;
let mut dec = RansDecoder::new(&bytes)?;
let sym = dec.peek(&table);       // look at slot without advancing
dec.advance(sym, &table)?;        // advance after external logic
let mut enc = dec.into_encoder();  // continue from the residual stack
enc.put(sym, &table)?;
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

## 64-bit variant

`Rans64Encoder`/`Rans64Decoder` (and batch helpers `encode64`/`decode64`) use a
64-bit state and emit 32-bit words during renormalization. This gives finer
frequency resolution and fewer renormalization steps per symbol -- the variant
used by production codecs (JPEG XL, LZFSE).

```rust
use ans::{encode64, decode64, FrequencyTable};

let table = FrequencyTable::from_counts(&[3, 7], 14)?;
let bytes = encode64(&[0u32, 1, 1, 0], &table)?;
let back = decode64(&bytes, &table, 4)?;
assert_eq!(back, &[0, 1, 1, 0]);
# Ok::<(), ans::AnsError>(())
```

## `no_std`

Zero dependencies. `no_std`-compatible (requires `alloc`). Builds on `wasm32-unknown-unknown`.

## Notes

- Encoding returns a byte vector in **stack format**: decoding consumes bytes from the end.
- `precision_bits` sets the frequency resolution ($T = 2^p$). Typical range: 12-16.
  The table allocates ~$4 \cdot 2^p$ bytes for the slot lookup.

## Examples

See [examples/README.md](examples/README.md) for runnable examples with
captured output.

## License

MIT OR Apache-2.0
