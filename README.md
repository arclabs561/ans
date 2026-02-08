# ans

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

## License

MIT OR Apache-2.0
