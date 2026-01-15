# ans

Asymmetric Numeral Systems: near-optimal entropy coding primitives.

Dual-licensed under MIT or the UNLICENSE.

```rust
use ans::tans::{TansEncoder, TansDecoder};

let counts = vec![2, 2, 4]; // 25%, 25%, 50%
let l_bits = 3; // L=8
let encoder = TansEncoder::new(&counts, l_bits).unwrap();
let decoder = TansDecoder::new(&counts, l_bits).unwrap();
```
