# ans examples

Each example is runnable from the repo root. Output excerpts below are real,
captured from release runs.

## Which example should I run?

| I want to... | Example |
|---|---|
| Round-trip symbols through batch and streaming rANS | `basic` |
| See the BB-ANS bits-back coding pattern | `bits_back` |
| Compare encoded size with Shannon entropy | `entropy_verification` |

## Core API

### `basic`: how do batch and streaming rANS round-trip a message?

Builds a frequency table, encodes and decodes the same message through the
batch API and the symbol-at-a-time streaming API, then demonstrates
`peek`/`advance`.

```bash
cargo run --release --example basic
```

```text
batch: encoded 6 symbols into 5 bytes
batch: roundtrip OK
streaming: encoded 6 symbols into 5 bytes
streaming: roundtrip OK
peek+advance: 0 2 1 2 2 0
```

## Bits-Back Coding

### `bits_back`: what does BB-ANS do with prior and posterior models?

Uses a two-symbol prior and posterior to show the bits-back pattern: decode
latents from the prior, then encode them under the posterior.

```bash
cargo run --release --example bits_back
```

```text
=== Bits-back coding (BB-ANS) ===

Seed: encoded 8 symbols
Encoder state after seed: 16872765

--- Bits-back encode (5 latents) ---
  latent[0]: decoded z=0 from prior
  latent[1]: decoded z=1 from prior
  latent[2]: decoded z=1 from prior
  latent[3]: decoded z=1 from prior
  latent[4]: decoded z=1 from prior

After prior decodes: state=134980868, remaining_bytes=0
Posterior encoding: 5 bytes

--- Bits-back decode ---
  latent[0]: decoded z=0 from posterior
  latent[1]: decoded z=1 from posterior
  latent[2]: decoded z=1 from posterior
  latent[3]: decoded z=1 from posterior
  latent[4]: decoded z=1 from posterior

Recovered latents match originals.

Bits used (posterior only): 40
Bits used (naive, same):   40
In a full BB-ANS pipeline, the prior-decode step extracts ~1.0 free bits per latent
```

## Entropy Check

### `entropy_verification`: how close is ANS to the entropy bound?

Uses `fingerprints` to compute Shannon entropy for several distributions, then
compares theoretical byte count with encoded byte count.

```bash
cargo run --release --example entropy_verification
```

```text
=== ANS Coding Efficiency vs Theoretical Entropy ===

  Distribution         | H (bits) |  Symbols |  Theor (B) |    ANS (B) | Redund
  ---------------------+----------+----------+------------+------------+-------
  Uniform(8)           |   3.0000 |      800 |        300 |        304 | +0.0400
  Zipf-like            |   1.9911 |     1999 |        498 |        501 | +0.0139
  Highly skewed        |   0.0986 |    10115 |        125 |        128 | +0.0026
  Binary (90/10)       |   0.4690 |     1000 |         59 |         62 | +0.0270
  Near-deterministic   |   0.0015 |    10000 |          2 |          5 | +0.0025

  Redundancy = (ANS bits/sym) - H(X). Closer to 0 = better.
  Negative values possible due to rounding (integer bytes).
```

Redundancy is extra bits per symbol beyond entropy. Values near zero mean the
coder is close to the theoretical bound for that quantized table.
