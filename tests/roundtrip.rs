//! Public-API integration tests for the rANS codec.
//!
//! Invariant under test: **decode inverts encode for every tested input.**
//! For any symbol sequence `m` and any valid `FrequencyTable` `t`,
//! `decode(encode(m, t), t, m.len()) == m` exactly (lossless round-trip).
//!
//! Unlike the inline `#[cfg(test)]` unit tests, this file consumes `ans`
//! as an external crate, so it exercises only the public surface a real
//! downstream user can reach. The codec is called directly (no
//! reimplementation, no mocks): the oracle is the identity `decode ∘ encode`.

use ans::{decode, encode, FrequencyTable};

/// Small known sequence with a known frequency model.
#[test]
fn roundtrip_small_known_sequence() {
    // Alphabet {A=0, B=1, C=2} with counts 10/20/70.
    let table = FrequencyTable::from_counts(&[10u32, 20, 70], 14).unwrap();
    let message = [0u32, 2, 1, 2, 2, 0]; // A C B C C A

    let bytes = encode(&message, &table).unwrap();
    let recovered = decode(&bytes, &table, message.len()).unwrap();

    assert_eq!(recovered, message, "decode must invert encode exactly");
}

/// Edge input: the empty message. rANS has no end-of-stream marker, so an
/// empty message still emits (and the decoder still validates) the 4-byte
/// final state.
#[test]
fn roundtrip_empty_message() {
    let table = FrequencyTable::from_counts(&[5u32, 3, 2], 12).unwrap();
    let message: [u32; 0] = [];

    let bytes = encode(&message, &table).unwrap();
    let recovered = decode(&bytes, &table, message.len()).unwrap();

    assert_eq!(
        bytes.len(),
        4,
        "empty message encodes to just the state word"
    );
    assert!(recovered.is_empty(), "decode of empty message is empty");
}

/// Skewed distribution: one dominant symbol plus rare symbols. Round-trip
/// must still be exact even though the model is far from uniform.
#[test]
fn roundtrip_skewed_distribution() {
    let table = FrequencyTable::from_counts(&[990u32, 5, 3, 1, 1], 14).unwrap();
    // Mostly symbol 0, with the rare symbols sprinkled in.
    let message: Vec<u32> = (0..500)
        .map(|i| match i % 100 {
            0 => 1,
            37 => 2,
            73 => 3,
            91 => 4,
            _ => 0,
        })
        .collect();

    let bytes = encode(&message, &table).unwrap();
    let recovered = decode(&bytes, &table, message.len()).unwrap();

    assert_eq!(recovered, message, "skewed-model round-trip must be exact");
}

/// Compression actually happens on a compressible input: a long, highly
/// skewed message must encode to fewer bytes than the naive one-byte-per-symbol
/// serialization for the same alphabet.
#[test]
fn compression_beats_naive_on_compressible_input() {
    let table = FrequencyTable::from_counts(&[990u32, 5, 3, 1, 1], 14).unwrap();
    // 99%-skewed toward symbol 0 over 1000 symbols: entropy ~0.1 bits/sym.
    let message: Vec<u32> = (0..1000)
        .map(|i| if i % 100 == 0 { 1 } else { 0 })
        .collect();

    let bytes = encode(&message, &table).unwrap();
    // Naive baseline: one byte per symbol (the obvious serialization for a
    // <=256-symbol alphabet). The codec includes a 4-byte state word, so the
    // win must clear that fixed overhead too.
    let naive_size = message.len();
    assert!(
        bytes.len() < naive_size,
        "expected compression: {} encoded bytes vs {} naive bytes",
        bytes.len(),
        naive_size
    );

    // And it must still decode back to the original.
    let recovered = decode(&bytes, &table, message.len()).unwrap();
    assert_eq!(recovered, message, "compressed bytes must round-trip");
}
