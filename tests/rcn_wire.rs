//! Compatibility with the 12-bit binary model in rcn's entropy coder.

use ans::{FrequencyTable, RansDecoder, RansEncoder};

// Recorded using rcn 1.2.0's BitEncoder::encode_bit and finish, with ans 0.4.1
// from its published Cargo.lock. Each tuple is (bit, predicted frequency of 1).
// The first five bytes are renormalization output and the final four are state.
const STREAM: [u8; 9] = [0x84, 0x0e, 0xb2, 0x00, 0xd0, 0xff, 0x6f, 0x9a, 0x05];
const BITS: [(u32, u32); 9] = [
    (1, 1),
    (0, 4095),
    (1, 3000),
    (1, 2048),
    (0, 1095),
    (1, 31),
    (0, 4081),
    (0, 2048),
    (1, 1500),
];

fn bit_table(p_one: u32) -> FrequencyTable {
    FrequencyTable::from_counts(&[4096 - p_one, p_one], 12).unwrap()
}

#[test]
fn decode_published_rcn_stream() {
    let mut decoder = RansDecoder::new(&STREAM).unwrap();
    for (bit, p_one) in BITS {
        assert_eq!(decoder.get(&bit_table(p_one)).unwrap(), bit);
    }
}

#[test]
fn produce_published_rcn_stream() {
    let mut encoder = RansEncoder::new();
    for (bit, p_one) in BITS.into_iter().rev() {
        encoder.put(bit, &bit_table(p_one)).unwrap();
    }
    assert_eq!(encoder.finish(), STREAM);
}
