use ans::rans::{RansDecoder, RansEncoder};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_rans_roundtrip(
        input in prop::collection::vec(0..3u32, 1..100),
        total_bits in 8..16u32,
    ) {
        let mut encoder = RansEncoder::new();

        // Setup 3 symbols with total power-of-2 frequency
        let total = 1u64 << total_bits;
        let freq0 = total / 2; // 50%
        let freq1 = total / 4; // 25%
        let freq2 = total / 4; // 25%

        let symbols = [
            (0, freq0 as u32),
            (freq0 as u32, freq1 as u32),
            ((freq0 + freq1) as u32, freq2 as u32),
        ];

        // Encode in reverse
        for &sym_idx in input.iter().rev() {
            let (cum_freq, freq) = symbols[sym_idx as usize];
            encoder.encode(cum_freq, freq, total_bits).unwrap();
        }

        let (final_state, stream) = encoder.finish();

        // Decode
        let mut decoder = RansDecoder::new(final_state, stream);
        let mut output = Vec::new();
        for _ in 0..input.len() {
            let cf = decoder.get_cum_freq(total_bits);
            let sym_idx = if cf < symbols[1].0 {
                0
            } else if cf < symbols[2].0 {
                1
            } else {
                2
            };
            output.push(sym_idx);
            let (cum_freq, freq) = symbols[sym_idx as usize];
            decoder.decode(cum_freq, freq, total_bits);
        }

        assert_eq!(input, output);
    }
}
