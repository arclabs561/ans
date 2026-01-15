use ans::rans::{InterleavedRansDecoder, InterleavedRansEncoder};

fn main() {
    let input = (0..10000).map(|i| (i % 3) as u32).collect::<Vec<_>>();
    let total_bits = 8;
    let symbols = [(0u32, 128u32), (128u32, 64u32), (192u32, 64u32)];

    for _ in 0..1000 {
        let mut encoder = InterleavedRansEncoder::new();
        for lane in 0..4 {
            for &idx in input.iter().rev() {
                let (cf, f) = symbols[idx as usize];
                encoder.encode(lane, cf, f, total_bits).unwrap();
            }
        }
        let (states, streams) = encoder.finish();

        let mut decoder = InterleavedRansDecoder::new(states, streams);
        for _ in 0..input.len() {
            for lane in 0..4 {
                let cf = decoder.get_cum_freq(lane, total_bits);
                let idx = if cf < 128 {
                    0
                } else if cf < 192 {
                    1
                } else {
                    2
                };
                let (ccf, f) = symbols[idx];
                decoder.decode(lane, ccf, f, total_bits);
            }
        }
    }
}
