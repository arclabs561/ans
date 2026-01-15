#![no_main]
use ans::rans::{RansDecoder, RansEncoder};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (Vec<u8>, u32)| {
    let (input_bytes, total_bits) = data;
    let total_bits = (total_bits % 8) + 8; // 8 to 15 bits
    let total = 1u64 << total_bits;

    if input_bytes.is_empty() {
        return;
    }

    // Create a simple model: 2 symbols, 50/50
    let f0 = total / 2;
    let f1 = total - f0;
    let symbols = [(0u32, f0 as u32), (f0 as u32, f1 as u32)];

    let mut input = Vec::new();
    for &b in &input_bytes {
        input.push((b % 2) as usize);
    }

    let mut encoder = RansEncoder::new();
    for &idx in input.iter().rev() {
        let (cf, f) = symbols[idx];
        if encoder.encode(cf, f, total_bits).is_err() {
            return;
        }
    }

    let (state, stream) = encoder.finish();
    let mut decoder = RansDecoder::new(state, stream);
    let mut output = Vec::new();
    for _ in 0..input.len() {
        let cf = decoder.get_cum_freq(total_bits);
        let idx = if cf < f0 as u32 { 0 } else { 1 };
        output.push(idx);
        let (ccf, f) = symbols[idx];
        decoder.decode(ccf, f, total_bits);
    }

    assert_eq!(input, output);
});
