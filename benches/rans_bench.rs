use ans::rans::{InterleavedRansDecoder, InterleavedRansEncoder, RansDecoder, RansEncoder};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_rans_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("rans_single");
    // Increase input size to 1000 symbols to see throughput benefits
    let input = (0..1000).map(|i| (i % 3) as u32).collect::<Vec<_>>();
    let total_bits = 8;
    let symbols = [(0, 128), (128, 64), (192, 64)];

    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut encoder = RansEncoder::new();
            for &idx in input.iter().rev() {
                let (cf, f) = symbols[idx as usize];
                encoder.encode(cf, f, total_bits).unwrap();
            }
            encoder.finish()
        })
    });

    let mut encoder = RansEncoder::new();
    for &idx in input.iter().rev() {
        let (cf, f) = symbols[idx as usize];
        encoder.encode(cf, f, total_bits).unwrap();
    }
    let (state, stream) = encoder.finish();

    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut decoder = RansDecoder::new(state, stream.clone());
            for _ in 0..input.len() {
                let cf = decoder.get_cum_freq(total_bits);
                let idx = if cf < 128 {
                    0
                } else if cf < 192 {
                    1
                } else {
                    2
                };
                let (ccf, f) = symbols[idx];
                decoder.decode(ccf, f, total_bits);
            }
        })
    });
}

fn bench_rans_interleaved(c: &mut Criterion) {
    let mut group = c.benchmark_group("rans_interleaved");
    let input = (0..1000).map(|i| (i % 3) as u32).collect::<Vec<_>>();
    let total_bits = 8;
    let symbols = [(0, 128), (128, 64), (192, 64)];

    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut encoder = InterleavedRansEncoder::new();
            for lane in 0..4 {
                for &idx in input.iter().rev() {
                    let (cf, f) = symbols[idx as usize];
                    encoder.encode(lane, cf, f, total_bits).unwrap();
                }
            }
            encoder.finish()
        })
    });

    let mut encoder = InterleavedRansEncoder::new();
    for lane in 0..4 {
        for &idx in input.iter().rev() {
            let (cf, f) = symbols[idx as usize];
            encoder.encode(lane, cf, f, total_bits).unwrap();
        }
    }
    let (states, streams) = encoder.finish();

    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut decoder = InterleavedRansDecoder::new(states, streams.clone());
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
        })
    });
}

criterion_group!(benches, bench_rans_single, bench_rans_interleaved);
criterion_main!(benches);
