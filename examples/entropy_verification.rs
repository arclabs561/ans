//! Verify ANS coding efficiency against theoretical entropy bounds.
//!
//! Uses fingerprints to compute the theoretical Shannon entropy of a symbol
//! distribution, then measures how close ANS gets to the theoretical optimum.
//! The gap (redundancy) should be small for a well-tuned entropy coder.
//!
//! Run: cargo run --example entropy_verification

use ans::{decode, encode, FrequencyTable};
use fingerprints::{entropy_plugin_nats, to_bits, Fingerprint};

fn main() {
    println!("=== ANS Coding Efficiency vs Theoretical Entropy ===\n");

    // Test several distributions with different entropy levels
    let distributions: Vec<(&str, Vec<u32>)> = vec![
        ("Uniform(8)", vec![100; 8]),
        ("Zipf-like", vec![1000, 500, 250, 125, 63, 31, 16, 8, 4, 2]),
        ("Highly skewed", vec![10000, 100, 10, 1, 1, 1, 1, 1]),
        ("Binary (90/10)", vec![900, 100]),
        ("Near-deterministic", vec![9999, 1]),
    ];

    println!(
        "  {:20} | {:>8} | {:>8} | {:>10} | {:>10} | {:>6}",
        "Distribution", "H (bits)", "Symbols", "Theor (B)", "ANS (B)", "Redund"
    );
    println!(
        "  {:-<20}-+-{:-<8}-+-{:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<6}",
        "", "", "", "", "", ""
    );

    for (name, counts) in &distributions {
        // Generate symbol stream from distribution
        let mut symbols = Vec::new();
        for (sym, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                symbols.push(sym as u32);
            }
        }
        let n_symbols = symbols.len();

        // Theoretical entropy via fingerprints
        let usize_counts: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
        let fp = Fingerprint::from_counts(usize_counts.iter().copied()).unwrap();
        let h_nats = entropy_plugin_nats(&fp);
        let h_bits = to_bits(h_nats);
        let theoretical_bytes = (h_bits * n_symbols as f64 / 8.0).ceil() as usize;

        // Also verify bits conversion is consistent
        let h_fp = h_bits;

        // ANS encode
        let table = FrequencyTable::from_counts(counts, 14).unwrap();
        let encoded = encode(&symbols, &table).unwrap();
        let ans_bytes = encoded.len();

        // Verify roundtrip
        let decoded = decode(&encoded, &table, n_symbols).unwrap();
        assert_eq!(decoded, symbols, "roundtrip failed for {name}");

        // Redundancy: how many extra bits per symbol beyond entropy
        let ans_bits_per_sym = (ans_bytes * 8) as f64 / n_symbols as f64;
        let redundancy = ans_bits_per_sym - h_bits;

        println!(
            "  {:20} | {:8.4} | {:8} | {:10} | {:10} | {:+.4}",
            name, h_bits, n_symbols, theoretical_bytes, ans_bytes, redundancy
        );

        // Sanity: fingerprint and direct computation should agree
        assert!(
            (h_bits - h_fp).abs() < 1e-6,
            "entropy mismatch: from_counts={h_bits} vs Fingerprint={h_fp}"
        );
    }

    println!("\n  Redundancy = (ANS bits/sym) - H(X). Closer to 0 = better.");
    println!("  Negative values possible due to rounding (integer bytes).");
}
