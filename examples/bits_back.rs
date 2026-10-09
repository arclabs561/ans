//! Bits-back coding (BB-ANS) demonstration.
//!
//! Shows the core bits-back trick: a latent z is *sampled* by decoding from
//! the posterior q(z|x) off the existing ANS state (those bits are borrowed,
//! not spent), then coded under the prior p(z). The net cost of z is
//! log q(z) - log p(z), which is KL(q||p) in expectation.
//!
//! The pattern (Townsend et al., "Practical lossless compression with latent
//! variables using bits back coding"):
//!   1. DECODE z from the posterior q(z|x) (pops bits from the state)
//!   2. ENCODE the data x under p(x|z) (omitted here)
//!   3. ENCODE z under the prior p(z) (pushes bits)
//!
//! The decoder runs the steps in reverse: decode z from the prior, decode x,
//! then ENCODE z under the posterior, which returns the borrowed bits.
//!
//! This example uses a toy two-symbol alphabet with known prior and posterior
//! to verify that bits-back round-trips correctly.

use ans::{AnsError, FrequencyTable, RansDecoder, RansEncoder};

fn main() -> Result<(), AnsError> {
    // Prior: P(z) = [0.5, 0.5] (uniform over two latent values)
    let prior = FrequencyTable::from_counts(&[1, 1], 12)?;
    // Posterior: Q(z|x) = [0.8, 0.2] (after observing data x)
    let posterior = FrequencyTable::from_counts(&[8, 2], 12)?;

    // Seed the ANS state with some initial "random" bits by encoding
    // a few symbols under an arbitrary model. In a real BB-ANS system
    // these would come from previously coded data.
    let seed_model = FrequencyTable::from_counts(&[3, 7], 12)?;
    let seed_message = [1u32, 0, 1, 1, 0, 1, 0, 0];

    let mut enc = RansEncoder::new();
    for &sym in seed_message.iter().rev() {
        enc.put(sym, &seed_model)?;
    }

    println!("=== Bits-back coding (BB-ANS) ===\n");
    println!("Seed: encoded {} symbols", seed_message.len());
    println!("Encoder state after seed: {}", enc.state());

    // --- Bits-back encode step ---
    // In BB-ANS, to encode a latent z ~ Q(z|x), we:
    //   1. Decode z from the POSTERIOR (this pops bits from the state)
    //   2. Encode z under the PRIOR (this pushes bits)
    // Net cost per latent = log q(z) - log p(z); KL(Q||P) in expectation.

    let num_latents = 5;
    let mut latent_samples = Vec::new();

    // To bits-back encode, we first need a decoder view of the current state.
    // Flush the encoder to bytes, build a decoder, do the bits-back ops,
    // then continue encoding.
    let midpoint_bytes = enc.finish();
    let mut dec = RansDecoder::new(&midpoint_bytes)?;

    println!("\n--- Bits-back encode ({num_latents} latents) ---");

    for i in 0..num_latents {
        // Step 1: DECODE from posterior (borrow bits from the state)
        let z = dec.peek(&posterior);
        dec.advance(z, &posterior)?;
        latent_samples.push(z);
        println!("  latent[{i}]: decoded z={z} from posterior");
    }

    // Now re-encode on the same stack: the decoder consumed bytes, leaving a
    // residual state plus any unread prefix bytes.
    let remaining = dec.remaining_bytes();
    let dec_state = dec.state();
    println!("\nAfter posterior decodes: state={dec_state}, remaining_bytes={remaining}");

    let mut enc2 = dec.into_encoder();
    for &z in latent_samples.iter().rev() {
        enc2.put(z, &prior)?;
    }
    let prior_bytes = enc2.finish();
    println!("Prior stack: {} bytes", prior_bytes.len());

    // --- Bits-back decode step ---
    // To recover: decode z from the prior, then encode it back under the
    // posterior, which returns the borrowed bits.
    println!("\n--- Bits-back decode ---");

    let mut dec2 = RansDecoder::new(&prior_bytes)?;
    let mut recovered = Vec::new();
    for i in 0..num_latents {
        let z = dec2.peek(&prior);
        dec2.advance(z, &prior)?;
        recovered.push(z);
        println!("  latent[{i}]: decoded z={z} from prior");
    }

    assert_eq!(recovered, latent_samples);
    println!("\nRecovered latents match originals.");

    let mut posterior_enc = dec2.into_encoder();
    for &z in recovered.iter().rev() {
        posterior_enc.put(z, &posterior)?;
    }
    assert_eq!(posterior_enc.finish(), midpoint_bytes);
    println!("Recovered seed stack matches original.");

    // --- Cost accounting ---
    // Each latent pops -log2 q(z) bits and pushes -log2 p(z) bits, so the
    // stack grows by log2 q(z) - log2 p(z) per latent (KL(q||p) in expectation).
    // Here p = [0.5, 0.5] and q = [0.8, 0.2].
    let q = [0.8f64, 0.2];
    let p = [0.5f64, 0.5];
    let popped: f64 = latent_samples.iter().map(|&z| -q[z as usize].log2()).sum();
    let pushed: f64 = latent_samples.iter().map(|&z| -p[z as usize].log2()).sum();
    println!("\nBits popped under posterior: {popped:.2}");
    println!("Bits pushed under prior:     {pushed:.2}");
    println!("Net stack growth (ideal):    {:.2} bits", pushed - popped);
    println!(
        "Stack bytes: {} before, {} after",
        midpoint_bytes.len(),
        prior_bytes.len()
    );

    Ok(())
}
