//! Bits-back coding (BB-ANS) demonstration.
//!
//! Shows the core bits-back trick: using "free" bits from ANS decoder state
//! to encode information from a prior distribution, reducing the net cost
//! of coding a latent variable below its posterior entropy.
//!
//! The pattern:
//!   1. DECODE a sample from the prior (extracts bits from the state)
//!   2. Observe the data given that latent sample
//!   3. ENCODE under the posterior (puts bits back, net cost = KL(q||p))
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
    //   1. Decode z from the PRIOR (this extracts bits from the state)
    //   2. Encode z under the POSTERIOR (this puts bits back)
    // Net cost = H(posterior) - H(prior) = KL(Q||P) in expectation.

    let num_latents = 5;
    let mut latent_samples = Vec::new();

    // To bits-back encode, we first need a decoder view of the current state.
    // Flush the encoder to bytes, build a decoder, do the bits-back ops,
    // then continue encoding.
    let midpoint_bytes = enc.finish();
    let mut dec = RansDecoder::new(&midpoint_bytes)?;

    println!("\n--- Bits-back encode ({num_latents} latents) ---");

    for i in 0..num_latents {
        // Step 1: DECODE from prior (extract "free" bits)
        let z = dec.peek(&prior);
        dec.advance(z, &prior)?;
        latent_samples.push(z);
        println!("  latent[{i}]: decoded z={z} from prior");
    }

    // Now re-encode on the same stack: the decoder consumed bytes, leaving a
    // residual state plus any unread prefix bytes.
    let remaining = dec.remaining_bytes();
    let dec_state = dec.state();
    println!("\nAfter prior decodes: state={dec_state}, remaining_bytes={remaining}");

    let mut enc2 = dec.into_encoder();
    for &z in latent_samples.iter().rev() {
        enc2.put(z, &posterior)?;
    }
    let posterior_bytes = enc2.finish();
    println!("Posterior stack: {} bytes", posterior_bytes.len());

    // --- Bits-back decode step ---
    // To recover: decode from posterior, then encode back into prior.
    println!("\n--- Bits-back decode ---");

    let mut dec2 = RansDecoder::new(&posterior_bytes)?;
    let mut recovered = Vec::new();
    for i in 0..num_latents {
        let z = dec2.peek(&posterior);
        dec2.advance(z, &posterior)?;
        recovered.push(z);
        println!("  latent[{i}]: decoded z={z} from posterior");
    }

    assert_eq!(recovered, latent_samples);
    println!("\nRecovered latents match originals.");

    let mut prior_enc = dec2.into_encoder();
    for &z in recovered.iter().rev() {
        prior_enc.put(z, &prior)?;
    }
    assert_eq!(prior_enc.finish(), midpoint_bytes);
    println!("Recovered seed stack matches original.");

    // --- Verify the information-theoretic property ---
    // With uniform prior (1 bit each) and posterior [0.8, 0.2] (~0.72 bits each),
    // bits-back should save ~0.28 bits per latent vs naive posterior coding.
    let posterior_bits = posterior_bytes.len() as f64 * 8.0;
    let naive_bits = {
        // Naive: encode each latent under the posterior only (no bits-back)
        let mut naive_enc = RansEncoder::new();
        for &z in latent_samples.iter().rev() {
            naive_enc.put(z, &posterior)?;
        }
        naive_enc.finish().len() as f64 * 8.0
    };
    println!("\nBits used (same stack):     {posterior_bits:.0}");
    println!("Bits used (posterior only): {naive_bits:.0}");
    println!(
        "In a full BB-ANS pipeline, the prior-decode step extracts ~{:.1} free bits per latent",
        1.0 // H(uniform prior) = 1 bit
    );

    Ok(())
}
