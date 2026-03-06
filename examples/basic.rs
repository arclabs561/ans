use ans::{decode, encode, FrequencyTable, RansDecoder, RansEncoder};

fn main() -> Result<(), ans::AnsError> {
    let counts = [10u32, 20, 70]; // symbols: A=0, B=1, C=2
    let table = FrequencyTable::from_counts(&counts, 14)?;

    // --- Batch API ---
    let message = [0u32, 2, 1, 2, 2, 0]; // A C B C C A
    let bytes = encode(&message, &table)?;
    println!(
        "batch: encoded {} symbols into {} bytes",
        message.len(),
        bytes.len()
    );
    let decoded = decode(&bytes, &table, message.len())?;
    assert_eq!(decoded, message);
    println!("batch: roundtrip OK");

    // --- Streaming API ---
    let mut enc = RansEncoder::new();
    for &sym in message.iter().rev() {
        enc.put(sym, &table)?;
    }
    let bytes = enc.finish();
    println!(
        "streaming: encoded {} symbols into {} bytes",
        message.len(),
        bytes.len()
    );

    let mut dec = RansDecoder::new(&bytes)?;
    let mut stream_decoded = Vec::new();
    for _ in 0..message.len() {
        stream_decoded.push(dec.get(&table)?);
    }
    assert_eq!(stream_decoded, message);
    println!("streaming: roundtrip OK");

    // --- Peek + advance (bits-back primitives) ---
    let bytes2 = encode(&message, &table)?;
    let mut dec2 = RansDecoder::new(&bytes2)?;
    print!("peek+advance: ");
    for _ in 0..message.len() {
        let sym = dec2.peek(&table);
        dec2.advance(sym, &table)?;
        print!("{sym} ");
    }
    println!();

    Ok(())
}
