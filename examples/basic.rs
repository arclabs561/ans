use ans::{decode, encode, FrequencyTable};

fn main() -> Result<(), ans::AnsError> {
    let counts = [10u32, 20, 70]; // symbols: A=0, B=1, C=2
    let table = FrequencyTable::from_counts(&counts, 14)?;

    let message = [0u32, 2, 1, 2, 2, 0]; // A C B C C A
    let bytes = encode(&message, &table)?;
    println!(
        "encoded {} symbols into {} bytes",
        message.len(),
        bytes.len()
    );

    let decoded = decode(&bytes, &table, message.len())?;
    assert_eq!(decoded, message);
    println!("roundtrip OK");

    Ok(())
}
