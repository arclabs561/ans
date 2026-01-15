//! Huffman coding baseline.
//!
//! Provides a classic prefix-based entropy coder for comparison with ANS.
//!
//! # Historical Context
//!
//! David Huffman (1952) developed this algorithm as a term paper at MIT.
//! It was the first practical algorithm for constructing optimal prefix codes.
//! It remained the standard for fast entropy coding until the rise of ANS.

use std::collections::BinaryHeap;

/// Huffman tree node.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Leaf {
        symbol: u8,
        freq: u32,
    },
    Internal {
        left: Box<Node>,
        right: Box<Node>,
        freq: u32,
    },
}

impl Node {
    fn freq(&self) -> u32 {
        match self {
            Node::Leaf { freq, .. } => *freq,
            Node::Internal { freq, .. } => *freq,
        }
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.freq().cmp(&self.freq()) // Min-priority queue
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Huffman encoder.
pub struct HuffmanEncoder {
    codes: Vec<Vec<u8>>, // symbol -> bit sequence
}

impl HuffmanEncoder {
    /// Create a new Huffman encoder from symbol frequencies.
    pub fn new(counts: &[u32]) -> Self {
        if counts.is_empty() {
            return Self {
                codes: vec![Vec::new(); 256],
            };
        }

        let mut pq = BinaryHeap::new();
        for (s, &f) in counts.iter().enumerate() {
            if f > 0 {
                pq.push(Node::Leaf {
                    symbol: s as u8,
                    freq: f,
                });
            }
        }

        while pq.len() > 1 {
            let left = pq.pop().unwrap();
            let right = pq.pop().unwrap();
            let freq = left.freq() + right.freq();
            pq.push(Node::Internal {
                left: Box::new(left),
                right: Box::new(right),
                freq,
            });
        }

        let mut codes = vec![Vec::new(); 256];
        if let Some(root) = pq.pop() {
            Self::build_codes(&root, Vec::new(), &mut codes);
        }

        Self { codes }
    }

    fn build_codes(node: &Node, prefix: Vec<u8>, codes: &mut Vec<Vec<u8>>) {
        match node {
            Node::Leaf { symbol, .. } => {
                codes[*symbol as usize] = if prefix.is_empty() { vec![0] } else { prefix };
            }
            Node::Internal { left, right, .. } => {
                let mut left_prefix = prefix.clone();
                left_prefix.push(0);
                Self::build_codes(left, left_prefix, codes);

                let mut right_prefix = prefix;
                right_prefix.push(1);
                Self::build_codes(right, right_prefix, codes);
            }
        }
    }

    /// Encode a symbol sequence into a bit stream.
    pub fn encode(&self, data: &[u8]) -> Vec<u8> {
        let mut bits = Vec::new();
        for &s in data {
            bits.extend_from_slice(&self.codes[s as usize]);
        }
        bits
    }
}

/// Huffman decoder.
pub struct HuffmanDecoder {
    root: Option<Node>,
}

impl HuffmanDecoder {
    /// Create a new Huffman decoder from frequencies.
    pub fn new(counts: &[u32]) -> Self {
        let mut pq = BinaryHeap::new();
        for (s, &f) in counts.iter().enumerate() {
            if f > 0 {
                pq.push(Node::Leaf {
                    symbol: s as u8,
                    freq: f,
                });
            }
        }

        while pq.len() > 1 {
            let left = pq.pop().unwrap();
            let right = pq.pop().unwrap();
            let freq = left.freq() + right.freq();
            pq.push(Node::Internal {
                left: Box::new(left),
                right: Box::new(right),
                freq,
            });
        }

        Self { root: pq.pop() }
    }

    /// Decode a bit stream into a symbol sequence.
    pub fn decode(&self, bits: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut curr = if let Some(ref r) = self.root {
            r
        } else {
            return Vec::new();
        };

        for &bit in bits {
            match curr {
                Node::Internal { left, right, .. } => {
                    curr = if bit == 0 { left } else { right };
                }
                Node::Leaf { .. } => unreachable!(),
            }

            if let Node::Leaf { symbol, .. } = curr {
                out.push(*symbol);
                curr = self.root.as_ref().unwrap();
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_roundtrip() {
        let data = b"abracadabra";
        let mut counts = [0u32; 256];
        for &b in data {
            counts[b as usize] += 1;
        }

        let encoder = HuffmanEncoder::new(&counts);
        let bits = encoder.encode(data);

        let decoder = HuffmanDecoder::new(&counts);
        let decoded = decoder.decode(&bits);

        assert_eq!(data.to_vec(), decoded);
    }
}
