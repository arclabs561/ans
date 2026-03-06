default:
    @just --list

check:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    cargo test
    cargo check --no-default-features

check-no-std:
    cargo check --no-default-features

test:
    cargo test

fmt:
    cargo fmt --all
