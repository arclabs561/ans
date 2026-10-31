# Changelog

## [Unreleased]

### Fixed

- `FrequencyTable::from_float_probs` gives every positive probability at least
  frequency 1, as documented; probabilities below about 2^-31 of the sum (for
  example the tail of a softmax) were rounded to 0 and could not be encoded.
- `FrequencyTable::from_normalized` sums frequencies in u64, so an overflowing
  sum is rejected instead of panicking in debug builds or wrapping to a valid
  total in release builds.

## [0.4.0] - 2026-07-05

### Added

- Added decoder-to-encoder residual stack handoff for bits-back workflows.
- Exposed rANS lower-bound constants for callers that size residual state.

## [0.3.2] - 2026-06-10

### Changed

- Bumped stale dependencies; fixed the fingerprints 0.2 API usage in the example.
