//! The shared `(b, g, r)` colour lookup table.
//!
//! Both generations consult the same 256³ boolean table, indexed
//! `(b << 16) | (g << 8) | r`.  The generated assets store it as alternating
//! false/true run lengths; decoding it once into a 2 MiB bitset keeps lookups
//! branch-free and avoids a run search per pixel.
//!
//! `min_spread` is derived from the table rather than assumed: a pixel whose
//! `max(b,g,r) - min(b,g,r)` is below it cannot be a hit, which is what lets a
//! gray page skip the lookup entirely without changing a result.  For the two
//! verified assets it is 5.

use crate::error::{Error, Result};

/// Number of entries in the flattened table: 256³.
pub const MASK_ELEMENTS: usize = 1 << 24;

/// `min_spread` when the table has no true entries at all.
pub const NO_ENTRIES_SPREAD: u16 = 256;

const BITS_PER_WORD: usize = 64;
const WORDS: usize = MASK_ELEMENTS / BITS_PER_WORD;

/// A decoded boolean lookup table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupTable {
    words: Box<[u64]>,
    true_entries: u32,
    min_spread: u16,
}

impl LookupTable {
    /// Decode alternating false/true run lengths.
    ///
    /// The runs must sum to exactly `256³`; anything else is a truncated or
    /// over-long asset and is rejected here rather than read out of bounds.
    pub fn from_runs(asset: &'static str, runs: &[u32], min_spread: u32) -> Result<Self> {
        if min_spread > NO_ENTRIES_SPREAD as u32 {
            return Err(Error::asset(
                asset,
                format!("min_spread {min_spread} is above {NO_ENTRIES_SPREAD}"),
            ));
        }
        if runs.is_empty() {
            return Err(Error::asset(asset, "the lookup has no runs"));
        }
        let mut words = vec![0u64; WORDS].into_boxed_slice();
        let mut position = 0usize;
        let mut true_entries = 0u32;
        for (index, &length) in runs.iter().enumerate() {
            let length = length as usize;
            let end = position
                .checked_add(length)
                .ok_or_else(|| Error::asset(asset, "run lengths overflow the table"))?;
            if end > MASK_ELEMENTS {
                return Err(Error::asset(
                    asset,
                    format!("run lengths exceed 256^3 at run {index} (offset {position})"),
                ));
            }
            if index % 2 == 1 {
                set_bit_range(&mut words, position, end);
                true_entries += length as u32;
            }
            position = end;
        }
        if position != MASK_ELEMENTS {
            return Err(Error::asset(
                asset,
                format!("run lengths cover {position} entries, expected {MASK_ELEMENTS}"),
            ));
        }
        Ok(LookupTable {
            words,
            true_entries,
            min_spread: min_spread as u16,
        })
    }

    /// Build a table directly from bytes, one entry per colour triple.
    ///
    /// Only the tests and the exporter round-trip check need this; the runtime
    /// path always decodes the run-length asset.
    pub fn from_bool_slice(values: &[bool], min_spread: u16) -> Self {
        let mut words = vec![0u64; WORDS].into_boxed_slice();
        let mut true_entries = 0u32;
        for (index, &value) in values.iter().enumerate() {
            if value {
                words[index / BITS_PER_WORD] |= 1 << (index % BITS_PER_WORD);
                true_entries += 1;
            }
        }
        LookupTable {
            words,
            true_entries,
            min_spread,
        }
    }

    /// Whether this colour triple is in the table.
    #[inline]
    pub fn contains(&self, blue: u8, green: u8, red: u8) -> bool {
        let index = ((blue as usize) << 16) | ((green as usize) << 8) | red as usize;
        (self.words[index / BITS_PER_WORD] >> (index % BITS_PER_WORD)) & 1 == 1
    }

    /// The smallest channel spread any true entry has.
    #[inline]
    pub fn min_spread(&self) -> u16 {
        self.min_spread
    }

    /// The spread a pixel must reach to be worth looking up.
    ///
    /// `full_frame` drops the threshold to zero, so every pixel of the frame is
    /// probed rather than only the ones that could be.  The pixels it adds can
    /// never be hits — the threshold is the smallest spread any true entry has —
    /// so the answer is the same and the work is strictly more.  It exists so
    /// that claim can be checked on real frames.
    #[inline]
    pub fn spread_threshold(&self, full_frame: bool) -> u16 {
        if full_frame { 0 } else { self.min_spread }
    }

    /// How many colour triples are true.
    #[inline]
    pub fn true_entries(&self) -> u32 {
        self.true_entries
    }

    /// Bytes per entry in memory, for the diagnostics and benchmarks.
    pub fn resident_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }

    /// Whether two decoded tables are bit-identical.
    ///
    /// The two generated assets carry the same table; the runtime shares one
    /// allocation only after this check, never on the assumption that the
    /// exporter's manifest said so.
    pub fn same_content(&self, other: &LookupTable) -> bool {
        self.min_spread == other.min_spread
            && self.true_entries == other.true_entries
            && self.words == other.words
    }

    /// The entry at a flat index, for round-trip checks.
    pub fn get_index(&self, index: usize) -> bool {
        (self.words[index / BITS_PER_WORD] >> (index % BITS_PER_WORD)) & 1 == 1
    }

    /// The raw bitset words, for diagnostics and round-trip checks.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }
}

/// Set every bit in `[start, end)` of a bitset.
fn set_bit_range(words: &mut [u64], start: usize, end: usize) {
    if start >= end {
        return;
    }
    let first = start / BITS_PER_WORD;
    let last = (end - 1) / BITS_PER_WORD;
    if first == last {
        let low = start % BITS_PER_WORD;
        let high = (end - 1) % BITS_PER_WORD;
        words[first] |= (!0u64 << low) & (!0u64 >> (BITS_PER_WORD - 1 - high));
        return;
    }
    words[first] |= !0u64 << (start % BITS_PER_WORD);
    for word in &mut words[first + 1..last] {
        *word = !0u64;
    }
    words[last] |= !0u64 >> (BITS_PER_WORD - 1 - ((end - 1) % BITS_PER_WORD));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_ranges_are_inclusive_at_both_ends() {
        let mut words = [0u64; 3];
        set_bit_range(&mut words, 0, 1);
        assert_eq!(words[0], 1);
        set_bit_range(&mut words, 63, 64);
        assert_eq!(words[0], 1 | (1 << 63));
        set_bit_range(&mut words, 64, 130);
        assert_eq!(words[1], !0u64);
        assert_eq!(words[2], 0b11);
    }

    #[test]
    fn runs_round_trip_through_a_bitset() {
        // A tiny synthetic table: only (b=0, g=0, r=1) and the last entry.
        let mut values = vec![false; MASK_ELEMENTS];
        values[1] = true;
        values[MASK_ELEMENTS - 1] = true;
        let table = LookupTable::from_bool_slice(&values, 1);
        assert!(table.contains(0, 0, 1));
        assert!(table.contains(255, 255, 255));
        assert!(!table.contains(0, 0, 0));
        assert_eq!(table.true_entries(), 2);
        assert!(table.get_index(1));
    }

    #[test]
    fn run_lengths_must_cover_the_whole_table() {
        let error = LookupTable::from_runs("test.bin", &[10, 20], 5).expect_err("must reject");
        assert!(matches!(error, Error::Asset { .. }));
    }

    #[test]
    fn run_lengths_may_not_overflow() {
        let error = LookupTable::from_runs("test.bin", &[0, u32::MAX, u32::MAX], 5)
            .expect_err("must reject");
        assert!(matches!(error, Error::Asset { .. }));
    }

    #[test]
    fn an_impossible_min_spread_is_rejected() {
        let error = LookupTable::from_runs("test.bin", &[MASK_ELEMENTS as u32], 300)
            .expect_err("must reject");
        assert!(matches!(error, Error::Asset { .. }));
    }
}
