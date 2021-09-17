// Copyright 2014-2016 Johannes Köster, Taylor Cramer.
// Licensed under the MIT license (http://opensource.org/licenses/MIT)
// This file may not be copied, modified, or distributed
// except according to those terms.

//! The [Burrows-Wheeler-Transform](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.37.6774) and related data structures.
//! The implementation is based on the lecture notes
//! "Algorithmen auf Sequenzen", Kopczynski, Marschall, Martin and Rahmann, 2008 - 2015.

use std::iter::repeat;

use crate::alphabets::Alphabet;
use crate::data_structures::suffix_array::RawSuffixArraySlice;
use crate::utils::prescan;

pub type BWT = Vec<u8>;
pub type BWTSlice = [u8];
pub type Less = Vec<usize>;
pub type BWTFind = Vec<usize>;

/// Calculate Burrows-Wheeler-Transform of the given text of length n.
/// Complexity: O(n).
///
/// # Arguments
///
/// * `text` - the text ended by sentinel symbol (being lexicographically smallest)
/// * `pos` - the suffix array for the text
///
/// # Example
///
/// ```
/// use bio::data_structures::bwt::bwt;
/// use bio::data_structures::suffix_array::suffix_array;
/// let text = b"GCCTTAACATTATTACGCCTA$";
/// let pos = suffix_array(text);
/// let bwt = bwt(text, &pos);
/// assert_eq!(bwt, b"ATTATTCAGGACCC$CTTTCAA");
/// ```
pub fn bwt(text: &[u8], pos: RawSuffixArraySlice) -> BWT {
    assert_eq!(text.len(), pos.len());
    let n = text.len();
    let mut bwt: BWT = vec![0u8; (n + 31) / 32 * 32];
    unsafe {
        bwt.set_len(n);
    }
    for r in 0..n {
        let p = pos[r];
        bwt[r] = if p > 0 { text[p - 1] } else { text[n - 1] };
    }

    bwt
}

/// Calculate the inverse of a BWT of length n, which is the original text.
/// Complexity: O(n).
///
/// This only works if the last sentinel in the original text is unique
/// and lexicographically the smallest.
///
/// # Arguments
///
/// * `bwt` - the BWT
pub fn invert_bwt(bwt: &BWTSlice) -> Vec<u8> {
    let alphabet = Alphabet::new(bwt);
    let n = bwt.len();
    let bwtfind = bwtfind(bwt, &alphabet);
    let mut inverse = Vec::with_capacity(n);

    let mut r = bwtfind[0];
    for _ in 0..n {
        r = bwtfind[r];
        inverse.push(bwt[r]);
    }

    inverse
}

const ALPHA: [u8; 6] = [b'$', b'A', b'C', b'G', b'N', b'T'];
const MAP: [u8; 128] = {
    let mut m = [0u8; 128];
    m[b'$' as usize] = 0;
    m[b'A' as usize] = 1;
    m[b'C' as usize] = 2;
    m[b'G' as usize] = 3;
    m[b'N' as usize] = 4;
    m[b'T' as usize] = 5;
    m
};

/// An occurrence array implementation.
#[derive(Clone, Serialize, Deserialize)]
pub struct Occ {
    occ: Vec<usize>,
    k: u32,
}

impl Occ {
    /// Calculate occ array with sampling from BWT of length n.
    /// Time complexity: O(n).
    /// Space complexity: O(n / k * A) with A being the alphabet size.
    /// The specified alphabet must match the alphabet of the text and its BWT.
    /// For large texts, it is advisable to transform
    /// the text before calculating the BWT (see alphabets::rank_transform).
    ///
    /// # Arguments
    ///
    /// * `bwt` - the BWT
    /// * `k` - the sampling rate: every k-th entry will be stored
    pub fn new(bwt: &BWTSlice, k: u32) -> Self {
        let n = bwt.len();
        let mut occ = Vec::with_capacity(n * ALPHA.len());
        let mut curr_occ = vec![0usize; ALPHA.len()];

        for (i, &c) in bwt.iter().enumerate() {
            let c_map = MAP[c as usize];
            curr_occ[c_map as usize] += 1;

            if i % k as usize == 0 {
                occ.extend_from_slice(&curr_occ);
            }
        }

        assert_eq!(k.count_ones(), 1);
        Occ {
            occ,
            k: k.trailing_zeros(),
        }
    }

    /// Get occurrence count of symbol a in BWT[..r+1].
    /// Complexity: O(k).
    pub fn get(&self, bwt: &BWTSlice, r: usize, a: u8) -> usize {
        // NOTE:
        //
        // Retrieving byte match counts in this function is critical to the performance of FM Index.
        //
        // The below manual count code is roughly equivalent to:
        // ```
        // let count = bwt[(i * self.k) + 1..r + 1]
        //     .iter()
        //     .filter(|&&c| c == a)
        //     .count();
        // self.occ[a as usize][i] + count
        // ```
        //
        // But there are a couple of reasons to do this manually:
        // 1) As of 2016, versions of rustc/LLVM vectorize this manual loop more reliably
        //    than the iterator adapter version.
        // 2) Manually accumulating the byte match count in a single chunk can allows
        //    us to use a `u32` for that count, which has faster arithmetic on common arches.
        //    This does necessitate storing `k` as a u32.
        //
        // See the conversation in these issues for some of the history here:
        //
        // https://github.com/rust-bio/rust-bio/pull/74
        // https://github.com/rust-bio/rust-bio/pull/76
        unsafe {
            let a_map = *MAP.as_ptr().add(a as usize);

            // self.k is our sampling rate, so find the checkpoints either side of r.
            let lo_checkpoint = r >> (self.k as usize);
            // Get the occurences at the low checkpoint
            let lo_occ = *self
                .occ
                .as_ptr()
                .add((a_map as usize) + lo_checkpoint * ALPHA.len());

            /*// If the sampling rate is infrequent it is worth checking if there is a closer
            // hi checkpoint.
            if self.k > 6 {
                // 64
                let hi_checkpoint = lo_checkpoint + 1;
                if let Some(&hi_occ) = self.occ.get((a_map as usize) + hi_checkpoint * ALPHA.len())
                {
                    // Its possible that there are no occurences between the low and high
                    // checkpoint in which case we bail early.
                    if lo_occ == hi_occ {
                        return lo_occ;
                    }

                    // If r is closer to the high checkpoint, count backwards from there.
                    let hi_idx = hi_checkpoint << (self.k as usize);
                    if (hi_idx - r) < ((self.k as usize) >> 1) {
                        return hi_occ
                            - bytecount::count(slice_unchecked(bwt, r + 1, hi_idx + 1), a)
                                as usize;
                    }
                }
            }*/

            // Otherwise the default case is to count from the low checkpoint.
            let lo_idx = lo_checkpoint << (self.k as usize);
            count_bytes(bwt.as_ptr(), lo_idx + 1, r + 1, a) + lo_occ
        }
    }

    // Prefetch into cache.
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch(&self, i: usize) {
        use std::arch::x86_64;

        let lo_checkpoint = i >> (self.k as usize);
        unsafe {
            let lo_occ = self.occ.as_ptr().add(lo_checkpoint * ALPHA.len());
            x86_64::_mm_prefetch::<{ x86_64::_MM_HINT_T1 }>(lo_occ as *const i8);
        }
    }
}

static MASK: [u8; 64] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

static L: usize = 32;

#[target_feature(enable = "avx2")]
unsafe fn count_bytes(ptr: *const u8, lo: usize, hi: usize, c: u8) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_si256();
    let cv = _mm256_set1_epi8(c as i8);
    let mut i = lo;

    while i + L <= hi {
        acc = _mm256_sub_epi8(
            acc,
            _mm256_cmpeq_epi8(_mm256_loadu_si256(ptr.add(i) as _), cv),
        );
        i += L;
    }

    let rem = (hi - lo) % L;

    if rem != 0 {
        let v = _mm256_loadu_si256(ptr.add(i) as _);
        let mask = _mm256_loadu_si256(MASK.as_ptr().add(L - rem) as _);
        acc = _mm256_sub_epi8(acc, _mm256_and_si256(mask, _mm256_cmpeq_epi8(v, cv)));
    }

    acc = _mm256_sad_epu8(acc, _mm256_setzero_si256());

    #[repr(align(32))]
    struct A([u64; 4]);
    let mut a = A([0; 4]);
    _mm256_store_si256(a.0.as_mut_ptr() as _, acc);
    (a.0[0] + a.0[1] + a.0[2] + a.0[3]) as usize
}

/// Calculate the less array for a given BWT. Complexity O(n).
pub fn less(bwt: &BWTSlice, alphabet: &Alphabet) -> Less {
    let m = alphabet
        .max_symbol()
        .expect("Expecting non-empty alphabet.") as usize
        + 2;
    let mut less: Less = repeat(0).take(m).collect();
    for &c in bwt.iter() {
        less[c as usize] += 1;
    }
    // calculate +-prescan
    prescan(&mut less[..], 0, |a, b| a + b);

    less
}

/// Calculate the bwtfind array needed for inverting the BWT. Complexity O(n).
pub fn bwtfind(bwt: &BWTSlice, alphabet: &Alphabet) -> BWTFind {
    let n = bwt.len();
    let mut less = less(bwt, alphabet);

    let mut bwtfind: BWTFind = repeat(0).take(n).collect();
    for (r, &c) in bwt.iter().enumerate() {
        bwtfind[less[c as usize]] = r;
        less[c as usize] += 1;
    }

    bwtfind
}

#[cfg(test)]
mod tests {
    use super::{bwt, bwtfind, invert_bwt, Occ};
    use crate::alphabets::dna;
    use crate::alphabets::Alphabet;
    use crate::data_structures::suffix_array::suffix_array;
    use crate::data_structures::wavelet_matrix::WaveletMatrix;

    #[test]
    fn test_bwtfind() {
        let text = b"cabca$";
        let alphabet = Alphabet::new(b"abc$");
        let pos = suffix_array(text);
        let bwt = bwt(text, &pos);
        let bwtfind = bwtfind(&bwt, &alphabet);
        assert_eq!(bwtfind, vec![5, 0, 3, 4, 1, 2]);
    }

    #[test]
    fn test_invert_bwt() {
        let text = b"cabca$";
        let pos = suffix_array(text);
        let bwt = bwt(text, &pos);
        let inverse = invert_bwt(&bwt);
        assert_eq!(inverse, text);
    }

    #[test]
    fn test_occ() {
        let bwt = vec![1u8, 3u8, 3u8, 1u8, 2u8, 0u8];
        let alphabet = Alphabet::new(&[0u8, 1u8, 2u8, 3u8]);
        let occ = Occ::new(&bwt, 3, &alphabet);
        assert_eq!(occ.occ, [[0, 0], [1, 2], [0, 0], [0, 2]]);
        assert_eq!(occ.get(&bwt, 4, 2u8), 1);
        assert_eq!(occ.get(&bwt, 4, 3u8), 2);
    }

    #[test]
    fn test_occwm() {
        let text = b"GCCTTAACATTATTACGCCTA$";
        let alphabet = {
            let mut a = dna::n_alphabet();
            a.insert(b'$');
            a
        };
        let sa = suffix_array(text);
        let bwt = bwt(text, &sa);
        let occ = Occ::new(&bwt, 3, &alphabet);
        let wm = WaveletMatrix::new(&bwt);

        for c in vec![b'A', b'C', b'G', b'T', b'$'] {
            for p in 0..text.len() {
                assert_eq!(occ.get(&bwt, p, c) as u64, wm.rank(c, p as u64));
            }
        }
    }
}
