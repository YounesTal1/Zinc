//! Poseidon2 over KoalaBear (p = 2^31 − 2^24 + 1), width 16, x^3 S-box,
//! 8 full + 20 partial rounds — the Plonky3 instance
//! (`p3_koala_bear::default_koalabear_poseidon2_16`), with Plonky3's round
//! constants and linear layers. A plain `u64` implementation is kept here
//! so that the trace builder and the constraint builder share one
//! description; `tests::matches_plonky3` pins it to Plonky3.
//!
//! Sponge (fixed-length, no padding): state = 0; for each block of `rate`
//! elements, `state[i] += block[i]`, then permute; digest = `state[0..8]`.

use p3_field::PrimeField32;
use p3_koala_bear::{
    KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL, KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    KOALABEAR_POSEIDON2_RC_16_INTERNAL, KoalaBear,
};
use std::sync::OnceLock;

pub const Q: u64 = 2_130_706_433; // 2^31 - 2^24 + 1
pub const WIDTH: usize = 16;
pub const HALF_FULL: usize = 4;
pub const PARTIAL: usize = 20;
pub const DIGEST_LANES: usize = 8;

/// `M4 = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]` (Plonky3 `MDSMat4`).
pub const M4: [[u64; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];

/// Internal diagonal `V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8,
/// 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]` as canonical residues mod Q.
pub fn internal_diag() -> [u64; WIDTH] {
    let inv2 = inv(2);
    let inv8 = inv(8);
    let inv16 = inv(16);
    let inv256 = inv(256);
    let inv2_24 = inv(1 << 24);
    [
        Q - 2,
        1,
        2,
        inv2,
        3,
        4,
        Q - inv2,
        Q - 3,
        Q - 4,
        inv256,
        inv8,
        inv2_24,
        Q - inv256,
        Q - inv8,
        Q - inv16,
        Q - inv2_24,
    ]
}

pub fn mulq(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % Q as u128) as u64
}
pub fn addq(a: u64, b: u64) -> u64 {
    (a + b) % Q
}
pub fn powq(mut a: u64, mut e: u64) -> u64 {
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = mulq(r, a);
        }
        a = mulq(a, a);
        e >>= 1;
    }
    r
}
pub fn inv(a: u64) -> u64 {
    powq(a % Q, Q - 2)
}

pub struct Constants {
    pub rc_initial: [[u64; WIDTH]; HALF_FULL],
    pub rc_final: [[u64; WIDTH]; HALF_FULL],
    pub rc_internal: [u64; PARTIAL],
    pub diag: [u64; WIDTH],
}

pub fn constants() -> &'static Constants {
    static C: OnceLock<Constants> = OnceLock::new();
    C.get_or_init(|| {
        let cv = |x: KoalaBear| x.as_canonical_u32() as u64;
        let mut rc_initial = [[0u64; WIDTH]; HALF_FULL];
        let mut rc_final = [[0u64; WIDTH]; HALF_FULL];
        for r in 0..HALF_FULL {
            for i in 0..WIDTH {
                rc_initial[r][i] = cv(KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL[r][i]);
                rc_final[r][i] = cv(KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL[r][i]);
            }
        }
        let mut rc_internal = [0u64; PARTIAL];
        for r in 0..PARTIAL {
            rc_internal[r] = cv(KOALABEAR_POSEIDON2_RC_16_INTERNAL[r]);
        }
        Constants { rc_initial, rc_final, rc_internal, diag: internal_diag() }
    })
}

/// External linear layer `M_E = circ(2·M4, M4, M4, M4)` on reduced inputs.
pub fn external_layer(state: &mut [u64; WIDTH]) {
    let mut out = [0u64; WIDTH];
    for c in 0..4 {
        for r in 0..4 {
            let mut acc = 0u64;
            for k in 0..4 {
                acc += M4[r][k] * state[4 * c + k];
            }
            out[4 * c + r] = acc % Q;
        }
    }
    let mut sums = [0u64; 4];
    for c in 0..4 {
        for r in 0..4 {
            sums[r] = (sums[r] + out[4 * c + r]) % Q;
        }
    }
    for c in 0..4 {
        for r in 0..4 {
            state[4 * c + r] = (out[4 * c + r] + sums[r]) % Q;
        }
    }
}

/// Internal linear layer `M_I = J + diag(V)`.
pub fn internal_layer(state: &mut [u64; WIDTH], diag: &[u64; WIDTH]) {
    let mut sum = 0u64;
    for v in state.iter() {
        sum = addq(sum, *v);
    }
    for i in 0..WIDTH {
        state[i] = addq(sum, mulq(diag[i], state[i]));
    }
}

pub fn sbox(x: u64) -> u64 {
    mulq(mulq(x, x), x)
}

pub fn permute(state: &mut [u64; WIDTH]) {
    let c = constants();
    external_layer(state);
    for r in 0..HALF_FULL {
        for i in 0..WIDTH {
            state[i] = sbox(addq(state[i], c.rc_initial[r][i]));
        }
        external_layer(state);
    }
    for r in 0..PARTIAL {
        state[0] = sbox(addq(state[0], c.rc_internal[r]));
        internal_layer(state, &c.diag);
    }
    for r in 0..HALF_FULL {
        for i in 0..WIDTH {
            state[i] = sbox(addq(state[i], c.rc_final[r][i]));
        }
        external_layer(state);
    }
}

/// Fixed-length sponge: `elems.len()` must be a multiple of `rate`.
pub fn sponge(elems: &[u64], rate: usize) -> [u64; DIGEST_LANES] {
    assert!(rate < WIDTH && elems.len() % rate == 0);
    let mut state = [0u64; WIDTH];
    for block in elems.chunks(rate) {
        for (i, e) in block.iter().enumerate() {
            assert!(*e < Q);
            state[i] = addq(state[i], *e);
        }
        permute(&mut state);
    }
    let mut d = [0u64; DIGEST_LANES];
    d.copy_from_slice(&state[..DIGEST_LANES]);
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::default_koalabear_poseidon2_16;
    use p3_symmetric::Permutation;

    #[test]
    fn matches_plonky3() {
        let p3 = default_koalabear_poseidon2_16();
        let mut rng_state = 12345u64;
        for _ in 0..20 {
            let mut input = [0u64; WIDTH];
            for v in input.iter_mut() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *v = (rng_state >> 11) % Q;
            }
            let p3_in: [KoalaBear; WIDTH] = core::array::from_fn(|i| KoalaBear::from_u64(input[i]));
            let p3_out = p3.permute(p3_in);
            let mut mine = input;
            permute(&mut mine);
            for i in 0..WIDTH {
                assert_eq!(mine[i], p3_out[i].as_canonical_u32() as u64, "lane {i}");
            }
        }
    }
}
