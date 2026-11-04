//! interfold DKG data for C2/C3 (secure preset, micro committee):
//! - threshold primes `q_j` (from `ring.rs`) and the micro-committee
//!   Shamir parity matrix (`configs/committee/micro/parity_secure.nr`:
//!   N_PARTIES = 9, T = 4, points 0..9, H = [-A | I]);
//! - the second BFV instance (`configs/secure/dkg.nr`): N = 8192, L = 2,
//!   q' = 2305843009214758913 / 2305843009214988289, t = 576460752317792257
//!   (= threshold q_0), k0_i = -t^{-1} mod q'_i, Q mod t = 3169161947774985;
//! - reference share generation, BFV key generation and encryption in the
//!   exact form C3 checks: `ct0 ≡ pk0·u + e0 + k0·k1`, `ct1 ≡ pk1·u + e1`
//!   (mod q'_i, X^N+1), `k1 = q_mod_t·m − t·z ∈ [0, t)`, `u` ternary,
//!   `e0, e1 ∈ [−20, 20]`.

use crate::ring::{self, PRIMES as QT, addmod, mulmod};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub const N_PARTIES: usize = 9;
pub const T: usize = 4;
pub const PARITY_ROWS: usize = N_PARTIES - T; // 5
pub const L_T: usize = 3;
pub const L_D: usize = 2;
pub const QD: [u64; L_D] = [2305843009214758913, 2305843009214988289];
pub const PT: u64 = 576460752317792257;
pub const Q_MOD_T: u64 = 3169161947774985;
pub const K0: [u64; L_D] = [493937176208329543, 1434253984466648362];
pub const E_BOUND: i64 = 20;

/// `H[j][row][col]`: rows `[-A | I]` with the C(5,k) alternating binomial pattern.
pub fn parity_matrix() -> [[[u64; N_PARTIES + 1]; PARITY_ROWS]; L_T] {
    core::array::from_fn(|j| {
        let q = QT[j];
        // Row r expresses y_{5+r} via y_0..y_4 by Lagrange extrapolation of a degree-T
        // polynomial through the points 0..T: y_{x} = Σ_k y_k · Π_{l≠k} (x − l)/(k − l), evaluated at
        // x = 5 + r. The basis values are integers (the repo stores them as such); the row is
        // `[-lagrange(x) | e_r]` so that H·y = 0.
        let lagrange = |x: i128| -> [i64; T + 1] {
            core::array::from_fn(|k| {
                let (mut num, mut den) = (1i128, 1i128);
                for l in 0..=T {
                    if l != k {
                        num *= x - l as i128;
                        den *= k as i128 - l as i128;
                    }
                }
                assert_eq!(num % den, 0);
                (num / den) as i64
            })
        };
        core::array::from_fn(|r| {
            let coeffs = lagrange((T + 1 + r) as i128);
            let mut row = [0u64; N_PARTIES + 1];
            for k in 0..=T {
                row[k] = (-coeffs[k]).rem_euclid(q as i64) as u64;
            }
            row[T + 1 + r] = 1;
            row
        })
    })
}

/// Shamir shares of `secret` (already reduced mod `q`): `y[0] = secret`,
/// `y[p] = f(p)` for a random degree-T polynomial.
pub fn shamir<R: RngCore>(secret: u64, q: u64, rng: &mut R) -> [u64; N_PARTIES + 1] {
    let mut coeffs = [0u64; T + 1];
    coeffs[0] = secret % q;
    for c in coeffs.iter_mut().skip(1) {
        *c = rng.random_range(0..q);
    }
    core::array::from_fn(|p| {
        let x = p as u64;
        let mut acc = 0u64;
        for c in coeffs.iter().rev() {
            acc = addmod(mulmod(acc, x, q), *c, q);
        }
        acc
    })
}

pub fn parity_holds(y: &[u64; N_PARTIES + 1], h: &[[u64; N_PARTIES + 1]; PARITY_ROWS], q: u64) -> bool {
    h.iter().all(|row| {
        let mut s = 0u64;
        for k in 0..=N_PARTIES {
            s = addmod(s, mulmod(row[k], y[k], q), q);
        }
        s == 0
    })
}

/// A DKG BFV public key `(pk0, pk1)` per modulus: `pk1 = a'` uniform,
/// `pk0 = −(a'·s' + e')`, `s'` ternary, `e'` CBD.
#[derive(Clone)]
pub struct DkgPk {
    pub pk0: [Vec<u64>; L_D],
    pub pk1: [Vec<u64>; L_D],
}

pub fn cbd20<R: RngCore>(rng: &mut R) -> i64 {
    let bits = rng.next_u64();
    let mut v = 0i64;
    for i in 0..20 {
        v += ((bits >> i) & 1) as i64 - ((bits >> (20 + i)) & 1) as i64;
    }
    v
}

pub fn keygen(n: usize, seed: u64) -> DkgPk {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let s: Vec<i64> = (0..n).map(|_| rng.random_range(-1..=1i64)).collect();
    let e: Vec<i64> = (0..n).map(|_| cbd20(&mut rng)).collect();
    let pk1: [Vec<u64>; L_D] = core::array::from_fn(|i| (0..n).map(|_| rng.random_range(0..QD[i])).collect());
    let pk0: [Vec<u64>; L_D] = core::array::from_fn(|i| {
        // -(a s + e) = compute_pk with e replaced by -e... compute_pk gives -a s + e, so use -e.
        let neg_e: Vec<i64> = e.iter().map(|x| -x).collect();
        ring::compute_pk(&pk1[i], &s, &neg_e, QD[i])
    });
    DkgPk { pk0, pk1 }
}

/// `(k1, z)` with `k1 = q_mod_t·m − t·z ∈ [0, t)`.
pub fn scale_message(m: u64) -> (u64, u64) {
    let prod = Q_MOD_T as u128 * m as u128;
    let z = (prod / PT as u128) as u64;
    let k1 = (prod % PT as u128) as u64;
    (k1, z)
}

#[derive(Clone)]
pub struct Encryption {
    pub u: Vec<i64>,
    pub e0: Vec<i64>,
    pub e1: Vec<i64>,
    pub k1: Vec<u64>,
    pub z: Vec<u64>,
    /// representatives in `[0, q'_i)`
    pub ct0: [Vec<u64>; L_D],
    pub ct1: [Vec<u64>; L_D],
}

/// BFV-encrypt the share polynomial `m` (coefficients in `[0, t)`) under `pk`.
pub fn encrypt<R: RngCore>(pk: &DkgPk, m: &[u64], rng: &mut R) -> Encryption {
    let n = m.len();
    let u: Vec<i64> = (0..n).map(|_| rng.random_range(-1..=1i64)).collect();
    let e0: Vec<i64> = (0..n).map(|_| cbd20(rng)).collect();
    let e1: Vec<i64> = (0..n).map(|_| cbd20(rng)).collect();
    let (k1, z): (Vec<u64>, Vec<u64>) = m.iter().map(|&x| scale_message(x)).unzip();
    let ct0 = core::array::from_fn(|i| {
        let q = QD[i];
        // pk0·u + e0 + k0·k1 : compute_pk(a=pk0, s=u, e=e0) gives -pk0 u + e0 → negate the product part
        let neg_u: Vec<i64> = u.iter().map(|x| -x).collect();
        let base = ring::compute_pk(&pk.pk0[i], &neg_u, &e0, q); // pk0·u + e0
        base.iter().zip(&k1).map(|(&b, &k)| addmod(b, mulmod(K0[i], k % q, q), q)).collect()
    });
    let ct1 = core::array::from_fn(|i| {
        let q = QD[i];
        let neg_u: Vec<i64> = u.iter().map(|x| -x).collect();
        ring::compute_pk(&pk.pk1[i], &neg_u, &e1, q) // pk1·u + e1
    });
    Encryption { u, e0, e1, k1, z, ct0, ct1 }
}

/// Reference check of the C3 identities and bounds (coefficient-wise).
pub fn encryption_holds(pk: &DkgPk, m: &[u64], enc: &Encryption) -> bool {
    let n = m.len();
    if !enc.u.iter().all(|x| x.abs() <= 1) || !enc.e0.iter().all(|x| x.abs() <= E_BOUND) || !enc.e1.iter().all(|x| x.abs() <= E_BOUND) {
        return false;
    }
    if !m.iter().all(|&x| x < PT) {
        return false;
    }
    for j in 0..n {
        let (k1, z) = scale_message(m[j]);
        if enc.k1[j] != k1 || enc.z[j] != z {
            return false;
        }
    }
    for i in 0..L_D {
        let q = QD[i];
        let neg_u: Vec<i64> = enc.u.iter().map(|x| -x).collect();
        let c0 = ring::compute_pk(&pk.pk0[i], &neg_u, &enc.e0, q);
        let c1 = ring::compute_pk(&pk.pk1[i], &neg_u, &enc.e1, q);
        for j in 0..n {
            let want0 = addmod(c0[j], mulmod(K0[i], enc.k1[j] % q, q), q);
            if enc.ct0[i][j] % q != want0 || enc.ct1[i][j] % q != c1[j] {
                return false;
            }
        }
    }
    true
}

pub fn center(x: u64, q: u64) -> i64 {
    if x > (q - 1) / 2 { x as i64 - q as i64 } else { x as i64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        for i in 0..L_D {
            assert_eq!(mulmod(K0[i], PT % QD[i], QD[i]), QD[i] - 1, "k0·t ≡ −1");
            assert_eq!((QD[i] - 1) % 16384, 0);
        }
        assert_eq!(((QD[0] as u128 * QD[1] as u128) % PT as u128) as u64, Q_MOD_T);
        let h = parity_matrix();
        // matches the repo constants for q_0, row 0 and row 4
        assert_eq!(h[0][0][..6], [576460752317792256, 5, 576460752317792247, 10, 576460752317792252, 1]);
        assert_eq!(h[0][4][..5], [576460752317792187, 315, 576460752317791717, 420, 576460752317792131]);
        assert_eq!(h[2][3][..5], [576460752316776414, 160, 576460752316776169, 224, 576460752316776379]);
        assert_eq!(h[1][1][..5], [576460752317399036, 24, 576460752317398996, 40, 576460752317399026]);
    }

    #[test]
    fn shares_satisfy_parity_and_reconstruct() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let h = parity_matrix();
        for j in 0..L_T {
            let q = QT[j];
            for _ in 0..20 {
                let s = rng.random_range(0..q);
                let y = shamir(s, q, &mut rng);
                assert_eq!(y[0], s);
                assert!(parity_holds(&y, &h[j], q));
                let mut bad = y;
                bad[3] = addmod(bad[3], 1, q);
                assert!(!parity_holds(&bad, &h[j], q));
            }
        }
    }

    #[test]
    fn encryption_reference() {
        let n = 64;
        let pk = keygen(n, 3);
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let m: Vec<u64> = (0..n).map(|_| rng.random_range(0..QT[1])).collect();
        let enc = encrypt(&pk, &m, &mut rng);
        assert!(encryption_holds(&pk, &m, &enc));
        let mut bad = enc.clone();
        bad.ct0[0][5] = addmod(bad.ct0[0][5], 1, QD[0]);
        assert!(!encryption_holds(&pk, &m, &bad));
        for j in 0..n {
            assert!(enc.k1[j] < PT && (enc.k1[j] as u128 + PT as u128 * enc.z[j] as u128) == Q_MOD_T as u128 * m[j] as u128);
            assert!(enc.z[j] < (1 << 52));
        }
    }
}
