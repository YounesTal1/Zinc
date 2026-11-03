//! The C1 ring relation and the public weights of its random-combination
//! check.
//!
//! Relation: `N = 8192`, three primes `q_j ≡ 1 mod 2N`,
//! `s ∈ {-1,0,1}^N`, `e_ek ∈ [-20, 20]^N`, `pk_j + a_j·s − e_ek ∈ (X^N + 1)`
//! in `F_{q_j}[X]`.
//!
//! Check: with `ω_i = ζ^{2i+1}` the roots of `X^N + 1` in `F_{q_j}` and a
//! Fiat–Shamir `β`, `Σ_i β^i · P_j(ω_i) ≡ 0 mod q_j` where
//! `P_j = pk_j + a_j·s − e_ek`. Expanding,
//! `Σ_m pk_{j,m}·U_m + s_m·V_{j,m} − e_m·U_m ≡ 0` with
//! `U_m = Σ_i β^i ω_i^m`, `V_{j,m} = Σ_i β^i a_j(ω_i) ω_i^m` — all public.
//! A nonzero `P_j mod (X^N+1)` has a nonzero evaluation vector, so the
//! combination vanishes for at most `N−1` values of `β`.

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub const N: usize = 8192;
pub const LOG_N: usize = 13;
pub const NUM_PRIMES: usize = 3;
/// interfold `configs/secure/threshold.nr` QIS (N = 8192, L = 3).
pub const PRIMES: [u64; NUM_PRIMES] = [576460752317792257, 576460752317399041, 576460752316776449];
/// Independent β's per prime: soundness of the ring check is
/// `(2(N−1)/q_j)^BETAS_PER_PRIME` ≈ 2^-90 per grinding attempt.
pub const BETAS_PER_PRIME: usize = 2;
pub const NUM_ACC: usize = NUM_PRIMES * BETAS_PER_PRIME;

#[inline]
pub fn mulmod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}
#[inline]
pub fn addmod(a: u64, b: u64, q: u64) -> u64 {
    let s = a + b;
    if s >= q { s - q } else { s }
}
#[inline]
pub fn submod(a: u64, b: u64, q: u64) -> u64 {
    if a >= b { a - b } else { a + q - b }
}
pub fn powmod(mut a: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1u64;
    a %= q;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, q);
        }
        a = mulmod(a, a, q);
        e >>= 1;
    }
    r
}
pub fn invmod(a: u64, q: u64) -> u64 {
    powmod(a, q - 2, q)
}

/// A primitive `2N`-th root of unity mod `q` (deterministic search).
pub fn root_2n(q: u64) -> u64 {
    root_2n_for(q, N)
}

/// A primitive `2n`-th root of unity mod `q`.
pub fn root_2n_for(q: u64, n: usize) -> u64 {
    assert_eq!((q - 1) % (2 * n as u64), 0);
    let e = (q - 1) / (2 * n as u64);
    for x in 2u64.. {
        let z = powmod(x, e, q);
        if powmod(z, n as u64, q) == q - 1 {
            return z;
        }
    }
    unreachable!()
}

/// In-place iterative NTT of length `N` with `root` of order `N`
/// (`out[i] = Σ_m in[m] root^{im}`).
pub fn ntt(v: &mut [u64], root: u64, q: u64) {
    let n = v.len();
    assert!(n.is_power_of_two());
    // bit reversal
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            v.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let wlen = powmod(root, (n / len) as u64, q);
        for start in (0..n).step_by(len) {
            let mut w = 1u64;
            for k in 0..len / 2 {
                let u = v[start + k];
                let t = mulmod(v[start + k + len / 2], w, q);
                v[start + k] = addmod(u, t, q);
                v[start + k + len / 2] = submod(u, t, q);
                w = mulmod(w, wlen, q);
            }
        }
        len <<= 1;
    }
}

/// Negacyclic forward transform: `â_i = a(ω_i)`, `ω_i = ζ^{2i+1}`.
pub fn nega_ntt(a: &[u64], zeta: u64, q: u64) -> Vec<u64> {
    let mut v: Vec<u64> = a.iter().enumerate().map(|(m, &x)| mulmod(x, powmod(zeta, m as u64, q), q)).collect();
    ntt(&mut v, mulmod(zeta, zeta, q), q);
    v
}

/// Inverse of [`nega_ntt`].
pub fn nega_intt(ahat: &[u64], zeta: u64, q: u64) -> Vec<u64> {
    let n = ahat.len() as u64;
    let mut v = ahat.to_vec();
    let root_inv = invmod(mulmod(zeta, zeta, q), q);
    ntt(&mut v, root_inv, q);
    let ninv = invmod(n, q);
    let zinv = invmod(zeta, q);
    for (m, x) in v.iter_mut().enumerate() {
        *x = mulmod(mulmod(*x, ninv, q), powmod(zinv, m as u64, q), q);
    }
    v
}

#[derive(Clone)]
pub struct Witness {
    pub s: Vec<i64>,    // {-1,0,1}
    pub e_ek: Vec<i64>, // CBD variance 10, in [-20, 20]
    pub pk: [Vec<u64>; NUM_PRIMES],
}

pub struct Public {
    pub a: [Vec<u64>; NUM_PRIMES],
}

pub fn sample_a(seed: u64) -> Public {
    sample_a_n(seed, N)
}

pub fn sample_a_n(seed: u64, n: usize) -> Public {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let a = core::array::from_fn(|j| (0..n).map(|_| rng.random_range(0..PRIMES[j])).collect());
    Public { a }
}

pub fn sample_witness(public: &Public, seed: u64) -> Witness {
    let n = public.a[0].len();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let s: Vec<i64> = (0..n).map(|_| rng.random_range(-1..=1i64)).collect();
    let e_ek: Vec<i64> = (0..n)
        .map(|_| {
            // centered binomial, variance 10: Σ_{i<20} (b_i − b'_i)
            let bits = rng.next_u64();
            let mut v = 0i64;
            for i in 0..20 {
                v += ((bits >> i) & 1) as i64 - ((bits >> (20 + i)) & 1) as i64;
            }
            v
        })
        .collect();
    let pk = core::array::from_fn(|j| compute_pk(&public.a[j], &s, &e_ek, PRIMES[j]));
    Witness { s, e_ek, pk }
}

/// `pk = −a·s + e mod (q, X^N+1)`.
pub fn compute_pk(a: &[u64], s: &[i64], e: &[i64], q: u64) -> Vec<u64> {
    let zeta = root_2n_for(q, a.len());
    let sq: Vec<u64> = s.iter().map(|&x| x.rem_euclid(q as i64) as u64).collect();
    let ahat = nega_ntt(a, zeta, q);
    let shat = nega_ntt(&sq, zeta, q);
    let prod: Vec<u64> = ahat.iter().zip(&shat).map(|(x, y)| mulmod(*x, *y, q)).collect();
    let as_poly = nega_intt(&prod, zeta, q);
    as_poly
        .iter()
        .zip(e)
        .map(|(&as_m, &e_m)| submod(e_m.rem_euclid(q as i64) as u64, as_m, q))
        .collect()
}

/// Reference check of the relation (coefficient-wise).
pub fn relation_holds(public: &Public, w: &Witness) -> bool {
    (0..NUM_PRIMES).all(|j| {
        let q = PRIMES[j];
        let expected = compute_pk(&public.a[j], &w.s, &w.e_ek, q);
        expected == w.pk[j]
    })
}

/// Public weights of one `(prime, β)` instance.
pub struct Weights {
    pub q: u64,
    pub u: Vec<u64>, // U_m
    pub v: Vec<u64>, // V_m (includes a_j)
}

pub fn weights(a: &[u64], q: u64, beta: u64) -> Weights {
    let n = a.len();
    let zeta = root_2n_for(q, n);
    let zeta2 = mulmod(zeta, zeta, q);
    let beta_n = powmod(beta, n as u64, q);
    let one_minus_beta_n = submod(1, beta_n, q);
    // U_m = ζ^m · (1 − β^n) / (1 − β ζ^{2m})  (β ζ^{2m} ≠ 1), else n·ζ^m
    let mut u = Vec::with_capacity(n);
    let mut zm = 1u64;
    let mut z2m = 1u64;
    for _ in 0..n {
        let d = submod(1, mulmod(beta, z2m, q), q);
        let um = if d == 0 {
            mulmod(n as u64 % q, zm, q)
        } else {
            mulmod(zm, mulmod(one_minus_beta_n, invmod(d, q), q), q)
        };
        u.push(um);
        zm = mulmod(zm, zeta, q);
        z2m = mulmod(z2m, zeta2, q);
    }
    // V_m = ζ^m Σ_i (β^i â_i) (ζ^2)^{im}
    let ahat = nega_ntt(a, zeta, q);
    let mut c: Vec<u64> = Vec::with_capacity(n);
    let mut bi = 1u64;
    for i in 0..n {
        c.push(mulmod(bi, ahat[i], q));
        bi = mulmod(bi, beta, q);
    }
    ntt(&mut c, zeta2, q);
    let mut zm = 1u64;
    for x in c.iter_mut() {
        *x = mulmod(*x, zm, q);
        zm = mulmod(zm, zeta, q);
    }
    Weights { q, u, v: c }
}

/// Reference evaluation of the combination `Σ_m pk_m U_m + s_m V_m − e_m U_m mod q`.
pub fn combination(wt: &Weights, pk: &[u64], s: &[i64], e: &[i64]) -> u64 {
    let q = wt.q;
    let mut acc = 0u64;
    for m in 0..pk.len() {
        acc = addmod(acc, mulmod(pk[m], wt.u[m], q), q);
        acc = addmod(acc, mulmod(s[m].rem_euclid(q as i64) as u64, wt.v[m], q), q);
        acc = submod(acc, mulmod(e[m].rem_euclid(q as i64) as u64, wt.u[m], q), q);
    }
    acc
}

/// Fiat–Shamir β's from the public commitments (any injective encoding).
pub fn derive_betas(domain: &[u8], commitments: &[&[u8]]) -> [[u64; BETAS_PER_PRIME]; NUM_PRIMES] {
    let mut out = [[0u64; BETAS_PER_PRIME]; NUM_PRIMES];
    for j in 0..NUM_PRIMES {
        for b in 0..BETAS_PER_PRIME {
            let mut h = blake3::Hasher::new();
            h.update(domain);
            h.update(&[j as u8, b as u8]);
            for c in commitments {
                h.update(&(c.len() as u64).to_le_bytes());
                h.update(c);
            }
            let bytes = h.finalize();
            let x = u128::from_le_bytes(bytes.as_bytes()[..16].try_into().unwrap());
            out[j][b] = (x % PRIMES[j] as u128) as u64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntt_roundtrip_and_relation() {
        let public = sample_a(1);
        let w = sample_witness(&public, 2);
        assert!(relation_holds(&public, &w));
        for j in 0..NUM_PRIMES {
            let q = PRIMES[j];
            let zeta = root_2n(q);
            let back = nega_intt(&nega_ntt(&public.a[j], zeta, q), zeta, q);
            assert_eq!(back, public.a[j]);
        }
    }

    #[test]
    fn combination_vanishes_iff_relation() {
        let public = sample_a(3);
        let mut w = sample_witness(&public, 4);
        let betas = derive_betas(b"test", &[b"x"]);
        for j in 0..NUM_PRIMES {
            for b in 0..BETAS_PER_PRIME {
                let wt = weights(&public.a[j], PRIMES[j], betas[j][b]);
                assert_eq!(combination(&wt, &w.pk[j], &w.s, &w.e_ek), 0);
            }
        }
        w.pk[1][77] = addmod(w.pk[1][77], 1, PRIMES[1]);
        let wt = weights(&public.a[1], PRIMES[1], betas[1][0]);
        assert_ne!(combination(&wt, &w.pk[1], &w.s, &w.e_ek), 0);
        // brute-force the weights definition on one β for a few m
        let q = PRIMES[0];
        let zeta = root_2n(q);
        let beta = betas[0][0];
        let wt0 = weights(&public.a[0], q, beta);
        let ahat = nega_ntt(&public.a[0], zeta, q);
        for &m in &[0usize, 1, 5, 4097, N - 1] {
            let (mut u, mut v, mut bi) = (0u64, 0u64, 1u64);
            for i in 0..N {
                let om = powmod(zeta, (2 * i + 1) as u64 * m as u64 % (2 * N as u64), q);
                u = addmod(u, mulmod(bi, om, q), q);
                v = addmod(v, mulmod(mulmod(bi, ahat[i], q), om, q), q);
                bi = mulmod(bi, beta, q);
            }
            assert_eq!(wt0.u[m], u, "U_{m}");
            assert_eq!(wt0.v[m], v, "V_{m}");
        }
    }
}
