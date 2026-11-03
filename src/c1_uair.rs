//! The interfold **C1** (`pk_generation`) statement, secure preset
//! (`circuits/lib/src/configs/secure/threshold.nr`: N = 8192, L = 3,
//! QIS = 576460752317792257 / 576460752317399041 / 576460752316776449,
//! EEK_BOUND = 20, SK_BOUND = 1, E_SM_BOUND = 13292279957849975063929256991047181926400000,
//! E_SM_LIMB_BOUNDS = (q_i−1)/2, E_SM_QUOTIENT_BOUNDS = (E_SM_BOUND + (q_i−1)/2)/q_i),
//! as one Zinc+ UAIR with Poseidon2 commitments.
//!
//! Witness (per coefficient m): `sk ∈ {−1,0,1}`, `eek ∈ [−20, 20]`,
//! `pk0[i] ∈ [−(q_i−1)/2, (q_i−1)/2]`, `e_sm_lifted ∈ [−B, B]`,
//! `e_sm[i] ∈ [−(q_i−1)/2, (q_i−1)/2]`, `e_sm_quot[i] ∈ [−Qb_i, Qb_i]`.
//! Constraints: all two-sided range checks (value offset into `[0, 2·bound]`
//! as 16-bit limbs plus a limb decomposition of the complement, as C1's
//! `range_check_2bounds`), `e_sm_lifted = e_sm[i] + e_sm_quot[i]·q_i`, and
//! `pk0[i] + a[i]·sk − eek ≡ 0 mod (q_i, X^N+1)` (C1 checks the equivalent
//! `pk0 = −a·sk + eek + q·r` with a bounded `r` at a Fiat–Shamir point; here
//! the identity is checked mod q_i directly by the random root-of-unity
//! combination of `ring.rs`, so `r` is not needed).
//! Outputs: `commit(sk)`, `commit(pk0)` (3 polynomials), `commit(e_sm)` (the 3
//! residue polynomials, as in C1), plus `cm_aux = commit(eek)` which binds eek
//! before the challenge (C1 absorbs eek into its in-circuit transcript).
//!
//! Hash: Poseidon2-KoalaBear-16 sponge (see `poseidon2.rs`), not C1's
//! BN254 Poseidon/SAFE — same inputs, different hash function.

use crate::circuit::{Builder, CellId, Form, Interp, Op, Val, ValId};
use crate::poseidon2::{self as p2, Q};
use crate::ring::{self, N, NUM_ACC, NUM_PRIMES, PRIMES};
use crypto_primitives::crypto_bigint_int::Int;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::HashMap;
use std::sync::OnceLock;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, LookupColumnSpec, LookupTableType, PublicColumnLayout, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace, ideal::DegreeOneIdeal,
};

pub const IDX_PER_ROW: usize = 8;
pub const ROWS: usize = N / IDX_PER_ROW;
pub const NUM_VARS: usize = 10;
pub const RATE_8: usize = 8;
pub const RATE_12: usize = 12;
pub const DEGREE_PLUS_ONE: usize = 32;

/// Sponges: 0 = sk, 1 = e_sm residues (4 perms), 2 = eek (aux), 3 = pk0 (4 perms).
pub const NUM_SPONGES: usize = 4;
pub const SP_SK: usize = 0;
pub const SP_ESM: usize = 1;
pub const SP_EEK: usize = 2;
pub const SP_PK: usize = 3;

pub const EEK_BOUND: i64 = 20;
pub const E_SM_BOUND_DEC: &str = "13292279957849975063929256991047181926400000";
pub const LIFTED_LIMBS: usize = 10; // 2B < 2^145
pub const QUOT_LIMBS: usize = 6; // 2Qb < 2^86
pub const RES_LIMBS: usize = 4; // 2c < 2^59

pub type ScalarInt = Int<3>;
pub type Scalar = DensePolynomial<ScalarInt, 1>;

pub fn e_sm_bound() -> BigInt {
    E_SM_BOUND_DEC.parse().unwrap()
}
pub fn half(q: u64) -> u64 {
    (q - 1) / 2
}
pub fn quot_bound(q: u64) -> BigInt {
    (e_sm_bound() + BigInt::from(half(q))) / BigInt::from(q)
}

// Public int columns.
pub const PUB_FIRST: usize = 0;
pub const PUB_LAST: usize = 1;
pub const PUB_D: usize = 2;
pub const PUB_W: usize = PUB_D + NUM_SPONGES * p2::DIGEST_LANES * 2; // 66
pub const W_PER_ACC: usize = 2 * IDX_PER_ROW + 1;
pub const NUM_PUB: usize = PUB_W + NUM_ACC * W_PER_ACC; // 168

pub fn pub_u(a: usize, k: usize) -> usize {
    PUB_W + a * W_PER_ACC + k
}
pub fn pub_v(a: usize, k: usize) -> usize {
    PUB_W + a * W_PER_ACC + IDX_PER_ROW + k
}
pub fn pub_c(a: usize) -> usize {
    PUB_W + a * W_PER_ACC + 2 * IDX_PER_ROW
}
pub fn pub_d(sp: usize, lane: usize) -> (usize, usize) {
    let b = PUB_D + (sp * p2::DIGEST_LANES + lane) * 2;
    (b, b + 1)
}

/// A two-sided-bounded value: `off = v + bound ∈ [0, 2·bound]` as 16-bit limbs
/// (`limbs`), and `comp = 2·bound − off` as 16-bit limbs. The first two limb
/// pairs also serve as hash limbs when `hash` is set.
#[derive(Clone, Debug)]
pub struct Bounded {
    pub limbs: Vec<CellId>,
    pub comp: Vec<CellId>,
}

pub struct RowProgram {
    pub b: Builder,
    pub t: [CellId; IDX_PER_ROW],
    pub eek: [(CellId, CellId); IDX_PER_ROW], // (e' = eek+20, 40 − e')
    /// `[k][j]`: pk0 offset `u = pk0 + c_j`: limbs (16,13,16,14) and complement (16,16,16,11)
    pub pk: [[Bounded; NUM_PRIMES]; IDX_PER_ROW],
    pub res: [[Bounded; NUM_PRIMES]; IDX_PER_ROW],
    pub lifted: [Bounded; IDX_PER_ROW],
    pub quot: [[Bounded; NUM_PRIMES]; IDX_PER_ROW],
    pub st: [[(CellId, CellId); p2::WIDTH]; NUM_SPONGES],
    pub outs: [[ValId; p2::WIDTH]; NUM_SPONGES],
    pub acc: [[CellId; 4]; NUM_ACC],
    pub u: [[CellId; 4]; NUM_ACC],
    pub num_cells: usize,
    pub down_index: HashMap<CellId, usize>,
}

pub fn program() -> &'static RowProgram {
    static P: OnceLock<RowProgram> = OnceLock::new();
    P.get_or_init(build_program)
}

/// 59-bit value `[0, 2c]`: hash limbs `L0 = l16 + 2^16 l13` (29 bits), `L1 = h16 + 2^16 h14` (30 bits).
fn bounded_59(b: &mut Builder) -> (Bounded, ValId, ValId) {
    let limbs = vec![b.word(16), b.word(13), b.word(16), b.word(14)];
    let comp = vec![b.word(16), b.word(16), b.word(16), b.word(11)];
    let l0 = b.val_pair(limbs[0], limbs[1]);
    let l1 = b.val_pair(limbs[2], limbs[3]);
    (Bounded { limbs, comp }, l0, l1)
}
fn bounded_w16(b: &mut Builder, n: usize) -> Bounded {
    Bounded { limbs: (0..n).map(|_| b.word(16)).collect(), comp: (0..n).map(|_| b.word(16)).collect() }
}

fn build_program() -> RowProgram {
    let mut b = Builder::default();
    let t: [CellId; 8] = core::array::from_fn(|_| b.cell(2, None));
    let eek: [(CellId, CellId); 8] = core::array::from_fn(|_| (b.word(6), b.word(6)));
    let mut pk_l0 = [[0; 3]; 8];
    let mut pk_l1 = [[0; 3]; 8];
    let pk: [[Bounded; 3]; 8] = core::array::from_fn(|k| {
        core::array::from_fn(|j| {
            let (bd, l0, l1) = bounded_59(&mut b);
            pk_l0[k][j] = l0;
            pk_l1[k][j] = l1;
            bd
        })
    });
    let mut res_l0 = [[0; 3]; 8];
    let mut res_l1 = [[0; 3]; 8];
    let res: [[Bounded; 3]; 8] = core::array::from_fn(|k| {
        core::array::from_fn(|j| {
            let (bd, l0, l1) = bounded_59(&mut b);
            res_l0[k][j] = l0;
            res_l1[k][j] = l1;
            bd
        })
    });
    let lifted: [Bounded; 8] = core::array::from_fn(|_| bounded_w16(&mut b, LIFTED_LIMBS));
    let quot: [[Bounded; 3]; 8] = core::array::from_fn(|_| core::array::from_fn(|_| bounded_w16(&mut b, QUOT_LIMBS)));
    let st: [[(CellId, CellId); 16]; 4] = core::array::from_fn(|_| core::array::from_fn(|_| (b.word(16), b.word(15))));
    let acc: [[CellId; 4]; NUM_ACC] = core::array::from_fn(|_| core::array::from_fn(|_| b.word(16)));
    let u: [[CellId; 4]; NUM_ACC] = core::array::from_fn(|_| core::array::from_fn(|_| b.word(16)));

    let t_val: [ValId; 8] = core::array::from_fn(|k| b.val_cell(t[k]));
    let eek_val: [ValId; 8] = core::array::from_fn(|k| b.val_cell(eek[k].0));
    let st_val: [[ValId; 16]; 4] = core::array::from_fn(|sp| core::array::from_fn(|l| b.val_pair(st[sp][l].0, st[sp][l].1)));

    let absorb = |b: &mut Builder, state: &[ValId; 16], block: &[ValId]| -> [ValId; 16] {
        core::array::from_fn(|l| if l < block.len() { b.val_sum(state[l], block[l]) } else { state[l] })
    };
    let mut outs = [[0; 16]; 4];
    let inp = absorb(&mut b, &st_val[SP_SK], &t_val);
    outs[SP_SK] = b.permutation(inp);
    let inp = absorb(&mut b, &st_val[SP_EEK], &eek_val);
    outs[SP_EEK] = b.permutation(inp);
    for (sp, l0, l1) in [(SP_PK, &pk_l0, &pk_l1), (SP_ESM, &res_l0, &res_l1)] {
        let mut limbs = Vec::with_capacity(48);
        for k in 0..8 {
            for j in 0..3 {
                limbs.push(l0[k][j]);
                limbs.push(l1[k][j]);
            }
        }
        let mut state = st_val[sp];
        for c in 0..4 {
            let inp = absorb(&mut b, &state, &limbs[c * RATE_12..(c + 1) * RATE_12]);
            state = b.permutation(inp);
        }
        outs[sp] = state;
    }

    let mut shifted: Vec<CellId> = Vec::new();
    for sp in 0..4 {
        for l in 0..16 {
            shifted.push(st[sp][l].0);
            shifted.push(st[sp][l].1);
        }
    }
    for a in 0..NUM_ACC {
        shifted.extend(acc[a]);
    }
    shifted.sort_unstable();
    let down_index = shifted.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let num_cells = b.cells.len();
    RowProgram { b, t, eek, pk, res, lifted, quot, st, outs, acc, u, num_cells, down_index }
}

// ---------------------------------------------------------------------------
// Scalars (Int<3>: the C1 bounds are up to 145 bits)
// ---------------------------------------------------------------------------

pub fn scalar_big(v: &BigInt) -> Scalar {
    let mag = v.magnitude();
    assert!(mag.bits() <= 190, "scalar too wide");
    let bytes = mag.to_bytes_le();
    let mut words = [0u64; 3];
    for (i, b) in bytes.iter().enumerate() {
        words[i / 8] |= (*b as u64) << (8 * (i % 8));
    }
    let u = crypto_primitives::crypto_bigint_uint::Uint::<3>::new(crypto_bigint::Uint::<3>::from_words(words));
    let x: ScalarInt = *u.as_int();
    let x = if v.sign() == Sign::Minus { -x } else { x };
    DensePolynomial::new([x])
}
pub fn scalar(v: u64) -> Scalar {
    scalar_big(&BigInt::from(v))
}

pub struct Scalars {
    pub two16: Scalar,
    pub two29: Scalar,
    pub qp: Scalar,
    pub qj: [Scalar; NUM_PRIMES],
    pub one: Scalar,
    pub forty: Scalar,
    pub two_c: [Scalar; NUM_PRIMES],
    pub two_b: Scalar,
    pub two_qb: [Scalar; NUM_PRIMES],
    /// `K_i = B − c_i − Qb_i·q_i` in `off_l − off_r − q_i·off_q = K_i`
    pub crt_k: [Scalar; NUM_PRIMES],
    table: HashMap<u64, usize>,
    vec: Vec<Scalar>,
}

impl Scalars {
    fn get(&self, v: u64) -> &Scalar {
        &self.vec[*self.table.get(&v).expect("scalar not registered")]
    }
}

pub fn scalars() -> &'static Scalars {
    static S: OnceLock<Scalars> = OnceLock::new();
    S.get_or_init(|| {
        let p = program();
        let mut table = HashMap::new();
        let mut vec = Vec::new();
        let mut reg = |v: u64| {
            if !table.contains_key(&v) {
                table.insert(v, vec.len());
                vec.push(scalar(v));
            }
        };
        for op in &p.b.ops {
            match op {
                Op::FullRound(fr) => fr.rc.iter().for_each(|&c| reg(c)),
                Op::Reduce { form, .. } => {
                    reg(form.konst);
                    form.terms.iter().for_each(|(c, _)| reg(*c));
                }
                Op::Sbox { rc, .. } => reg(*rc),
            }
        }
        let bb = e_sm_bound();
        Scalars {
            two16: scalar(1 << 16),
            two29: scalar(1 << 29),
            qp: scalar(Q),
            qj: core::array::from_fn(|j| scalar(PRIMES[j])),
            one: scalar(1),
            forty: scalar(2 * EEK_BOUND as u64),
            two_c: core::array::from_fn(|j| scalar(2 * half(PRIMES[j]))),
            two_b: scalar_big(&(&bb * BigInt::from(2))),
            two_qb: core::array::from_fn(|j| scalar_big(&(quot_bound(PRIMES[j]) * BigInt::from(2)))),
            crt_k: core::array::from_fn(|j| {
                let k = &bb - BigInt::from(half(PRIMES[j])) - quot_bound(PRIMES[j]) * BigInt::from(PRIMES[j]);
                scalar_big(&k)
            }),
            table,
            vec,
        }
    })
}

pub fn one_ideal() -> &'static DegreeOneIdeal<ScalarInt> {
    static I: OnceLock<DegreeOneIdeal<ScalarInt>> = OnceLock::new();
    I.get_or_init(|| DegreeOneIdeal::new(ScalarInt::from(1i64)))
}

// ---------------------------------------------------------------------------
// UAIR
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct C1Uair;

impl Uair for C1Uair {
    type Ideal = DegreeOneIdeal<ScalarInt>;
    type Scalar = Scalar;

    fn signature() -> UairSignature {
        let p = program();
        let total = TotalColumnLayout::new(0, 0, NUM_PUB + p.num_cells);
        let public = PublicColumnLayout::new(0, 0, NUM_PUB);
        let mut shifts: Vec<ShiftSpec> = p.down_index.keys().map(|c| ShiftSpec::new(NUM_PUB + c, 1)).collect();
        shifts.sort_by_key(|s| s.source_col());
        let lookups: Vec<LookupColumnSpec> = p
            .b
            .cells
            .iter()
            .enumerate()
            .filter_map(|(c, info)| {
                info.lookup.map(|w| LookupColumnSpec {
                    column_index: NUM_PUB + c,
                    table_type: LookupTableType::Word { width: w, chunk_width: None },
                })
            })
            .collect();
        UairSignature::new(total, public, shifts, lookups, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let p = program();
        let sc = scalars();
        let ideal = ideal_from_ref(one_ideal());
        let cell = |c: CellId| -> B::Expr { up.int[NUM_PUB + c].clone() };
        let pubc = |i: usize| -> B::Expr { up.int[i].clone() };
        let dcell = |c: CellId| -> B::Expr { down.int[p.down_index[&c]].clone() };
        let mbs = |e: &B::Expr, s: &Scalar| mbs(e, s).expect("scalar mul overflow");
        let pair = |lo: &B::Expr, hi: &B::Expr| -> B::Expr { mbs(hi, &sc.two16) + lo };
        let limbs = |cells: &[CellId], get: &dyn Fn(CellId) -> B::Expr| -> B::Expr {
            let mut acc = get(*cells.last().unwrap());
            for &c in cells[..cells.len() - 1].iter().rev() {
                acc = mbs(&acc, &sc.two16) + &get(c);
            }
            acc
        };
        // value of a 59-bit bounded (16,13,16,14) layout
        let val59 = |bd: &Bounded| -> B::Expr {
            let l0 = pair(&cell(bd.limbs[0]), &cell(bd.limbs[1]));
            let l1 = pair(&cell(bd.limbs[2]), &cell(bd.limbs[3]));
            mbs(&l1, &sc.two29) + &l0
        };

        // ---- ring accumulators over pk0 offsets u = pk0 + c ----
        let u_val: Vec<Vec<B::Expr>> =
            (0..IDX_PER_ROW).map(|k| (0..NUM_PRIMES).map(|j| val59(&p.pk[k][j])).collect()).collect();
        for a in 0..NUM_ACC {
            let j = a / ring::BETAS_PER_PRIME;
            let acc_cur = limbs(&p.acc[a], &cell);
            let acc_next = limbs(&p.acc[a], &dcell);
            let uq = limbs(&p.u[a], &cell);
            let mut s = pubc(pub_c(a));
            for k in 0..IDX_PER_ROW {
                let uk = pubc(pub_u(a, k));
                let vk = pubc(pub_v(a, k));
                let e = cell(p.eek[k].0);
                s += &(uk.clone() * &u_val[k][j]);
                s += &(vk * &cell(p.t[k]));
                s += &(mbs(&e, &sc.qj[j]) - &(uk * &e));
            }
            b.assert_in_ideal(acc_next - &acc_cur - &s + &mbs(&uq, &sc.qj[j]), &ideal);
            b.assert_in_ideal(pubc(PUB_FIRST) * &acc_cur, &ideal);
        }

        // ---- range checks: ternary sk, eek two-sided, and the complements ----
        let one = from_ref(&sc.one);
        let two = one.clone() + &one;
        for k in 0..IDX_PER_ROW {
            let t = cell(p.t[k]);
            b.assert_in_ideal(t.clone() * &(t.clone() - &one) * &(t - &two), &ideal);
            // e' + e'' = 40
            b.assert_in_ideal(cell(p.eek[k].0) + &cell(p.eek[k].1) - &from_ref(&sc.forty), &ideal);
            // lifted: off + comp = 2B
            let lo = limbs(&p.lifted[k].limbs, &cell);
            let lc = limbs(&p.lifted[k].comp, &cell);
            b.assert_in_ideal(lo.clone() + &lc - &from_ref(&sc.two_b), &ideal);
            for j in 0..NUM_PRIMES {
                // pk0: off(u) + comp = 2c
                let uc = limbs(&p.pk[k][j].comp, &cell);
                b.assert_in_ideal(u_val[k][j].clone() + &uc - &from_ref(&sc.two_c[j]), &ideal);
                // residue: off + comp = 2c
                let ro = val59(&p.res[k][j]);
                let rc = limbs(&p.res[k][j].comp, &cell);
                b.assert_in_ideal(ro.clone() + &rc - &from_ref(&sc.two_c[j]), &ideal);
                // quotient: off + comp = 2Qb
                let qo = limbs(&p.quot[k][j].limbs, &cell);
                let qc = limbs(&p.quot[k][j].comp, &cell);
                b.assert_in_ideal(qo.clone() + &qc - &from_ref(&sc.two_qb[j]), &ideal);
                // CRT: off_l − off_r − q·off_q = K_j
                b.assert_in_ideal(lo.clone() - &ro - &mbs(&qo, &sc.qj[j]) - &from_ref(&sc.crt_k[j]), &ideal);
            }
        }

        // ---- Poseidon2 ops ----
        let mut vals: Vec<Option<B::Expr>> = vec![None; p.b.vals.len()];
        fn val_expr<E: Clone>(
            v: ValId,
            vals: &mut Vec<Option<E>>,
            p: &RowProgram,
            cell: &dyn Fn(CellId) -> E,
            pair: &dyn Fn(&E, &E) -> E,
            add: &dyn Fn(&E, &E) -> E,
        ) -> E {
            if let Some(e) = &vals[v] {
                return e.clone();
            }
            let e = match p.b.vals[v] {
                Val::Cell(c) => cell(c),
                Val::Pair(lo, hi) => pair(&cell(lo), &cell(hi)),
                Val::Sum(x, y) => {
                    let ex = val_expr(x, vals, p, cell, pair, add);
                    let ey = val_expr(y, vals, p, cell, pair, add);
                    add(&ex, &ey)
                }
            };
            vals[v] = Some(e.clone());
            e
        }
        let add = |x: &B::Expr, y: &B::Expr| x.clone() + y;
        let ve = |v: ValId, vals: &mut Vec<Option<B::Expr>>| val_expr(v, vals, p, &cell, &pair, &add);
        let form_expr = |f: &Form, vals: &mut Vec<Option<B::Expr>>| -> B::Expr {
            let mut acc = from_ref(sc.get(f.konst));
            for (c, v) in &f.terms {
                let e = val_expr(*v, vals, p, &cell, &pair, &add);
                acc += &mbs(&e, sc.get(*c));
            }
            acc
        };
        let cube_eq = |b: &mut B, lin: B::Expr, out: (CellId, CellId), k: &[CellId]| {
            let cube = lin.clone() * &lin * &lin;
            let outv = pair(&cell(out.0), &cell(out.1));
            let kv = limbs(k, &cell);
            b.assert_in_ideal(cube - &outv - &mbs(&kv, &sc.qp), &ideal);
        };
        for op in &p.b.ops {
            match op {
                Op::FullRound(fr) => {
                    let xs: Vec<B::Expr> = fr.vals.iter().map(|v| ve(*v, &mut vals)).collect();
                    let lin: Vec<B::Expr> = if fr.apply_me { crate::circuit::me_expr(&xs) } else { xs };
                    for i in 0..16 {
                        let l = lin[i].clone() + &from_ref(sc.get(fr.rc[i]));
                        cube_eq(b, l, fr.outs[i], &fr.ks[i]);
                    }
                }
                Op::Reduce { form, x, m } => {
                    let f = form_expr(form, &mut vals);
                    let xv = pair(&cell(x.0), &cell(x.1));
                    let mv = limbs(m, &cell);
                    b.assert_in_ideal(f - &xv - &mbs(&mv, &sc.qp), &ideal);
                }
                Op::Sbox { x, rc, out, k } => {
                    let mut l = ve(*x, &mut vals);
                    if *rc != 0 {
                        l += &from_ref(sc.get(*rc));
                    }
                    cube_eq(b, l, *out, k);
                }
            }
        }

        // ---- state carry / digest, first-row zero ----
        let first = pubc(PUB_FIRST);
        let last = pubc(PUB_LAST);
        for sp in 0..NUM_SPONGES {
            for l in 0..p2::WIDTH {
                let (lo, hi) = p.st[sp][l];
                let st_now = pair(&cell(lo), &cell(hi));
                b.assert_in_ideal(first.clone() * &st_now, &ideal);
                let st_next = pair(&dcell(lo), &dcell(hi));
                let out = ve(p.outs[sp][l], &mut vals);
                let diff = st_next - &out;
                if l < p2::DIGEST_LANES {
                    let (dlo, dhi) = pub_d(sp, l);
                    b.assert_in_ideal(diff + &pair(&pubc(dlo), &pubc(dhi)), &ideal);
                } else {
                    b.assert_in_ideal(diff.clone() - &(last.clone() * &diff), &ideal);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Witness, commitments, traces
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct C1Witness {
    pub sk: Vec<i64>,
    pub eek: Vec<i64>,
    /// centered: `[−(q_j−1)/2, (q_j−1)/2]`
    pub pk0: [Vec<i64>; NUM_PRIMES],
    pub e_sm_lifted: Vec<BigInt>,
    /// centered residues
    pub e_sm: [Vec<i64>; NUM_PRIMES],
    pub e_sm_quot: [Vec<BigInt>; NUM_PRIMES],
}

fn center(x: u64, q: u64) -> i64 {
    if x > half(q) { x as i64 - q as i64 } else { x as i64 }
}

pub fn sample_witness(public: &ring::Public, seed: u64) -> C1Witness {
    let w = ring::sample_witness(public, seed);
    let n = w.s.len();
    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x5eed);
    let bb = e_sm_bound();
    let two_b: BigUint = (&bb * BigInt::from(2)).to_biguint().unwrap();
    let e_sm_lifted: Vec<BigInt> = (0..n)
        .map(|_| {
            // uniform in [−B, B]
            let mut bytes = [0u8; 24];
            rng.fill(&mut bytes);
            let r = BigUint::from_bytes_le(&bytes) % (&two_b + BigUint::one());
            BigInt::from(r) - &bb
        })
        .collect();
    let e_sm: [Vec<i64>; 3] = core::array::from_fn(|j| {
        let q = BigInt::from(PRIMES[j]);
        e_sm_lifted
            .iter()
            .map(|x| {
                let r = ((x % &q) + &q) % &q;
                center(r.to_u64().unwrap(), PRIMES[j])
            })
            .collect()
    });
    let e_sm_quot: [Vec<BigInt>; 3] = core::array::from_fn(|j| {
        let q = BigInt::from(PRIMES[j]);
        e_sm_lifted
            .iter()
            .zip(&e_sm[j])
            .map(|(x, r)| {
                let d = x - BigInt::from(*r);
                assert!((&d % &q).is_zero());
                d / &q
            })
            .collect()
    });
    let pk0 = core::array::from_fn(|j| w.pk[j].iter().map(|&x| center(x, PRIMES[j])).collect());
    C1Witness { sk: w.s, eek: w.e_ek, pk0, e_sm_lifted, e_sm, e_sm_quot }
}

pub fn pk_limbs(off: u64) -> (u64, u64) {
    (off & ((1 << 29) - 1), off >> 29)
}

pub struct Commitments {
    pub sk: [u64; 8],
    pub pk: [u64; 8],
    pub esm: [u64; 8],
    pub aux: [u64; 8],
}

impl Commitments {
    pub fn bytes(d: &[u64; 8]) -> Vec<u8> {
        d.iter().flat_map(|x| (*x as u32).to_le_bytes()).collect()
    }
    pub fn all_bytes(&self) -> Vec<Vec<u8>> {
        vec![Self::bytes(&self.sk), Self::bytes(&self.pk), Self::bytes(&self.esm), Self::bytes(&self.aux)]
    }
}

fn limb_stream(polys: &[Vec<i64>; NUM_PRIMES]) -> Vec<u64> {
    let n = polys[0].len();
    let mut out = Vec::with_capacity(2 * NUM_PRIMES * n);
    for m in 0..n {
        for j in 0..NUM_PRIMES {
            let off = (polys[j][m] + half(PRIMES[j]) as i64) as u64;
            let (l0, l1) = pk_limbs(off);
            out.push(l0);
            out.push(l1);
        }
    }
    out
}

pub fn commit(w: &C1Witness) -> Commitments {
    let t: Vec<u64> = w.sk.iter().map(|&x| (x + 1) as u64).collect();
    let e: Vec<u64> = w.eek.iter().map(|&x| (x + EEK_BOUND) as u64).collect();
    Commitments {
        sk: p2::sponge(&t, RATE_8),
        pk: p2::sponge(&limb_stream(&w.pk0), RATE_12),
        esm: p2::sponge(&limb_stream(&w.e_sm), RATE_12),
        aux: p2::sponge(&e, RATE_8),
    }
}

pub struct Statement {
    pub a: [Vec<u64>; NUM_PRIMES],
    pub cm: Commitments,
}

pub const FS_DOMAIN: &[u8] = b"zincbench/interfold-c1-poseidon/v1";

pub fn weights_for(stmt: &Statement) -> Vec<ring::Weights> {
    let cb = stmt.cm.all_bytes();
    let refs: Vec<&[u8]> = cb.iter().map(|v| v.as_slice()).collect();
    let betas = ring::derive_betas(FS_DOMAIN, &refs);
    let mut out = Vec::with_capacity(NUM_ACC);
    for j in 0..NUM_PRIMES {
        for b in 0..ring::BETAS_PER_PRIME {
            out.push(ring::weights(&stmt.a[j], PRIMES[j], betas[j][b]));
        }
    }
    out
}

pub fn public_columns(stmt: &Statement, wts: &[ring::Weights]) -> Vec<Vec<i64>> {
    let mut cols = vec![vec![0i64; ROWS]; NUM_PUB];
    cols[PUB_FIRST][0] = 1;
    cols[PUB_LAST][ROWS - 1] = 1;
    let digests = [&stmt.cm.sk, &stmt.cm.esm, &stmt.cm.aux, &stmt.cm.pk];
    for sp in 0..NUM_SPONGES {
        for l in 0..p2::DIGEST_LANES {
            let (lo, hi) = pub_d(sp, l);
            cols[lo][ROWS - 1] = (digests[sp][l] & 0xFFFF) as i64;
            cols[hi][ROWS - 1] = (digests[sp][l] >> 16) as i64;
        }
    }
    for a in 0..NUM_ACC {
        let wt = &wts[a];
        let q = wt.q;
        let c_off = half(q) % q;
        for i in 0..ROWS {
            let mut c = 0u64;
            for k in 0..IDX_PER_ROW {
                let m = i * IDX_PER_ROW + k;
                cols[pub_u(a, k)][i] = wt.u[m] as i64;
                cols[pub_v(a, k)][i] = wt.v[m] as i64;
                // pk0 = u − c, s = t − 1, eek = e' − 20:
                // constant = −U·c − V + 20·U  (mod q)
                let term = ring::submod(
                    ring::mulmod(EEK_BOUND as u64, wt.u[m], q),
                    ring::addmod(ring::mulmod(wt.u[m], c_off, q), wt.v[m], q),
                    q,
                );
                c = ring::addmod(c, term, q);
            }
            cols[pub_c(a)][i] = c as i64;
        }
    }
    cols
}

fn set_limbs(it: &mut Interp, cells: &[CellId], v: &BigUint) {
    let mut v = v.clone();
    let mask = BigUint::from(0xFFFFu32);
    for &c in cells {
        it.set(c, (&v & &mask).to_u64().unwrap());
        v >>= 16;
    }
    assert!(v.is_zero(), "limb overflow");
}

fn set_bounded(it: &mut Interp, bd: &Bounded, off: &BigUint, two_bound: &BigUint) {
    assert!(off <= two_bound);
    set_limbs(it, &bd.limbs, off);
    set_limbs(it, &bd.comp, &(two_bound - off));
}

/// Sets a 59-bit bounded value with the (16,13,16,14) hash-limb layout.
fn set_bounded59(it: &mut Interp, bd: &Bounded, off: u64, two_c: u64) {
    assert!(off <= two_c);
    let (l0, l1) = pk_limbs(off);
    it.set(bd.limbs[0], l0 & 0xFFFF);
    it.set(bd.limbs[1], l0 >> 16);
    it.set(bd.limbs[2], l1 & 0xFFFF);
    it.set(bd.limbs[3], l1 >> 16);
    set_limbs(it, &bd.comp, &BigUint::from(two_c - off));
}

pub fn build_trace(stmt: &Statement, wts: &[ring::Weights], w: &C1Witness) -> UairTrace<'static, i64, i64, DEGREE_PLUS_ONE> {
    let p = program();
    let pubcols = public_columns(stmt, wts);
    let mut wit = vec![vec![0i64; ROWS]; p.num_cells];
    let mut states = [[0u64; 16]; NUM_SPONGES];
    let mut accs = [0u64; NUM_ACC];
    let bb = e_sm_bound();
    let two_b = (&bb * BigInt::from(2)).to_biguint().unwrap();
    let two_qb: [BigUint; 3] = core::array::from_fn(|j| (quot_bound(PRIMES[j]) * BigInt::from(2)).to_biguint().unwrap());
    for i in 0..ROWS {
        let mut it = Interp::new(&p.b);
        for k in 0..IDX_PER_ROW {
            let m = i * IDX_PER_ROW + k;
            it.set(p.t[k], (w.sk[m] + 1) as u64);
            let e = (w.eek[m] + EEK_BOUND) as u64;
            it.set(p.eek[k].0, e);
            it.set(p.eek[k].1, 2 * EEK_BOUND as u64 - e);
            let lifted_off = (&w.e_sm_lifted[m] + &bb).to_biguint().unwrap();
            set_bounded(&mut it, &p.lifted[k], &lifted_off, &two_b);
            for j in 0..NUM_PRIMES {
                let c = half(PRIMES[j]);
                set_bounded59(&mut it, &p.pk[k][j], (w.pk0[j][m] + c as i64) as u64, 2 * c);
                set_bounded59(&mut it, &p.res[k][j], (w.e_sm[j][m] + c as i64) as u64, 2 * c);
                let qoff = (&w.e_sm_quot[j][m] + quot_bound(PRIMES[j])).to_biguint().unwrap();
                set_bounded(&mut it, &p.quot[k][j], &qoff, &two_qb[j]);
            }
        }
        for sp in 0..NUM_SPONGES {
            for l in 0..16 {
                it.set(p.st[sp][l].0, states[sp][l] & 0xFFFF);
                it.set(p.st[sp][l].1, states[sp][l] >> 16);
            }
        }
        for a in 0..NUM_ACC {
            let j = a / ring::BETAS_PER_PRIME;
            let q = PRIMES[j] as u128;
            let c = half(PRIMES[j]) as i64;
            let mut total: u128 = accs[a] as u128 + pubcols[pub_c(a)][i] as u128;
            for k in 0..IDX_PER_ROW {
                let m = i * IDX_PER_ROW + k;
                let uk = pubcols[pub_u(a, k)][i] as u128;
                let vk = pubcols[pub_v(a, k)][i] as u128;
                let e = (w.eek[m] + EEK_BOUND) as u128;
                let uoff = (w.pk0[j][m] + c) as u128;
                total += uk * uoff + vk * (w.sk[m] + 1) as u128 + q * e - uk * e;
            }
            let next = total % q;
            let uq = total / q;
            let mut acc_v = accs[a];
            for cc in 0..4 {
                it.set(p.acc[a][cc], acc_v & 0xFFFF);
                acc_v >>= 16;
            }
            let mut uv = uq;
            for cc in 0..4 {
                it.set(p.u[a][cc], (uv & 0xFFFF) as u64);
                uv >>= 16;
            }
            assert_eq!(uv, 0);
            accs[a] = next as u64;
        }
        it.run(&p.b.ops);
        for sp in 0..NUM_SPONGES {
            for l in 0..16 {
                states[sp][l] = it.val(p.outs[sp][l]) as u64;
            }
        }
        for (c, col) in wit.iter_mut().enumerate() {
            col[i] = it.get(c) as i64;
        }
    }
    for a in 0..NUM_ACC {
        assert_eq!(accs[a], 0, "ring combination {a} does not vanish: witness violates the relation");
    }
    let digests = [&stmt.cm.sk, &stmt.cm.esm, &stmt.cm.aux, &stmt.cm.pk];
    for sp in 0..NUM_SPONGES {
        assert_eq!(&states[sp][..8], &digests[sp][..], "sponge {sp} digest mismatch");
    }
    let int: Vec<DenseMultilinearExtension<i64>> =
        pubcols.into_iter().chain(wit).map(|v| v.into_iter().collect::<DenseMultilinearExtension<_>>()).collect();
    UairTrace { int: int.into(), ..Default::default() }
}

pub fn public_trace(stmt: &Statement, wts: &[ring::Weights]) -> UairTrace<'static, i64, i64, DEGREE_PLUS_ONE> {
    let int: Vec<DenseMultilinearExtension<i64>> =
        public_columns(stmt, wts).into_iter().map(|v| v.into_iter().collect::<DenseMultilinearExtension<_>>()).collect();
    UairTrace { int: int.into(), ..Default::default() }
}

/// Reference check of the C1 statement on a witness (coefficient-wise).
pub fn c1_holds(public: &ring::Public, w: &C1Witness) -> bool {
    let n = w.sk.len();
    let bb = e_sm_bound();
    if !w.sk.iter().all(|x| x.abs() <= 1) || !w.eek.iter().all(|x| x.abs() <= EEK_BOUND) {
        return false;
    }
    if !w.e_sm_lifted.iter().all(|x| x.abs() <= bb) {
        return false;
    }
    for j in 0..NUM_PRIMES {
        let q = PRIMES[j];
        let c = half(q) as i64;
        let qb = quot_bound(q);
        if !w.pk0[j].iter().all(|x| x.abs() <= c) || !w.e_sm[j].iter().all(|x| x.abs() <= c) {
            return false;
        }
        if !w.e_sm_quot[j].iter().all(|x| x.abs() <= qb) {
            return false;
        }
        for m in 0..n {
            if &w.e_sm_lifted[m] != &(BigInt::from(w.e_sm[j][m]) + &w.e_sm_quot[j][m] * BigInt::from(q)) {
                return false;
            }
        }
        let expected = ring::compute_pk(&public.a[j], &w.sk, &w.eek, q);
        for m in 0..n {
            if (w.pk0[j][m] as i128 - expected[m] as i128).rem_euclid(q as i128) != 0 {
                return false;
            }
        }
    }
    true
}
