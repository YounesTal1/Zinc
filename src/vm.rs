//! Coprocessor-style Zinc+ UAIR for interfold **C2** (share computation),
//! **C3** (share encryption) and the merged dealer circuit C2+C3.
//!
//! Rows come in three kinds, chosen per row by public one-hot selectors
//! `S_ENC`, `S_SHR`, `S_HASH` (all-zero separator rows sit between the
//! blocks). The kinds share the physical int columns — pooled by lookup
//! table — and every constraint of a kind is multiplied by its selector.
//!
//! * **Hash rows** — one Poseidon2-KoalaBear-16 permutation (rate 8) per row:
//!   lane `l < 8` absorbs `A_l = a16 + 2^16·a15 < 2^31`; `start·ST = 0`,
//!   `cont·(ST_next − OUT) = 0`, `end·(OUT_l − D_l) = 0` (digest, public),
//!   `ispub·(A − PUBA) = 0` on sponges over public data (recipient keys).
//!   Sponges over witness data receive their chunks over the bus.
//! * **Bus** — three accumulators `acc' = acc + Σ_runs base_run·Σ_k γ^k·v_k − M·u`
//!   with `M = 2^61−1` and `γ` derived from the Zip+ commitment root of the
//!   int lane. Producers add `γ^idx·v`, consumers `(M−γ^idx)·v`; each
//!   element index has one producer and one consumer, so the sum vanishes
//!   mod `M` iff the two sides agree (Schwartz–Zippel; `≤ 2^-39` per
//!   accumulator, three of them ⇒ `2^-117`).
//! * **Enc rows** — `E` encryptions × `C` coefficients per row: `u` ternary,
//!   `e0, e1 ∈ [−20, 20]`, `k1 ∈ [0, t)` (+ `m ∈ [0,t)`, `z < 2^64` and
//!   `k1 = q_mod_t·m − t·z` in C3 mode; `k1` arrives over the bus in merged
//!   mode), `ct0/ct1` per DKG modulus centered in `[−(q'−1)/2, (q'−1)/2]`,
//!   and the random root-of-unity accumulators (one per (DKG modulus, β),
//!   all encryptions and both legs folded with public `ρ`) of
//!   `ct0 ≡ pk0·u + e0 + k0·k1`, `ct1 ≡ pk1·u + e1 (mod q'_i, X^N+1)`.
//! * **Shr rows** — one coefficient of one track: the secret (`sk+1` ternary
//!   or a centered `e_sm` residue per threshold modulus), the 9 shares
//!   `∈ [0, q_j)`, the 5 parity quotients per modulus (`H_j·y ≡ 0 mod q_j`
//!   with `y_0` the secret), and in merged mode `z, k1` per share.
//!
//! The accumulators live in the *arbitrary-polynomial* lane (degree-0
//! polynomials), which Zip+ commits separately: the prover commits the int
//! lane first, derives `γ, β, ρ` from that root, fills the accumulators and
//! proves; the verifier derives the same challenges from the int commitment
//! inside the proof. Everything that enters an identity is therefore fixed
//! before its challenge.

use crate::circuit::{Builder, CellId as BCell, Form, Interp, Op, Val, ValId, me_expr};
use crate::dkg::{self, K0, L_D, L_T, N_PARTIES, PARITY_ROWS, PT, QD, Q_MOD_T};
use crate::poseidon2::{self as p2, Q};
use crate::ring::{self, PRIMES as QT, addmod, mulmod, submod};
use crypto_primitives::crypto_bigint_int::Int;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::HashMap;
use std::sync::OnceLock;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, LookupColumnSpec, LookupTableType, PublicColumnLayout, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace, ideal::DegreeOneIdeal,
};

/// Trace polynomial degree bound (`D`); only the arbitrary lane cares and
/// it holds constants, so 1.
pub const D: usize = 1;
pub type ScalarInt = Int<3>;
pub type Scalar = DensePolynomial<ScalarInt, 1>;
pub type ArbCell = DensePolynomial<i64, D>;
pub const RATE: usize = 8;
pub const BUS_M: u64 = (1u64 << 61) - 1;
pub const NUM_BUS: usize = 3;
pub const BETAS: usize = 2;
pub const NUM_ENC_ACC: usize = L_D * BETAS;
/// Arbitrary-lane limbs are kept below 2^31 so that Zip+'s integer IPRS
/// encoding (i128 codewords) cannot overflow: an accumulator is
/// `acc = a0 + 2^31·a1` and its quotient `u = Σ u_i 2^{31 i}`.
pub const ARB_LIMB: u32 = 31;
pub const BUS_U_LIMBS: usize = 4; // u < 2^94
pub const ENC_U_LIMBS: usize = 3; // u < 2^68
pub const NUM_ARB: usize = NUM_BUS * (2 + BUS_U_LIMBS) + NUM_ENC_ACC * (2 + ENC_U_LIMBS);


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    Sk,
    Esm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// C2a (sk) / C2b (e_sm): one track, 3 moduli, 9 parties, 27 share commitments.
    C2(Track),
    /// C3: one encryption (message commitment in, ciphertext commitment out).
    C3,
    /// Both tracks' shares and all 54 encryptions in one circuit.
    Merged,
}

// ---------------------------------------------------------------------------
// Column pools
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tbl {
    W16,
    W15,
    W8,
    W6,
    W12,
    W14,
    W13,
    Free,
}
pub const NUM_TBL: usize = 8;
pub const TBLS: [Tbl; NUM_TBL] = [Tbl::W16, Tbl::W15, Tbl::W8, Tbl::W6, Tbl::W12, Tbl::W14, Tbl::W13, Tbl::Free];
impl Tbl {
    pub fn width(self) -> Option<usize> {
        match self {
            Tbl::W16 => Some(16),
            Tbl::W15 => Some(15),
            Tbl::W8 => Some(8),
            Tbl::W6 => Some(6),
            Tbl::W12 => Some(12),
            Tbl::W14 => Some(14),
            Tbl::W13 => Some(13),
            Tbl::Free => None,
        }
    }
    fn of_width(w: Option<usize>) -> Tbl {
        match w {
            Some(16) => Tbl::W16,
            Some(15) => Tbl::W15,
            Some(8) => Tbl::W8,
            Some(6) => Tbl::W6,
            Some(12) => Tbl::W12,
            Some(14) => Tbl::W14,
            Some(13) => Tbl::W13,
            None => Tbl::Free,
            Some(w) => panic!("no pool for width {w}"),
        }
    }
    fn index(self) -> usize {
        TBLS.iter().position(|t| *t == self).unwrap()
    }
}

/// A logical int cell of a row kind: (pool, index within the pool).
#[derive(Clone, Copy, Debug)]
pub struct Cell(pub Tbl, pub usize);

#[derive(Default)]
struct Alloc {
    counts: [usize; NUM_TBL],
}
impl Alloc {
    fn cell(&mut self, t: Tbl) -> Cell {
        let i = self.counts[t.index()];
        self.counts[t.index()] += 1;
        Cell(t, i)
    }
    /// `[0, 2^60)`: chunks (24, 24, 12) as cells (16, 8, 16, 8, 12).
    fn v60(&mut self) -> [Cell; 5] {
        [self.cell(Tbl::W16), self.cell(Tbl::W8), self.cell(Tbl::W16), self.cell(Tbl::W8), self.cell(Tbl::W12)]
    }
    /// C1's `bounded_59` layout for `e_sm` residue offsets: limbs (16, 13, 16, 15)
    /// — hash elements `l0 = c0 + 2^16 c1` (29 bits), `l1 = c2 + 2^16 c3` — and
    /// complement limbs (16, 16, 16, 12).
    fn res59(&mut self) -> ([Cell; 4], [Cell; 4]) {
        (
            [self.cell(Tbl::W16), self.cell(Tbl::W13), self.cell(Tbl::W16), self.cell(Tbl::W15)],
            [self.cell(Tbl::W16), self.cell(Tbl::W16), self.cell(Tbl::W16), self.cell(Tbl::W12)],
        )
    }
    /// `[0, 2^62)`: chunks (24, 24, 14).
    fn v62(&mut self) -> [Cell; 5] {
        [self.cell(Tbl::W16), self.cell(Tbl::W8), self.cell(Tbl::W16), self.cell(Tbl::W8), self.cell(Tbl::W14)]
    }
    fn w16s(&mut self, n: usize) -> Vec<Cell> {
        (0..n).map(|_| self.cell(Tbl::W16)).collect()
    }
}

fn split5(v: u64) -> [u64; 5] {
    [v & 0xFFFF, (v >> 16) & 0xFF, (v >> 24) & 0xFFFF, (v >> 40) & 0xFF, v >> 48]
}
/// The three 24-bit hash chunks of a value `< 2^62`.
pub fn chunks3(v: u64) -> [u64; 3] {
    [v & 0xFFFFFF, (v >> 24) & 0xFFFFFF, v >> 48]
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

pub struct EncCell {
    pub tu: Cell,
    pub e0: (Cell, Cell),
    pub e1: (Cell, Cell),
    pub k1: [Cell; 5],
    pub k1c: [Cell; 5],
    /// C3: message (limbs, complement) and z limbs
    pub m: Option<([Cell; 5], [Cell; 5], Vec<Cell>)>,
    /// `[i][leg]` = (offset limbs, complement limbs)
    pub ct: [[([Cell; 5], [Cell; 5]); 2]; L_D],
}

pub struct EncLayout {
    pub e_per_row: usize,
    pub c_per_row: usize,
    /// `[e][c]`
    pub cells: Vec<Vec<EncCell>>,
}

pub struct ShrCell {
    /// `[j][p]` shares as (limbs, complement)
    pub y: [Vec<([Cell; 5], [Cell; 5])>; L_T],
    /// `[j][r]` parity quotient limbs (4 × 16)
    pub w: [Vec<Vec<Cell>>; L_T],
    pub sk: Option<Cell>,
    pub res: Option<[([Cell; 4], [Cell; 4]); L_T]>,
    /// merged: `[j][p]` = (z limbs, k1 limbs, k1 complement)
    pub k1: Option<[Vec<(Vec<Cell>, [Cell; 5], [Cell; 5])>; L_T]>,
}

pub struct HashLayout {
    pub b: Builder,
    pub map: Vec<Cell>,
    pub a: [(BCell, BCell); RATE],
    pub st: [(BCell, BCell); p2::WIDTH],
    pub outs: [ValId; p2::WIDTH],
}

pub struct Layout {
    pub mode: Mode,
    pub enc: Option<EncLayout>,
    pub shr: Option<ShrCell>,
    pub hash: HashLayout,
    pub pool_base: [usize; NUM_TBL],
    pub pool_count: [usize; NUM_TBL],
    pub num_wit: usize,
    /// shifted int witness column → index in `down.int`
    pub down_int: HashMap<usize, usize>,
    pub runs_enc: usize,
    pub runs_shr: usize,
    pub max_runs: usize,
}

impl Layout {
    pub fn phys(&self, c: Cell) -> usize {
        self.pool_base[c.0.index()] + c.1
    }
}

/// Arbitrary-lane columns: `[acc_lo, acc_hi, u_0, u_1, …]` per accumulator.
pub fn arb_bus(k: usize, part: usize) -> usize {
    k * (2 + BUS_U_LIMBS) + part
}
pub fn arb_enc(i: usize, b: usize, part: usize) -> usize {
    NUM_BUS * (2 + BUS_U_LIMBS) + (i * BETAS + b) * (2 + ENC_U_LIMBS) + part
}
/// Index of a shifted acc limb (`part ∈ {0,1}`) in `down.arbitrary_poly`.
pub fn darb_index(col: usize) -> usize {
    if col < NUM_BUS * (2 + BUS_U_LIMBS) {
        let k = col / (2 + BUS_U_LIMBS);
        2 * k + col % (2 + BUS_U_LIMBS)
    } else {
        let c = col - NUM_BUS * (2 + BUS_U_LIMBS);
        let a = c / (2 + ENC_U_LIMBS);
        2 * NUM_BUS + 2 * a + c % (2 + ENC_U_LIMBS)
    }
}

// Public int columns.
pub const PUB_S_ENC: usize = 0;
pub const PUB_S_SHR: usize = 1;
pub const PUB_S_HASH: usize = 2;
pub const PUB_START: usize = 3;
pub const PUB_CONT: usize = 4;
pub const PUB_END: usize = 5;
/// hash rows: public-data sponge; shr rows (merged): sk-track flag
pub const PUB_FLAG: usize = 6;
pub const PUB_FIRST: usize = 7;
/// 1 on row 2^nv − 2: the accumulators' next value (row 2^nv − 1) must be 0
/// (Zinc+ does not constrain the last row, so the wrap-around is explicit).
pub const PUB_LAST: usize = 8;
pub const PUB_D: usize = 9; // 16: digest lanes (lo16, hi15)
pub const PUB_A: usize = PUB_D + 16; // 16: public absorbed chunks (lo16, hi15)
pub const PUB_BUS: usize = PUB_A + 16;
pub fn pub_bus(l: &Layout, k: usize, run: usize) -> usize {
    PUB_BUS + k * l.max_runs + run
}
pub fn pub_w_base(l: &Layout) -> usize {
    PUB_BUS + NUM_BUS * l.max_runs
}
/// Enc weights per (e, c, i, b): `[W_ct0, W_ct1, W_u, W_k1]`; then `C` per (i, b).
pub fn pub_w(l: &Layout, e: usize, c: usize, i: usize, b: usize, which: usize) -> usize {
    let el = l.enc.as_ref().unwrap();
    pub_w_base(l) + (((e * el.c_per_row + c) * L_D + i) * BETAS + b) * 4 + which
}
pub fn pub_c(l: &Layout, i: usize, b: usize) -> usize {
    let el = l.enc.as_ref().unwrap();
    pub_w_base(l) + el.e_per_row * el.c_per_row * NUM_ENC_ACC * 4 + i * BETAS + b
}
pub fn num_pub(l: &Layout) -> usize {
    match &l.enc {
        Some(_) => pub_c(l, L_D - 1, BETAS - 1) + 1,
        None => pub_w_base(l),
    }
}

pub fn build_layout(mode: Mode) -> Layout {
    // ---- hash kind (circuit::Builder program) ----
    let mut b = Builder::default();
    let a: [(BCell, BCell); RATE] = core::array::from_fn(|_| (b.word(16), b.word(15)));
    let st: [(BCell, BCell); 16] = core::array::from_fn(|_| (b.word(16), b.word(15)));
    let st_val: [ValId; 16] = core::array::from_fn(|l| b.val_pair(st[l].0, st[l].1));
    let a_val: [ValId; RATE] = core::array::from_fn(|k| b.val_pair(a[k].0, a[k].1));
    let inputs: [ValId; 16] =
        core::array::from_fn(|l| if l < RATE { b.val_sum(st_val[l], a_val[l]) } else { st_val[l] });
    let outs = b.permutation(inputs);
    let mut ha = Alloc::default();
    let map: Vec<Cell> = b.cells.iter().map(|info| ha.cell(Tbl::of_width(info.lookup))).collect();
    let hash = HashLayout { b, map, a, st, outs };

    // ---- enc kind ----
    let (e_per_row, c_per_row) = match mode {
        Mode::C3 => (1, 8),
        Mode::Merged => (N_PARTIES, 1),
        Mode::C2(_) => (0, 0),
    };
    let mut ea = Alloc::default();
    let enc = (e_per_row > 0).then(|| EncLayout {
        e_per_row,
        c_per_row,
        cells: (0..e_per_row)
            .map(|_| {
                (0..c_per_row)
                    .map(|_| EncCell {
                        tu: ea.cell(Tbl::Free),
                        e0: (ea.cell(Tbl::W6), ea.cell(Tbl::W6)),
                        e1: (ea.cell(Tbl::W6), ea.cell(Tbl::W6)),
                        k1: ea.v60(),
                        k1c: ea.v60(),
                        m: (mode == Mode::C3).then(|| (ea.v60(), ea.v60(), ea.w16s(4))),
                        ct: core::array::from_fn(|_| core::array::from_fn(|_| (ea.v62(), ea.v62()))),
                    })
                    .collect()
            })
            .collect(),
    });

    // ---- shr kind ----
    let mut sa = Alloc::default();
    let shr = (mode != Mode::C3).then(|| {
        let track = match mode {
            Mode::C2(t) => Some(t),
            _ => None,
        };
        ShrCell {
            y: core::array::from_fn(|_| (0..N_PARTIES).map(|_| (sa.v60(), sa.v60())).collect()),
            w: core::array::from_fn(|_| (0..PARITY_ROWS).map(|_| sa.w16s(4)).collect()),
            sk: (track != Some(Track::Esm)).then(|| sa.cell(Tbl::Free)),
            res: (track != Some(Track::Sk)).then(|| core::array::from_fn(|_| sa.res59())),
            k1: (mode == Mode::Merged)
                .then(|| core::array::from_fn(|_| (0..N_PARTIES).map(|_| (sa.w16s(4), sa.v60(), sa.v60())).collect())),
        }
    });

    // ---- pools ----
    let mut pool_count = [0usize; NUM_TBL];
    for t in 0..NUM_TBL {
        pool_count[t] = ha.counts[t].max(ea.counts[t]).max(sa.counts[t]);
    }
    let mut pool_base = [0usize; NUM_TBL];
    let mut acc = 0;
    for t in 0..NUM_TBL {
        pool_base[t] = acc;
        acc += pool_count[t];
    }
    let num_wit = acc;

    let runs_enc = e_per_row * c_per_row * 2;
    let runs_shr = match &shr {
        Some(sh) => sh.sk.is_some() as usize + sh.res.is_some() as usize + L_T * N_PARTIES,
        None => 0,
    };
    let max_runs = runs_enc + runs_shr + 1;

    let mut l = Layout {
        mode,
        enc,
        shr,
        hash,
        pool_base,
        pool_count,
        num_wit,
        down_int: HashMap::new(),
        runs_enc,
        runs_shr,
        max_runs,
    };
    let mut shifted: Vec<usize> = Vec::new();
    for lane in 0..16 {
        shifted.push(l.phys(l.hash.map[l.hash.st[lane].0]));
        shifted.push(l.phys(l.hash.map[l.hash.st[lane].1]));
    }
    shifted.sort_unstable();
    shifted.dedup();
    l.down_int = shifted.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    l
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

pub fn scalar(v: u64) -> Scalar {
    crate::c1_uair::scalar(v)
}

pub struct Scalars {
    pub two16: Scalar,
    pub two24: Scalar,
    pub two29: Scalar,
    pub two48: Scalar,
    pub two31: Scalar,
    pub one: Scalar,
    pub qp: Scalar,
    pub forty: Scalar,
    pub qd: [Scalar; L_D],
    pub qd_m1: [Scalar; L_D],
    pub qt: [Scalar; L_T],
    pub qt_m1: [Scalar; L_T],
    pub t_m1: Scalar,
    pub t: Scalar,
    pub q_mod_t: Scalar,
    pub bus_m: Scalar,
    pub gpow: Vec<Vec<Scalar>>,
    pub h: [[[Scalar; N_PARTIES + 1]; PARITY_ROWS]; L_T],
    pub k_sk: [[Scalar; PARITY_ROWS]; L_T],
    pub k_esm: [[Scalar; PARITY_ROWS]; L_T],
    table: HashMap<u64, usize>,
    vec: Vec<Scalar>,
}
impl Scalars {
    fn get(&self, v: u64) -> &Scalar {
        &self.vec[*self.table.get(&v).expect("scalar not registered")]
    }
}

/// `K_sk[j][r] = q − H_0`  (secret `sk = t − 1`), `K_esm[j][r] = −H_0·c mod q` (secret `= off − c`).
pub fn make_scalars(l: &Layout, gammas: &[u64; NUM_BUS]) -> Scalars {
    let mut table = HashMap::new();
    let mut vec = Vec::new();
    let mut reg = |v: u64| {
        if !table.contains_key(&v) {
            table.insert(v, vec.len());
            vec.push(scalar(v));
        }
    };
    for op in &l.hash.b.ops {
        match op {
            Op::FullRound(fr) => fr.rc.iter().for_each(|&c| reg(c)),
            Op::Reduce { form, .. } => {
                reg(form.konst);
                form.terms.iter().for_each(|(c, _)| reg(*c));
            }
            Op::Sbox { rc, .. } => reg(*rc),
        }
    }
    let h = dkg::parity_matrix();
    Scalars {
        two16: scalar(1 << 16),
        two24: scalar(1 << 24),
        two29: scalar(1 << 29),
        two48: scalar(1 << 48),
        two31: scalar(1 << 31),
        one: scalar(1),
        qp: scalar(Q),
        forty: scalar(40),
        qd: core::array::from_fn(|i| scalar(QD[i])),
        qd_m1: core::array::from_fn(|i| scalar(QD[i] - 1)),
        qt: core::array::from_fn(|j| scalar(QT[j])),
        qt_m1: core::array::from_fn(|j| scalar(QT[j] - 1)),
        t_m1: scalar(PT - 1),
        t: scalar(PT),
        q_mod_t: scalar(Q_MOD_T),
        bus_m: scalar(BUS_M),
        gpow: (0..NUM_BUS).map(|k| (0..12).map(|e| scalar(ring::powmod(gammas[k], e as u64, BUS_M))).collect()).collect(),
        h: core::array::from_fn(|j| core::array::from_fn(|r| core::array::from_fn(|k| scalar(h[j][r][k])))),
        k_sk: core::array::from_fn(|j| core::array::from_fn(|r| scalar(QT[j] - h[j][r][0]))),
        k_esm: core::array::from_fn(|j| {
            core::array::from_fn(|r| scalar(submod(0, mulmod(h[j][r][0], (QT[j] - 1) / 2, QT[j]), QT[j])))
        }),
        table,
        vec,
    }
}

// ---------------------------------------------------------------------------
// UAIR
// ---------------------------------------------------------------------------

pub struct Globals {
    pub layout: Layout,
    pub scalars: Scalars,
    pub gammas: [u64; NUM_BUS],
}
impl Globals {
    pub fn scalars_gammas(&self) -> [u64; NUM_BUS] {
        self.gammas
    }
}

static GLOBALS: OnceLock<Globals> = OnceLock::new();
/// The UAIR is static in Zinc+, so the layout and the γ-dependent scalars
/// are installed once per process (one mode, one proof per process).
pub fn install(mode: Mode, gammas: [u64; NUM_BUS]) -> &'static Globals {
    let g = GLOBALS.get_or_init(|| {
        let layout = build_layout(mode);
        let scalars = make_scalars(&layout, &gammas);
        Globals { layout, scalars, gammas }
    });
    assert!(g.layout.mode == mode, "vm::install called with a different mode");
    g
}
pub fn globals() -> &'static Globals {
    GLOBALS.get().expect("vm::install first")
}
pub fn one_ideal() -> &'static DegreeOneIdeal<ScalarInt> {
    static I: OnceLock<DegreeOneIdeal<ScalarInt>> = OnceLock::new();
    I.get_or_init(|| DegreeOneIdeal::new(ScalarInt::from(1i64)))
}

#[derive(Clone, Debug)]
pub struct VmUair;

impl Uair for VmUair {
    type Ideal = DegreeOneIdeal<ScalarInt>;
    type Scalar = Scalar;

    fn signature() -> UairSignature {
        let l = &globals().layout;
        let np = num_pub(l);
        let (n_arb, n_int) = (NUM_ARB, np + l.num_wit);
        let total = TotalColumnLayout::new(0, n_arb, n_int);
        let public = PublicColumnLayout::new(0, 0, np);
        // flat indexing: arbitrary_poly || int
        let mut shifts: Vec<ShiftSpec> = Vec::new();
        for k in 0..NUM_BUS {
            shifts.push(ShiftSpec::new(arb_bus(k, 0), 1));
            shifts.push(ShiftSpec::new(arb_bus(k, 1), 1));
        }
        for i in 0..L_D {
            for b in 0..BETAS {
                shifts.push(ShiftSpec::new(arb_enc(i, b, 0), 1));
                shifts.push(ShiftSpec::new(arb_enc(i, b, 1), 1));
            }
        }
        for c in l.down_int.keys() {
            shifts.push(ShiftSpec::new(n_arb + np + c, 1));
        }
        shifts.sort_by_key(|s| s.source_col());
        let mut lookups = Vec::new();
        for t in 0..NUM_TBL {
            if let Some(w) = TBLS[t].width() {
                for i in 0..l.pool_count[t] {
                    lookups.push(LookupColumnSpec {
                        column_index: n_arb + np + l.pool_base[t] + i,
                        table_type: LookupTableType::Word { width: w, chunk_width: None },
                    });
                }
            }
        }
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
        let g = globals();
        let l = &g.layout;
        let sc = &g.scalars;
        let np = num_pub(l);
        let ideal = ideal_from_ref(one_ideal());
        let mbs = |e: &B::Expr, s: &Scalar| mbs(e, s).expect("scalar mul overflow");
        let pubc = |i: usize| -> B::Expr { up.int[i].clone() };
        let wit = |c: usize| -> B::Expr { up.int[np + c].clone() };
        let cell = |c: Cell| -> B::Expr { wit(l.phys(c)) };
        let dwit = |c: usize| -> B::Expr { down.int[l.down_int[&c]].clone() };
        let arb = |a: usize| -> B::Expr { up.arbitrary_poly[a].clone() };
        // shifted acc columns come first in their lane (lowest flat indices), in column order
        let darb = |col: usize| -> B::Expr { down.arbitrary_poly[darb_index(col)].clone() };
        // accumulator value `a0 + 2^31 a1` (current / next row) and quotient
        let acc_cur = |a: usize| -> B::Expr { mbs(&arb(a + 1), &sc.two31) + &arb(a) };
        let darb_acc = |a: usize| -> B::Expr { mbs(&darb(a + 1), &sc.two31) + &darb(a) };
        let limbs16 = |cells: &[Cell], get: &dyn Fn(Cell) -> B::Expr| -> B::Expr {
            let mut acc = get(*cells.last().unwrap());
            for &c in cells[..cells.len() - 1].iter().rev() {
                acc = mbs(&acc, &sc.two16) + &get(c);
            }
            acc
        };
        let chunk = |lo: &B::Expr, hi: &B::Expr| -> B::Expr { mbs(hi, &sc.two16) + lo };
        let v5_chunks =
            |c: &[Cell; 5]| -> [B::Expr; 3] { [chunk(&cell(c[0]), &cell(c[1])), chunk(&cell(c[2]), &cell(c[3])), cell(c[4])] };
        let v5 = |c: &[Cell; 5]| -> B::Expr {
            let ch = v5_chunks(c);
            mbs(&ch[2], &sc.two48) + &mbs(&ch[1], &sc.two24) + &ch[0]
        };
        let gate = |b: &mut B, s: &B::Expr, e: B::Expr| b.assert_in_ideal(s.clone() * &e, &ideal);
        // C1 residue layout: value = l0 + 2^29 l1, l0 = c0 + 2^16 c1, l1 = c2 + 2^16 c3
        let res_chunks = |c: &[Cell; 4]| -> [B::Expr; 2] { [chunk(&cell(c[0]), &cell(c[1])), chunk(&cell(c[2]), &cell(c[3]))] };
        let res_val = |c: &[Cell; 4]| -> B::Expr {
            let ch = res_chunks(c);
            mbs(&ch[1], &sc.two29) + &ch[0]
        };
        let res_comp = |c: &[Cell; 4]| -> B::Expr { limbs16(c, &cell) };
        let uval = |a: usize, limbs: usize| -> B::Expr {
            let mut acc = arb(a + 1 + limbs);
            for i in (0..limbs - 1).rev() {
                acc = mbs(&acc, &sc.two31) + &arb(a + 2 + i);
            }
            acc
        };

        let s_enc = pubc(PUB_S_ENC);
        let s_shr = pubc(PUB_S_SHR);
        let s_hash = pubc(PUB_S_HASH);
        let one = from_ref(&sc.one);
        let two = one.clone() + &one;
        let zero = || one.clone() - &one;

        // bus runs: (position, elements) collected per kind, assembled at the end
        let mut runs: Vec<(usize, Vec<B::Expr>)> = Vec::new();
        let mut pos_enc = 0usize;
        let mut pos_shr = l.runs_enc;
        let pos_hash = l.runs_enc + l.runs_shr;

        // ================= hash rows =================
        {
            let hl = &l.hash;
            let mut vals: Vec<Option<B::Expr>> = vec![None; hl.b.vals.len()];
            fn val_expr<E: Clone>(
                v: ValId,
                vals: &mut Vec<Option<E>>,
                bl: &Builder,
                cell: &dyn Fn(BCell) -> E,
                pair: &dyn Fn(&E, &E) -> E,
                add: &dyn Fn(&E, &E) -> E,
            ) -> E {
                if let Some(e) = &vals[v] {
                    return e.clone();
                }
                let e = match bl.vals[v] {
                    Val::Cell(c) => cell(c),
                    Val::Pair(lo, hi) => pair(&cell(lo), &cell(hi)),
                    Val::Sum(x, y) => {
                        let ex = val_expr(x, vals, bl, cell, pair, add);
                        let ey = val_expr(y, vals, bl, cell, pair, add);
                        add(&ex, &ey)
                    }
                };
                vals[v] = Some(e.clone());
                e
            }
            let hcell = |c: BCell| -> B::Expr { cell(hl.map[c]) };
            let add = |x: &B::Expr, y: &B::Expr| x.clone() + y;
            let form_expr = |f: &Form, vals: &mut Vec<Option<B::Expr>>| -> B::Expr {
                let mut acc = from_ref(sc.get(f.konst));
                for (c, v) in &f.terms {
                    let e = val_expr(*v, vals, &hl.b, &hcell, &chunk, &add);
                    acc += &mbs(&e, sc.get(*c));
                }
                acc
            };
            let hlimbs = |cells: &[BCell]| -> B::Expr {
                let mut acc = hcell(*cells.last().unwrap());
                for &c in cells[..cells.len() - 1].iter().rev() {
                    acc = mbs(&acc, &sc.two16) + &hcell(c);
                }
                acc
            };
            let cube_eq = |b: &mut B, lin: B::Expr, out: (BCell, BCell), k: &[BCell]| {
                let cube = lin.clone() * &lin * &lin;
                let outv = chunk(&hcell(out.0), &hcell(out.1));
                let kv = hlimbs(k);
                gate(b, &s_hash, cube - &outv - &mbs(&kv, &sc.qp));
            };
            for op in &hl.b.ops {
                match op {
                    Op::FullRound(fr) => {
                        let xs: Vec<B::Expr> =
                            fr.vals.iter().map(|v| val_expr(*v, &mut vals, &hl.b, &hcell, &chunk, &add)).collect();
                        let lin: Vec<B::Expr> = if fr.apply_me { me_expr(&xs) } else { xs };
                        for i in 0..16 {
                            let lx = lin[i].clone() + &from_ref(sc.get(fr.rc[i]));
                            cube_eq(b, lx, fr.outs[i], &fr.ks[i]);
                        }
                    }
                    Op::Reduce { form, x, m } => {
                        let f = form_expr(form, &mut vals);
                        let xv = chunk(&hcell(x.0), &hcell(x.1));
                        let mv = hlimbs(m);
                        gate(b, &s_hash, f - &xv - &mbs(&mv, &sc.qp));
                    }
                    Op::Sbox { x, rc, out, k } => {
                        let mut lx = val_expr(*x, &mut vals, &hl.b, &hcell, &chunk, &add);
                        if *rc != 0 {
                            lx += &from_ref(sc.get(*rc));
                        }
                        cube_eq(b, lx, *out, k);
                    }
                }
            }
            let start = pubc(PUB_START);
            let cont = pubc(PUB_CONT);
            let end = pubc(PUB_END);
            let ispub = pubc(PUB_FLAG);
            for lane in 0..16 {
                let (lo, hi) = hl.st[lane];
                let st_now = chunk(&hcell(lo), &hcell(hi));
                gate(b, &s_hash, start.clone() * &st_now);
                let st_next = chunk(&dwit(l.phys(hl.map[lo])), &dwit(l.phys(hl.map[hi])));
                let out = val_expr(hl.outs[lane], &mut vals, &hl.b, &hcell, &chunk, &add);
                gate(b, &s_hash, cont.clone() * &(st_next - &out));
                if lane < p2::DIGEST_LANES {
                    let d = chunk(&pubc(PUB_D + 2 * lane), &pubc(PUB_D + 2 * lane + 1));
                    gate(b, &s_hash, end.clone() * &(out - &d));
                }
            }
            let mut a_elems = Vec::with_capacity(RATE);
            for k in 0..RATE {
                let av = chunk(&hcell(hl.a[k].0), &hcell(hl.a[k].1));
                let pa = chunk(&pubc(PUB_A + 2 * k), &pubc(PUB_A + 2 * k + 1));
                gate(b, &s_hash, ispub.clone() * &(av.clone() - &pa));
                a_elems.push(av);
            }
            runs.push((pos_hash, a_elems));
        }

        // ================= enc rows =================
        if let Some(el) = &l.enc {
            // per-accumulator row sums
            let mut acc_sum: Vec<B::Expr> = (0..NUM_ENC_ACC).map(|a| pubc(pub_c(l, a / BETAS, a % BETAS))).collect();
            for e in 0..el.e_per_row {
                for c in 0..el.c_per_row {
                    let ec = &el.cells[e][c];
                    let tu = cell(ec.tu);
                    gate(b, &s_enc, tu.clone() * &(tu.clone() - &one) * &(tu.clone() - &two));
                    gate(b, &s_enc, cell(ec.e0.0) + &cell(ec.e0.1) - &from_ref(&sc.forty));
                    gate(b, &s_enc, cell(ec.e1.0) + &cell(ec.e1.1) - &from_ref(&sc.forty));
                    let k1 = v5(&ec.k1);
                    gate(b, &s_enc, k1.clone() + &v5(&ec.k1c) - &from_ref(&sc.t_m1));
                    if let Some((m, mc, z)) = &ec.m {
                        let mv = v5(m);
                        gate(b, &s_enc, mv.clone() + &v5(mc) - &from_ref(&sc.t_m1));
                        let zv = limbs16(z, &cell);
                        gate(b, &s_enc, k1.clone() + &mbs(&zv, &sc.t) - &mbs(&mv, &sc.q_mod_t));
                        runs.push((pos_enc, v5_chunks(m).to_vec())); // producer → msg sponge
                    } else {
                        runs.push((pos_enc, v5_chunks(&ec.k1).to_vec())); // consumer ← shr rows
                    }
                    pos_enc += 1;
                    let mut ct_run = Vec::with_capacity(12);
                    for i in 0..L_D {
                        for leg in 0..2 {
                            let (lim, comp) = &ec.ct[i][leg];
                            gate(b, &s_enc, v5(lim) + &v5(comp) - &from_ref(&sc.qd_m1[i]));
                            ct_run.extend(v5_chunks(lim));
                        }
                    }
                    runs.push((pos_enc, ct_run)); // producer → ct sponge
                    pos_enc += 1;
                    for i in 0..L_D {
                        let ct0 = v5(&ec.ct[i][0].0) + &cell(ec.e0.1);
                        let ct1 = v5(&ec.ct[i][1].0) + &cell(ec.e1.1);
                        for bb in 0..BETAS {
                            let a = i * BETAS + bb;
                            acc_sum[a] += &(pubc(pub_w(l, e, c, i, bb, 0)) * &ct0);
                            acc_sum[a] += &(pubc(pub_w(l, e, c, i, bb, 1)) * &ct1);
                            acc_sum[a] += &(pubc(pub_w(l, e, c, i, bb, 2)) * &tu);
                            acc_sum[a] += &(pubc(pub_w(l, e, c, i, bb, 3)) * &k1);
                        }
                    }
                }
            }
            for i in 0..L_D {
                for bb in 0..BETAS {
                    let a = arb_enc(i, bb, 0);
                    let s = core::mem::replace(&mut acc_sum[i * BETAS + bb], zero());
                    b.assert_in_ideal(darb_acc(a) - &acc_cur(a) - &s + &mbs(&uval(a, ENC_U_LIMBS), &sc.qd[i]), &ideal);
                    b.assert_in_ideal(pubc(PUB_FIRST) * &acc_cur(a), &ideal);
                    b.assert_in_ideal(pubc(PUB_LAST) * &darb_acc(a), &ideal);
                }
            }
        }

        // ================= shr rows =================
        if let Some(sh) = &l.shr {
            let f = pubc(PUB_FLAG); // sk-track flag (merged); constant per mode otherwise
            if let Some(skc) = sh.sk {
                let t = cell(skc);
                gate(b, &s_shr, t.clone() * &(t.clone() - &one) * &(t.clone() - &two));
                if sh.res.is_some() {
                    gate(b, &s_shr, (one.clone() - &f) * &t);
                }
                runs.push((pos_shr, vec![t]));
                pos_shr += 1;
            }
            if let Some(res) = &sh.res {
                let mut run = Vec::new();
                for j in 0..L_T {
                    let off = res_val(&res[j].0);
                    gate(b, &s_shr, off.clone() + &res_comp(&res[j].1) - &from_ref(&sc.qt_m1[j]));
                    if sh.sk.is_some() {
                        gate(b, &s_shr, f.clone() * &off);
                    }
                    run.extend(res_chunks(&res[j].0));
                }
                runs.push((pos_shr, run));
                pos_shr += 1;
            }
            for j in 0..L_T {
                let y: Vec<B::Expr> = sh.y[j].iter().map(|(lim, _)| v5(lim)).collect();
                for p in 0..N_PARTIES {
                    let (lim, comp) = &sh.y[j][p];
                    gate(b, &s_shr, y[p].clone() + &v5(comp) - &from_ref(&sc.qt_m1[j]));
                    if sh.k1.is_none() {
                        runs.push((pos_shr, v5_chunks(lim).to_vec())); // producer → share sponge
                        pos_shr += 1;
                    }
                }
                for r in 0..PARITY_ROWS {
                    let mut s = zero();
                    if let Some(skc) = sh.sk {
                        s += &mbs(&cell(skc), &sc.h[j][r][0]);
                    }
                    if let Some(res) = &sh.res {
                        s += &mbs(&res_val(&res[j].0), &sc.h[j][r][0]);
                    }
                    for p in 0..N_PARTIES {
                        s += &mbs(&y[p], &sc.h[j][r][p + 1]);
                    }
                    let kc = match (sh.sk.is_some(), sh.res.is_some()) {
                        (true, false) => from_ref(&sc.k_sk[j][r]),
                        (false, true) => from_ref(&sc.k_esm[j][r]),
                        _ => f.clone() * &from_ref(&sc.k_sk[j][r]) + &((one.clone() - &f) * &from_ref(&sc.k_esm[j][r])),
                    };
                    let w = limbs16(&sh.w[j][r], &cell);
                    gate(b, &s_shr, s + &kc - &mbs(&w, &sc.qt[j]));
                }
                if let Some(k1s) = &sh.k1 {
                    for p in 0..N_PARTIES {
                        let (z, k1, k1c) = &k1s[j][p];
                        let k1v = v5(k1);
                        gate(b, &s_shr, k1v.clone() + &v5(k1c) - &from_ref(&sc.t_m1));
                        let zv = limbs16(z, &cell);
                        gate(b, &s_shr, k1v + &mbs(&zv, &sc.t) - &mbs(&y[p], &sc.q_mod_t));
                        runs.push((pos_shr, v5_chunks(k1).to_vec())); // producer → enc rows
                        pos_shr += 1;
                    }
                }
            }
            assert_eq!(pos_shr, l.runs_enc + l.runs_shr);
        }
        assert_eq!(pos_enc, l.runs_enc);

        // ================= bus =================
        for k in 0..NUM_BUS {
            let mut sum = zero();
            for (pos, elems) in &runs {
                let base = pubc(pub_bus(l, k, *pos));
                let mut inner = elems[0].clone();
                for (e, v) in elems.iter().enumerate().skip(1) {
                    inner += &mbs(v, &sc.gpow[k][e]);
                }
                sum += &(base * &inner);
            }
            let a = arb_bus(k, 0);
            b.assert_in_ideal(darb_acc(a) - &acc_cur(a) - &sum + &mbs(&uval(a, BUS_U_LIMBS), &sc.bus_m), &ideal);
            b.assert_in_ideal(pubc(PUB_FIRST) * &acc_cur(a), &ideal);
            b.assert_in_ideal(pubc(PUB_LAST) * &darb_acc(a), &ideal);
        }
    }
}

// ---------------------------------------------------------------------------
// Dealer data, schedule
// ---------------------------------------------------------------------------

/// Everything the dealer holds (both tracks, all recipients).
pub struct Dealer {
    pub n: usize,
    pub c1: crate::c1_uair::C1Witness,
    /// `shares[track][j][m]` = `[secret, y_1, …, y_9]`
    pub shares: [[Vec<[u64; N_PARTIES + 1]>; L_T]; 2],
    /// recipient keys, party `p+1`
    pub pks: Vec<dkg::DkgPk>,
    /// `encs[track][j][p]`
    pub encs: [[Vec<dkg::Encryption>; L_T]; 2],
}

pub fn make_dealer(n: usize, seed: u64) -> Dealer {
    let public = ring::sample_a_n(seed, n);
    let c1 = crate::c1_uair::sample_witness(&public, seed + 1000);
    let mut rng = ChaCha20Rng::seed_from_u64(seed + 77);
    let secret = |track: usize, j: usize, m: usize| -> u64 {
        let q = QT[j] as i64;
        if track == 0 { c1.sk[m].rem_euclid(q) as u64 } else { c1.e_sm[j][m].rem_euclid(q) as u64 }
    };
    let shares: [[Vec<[u64; N_PARTIES + 1]>; L_T]; 2] = core::array::from_fn(|track| {
        core::array::from_fn(|j| (0..n).map(|m| dkg::shamir(secret(track, j, m), QT[j], &mut rng)).collect())
    });
    let pks: Vec<dkg::DkgPk> = (0..N_PARTIES).map(|p| dkg::keygen(n, seed + 500 + p as u64)).collect();
    let encs = core::array::from_fn(|track| {
        core::array::from_fn(|j| {
            (0..N_PARTIES)
                .map(|p| {
                    let msg: Vec<u64> = (0..n).map(|m| shares[track][j][m][p + 1]).collect();
                    dkg::encrypt(&pks[p], &msg, &mut rng)
                })
                .collect()
        })
    });
    Dealer { n, c1, shares, pks, encs }
}

#[derive(Clone, Debug)]
pub struct SpongeSpec {
    pub name: String,
    pub public: bool,
    /// number of absorbed elements (multiple of RATE)
    pub len: usize,
    /// the elements; for the verifier's schedule, witness sponges carry none
    pub elems: Vec<u64>,
    pub digest: [u64; 8],
    pub bus_off: usize,
}

pub struct Schedule {
    pub n: usize,
    pub mode: Mode,
    pub enc_list: Vec<(usize, usize, usize)>,
    pub enc_rows: usize,
    pub shr_rows: usize,
    pub sponges: Vec<SpongeSpec>,
    pub hash_rows: usize,
    pub num_vars: usize,
    pub row_enc0: usize,
    pub row_shr0: usize,
    pub row_hash0: usize,
    pub k1_off: usize,
}

impl Schedule {
    pub fn sponge(&self, name: &str) -> &SpongeSpec {
        self.sponges.iter().find(|s| s.name == name).unwrap_or_else(|| panic!("no sponge {name}"))
    }
    pub fn k1_idx(&self, t: usize, j: usize, p: usize, m: usize) -> usize {
        self.k1_off + ((t * L_T + j) * N_PARTIES + p) * 3 * self.n + 3 * m
    }
}

fn sponge_of(name: String, public: bool, elems: Vec<u64>, bus_off: &mut usize) -> SpongeSpec {
    assert!(elems.len() % RATE == 0, "sponge {name}: {} elements", elems.len());
    assert!(elems.iter().all(|&e| e < Q));
    let digest = p2::sponge(&elems, RATE);
    let s = SpongeSpec { name, public, len: elems.len(), elems, digest, bus_off: *bus_off };
    if !public {
        *bus_off += s.len;
    }
    s
}

/// Encryptions of a mode as `(track, j, p)`.
pub fn enc_list(mode: Mode) -> Vec<(usize, usize, usize)> {
    match mode {
        Mode::C3 => vec![(0, 0, 0)],
        Mode::Merged => {
            let mut v = Vec::new();
            for t in 0..2 {
                for j in 0..L_T {
                    for p in 0..N_PARTIES {
                        v.push((t, j, p));
                    }
                }
            }
            v
        }
        Mode::C2(_) => vec![],
    }
}

pub fn sk_name() -> String {
    "sk".into()
}
pub fn esm_name() -> String {
    "esm".into()
}
pub fn share_name(t: usize, j: usize, p: usize) -> String {
    format!("share[{t}][{j}][{}]", p + 1)
}
pub fn pk_name(p: usize) -> String {
    format!("pk[{}]", p + 1)
}
pub fn ct_name(t: usize, j: usize, p: usize) -> String {
    format!("ct[{t}][{j}][{}]", p + 1)
}

/// Hash serializations (chunks `< 2^24`): `sk ↦ sk+1`; `[0,q)` values and
/// centered values offset by `(q−1)/2` as (24, 24, rest) chunks.
pub fn schedule(d: &Dealer, mode: Mode) -> Schedule {
    let n = d.n;
    let c = |q: u64| (q - 1) / 2;
    let mut bus_off = 0usize;
    let mut sponges = Vec::new();
    let sk_elems = || -> Vec<u64> { d.c1.sk.iter().map(|&x| (x + 1) as u64).collect() };
    // C1's `limb_stream`: per m, per j, the (29, 30)-bit limbs of the offset residue
    let esm_elems = || -> Vec<u64> {
        let mut v = Vec::new();
        for m in 0..n {
            for j in 0..L_T {
                let (l0, l1) = crate::c1_uair::pk_limbs((d.c1.e_sm[j][m] + c(QT[j]) as i64) as u64);
                v.push(l0);
                v.push(l1);
            }
        }
        v
    };
    let share_elems = |t: usize, j: usize, p: usize| -> Vec<u64> {
        (0..n).flat_map(|m| chunks3(d.shares[t][j][m][p + 1])).collect()
    };
    let pk_elems = |p: usize| -> Vec<u64> {
        let pk = &d.pks[p];
        let mut v = Vec::new();
        for m in 0..n {
            for i in 0..L_D {
                v.extend(chunks3((dkg::center(pk.pk0[i][m], QD[i]) + c(QD[i]) as i64) as u64));
                v.extend(chunks3((dkg::center(pk.pk1[i][m], QD[i]) + c(QD[i]) as i64) as u64));
            }
        }
        v
    };
    let ct_elems = |t: usize, j: usize, p: usize| -> Vec<u64> {
        let e = &d.encs[t][j][p];
        let mut v = Vec::new();
        for m in 0..n {
            for i in 0..L_D {
                v.extend(chunks3((dkg::center(e.ct0[i][m], QD[i]) + c(QD[i]) as i64) as u64));
                v.extend(chunks3((dkg::center(e.ct1[i][m], QD[i]) + c(QD[i]) as i64) as u64));
            }
        }
        v
    };
    let encs = enc_list(mode);
    let (enc_rows, shr_rows);
    match mode {
        Mode::C2(track) => {
            let t = if track == Track::Sk { 0 } else { 1 };
            if t == 0 {
                sponges.push(sponge_of(sk_name(), false, sk_elems(), &mut bus_off));
            } else {
                sponges.push(sponge_of(esm_name(), false, esm_elems(), &mut bus_off));
            }
            for j in 0..L_T {
                for p in 0..N_PARTIES {
                    sponges.push(sponge_of(share_name(t, j, p), false, share_elems(t, j, p), &mut bus_off));
                }
            }
            enc_rows = 0;
            shr_rows = n;
        }
        Mode::C3 => {
            let (t, j, p) = encs[0];
            sponges.push(sponge_of("msg".into(), false, share_elems(t, j, p), &mut bus_off));
            sponges.push(sponge_of(pk_name(p), true, pk_elems(p), &mut bus_off));
            sponges.push(sponge_of(ct_name(t, j, p), false, ct_elems(t, j, p), &mut bus_off));
            enc_rows = n / 8;
            shr_rows = 0;
        }
        Mode::Merged => {
            sponges.push(sponge_of(sk_name(), false, sk_elems(), &mut bus_off));
            sponges.push(sponge_of(esm_name(), false, esm_elems(), &mut bus_off));
            for p in 0..N_PARTIES {
                sponges.push(sponge_of(pk_name(p), true, pk_elems(p), &mut bus_off));
            }
            for &(t, j, p) in &encs {
                sponges.push(sponge_of(ct_name(t, j, p), false, ct_elems(t, j, p), &mut bus_off));
            }
            enc_rows = encs.len() / N_PARTIES * n;
            shr_rows = 2 * n;
        }
    }
    // the opened commitments are exactly C1's outputs
    let cm = crate::c1_uair::commit(&d.c1);
    for sp in &sponges {
        if sp.name == sk_name() {
            assert_eq!(sp.digest, cm.sk, "sk sponge differs from C1's commit(sk)");
        }
        if sp.name == esm_name() {
            assert_eq!(sp.digest, cm.esm, "e_sm sponge differs from C1's commit(e_sm)");
        }
    }
    let k1_off = bus_off;
    let hash_rows: usize = sponges.iter().map(|s| s.len / RATE).sum();
    let row_enc0 = 0;
    let row_shr0 = row_enc0 + enc_rows + 1;
    let row_hash0 = row_shr0 + shr_rows + 1;
    let total = row_hash0 + hash_rows + 2; // two trailing separator rows (PUB_LAST sits on the first of them)
    let mut nv = 6; // Zip+ needs ≥ 64 rows
    while (1usize << nv) < total {
        nv += 1;
    }
    Schedule { n, mode, enc_list: encs, enc_rows, shr_rows, sponges, hash_rows, num_vars: nv, row_enc0, row_shr0, row_hash0, k1_off }
}

// ---------------------------------------------------------------------------
// Statement, challenges, public columns
// ---------------------------------------------------------------------------

/// What the verifier knows: the commitments (sponge digests) and the
/// recipients' public keys (public in C3: `pk` is a public input hashed
/// in-circuit, and the identity weights need it in the clear).
#[derive(Clone)]
pub struct Statement {
    pub n: usize,
    pub digests: Vec<(String, [u64; 8])>,
    pub pks: Vec<dkg::DkgPk>,
}

pub fn statement(d: &Dealer, s: &Schedule) -> Statement {
    Statement { n: d.n, digests: s.sponges.iter().map(|sp| (sp.name.clone(), sp.digest)).collect(), pks: d.pks.clone() }
}

#[derive(Clone, Debug)]
pub struct Challenges {
    pub gammas: [u64; NUM_BUS],
    pub betas: [[u64; BETAS]; L_D],
    /// `rhos[i][b][enc_idx][leg]`
    pub rhos: Vec<Vec<Vec<[u64; 2]>>>,
}

fn fs(domain: &[u8], tags: &[u8], root: &[u8], modulus: u64) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(b"zincbench/vm/");
    h.update(domain);
    h.update(tags);
    h.update(root);
    let x = u128::from_le_bytes(h.finalize().as_bytes()[..16].try_into().unwrap());
    (x % modulus as u128) as u64
}

/// All challenges from the int-lane commitment root (and the statement).
pub fn challenges(root: &[u8], stmt: &Statement, num_encs: usize) -> Challenges {
    let mut h = blake3::Hasher::new();
    h.update(root);
    for (name, dg) in &stmt.digests {
        h.update(name.as_bytes());
        for x in dg {
            h.update(&x.to_le_bytes());
        }
    }
    let bound = h.finalize();
    let r = bound.as_bytes();
    let gammas = core::array::from_fn(|k| 2 + fs(b"bus", &[k as u8], r, BUS_M - 2));
    let betas = core::array::from_fn(|i| core::array::from_fn(|b| fs(b"beta", &[i as u8, b as u8], r, QD[i])));
    let rhos = (0..L_D)
        .map(|i| {
            (0..BETAS)
                .map(|b| {
                    (0..num_encs)
                        .map(|e| core::array::from_fn(|leg| fs(b"rho", &[i as u8, b as u8, e as u8, leg as u8], r, QD[i])))
                        .collect()
                })
                .collect()
        })
        .collect();
    Challenges { gammas, betas, rhos }
}

/// Public int columns (verifier side).
pub fn public_columns(l: &Layout, s: &Schedule, stmt: &Statement, ch: &Challenges) -> Vec<Vec<i64>> {
    let np = num_pub(l);
    let rows = 1usize << s.num_vars;
    let mut cols = vec![vec![0i64; rows]; np];
    cols[PUB_FIRST][0] = 1;
    cols[PUB_LAST][rows - 2] = 1;
    let n = s.n;
    let gp = |k: usize, e: usize| ring::powmod(ch.gammas[k], e as u64, BUS_M);
    let pos_hash = l.runs_enc + l.runs_shr;

    // ---- enc rows ----
    if let Some(el) = &l.enc {
        let groups = s.enc_list.len() / el.e_per_row;
        let rows_per_group = n / el.c_per_row;
        let u_w: Vec<Vec<ring::Weights>> =
            (0..L_D).map(|i| (0..BETAS).map(|b| ring::weights(&vec![0u64; n], QD[i], ch.betas[i][b])).collect()).collect();
        for g in 0..groups {
            let encs = &s.enc_list[g * el.e_per_row..(g + 1) * el.e_per_row];
            // V weights per (e, i, leg, b)
            let v_w: Vec<Vec<Vec<Vec<ring::Weights>>>> = encs
                .iter()
                .map(|&(_, _, p)| {
                    (0..L_D)
                        .map(|i| {
                            (0..2)
                                .map(|leg| {
                                    let pk = if leg == 0 { &stmt.pks[p].pk0[i] } else { &stmt.pks[p].pk1[i] };
                                    (0..BETAS).map(|b| ring::weights(pk, QD[i], ch.betas[i][b])).collect()
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();
            for rr in 0..rows_per_group {
                let r = s.row_enc0 + g * rows_per_group + rr;
                cols[PUB_S_ENC][r] = 1;
                for e in 0..el.e_per_row {
                    let (t, j, p) = encs[e];
                    let eidx = g * el.e_per_row + e;
                    for c in 0..el.c_per_row {
                        let m = rr * el.c_per_row + c;
                        for i in 0..L_D {
                            let q = QD[i];
                            let cq = (q - 1) / 2;
                            for b in 0..BETAS {
                                let um = u_w[i][b].u[m];
                                let [r0, r1] = ch.rhos[i][b][eidx];
                                let w0 = mulmod(r0, um, q);
                                let w1 = mulmod(r1, um, q);
                                let v0 = v_w[e][i][0][b].v[m];
                                let v1 = v_w[e][i][1][b].v[m];
                                let wv = addmod(mulmod(r0, v0, q), mulmod(r1, v1, q), q);
                                cols[pub_w(l, e, c, i, b, 0)][r] = w0 as i64;
                                cols[pub_w(l, e, c, i, b, 1)][r] = w1 as i64;
                                cols[pub_w(l, e, c, i, b, 2)][r] = submod(0, wv, q) as i64;
                                cols[pub_w(l, e, c, i, b, 3)][r] = submod(0, mulmod(w0, K0[i], q), q) as i64;
                                // constants: −(w0 + w1)(c + 20) + wv
                                let k = addmod(submod(0, mulmod(addmod(w0, w1, q), cq + 20, q), q), wv, q);
                                let cc = &mut cols[pub_c(l, i, b)][r];
                                *cc = addmod(*cc as u64, k, q) as i64;
                            }
                        }
                        let run0 = (e * el.c_per_row + c) * 2;
                        for k in 0..NUM_BUS {
                            let (idx_a, consumer) = match s.mode {
                                Mode::C3 => (s.sponge("msg").bus_off + 3 * m, false),
                                _ => (s.k1_idx(t, j, p, m), true),
                            };
                            let base_a = gp(k, idx_a);
                            cols[pub_bus(l, k, run0)][r] = if consumer { BUS_M - base_a } else { base_a } as i64;
                            cols[pub_bus(l, k, run0 + 1)][r] = gp(k, s.sponge(&ct_name(t, j, p)).bus_off + 12 * m) as i64;
                        }
                    }
                }
            }
        }
    }
    // ---- shr rows ----
    if let Some(sh) = &l.shr {
        let tracks: Vec<usize> = match s.mode {
            Mode::C2(Track::Sk) => vec![0],
            Mode::C2(Track::Esm) => vec![1],
            _ => vec![0, 1],
        };
        for (ti, &t) in tracks.iter().enumerate() {
            for m in 0..n {
                let r = s.row_shr0 + ti * n + m;
                cols[PUB_S_SHR][r] = 1;
                if t == 0 {
                    cols[PUB_FLAG][r] = 1;
                }
                let mut pos = l.runs_enc;
                let set = |cols: &mut Vec<Vec<i64>>, pos: &mut usize, idx: Option<usize>| {
                    for k in 0..NUM_BUS {
                        cols[pub_bus(l, k, *pos)][r] = idx.map_or(0, |x| gp(k, x) as i64);
                    }
                    *pos += 1;
                };
                if sh.sk.is_some() {
                    set(&mut cols, &mut pos, (t == 0).then(|| s.sponge(&sk_name()).bus_off + m));
                }
                if sh.res.is_some() {
                    set(&mut cols, &mut pos, (t == 1).then(|| s.sponge(&esm_name()).bus_off + 6 * m));
                }
                for j in 0..L_T {
                    for p in 0..N_PARTIES {
                        if sh.k1.is_none() {
                            set(&mut cols, &mut pos, Some(s.sponge(&share_name(t, j, p)).bus_off + 3 * m));
                        } else {
                            set(&mut cols, &mut pos, Some(s.k1_idx(t, j, p, m)));
                        }
                    }
                }
            }
        }
    }
    // ---- hash rows ----
    {
        let mut r = s.row_hash0;
        for sp in &s.sponges {
            let blocks = sp.len / RATE;
            let dg = stmt.digests.iter().find(|(nm, _)| *nm == sp.name).map(|x| x.1).expect("digest in statement");
            for blk in 0..blocks {
                cols[PUB_S_HASH][r] = 1;
                if blk == 0 {
                    cols[PUB_START][r] = 1;
                }
                if blk + 1 < blocks {
                    cols[PUB_CONT][r] = 1;
                } else {
                    cols[PUB_END][r] = 1;
                    for lane in 0..8 {
                        cols[PUB_D + 2 * lane][r] = (dg[lane] & 0xFFFF) as i64;
                        cols[PUB_D + 2 * lane + 1][r] = (dg[lane] >> 16) as i64;
                    }
                }
                if sp.public {
                    cols[PUB_FLAG][r] = 1;
                    for k in 0..RATE {
                        let v = sp.elems[blk * RATE + k];
                        cols[PUB_A + 2 * k][r] = (v & 0xFFFF) as i64;
                        cols[PUB_A + 2 * k + 1][r] = (v >> 16) as i64;
                    }
                } else {
                    for k in 0..NUM_BUS {
                        cols[pub_bus(l, k, pos_hash)][r] = (BUS_M - gp(k, sp.bus_off + RATE * blk)) as i64;
                    }
                }
                r += 1;
            }
        }
    }
    cols
}

/// Public-data sponge contents needed by the verifier to build `PUB_A`
/// come from the statement's `pks`; `schedule` recomputes them from the
/// dealer. For the verifier we rebuild only the public parts.
pub fn verifier_schedule(stmt: &Statement, mode: Mode) -> Schedule {
    // Build a dealer-free schedule: witness sponges only need their length
    // and bus offsets; public ones need their elements. We reuse `schedule`
    // with a dummy dealer sharing `n` and `pks` — the witness-sponge elements
    // are never read by `public_columns`.
    let d = dealer_shell(stmt);
    let mut s = schedule(&d, mode);
    for sp in s.sponges.iter_mut() {
        if !sp.public {
            sp.elems.clear(); // not the verifier's business
        }
    }
    s
}

fn dealer_shell(stmt: &Statement) -> Dealer {
    let n = stmt.n;
    let zero_enc = || dkg::Encryption {
        u: vec![0; n],
        e0: vec![0; n],
        e1: vec![0; n],
        k1: vec![0; n],
        z: vec![0; n],
        ct0: core::array::from_fn(|_| vec![0; n]),
        ct1: core::array::from_fn(|_| vec![0; n]),
    };
    Dealer {
        n,
        c1: crate::c1_uair::C1Witness {
            sk: vec![0; n],
            eek: vec![0; n],
            pk0: core::array::from_fn(|_| vec![0; n]),
            e_sm_lifted: vec![num_bigint::BigInt::zero(); n],
            e_sm: core::array::from_fn(|_| vec![0; n]),
            e_sm_quot: core::array::from_fn(|_| vec![num_bigint::BigInt::zero(); n]),
        },
        shares: core::array::from_fn(|_| core::array::from_fn(|_| vec![[0u64; N_PARTIES + 1]; n])),
        pks: stmt.pks.clone(),
        encs: core::array::from_fn(|_| core::array::from_fn(|_| (0..N_PARTIES).map(|_| zero_enc()).collect())),
    }
}

// ---------------------------------------------------------------------------
// Witness columns (prover side)
// ---------------------------------------------------------------------------

pub fn witness_columns(l: &Layout, s: &Schedule, d: &Dealer) -> Vec<Vec<i64>> {
    let rows = 1usize << s.num_vars;
    let mut cols = vec![vec![0i64; rows]; l.num_wit];
    let n = s.n;
    let c = |q: u64| (q - 1) / 2;
    let set = |cols: &mut Vec<Vec<i64>>, r: usize, cell: Cell, v: u64| {
        if let Some(w) = cell.0.width() {
            assert!(v < (1u64 << w), "cell {cell:?} = {v} overflows {w} bits");
        }
        cols[l.phys(cell)][r] = v as i64;
    };
    let set5 = |cols: &mut Vec<Vec<i64>>, r: usize, cells: &[Cell; 5], v: u64| {
        for (k, x) in split5(v).iter().enumerate() {
            set(cols, r, cells[k], *x);
        }
    };
    let set16 = |cols: &mut Vec<Vec<i64>>, r: usize, cells: &[Cell], mut v: u128| {
        for &cc in cells {
            set(cols, r, cc, (v & 0xFFFF) as u64);
            v >>= 16;
        }
        assert_eq!(v, 0, "limb overflow");
    };
    // ---- enc rows ----
    if let Some(el) = &l.enc {
        let groups = s.enc_list.len() / el.e_per_row;
        let rows_per_group = n / el.c_per_row;
        for g in 0..groups {
            let encs = &s.enc_list[g * el.e_per_row..(g + 1) * el.e_per_row];
            for rr in 0..rows_per_group {
                let r = s.row_enc0 + g * rows_per_group + rr;
                for e in 0..el.e_per_row {
                    let (t, j, p) = encs[e];
                    let en = &d.encs[t][j][p];
                    for cc in 0..el.c_per_row {
                        let m = rr * el.c_per_row + cc;
                        let ec = &el.cells[e][cc];
                        set(&mut cols, r, ec.tu, (en.u[m] + 1) as u64);
                        let e0 = (en.e0[m] + 20) as u64;
                        set(&mut cols, r, ec.e0.0, e0);
                        set(&mut cols, r, ec.e0.1, 40 - e0);
                        let e1 = (en.e1[m] + 20) as u64;
                        set(&mut cols, r, ec.e1.0, e1);
                        set(&mut cols, r, ec.e1.1, 40 - e1);
                        set5(&mut cols, r, &ec.k1, en.k1[m]);
                        set5(&mut cols, r, &ec.k1c, PT - 1 - en.k1[m]);
                        if let Some((mm, mc, z)) = &ec.m {
                            let msg = d.shares[t][j][m][p + 1];
                            set5(&mut cols, r, mm, msg);
                            set5(&mut cols, r, mc, PT - 1 - msg);
                            set16(&mut cols, r, z, en.z[m] as u128);
                        }
                        for i in 0..L_D {
                            for leg in 0..2 {
                                let ctv = if leg == 0 { en.ct0[i][m] } else { en.ct1[i][m] };
                                let off = (dkg::center(ctv, QD[i]) + c(QD[i]) as i64) as u64;
                                set5(&mut cols, r, &ec.ct[i][leg].0, off);
                                set5(&mut cols, r, &ec.ct[i][leg].1, QD[i] - 1 - off);
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- shr rows ----
    if let Some(sh) = &l.shr {
        let tracks: Vec<usize> = match s.mode {
            Mode::C2(Track::Sk) => vec![0],
            Mode::C2(Track::Esm) => vec![1],
            _ => vec![0, 1],
        };
        let h = dkg::parity_matrix();
        for (ti, &t) in tracks.iter().enumerate() {
            for m in 0..n {
                let r = s.row_shr0 + ti * n + m;
                if let Some(skc) = sh.sk {
                    set(&mut cols, r, skc, if t == 0 { (d.c1.sk[m] + 1) as u64 } else { 0 });
                }
                if let Some(res) = &sh.res {
                    for j in 0..L_T {
                        let off = if t == 1 { (d.c1.e_sm[j][m] + c(QT[j]) as i64) as u64 } else { 0 };
                        assert!(off < (1 << 59), "e_sm residue offset outside C1's 59-bit layout");
                        let (l0, l1) = crate::c1_uair::pk_limbs(off);
                        set(&mut cols, r, res[j].0[0], l0 & 0xFFFF);
                        set(&mut cols, r, res[j].0[1], l0 >> 16);
                        set(&mut cols, r, res[j].0[2], l1 & 0xFFFF);
                        set(&mut cols, r, res[j].0[3], l1 >> 16);
                        set16(&mut cols, r, &res[j].1, (QT[j] - 1 - off) as u128);
                    }
                }
                for j in 0..L_T {
                    let q = QT[j];
                    let y = &d.shares[t][j][m];
                    for p in 0..N_PARTIES {
                        set5(&mut cols, r, &sh.y[j][p].0, y[p + 1]);
                        set5(&mut cols, r, &sh.y[j][p].1, q - 1 - y[p + 1]);
                    }
                    for rr in 0..PARITY_ROWS {
                        let h0 = h[j][rr][0] as u128;
                        let mut sum: u128 = if t == 0 {
                            h0 * (d.c1.sk[m] + 1) as u128 + (q - h[j][rr][0]) as u128
                        } else {
                            h0 * (d.c1.e_sm[j][m] + c(q) as i64) as u128 + submod(0, mulmod(h[j][rr][0], c(q), q), q) as u128
                        };
                        for p in 0..N_PARTIES {
                            sum += h[j][rr][p + 1] as u128 * y[p + 1] as u128;
                        }
                        assert_eq!(sum % q as u128, 0, "parity violated");
                        set16(&mut cols, r, &sh.w[j][rr], sum / q as u128);
                    }
                    if let Some(k1s) = &sh.k1 {
                        for p in 0..N_PARTIES {
                            let (k1, z) = dkg::scale_message(y[p + 1]);
                            let (zc, k1c, k1cc) = &k1s[j][p];
                            set16(&mut cols, r, zc, z as u128);
                            set5(&mut cols, r, k1c, k1);
                            set5(&mut cols, r, k1cc, PT - 1 - k1);
                        }
                    }
                }
            }
        }
    }
    // ---- hash rows ----
    {
        let hl = &l.hash;
        let mut r = s.row_hash0;
        for sp in &s.sponges {
            let blocks = sp.elems.len() / RATE;
            let mut state = [0u64; 16];
            for blk in 0..blocks {
                let mut it = Interp::new(&hl.b);
                for k in 0..RATE {
                    let v = sp.elems[blk * RATE + k];
                    it.set(hl.a[k].0, v & 0xFFFF);
                    it.set(hl.a[k].1, v >> 16);
                }
                for lane in 0..16 {
                    it.set(hl.st[lane].0, state[lane] & 0xFFFF);
                    it.set(hl.st[lane].1, state[lane] >> 16);
                }
                it.run(&hl.b.ops);
                for lane in 0..16 {
                    state[lane] = it.val(hl.outs[lane]) as u64;
                }
                for (bc, &pc) in hl.map.iter().enumerate() {
                    cols[l.phys(pc)][r] = it.get(bc) as i64;
                }
                r += 1;
            }
            assert_eq!(&state[..8], &sp.digest[..], "sponge {} digest", sp.name);
        }
    }
    cols
}

/// Arbitrary-lane columns: bus and encryption accumulators, from the public
/// and int witness columns (prover side, after the challenges are known).
pub fn arb_columns(l: &Layout, s: &Schedule, pubcols: &[Vec<i64>], wit: &[Vec<i64>], ch: &Challenges) -> Vec<Vec<i64>> {
    arb_columns_opt(l, s, pubcols, wit, ch, true)
}

/// `strict = false` skips the final balance assertions (tamper tests).
pub fn arb_columns_opt(l: &Layout, s: &Schedule, pubcols: &[Vec<i64>], wit: &[Vec<i64>], ch: &Challenges, strict: bool) -> Vec<Vec<i64>> {
    let rows = 1usize << s.num_vars;
    let mut arb = vec![vec![0i64; rows]; NUM_ARB];
    let pos_hash = l.runs_enc + l.runs_shr;
    let gp: Vec<Vec<u64>> =
        (0..NUM_BUS).map(|k| (0..12).map(|e| ring::powmod(ch.gammas[k], e as u64, BUS_M)).collect()).collect();
    let big = |x: u64| BigUint::from(x);
    // bus run elements of a row, by position (mirrors constrain_general)
    let row_runs = |r: usize| -> Vec<(usize, Vec<u64>)> {
        let get = |c: Cell| wit[l.phys(c)][r] as u64;
        let ch3 = |c: &[Cell; 5]| -> [u64; 3] { [get(c[0]) + (get(c[1]) << 16), get(c[2]) + (get(c[3]) << 16), get(c[4])] };
        let mut runs = Vec::new();
        let mut pos = 0usize;
        if let Some(el) = &l.enc {
            for e in 0..el.e_per_row {
                for c in 0..el.c_per_row {
                    let ec = &el.cells[e][c];
                    runs.push((pos, match &ec.m {
                        Some((m, _, _)) => ch3(m).to_vec(),
                        None => ch3(&ec.k1).to_vec(),
                    }));
                    pos += 1;
                    let mut ct = Vec::new();
                    for i in 0..L_D {
                        for leg in 0..2 {
                            ct.extend(ch3(&ec.ct[i][leg].0));
                        }
                    }
                    runs.push((pos, ct));
                    pos += 1;
                }
            }
        }
        if let Some(sh) = &l.shr {
            if let Some(skc) = sh.sk {
                runs.push((pos, vec![get(skc)]));
                pos += 1;
            }
            if let Some(res) = &sh.res {
                let ch2 = |c: &[Cell; 4]| [get(c[0]) + (get(c[1]) << 16), get(c[2]) + (get(c[3]) << 16)];
                runs.push((pos, (0..L_T).flat_map(|j| ch2(&res[j].0)).collect()));
                pos += 1;
            }
            for j in 0..L_T {
                for p in 0..N_PARTIES {
                    match &sh.k1 {
                        None => runs.push((pos, ch3(&sh.y[j][p].0).to_vec())),
                        Some(k1s) => runs.push((pos, ch3(&k1s[j][p].1).to_vec())),
                    }
                    pos += 1;
                }
            }
        }
        assert_eq!(pos, pos_hash);
        let hl = &l.hash;
        runs.push((pos, (0..RATE).map(|k| get(hl.map[hl.a[k].0]) + (get(hl.map[hl.a[k].1]) << 16)).collect()));
        runs
    };
    let put = |arb: &mut Vec<Vec<i64>>, a: usize, limbs: usize, r: usize, acc: u64, u: &BigUint| {
        arb[a][r] = (acc & ((1 << ARB_LIMB) - 1)) as i64;
        arb[a + 1][r] = (acc >> ARB_LIMB) as i64;
        let mut u = u.clone();
        for i in 0..limbs {
            arb[a + 2 + i][r] = (&u % (1u64 << ARB_LIMB)).to_u64().unwrap() as i64;
            u >>= ARB_LIMB;
        }
        assert!(u.is_zero(), "quotient overflow");
    };
    let m_big = big(BUS_M);
    let mut bus_acc = [0u64; NUM_BUS];
    let mut enc_acc = [0u64; NUM_ENC_ACC];
    for r in 0..rows {
        let runs = row_runs(r);
        for k in 0..NUM_BUS {
            let mut total = big(bus_acc[k]);
            for (pos, elems) in &runs {
                let base = pubcols[pub_bus(l, k, *pos)][r] as u64;
                if base == 0 {
                    continue;
                }
                let mut inner = BigUint::zero();
                for (e, v) in elems.iter().enumerate() {
                    inner += big(gp[k][e]) * big(*v);
                }
                total += big(base) * inner;
            }
            let u = &total / &m_big;
            let acc = (&total % &m_big).to_u64().unwrap();
            put(&mut arb, arb_bus(k, 0), BUS_U_LIMBS, r, bus_acc[k], &u);
            bus_acc[k] = acc;
        }
        if let Some(el) = &l.enc {
            for i in 0..L_D {
                for b in 0..BETAS {
                    let a = i * BETAS + b;
                    let q = big(QD[i]);
                    let mut total = big(enc_acc[a]) + big(pubcols[pub_c(l, i, b)][r] as u64);
                    for e in 0..el.e_per_row {
                        for c in 0..el.c_per_row {
                            let ec = &el.cells[e][c];
                            let get = |cc: Cell| wit[l.phys(cc)][r] as u64;
                            let v5 = |cs: &[Cell; 5]| -> u64 {
                                let ch = [get(cs[0]) + (get(cs[1]) << 16), get(cs[2]) + (get(cs[3]) << 16), get(cs[4])];
                                ch[0] + (ch[1] << 24) + (ch[2] << 48)
                            };
                            let w = |which: usize| big(pubcols[pub_w(l, e, c, i, b, which)][r] as u64);
                            total += w(0) * big(v5(&ec.ct[i][0].0) + get(ec.e0.1));
                            total += w(1) * big(v5(&ec.ct[i][1].0) + get(ec.e1.1));
                            total += w(2) * big(get(ec.tu));
                            total += w(3) * big(v5(&ec.k1));
                        }
                    }
                    let u = &total / &q;
                    let acc = (&total % &q).to_u64().unwrap();
                    put(&mut arb, arb_enc(i, b, 0), ENC_U_LIMBS, r, enc_acc[a], &u);
                    enc_acc[a] = acc;
                }
            }
        }
    }
    if strict {
        for k in 0..NUM_BUS {
            assert_eq!(bus_acc[k], 0, "bus {k} does not balance (producer/consumer mismatch)");
        }
        for a in 0..NUM_ENC_ACC {
            assert_eq!(enc_acc[a], 0, "encryption accumulator {a} does not vanish");
        }
    }
    arb
}

pub fn int_mles(cols: Vec<Vec<i64>>) -> Vec<DenseMultilinearExtension<i64>> {
    cols.into_iter().map(|v| v.into_iter().collect()).collect()
}
pub fn arb_mles(cols: Vec<Vec<i64>>) -> Vec<DenseMultilinearExtension<ArbCell>> {
    cols.into_iter().map(|v| v.into_iter().map(|x| DensePolynomial::new([x])).collect()).collect()
}

pub fn full_trace(
    pubcols: Vec<Vec<i64>>,
    wit: Vec<Vec<i64>>,
    arb: Vec<Vec<i64>>,
) -> UairTrace<'static, i64, i64, D> {
    let int: Vec<Vec<i64>> = pubcols.into_iter().chain(wit).collect();
    UairTrace { int: int_mles(int).into(), arbitrary_poly: arb_mles(arb).into(), ..Default::default() }
}
pub fn public_trace(pubcols: Vec<Vec<i64>>) -> UairTrace<'static, i64, i64, D> {
    UairTrace { int: int_mles(pubcols).into(), ..Default::default() }
}

/// Reference check of the whole dealer statement (what the circuits prove).
pub fn dealer_holds(d: &Dealer) -> bool {
    let h = dkg::parity_matrix();
    for t in 0..2 {
        for j in 0..L_T {
            for m in 0..d.n {
                let y = &d.shares[t][j][m];
                let sec = if t == 0 { d.c1.sk[m] } else { d.c1.e_sm[j][m] }.rem_euclid(QT[j] as i64) as u64;
                if y[0] != sec || !dkg::parity_holds(y, &h[j], QT[j]) {
                    return false;
                }
            }
            for p in 0..N_PARTIES {
                let msg: Vec<u64> = (0..d.n).map(|m| d.shares[t][j][m][p + 1]).collect();
                if !dkg::encryption_holds(&d.pks[p], &msg, &d.encs[t][j][p]) {
                    return false;
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Exact integer constraint checker (debugging / tests)
// ---------------------------------------------------------------------------

pub mod check {
    use super::*;
    use zinc_uair::ideal::{Ideal, IdealCheck, IdealCheckError};
    use zinc_utils::from_ref::FromRef;

    pub type X = Int<4>;

    #[derive(Clone, Debug)]
    pub struct ZeroIdeal;
    impl Ideal for ZeroIdeal {}
    impl FromRef<ZeroIdeal> for ZeroIdeal {
        fn from_ref(_: &ZeroIdeal) -> Self {
            ZeroIdeal
        }
    }
    impl IdealCheck<X> for ZeroIdeal {
        fn contains(&self, v: &X) -> Result<bool, IdealCheckError> {
            Ok(*v == X::from(0i64))
        }
    }

    pub struct Checker {
        pub idx: usize,
        pub failed: Vec<(usize, X)>,
    }
    impl ConstraintBuilder for Checker {
        type Expr = X;
        type Ideal = ZeroIdeal;
        fn assert_in_ideal(&mut self, e: X, _: &ZeroIdeal) {
            if e != X::from(0i64) {
                self.failed.push((self.idx, e));
            }
            self.idx += 1;
        }
        fn assert_zero(&mut self, e: X) {
            self.assert_in_ideal(e, &ZeroIdeal);
        }
    }

    /// Evaluate every constraint of `VmUair` on every row of the trace over
    /// the integers. Returns `(row, constraint index, value)` for failures.
    pub fn check_trace(trace: &UairTrace<'static, i64, i64, D>) -> Vec<(usize, usize, X)> {
        let sig = VmUair::signature();
        let rows = trace.int[0].evaluations.len();
        let n_arb = trace.arbitrary_poly.len();
        let mut fails = Vec::new();
        let xi = |v: i64| X::from(v);
        for r in 0..rows {
            let arb: Vec<X> = trace.arbitrary_poly.iter().map(|c| xi(c.evaluations[r].coeffs[0])).collect();
            let int: Vec<X> = trace.int.iter().map(|c| xi(c.evaluations[r])).collect();
            let mut d_arb = Vec::new();
            let mut d_int = Vec::new();
            for sp in sig.shifts() {
                let rr = r + sp.shift_amount();
                let src = sp.source_col();
                if src < n_arb {
                    d_arb.push(if rr < rows { xi(trace.arbitrary_poly[src].evaluations[rr].coeffs[0]) } else { xi(0) });
                } else {
                    d_int.push(if rr < rows { xi(trace.int[src - n_arb].evaluations[rr]) } else { xi(0) });
                }
            }
            let up = TraceRow { binary_poly: &[], arbitrary_poly: &arb, int: &int, bit_op: &[] };
            let down = TraceRow { binary_poly: &[], arbitrary_poly: &d_arb, int: &d_int, bit_op: &[] };
            let mut b = Checker { idx: 0, failed: Vec::new() };
            VmUair::constrain_general(
                &mut b,
                up,
                down,
                |s: &Scalar| s.coeffs[0].resize::<4>(),
                |e: &X, s: &Scalar| Some(*e * s.coeffs[0].resize::<4>()),
                |_| ZeroIdeal,
            );
            for (i, v) in b.failed {
                fails.push((r, i, v));
            }
        }
        fails
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(mode: Mode, n: usize, tamper: impl Fn(&Layout, &Schedule, &mut Vec<Vec<i64>>)) -> usize {
        let d = make_dealer(n, 7);
        assert!(dealer_holds(&d));
        let s = schedule(&d, mode);
        let stmt = statement(&d, &s);
        let l = build_layout(mode);
        let mut wit = witness_columns(&l, &s, &d);
        tamper(&l, &s, &mut wit);
        let ch = challenges(b"test-root", &stmt, s.enc_list.len());
        // the global layout/scalars are per process: tests of one mode only
        let g = install(mode, ch.gammas);
        let pubcols = public_columns(&g.layout, &s, &stmt, &ch);
        let arb = arb_columns_opt(&g.layout, &s, &pubcols, &wit, &ch, false);
        // verifier-side public columns must agree with the prover's
        let sv = verifier_schedule(&stmt, mode);
        assert_eq!(public_columns(&g.layout, &sv, &stmt, &ch), pubcols);
        let trace = full_trace(pubcols, wit, arb);
        check::check_trace(&trace).len()
    }

    // One mode per test binary run: `install` is process-global, so these are
    // gated by an env var and run one at a time (`VM_TEST_MODE=c3 cargo test vm`).
    #[test]
    fn honest_and_tampered() {
        let mode = match std::env::var("VM_TEST_MODE").as_deref() {
            Ok("c2a") => Mode::C2(Track::Sk),
            Ok("c2b") => Mode::C2(Track::Esm),
            Ok("merged") => Mode::Merged,
            _ => Mode::C3,
        };
        let n = if mode == Mode::Merged { 8 } else { 16 };
        assert_eq!(build(mode, n, |_, _, _| {}), 0, "honest trace must satisfy every constraint");
        // first witness cell of the first non-separator row of each kind
        let f = build(mode, n, |l, s, wit| {
            if let Some(el) = &l.enc {
                wit[l.phys(el.cells[0][0].tu)][s.row_enc0] ^= 1;
            } else if let Some(sh) = &l.shr {
                wit[l.phys(sh.y[0][0].0[0])][s.row_shr0] += 1;
            }
        });
        assert!(f > 0, "tampered witness must violate a constraint");
        let f = build(mode, n, |l, s, wit| {
            wit[l.phys(l.hash.map[l.hash.a[0].0])][s.row_hash0] += 1;
        });
        assert!(f > 0, "tampered sponge input must violate a constraint");
    }
}
