//! A tiny "program" model for the Poseidon2 part of the UAIR: one
//! description, interpreted twice — by the trace builder (concrete `u128`
//! values) and by `Uair::constrain_general` (expressions). Keeps the two
//! in lockstep.
//!
//! Cells are witness int columns; every cell carries a bound and (except
//! the ternary `t` cells) a `Word{width}` lookup. A *value* is a cell or a
//! pair `lo + 2^16·hi`.

use crate::poseidon2::{self as p2, Q};
use std::collections::HashMap;

pub type CellId = usize;
pub type ValId = usize;

#[derive(Clone, Copy, Debug)]
pub enum Val {
    Cell(CellId),
    /// `lo + 2^16 · hi`
    Pair(CellId, CellId),
    /// `a + b`
    Sum(ValId, ValId),
}

#[derive(Clone, Debug, Default)]
pub struct Form {
    /// `Σ coef · value`; coefficients are nonnegative integers (< 2^64).
    pub terms: Vec<(u64, ValId)>,
    pub konst: u64,
    /// Upper bound on the integer value.
    pub max: u128,
}

#[derive(Clone, Debug)]
pub struct FullRound {
    /// Apply `M_E` to `vals` before adding the round constants (the
    /// initial rounds); the terminal rounds take the partial-round state
    /// as is.
    pub apply_me: bool,
    pub vals: [ValId; p2::WIDTH],
    pub rc: [u64; p2::WIDTH],
    pub outs: [(CellId, CellId); p2::WIDTH],
    pub ks: Vec<Vec<CellId>>,
}

#[derive(Clone, Debug)]
pub enum Op {
    /// `lin_i^3 = out_i + Q·K_i` for all 16 lanes, `lin = M_E(vals) + rc`.
    FullRound(FullRound),
    /// `form = x + Q·M`, `x = lo + 2^16 hi < 2^31`.
    Reduce { form: Form, x: (CellId, CellId), m: Vec<CellId> },
    /// `x^3 = out + Q·K` for a reduced value `x`.
    Sbox { x: ValId, rc: u64, out: (CellId, CellId), k: Vec<CellId> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellInfo {
    pub bound: u128,
    /// `Some(w)` ⇒ `Word{w}` lookup; `None` ⇒ constrained otherwise.
    pub lookup: Option<usize>,
}

#[derive(Default)]
pub struct Builder {
    pub cells: Vec<CellInfo>,
    pub vals: Vec<Val>,
    pub ops: Vec<Op>,
}

pub const ME: [[u64; 16]; 16] = me_matrix();

/// `M_E` on 16 expressions using additions only.
pub fn me_expr<E>(x: &[E]) -> Vec<E>
where
    E: Clone + for<'a> std::ops::Add<&'a E, Output = E>,
{
    let mut y: Vec<E> = Vec::with_capacity(16);
    for c in 0..4 {
        let x0 = &x[4 * c];
        let x1 = &x[4 * c + 1];
        let x2 = &x[4 * c + 2];
        let x3 = &x[4 * c + 3];
        let t01 = x0.clone() + x1;
        let t23 = x2.clone() + x3;
        let t0123 = t01.clone() + &t23;
        let t01123 = t0123.clone() + x1;
        let t01233 = t0123 + x3;
        let y3 = t01233.clone() + x0 + x0;
        let y1 = t01123.clone() + x2 + x2;
        let y0 = t01123 + &t01;
        let y2 = t01233 + &t23;
        y.extend([y0, y1, y2, y3]);
    }
    let mut sums: Vec<E> = (0..4).map(|r| y[r].clone()).collect();
    for c in 1..4 {
        for r in 0..4 {
            sums[r] = sums[r].clone() + &y[4 * c + r];
        }
    }
    (0..16).map(|i| y[i].clone() + &sums[i % 4]).collect()
}

const fn me_matrix() -> [[u64; 16]; 16] {
    let mut m = [[0u64; 16]; 16];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 16 {
            let base = p2::M4[i % 4][j % 4];
            m[i][j] = if i / 4 == j / 4 { 2 * base } else { base };
            j += 1;
        }
        i += 1;
    }
    m
}

fn bits(x: u128) -> usize {
    128 - x.leading_zeros() as usize
}

impl Builder {
    pub fn cell(&mut self, bound: u128, lookup: Option<usize>) -> CellId {
        self.cells.push(CellInfo { bound, lookup });
        self.cells.len() - 1
    }
    pub fn word(&mut self, w: usize) -> CellId {
        self.cell((1u128 << w) - 1, Some(w))
    }
    pub fn val_cell(&mut self, c: CellId) -> ValId {
        self.vals.push(Val::Cell(c));
        self.vals.len() - 1
    }
    pub fn val_pair(&mut self, lo: CellId, hi: CellId) -> ValId {
        self.vals.push(Val::Pair(lo, hi));
        self.vals.len() - 1
    }
    pub fn val_sum(&mut self, a: ValId, b: ValId) -> ValId {
        self.vals.push(Val::Sum(a, b));
        self.vals.len() - 1
    }
    /// A reduced value: `lo` 16 bits, `hi` 15 bits (so `< 2^31`).
    pub fn reduced_pair(&mut self) -> ((CellId, CellId), ValId) {
        let lo = self.word(16);
        let hi = self.word(15);
        let v = self.val_pair(lo, hi);
        ((lo, hi), v)
    }
    pub fn val_bound(&self, v: ValId) -> u128 {
        match self.vals[v] {
            Val::Cell(c) => self.cells[c].bound,
            Val::Pair(lo, hi) => self.cells[lo].bound + (self.cells[hi].bound << 16),
            Val::Sum(a, b) => self.val_bound(a) + self.val_bound(b),
        }
    }
    pub fn form(&self, terms: Vec<(u64, ValId)>, konst: u64) -> Form {
        let mut max = konst as u128;
        for (c, v) in &terms {
            max += *c as u128 * self.val_bound(*v);
        }
        Form { terms, konst, max }
    }
    fn limb_cells(&mut self, max: u128) -> Vec<CellId> {
        let n = (bits(max) + 15) / 16;
        (0..n.max(1)).map(|_| self.word(16)).collect()
    }

    /// Emit `Reduce(form)`; returns the reduced value.
    pub fn reduce(&mut self, form: Form) -> ValId {
        let (x, v) = self.reduced_pair();
        let m = self.limb_cells(form.max / Q as u128);
        self.ops.push(Op::Reduce { form, x, m });
        v
    }

    /// One Poseidon2 permutation on 16 input values (each `< 2^32`).
    /// Returns the 16 reduced output values.
    pub fn permutation(&mut self, inputs: [ValId; 16]) -> [ValId; 16] {
        let c = p2::constants();
        let mut cur = inputs;
        for r in 0..p2::HALF_FULL {
            cur = self.full_round(cur, c.rc_initial[r], true);
        }
        // Lanes entering the partial rounds: M_E(cur) as integer forms.
        let mut forms: Vec<Form> = (0..16)
            .map(|i| self.form((0..16).map(|l| (ME[i][l], cur[l])).collect(), 0))
            .collect();
        for r in 0..p2::PARTIAL {
            let mut f0 = forms[0].clone();
            f0.konst = (f0.konst + c.rc_internal[r]) % Q;
            f0.max = f0.max - forms[0].konst as u128 + f0.konst as u128;
            let x = self.reduce(f0);
            let (out, outv) = self.reduced_pair();
            let x_bound = self.val_bound(x);
            let k = self.limb_cells(x_bound * x_bound * x_bound / Q as u128);
            self.ops.push(Op::Sbox { x, rc: 0, out, k });
            forms[0] = self.form(vec![(1, outv)], 0);
            // M_I: new_i = Σ_l form_l + d_i · form_i, coefficients mod Q.
            let sum = self.sum_forms(&forms);
            let mut new_forms = Vec::with_capacity(16);
            for i in 0..16 {
                let scaled = self.scale_form(&forms[i], c.diag[i]);
                new_forms.push(self.add_forms(&sum, &scaled));
            }
            forms = new_forms;
        }
        let mut cur: [ValId; 16] = [0; 16];
        for i in 0..16 {
            cur[i] = self.reduce(forms[i].clone());
        }
        for r in 0..p2::HALF_FULL {
            cur = self.full_round(cur, c.rc_final[r], r > 0);
        }
        let mut outs = [0; 16];
        for i in 0..16 {
            let f = self.form((0..16).map(|l| (ME[i][l], cur[l])).collect(), 0);
            outs[i] = self.reduce(f);
        }
        outs
    }

    fn full_round(&mut self, vals: [ValId; 16], rc: [u64; 16], apply_me: bool) -> [ValId; 16] {
        let mut outs = [(0, 0); 16];
        let mut outv = [0; 16];
        let mut ks = Vec::with_capacity(16);
        for i in 0..16 {
            let mut lin_max: u128 = rc[i] as u128;
            if apply_me {
                for l in 0..16 {
                    lin_max += ME[i][l] as u128 * self.val_bound(vals[l]);
                }
            } else {
                lin_max += self.val_bound(vals[i]);
            }
            assert!(bits(lin_max) * 3 < 127, "S-box input too wide");
            let (o, v) = self.reduced_pair();
            outs[i] = o;
            outv[i] = v;
            ks.push(self.limb_cells(lin_max * lin_max * lin_max / Q as u128));
        }
        self.ops.push(Op::FullRound(FullRound { apply_me, vals, rc, outs, ks }));
        outv
    }

    fn sum_forms(&self, forms: &[Form]) -> Form {
        let mut acc: HashMap<ValId, u64> = HashMap::new();
        let mut konst = 0u64;
        for f in forms {
            for (c, v) in &f.terms {
                let e = acc.entry(*v).or_insert(0);
                *e = (*e + *c) % Q;
            }
            konst = (konst + f.konst) % Q;
        }
        let mut terms: Vec<(u64, ValId)> = acc.into_iter().map(|(v, c)| (c, v)).collect();
        terms.sort_by_key(|t| t.1);
        self.form(terms, konst)
    }
    fn scale_form(&self, f: &Form, d: u64) -> Form {
        let terms = f.terms.iter().map(|(c, v)| (p2::mulq(*c, d), *v)).collect();
        self.form(terms, p2::mulq(f.konst, d))
    }
    fn add_forms(&self, a: &Form, b: &Form) -> Form {
        self.sum_forms(&[a.clone(), b.clone()])
    }
}

/// Concrete evaluation of a program over one row.
pub struct Interp<'a> {
    pub cells: Vec<Option<u64>>,
    pub b: &'a Builder,
}

impl<'a> Interp<'a> {
    pub fn new(b: &'a Builder) -> Self {
        Self { cells: vec![None; b.cells.len()], b }
    }
    pub fn set(&mut self, c: CellId, v: u64) {
        assert!((v as u128) <= self.b.cells[c].bound, "cell {c} = {v} exceeds bound {}", self.b.cells[c].bound);
        self.cells[c] = Some(v);
    }
    pub fn get(&self, c: CellId) -> u64 {
        self.cells[c].unwrap_or_else(|| panic!("cell {c} unset"))
    }
    pub fn val(&self, v: ValId) -> u128 {
        match self.b.vals[v] {
            Val::Cell(c) => self.get(c) as u128,
            Val::Pair(lo, hi) => self.get(lo) as u128 + ((self.get(hi) as u128) << 16),
            Val::Sum(a, b) => self.val(a) + self.val(b),
        }
    }
    pub fn form(&self, f: &Form) -> u128 {
        let mut acc = f.konst as u128;
        for (c, v) in &f.terms {
            acc += *c as u128 * self.val(*v);
        }
        debug_assert!(acc <= f.max);
        acc
    }
    fn set_pair(&mut self, p: (CellId, CellId), v: u64) {
        self.set(p.0, v & 0xFFFF);
        self.set(p.1, v >> 16);
    }
    fn set_limbs(&mut self, cells: &[CellId], mut v: u128) {
        for &c in cells {
            self.set(c, (v & 0xFFFF) as u64);
            v >>= 16;
        }
        assert_eq!(v, 0, "limb overflow");
    }
    /// Run ops `range` in order.
    pub fn run(&mut self, ops: &[Op]) {
        for op in ops {
            match op {
                Op::FullRound(fr) => {
                    let vals: Vec<u128> = fr.vals.iter().map(|v| self.val(*v)).collect();
                    for i in 0..16 {
                        let mut lin: u128 = fr.rc[i] as u128;
                        if fr.apply_me {
                            for l in 0..16 {
                                lin += ME[i][l] as u128 * vals[l];
                            }
                        } else {
                            lin += vals[i];
                        }
                        let cube = lin * lin * lin;
                        let out = (cube % Q as u128) as u64;
                        self.set_pair(fr.outs[i], out);
                        self.set_limbs(&fr.ks[i], cube / Q as u128);
                    }
                }
                Op::Reduce { form, x, m } => {
                    let v = self.form(form);
                    let xv = (v % Q as u128) as u64;
                    self.set_pair(*x, xv);
                    self.set_limbs(m, v / Q as u128);
                }
                Op::Sbox { x, rc, out, k } => {
                    let xv = self.val(*x) + *rc as u128;
                    let cube = xv * xv * xv;
                    self.set_pair(*out, (cube % Q as u128) as u64);
                    self.set_limbs(k, cube / Q as u128);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_program_matches_reference() {
        let mut b = Builder::default();
        let mut inputs = [0; 16];
        let mut in_cells = Vec::new();
        for i in 0..16 {
            let (p, v) = b.reduced_pair();
            inputs[i] = v;
            in_cells.push(p);
        }
        let outs = b.permutation(inputs);
        let mut it = Interp::new(&b);
        let mut state = [0u64; 16];
        for i in 0..16 {
            state[i] = (i as u64 * 1234567 + 89) % Q;
            it.set(in_cells[i].0, state[i] & 0xFFFF);
            it.set(in_cells[i].1, state[i] >> 16);
        }
        it.run(&b.ops);
        p2::permute(&mut state);
        for i in 0..16 {
            assert_eq!(it.val(outs[i]) as u64, state[i], "lane {i}");
        }
        eprintln!("cells per permutation: {}", b.cells.len() - 32);
    }
}
