//! Zinc+/Zip+ type scaffolding, copied from `protocol/benches/limber_multiswap.rs`
//! (same bounds), instantiated for 16-bit `i64` int cells with the
//! narrow-stage IPRS lane, exactly like the Limber "sound row".

#![allow(clippy::type_complexity)]

use crypto_primitives::{
    ConstIntRing, ConstIntSemiring, FixedSemiring, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use std::{fmt::Debug, marker::PhantomData, ops::Neg};
use zinc_poly::{
    ConstCoeffBitWidth, Polynomial,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::{DensePolyInnerProduct, DensePolynomial},
    },
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_protocol::ZincTypes;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    field::runtime_monty::Fp,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct, ScalarProduct},
    mul_by_scalar::MulByScalar,
    named::Named,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, IprsCodeNarrow, PnttConfigF65537},
    },
    pcs::structs::ZipTypes,
};

pub const PERFORM_CHECKS: bool = if cfg!(feature = "unchecked") {
    zinc_utils::UNCHECKED
} else {
    zinc_utils::CHECKED
};
/// IPRS inverse rate (rate 1/8, as the paper's benchmarks).
pub const REP: usize = 8;
pub const SECURITY_BITS: usize = zinc_protocol::SECURITY_BITS;
pub const NUM_COL_OPENINGS_FOR_REP: usize = zip_plus::pcs::structs::num_column_openings(REP, SECURITY_BITS);
pub const FIELD_LIMBS: usize = if SECURITY_BITS > 100 { 3 } else { 2 };

zinc_utils::define_modulus!(pub BenchSlot, FIELD_LIMBS);
pub type F = Fp<BenchSlot, FIELD_LIMBS>;

//
// Generic Zip/Zinc type scaffolding — copied from `protocol/benches/e2e.rs`
// on this branch (same bounds), so the concrete instantiation below is the
// only new part.
//

#[allow(clippy::type_complexity)]
#[derive(Debug, Clone, Copy)]
pub struct GenericBenchZipTypes<
    Eval,
    Cw,
    Fmod,
    PrimeTest,
    Chal,
    Pt,
    CombR,
    Comb,
    EvalDotChal,
    CombDotChal,
    ArrCombRDotChal,
>(
    PhantomData<(
        Eval,
        Cw,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        Comb,
        EvalDotChal,
        CombDotChal,
        ArrCombRDotChal,
    )>,
);

impl<Eval, Cw, Fmod, PrimeTest, Chal, Pt, CombR, Comb, EvalDotChal, CombDotChal, ArrCombRDotChal>
    ZipTypes
    for GenericBenchZipTypes<
        Eval,
        Cw,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        Comb,
        EvalDotChal,
        CombDotChal,
        ArrCombRDotChal,
    >
where
    Eval: ConstCoeffBitWidth + Default + Named + Clone + Debug + Send + Sync,
    Cw: FixedSemiring + ConstCoeffBitWidth + ConstTranscribable + FromRef<Eval> + Named + Copy,
    Fmod: ConstIntSemiring + ConstTranscribable + Named,
    PrimeTest: PrimalityTest<Fmod> + Send + Sync,
    Chal: ConstIntRing + ConstTranscribable + Named,
    Pt: ConstIntRing,
    CombR: ConstIntRing
        + Neg<Output = CombR>
        + ConstTranscribable
        + FromRef<CombR>
        + for<'a> MulByScalar<&'a Chal>,
    Comb: FixedSemiring + Polynomial<CombR> + FromRef<Eval> + FromRef<Cw> + Named,
    EvalDotChal: InnerProduct<Eval, Chal, CombR> + Clone + Debug + Send + Sync,
    CombDotChal: InnerProduct<Comb, Chal, CombR> + Clone + Debug + Send + Sync,
    ArrCombRDotChal: InnerProduct<[CombR], Chal, CombR> + Clone + Debug + Send + Sync,
{
    const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
    type Eval = Eval;
    type Cw = Cw;
    type Fmod = Fmod;
    type PrimeTest = PrimeTest;
    type Chal = Chal;
    type Pt = Pt;
    type CombR = CombR;
    type Comb = Comb;
    type EvalDotChal = EvalDotChal;
    type CombDotChal = CombDotChal;
    type ArrCombRDotChal = ArrCombRDotChal;
}

/// Marker selecting the int lane's code. The 16-bit cells use the
/// narrow-stage IPRS variant.
pub trait IntLane<Zt: ZipTypes> {
    type Code: LinearCode<Zt> + BenchCode;
}
#[derive(Clone, Debug)]
pub struct NarrowIprs;
impl<Zt: ZipTypes> IntLane<Zt> for NarrowIprs
where
    IprsCodeNarrow<Zt, PnttConfigF65537, REP, PERFORM_CHECKS>: LinearCode<Zt>,
{
    type Code = IprsCodeNarrow<Zt, PnttConfigF65537, REP, PERFORM_CHECKS>;
}

#[derive(Clone, Debug)]
pub struct GenericBenchZincTypes<
    Int,
    CwR,
    Chal,
    Pt,
    BinaryCombR,
    CombR,
    IntCombR,
    Fmod,
    PrimeTest,
    const D: usize,
    IntCode = NarrowIprs,
>(
    PhantomData<(
        Int,
        CwR,
        Chal,
        Pt,
        BinaryCombR,
        CombR,
        IntCombR,
        Fmod,
        PrimeTest,
        IntCode,
    )>,
);

impl<Int, CwR, Chal, Pt, BinaryCombR, CombR, IntCombR, Fmod, PrimeTest, const D: usize, IntCode>
    ZincTypes<D>
    for GenericBenchZincTypes<
        Int,
        CwR,
        Chal,
        Pt,
        BinaryCombR,
        CombR,
        IntCombR,
        Fmod,
        PrimeTest,
        D,
        IntCode,
    >
where
    IntCode: IntLane<
            GenericBenchZipTypes<
                Int,
                CwR,
                Fmod,
                PrimeTest,
                Chal,
                Pt,
                IntCombR,
                IntCombR,
                ScalarProduct,
                ScalarProduct,
                MBSInnerProduct,
            >,
        > + Clone
        + Debug
        + Send
        + Sync
        + 'static,
    Int: ConstIntSemiring
        + for<'a> MulByScalar<&'a i64, CwR>
        + Named
        + ConstCoeffBitWidth
        + ConstTranscribable
        + Default
        + Clone
        + Send
        + Sync
        + 'static,
    CwR: FixedSemiring
        + for<'a> MulByScalar<&'a i64>
        + ConstCoeffBitWidth
        + ConstTranscribable
        + Named
        + FromRef<Int>
        + FromRef<CwR>
        + Copy,
    Chal: ConstIntRing + ConstTranscribable + Named,
    Pt: ConstIntRing,
    BinaryCombR: ConstIntRing
        + Polynomial<BinaryCombR>
        + Neg<Output = BinaryCombR>
        + for<'a> MulByScalar<&'a i64>
        + for<'a> MulByScalar<&'a Chal>
        + ConstTranscribable
        + Named
        + FromRef<i64>
        + FromRef<Int>
        + FromRef<CwR>
        + FromRef<Chal>
        + FromRef<BinaryCombR>,
    CombR: ConstIntRing
        + Polynomial<CombR>
        + Neg<Output = CombR>
        + for<'a> MulByScalar<&'a i64>
        + for<'a> MulByScalar<&'a Chal>
        + ConstTranscribable
        + Named
        + FromRef<i64>
        + FromRef<Int>
        + FromRef<CwR>
        + FromRef<Chal>
        + FromRef<CombR>,
    IntCombR: ConstIntRing
        + Polynomial<IntCombR>
        + Neg<Output = IntCombR>
        + for<'a> MulByScalar<&'a i64>
        + for<'a> MulByScalar<&'a Chal>
        + ConstTranscribable
        + Named
        + FromRef<i64>
        + FromRef<Int>
        + FromRef<CwR>
        + FromRef<Chal>
        + FromRef<IntCombR>,
    Fmod: ConstIntSemiring + ConstTranscribable + Named,
    PrimeTest: PrimalityTest<Fmod> + Debug + Send + Sync,
{
    type Int = Int;
    type Chal = Chal;
    type Pt = Pt;
    type Fmod = Fmod;
    type PrimeTest = PrimeTest;

    type BinaryZt = GenericBenchZipTypes<
        BinaryPoly<D>,
        DensePolynomial<i64, D>,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        BinaryCombR,
        DensePolynomial<BinaryCombR, D>,
        BinaryPolyInnerProduct<Chal, D>,
        DensePolyInnerProduct<BinaryCombR, Chal, BinaryCombR, MBSInnerProduct, D>,
        MBSInnerProduct,
    >;
    type ArbitraryZt = GenericBenchZipTypes<
        DensePolynomial<Int, D>,
        DensePolynomial<CwR, D>,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        CombR,
        DensePolynomial<CombR, D>,
        DensePolyInnerProduct<Int, Chal, CombR, MBSInnerProduct, D>,
        DensePolyInnerProduct<CombR, Chal, CombR, MBSInnerProduct, D>,
        MBSInnerProduct,
    >;
    type IntZt = GenericBenchZipTypes<
        Int,
        CwR,
        Fmod,
        PrimeTest,
        Chal,
        Pt,
        IntCombR,
        IntCombR,
        ScalarProduct,
        ScalarProduct,
        MBSInnerProduct,
    >;

    type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
    type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, PERFORM_CHECKS>;
    type IntLc = <IntCode as IntLane<Self::IntZt>>::Code;
}

/// How the harness builds a lane's linear code for a given row length.
pub trait BenchCode: Sized {
    fn bench_new(row_len: usize) -> Self;
}

impl<Zt: ZipTypes> BenchCode for IprsCode<Zt, PnttConfigF65537, REP, PERFORM_CHECKS> {
    fn bench_new(row_len: usize) -> Self {
        IprsCode::new_with_optimal_depth(row_len).expect("IPRS params")
    }
}

/// Narrow-stage IPRS for 16-bit cells (base layer + first radix-8 stage on
/// `i64`), as validated for the Limber limb lane.
impl<Zt: ZipTypes> BenchCode for IprsCodeNarrow<Zt, PnttConfigF65537, REP, PERFORM_CHECKS> {
    fn bench_new(row_len: usize) -> Self {
        IprsCodeNarrow::new(IprsCode::new_with_optimal_depth(row_len).expect("IPRS params"), 1)
            .expect("narrow IPRS params")
    }
}

pub const DEGREE_PLUS_ONE: usize = 32;
/// Combination-ring limbs: with ≤ 2^14 columns of 16-bit cells and 128-bit
/// alphas a combined-row entry is `< 2^158`, the verifier's `encode_wide`
/// of it `< 2^231`, and the opened columns' alpha-combination `< 2^231`;
/// all below the 255 signed bits of `Int<4>`.
pub const LIMB_M: usize = 4;

pub type LimbZincTypes = GenericBenchZincTypes<
    /* Int         = */ i64,
    /* CwR         = */ i128,
    /* Chal        = */ i128,
    /* Pt          = */ i128,
    /* BinaryCombR = */ Int<5>,
    /* CombR       = */ Int<LIMB_M>,
    /* IntCombR    = */ Int<LIMB_M>,
    /* Fmod        = */ Uint<FIELD_LIMBS>,
    MillerRabin,
    DEGREE_PLUS_ONE,
    /* int code    = */ NarrowIprs,
>;
