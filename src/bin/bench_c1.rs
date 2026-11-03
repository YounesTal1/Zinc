//! Prove + verify the lattice relation with Poseidon2 commitments, one
//! Zinc+ proof. Env: `SEED`, `REPS`, `TAMPER=1`, `STEPS=1`.

#![allow(clippy::type_complexity)]

use crypto_primitives::PrimeField;
use std::{hint::black_box, time::Instant};
use zinc_protocol::{Proof, ZincPlusPiop, ZincTypes};
use zinc_uair::{Uair, ideal::DegreeOneIdeal, ideal_collector::IdealOrZero};
use zincbench::c1_uair::{self as uair, C1Uair as LatticePoseidonUair, NUM_PUB, NUM_VARS, PUB_LAST, ROWS, Statement};
use zincbench::zinc_types::{
    BenchCode, DEGREE_PLUS_ONE, F, LimbZincTypes as Zt, NUM_COL_OPENINGS_FOR_REP, PERFORM_CHECKS, REP,
    SECURITY_BITS,
};
use zincbench::{ring, c1_uair::program};
use zip_plus::{code::iprs::IprsCode, pcs::structs::ZipPlus, utils::eprint_proof_size};

type Piop = ZincPlusPiop<Zt, LatticePoseidonUair, F, DEGREE_PLUS_ONE>;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let seed = env_usize("SEED", 1) as u64;
    let reps = env_usize("REPS", 1);
    let tamper = std::env::var("TAMPER").is_ok();
    let steps = std::env::var("STEPS").is_ok();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    let p = program();
    let sig = LatticePoseidonUair::signature();
    eprintln!(
        "interfold-C1+Poseidon2 UAIR: {} rows x ({} public + {} witness) int cols, {} lookups, {} shifts, {} Poseidon ops/row, max degree {}, checks={PERFORM_CHECKS}, rate 1/{REP}, {SECURITY_BITS}-bit target ({NUM_COL_OPENINGS_FOR_REP} openings), transcript-drawn 128-bit prime, threads={threads}",
        ROWS,
        NUM_PUB,
        p.num_cells,
        sig.lookup_specs().len(),
        sig.shifts().len(),
        p.b.ops.len(),
        zinc_uair::degree_counter::count_max_degree::<LatticePoseidonUair>(),
    );

    // ---- witness generation (sampling + pk) and hashing ----
    let t0 = Instant::now();
    let public = ring::sample_a(seed);
    let witness = uair::sample_witness(&public, seed + 1000);
    let t_sample = t0.elapsed().as_secs_f64();
    assert!(uair::c1_holds(&public, &witness), "sampled witness violates C1");
    let t0 = Instant::now();
    let cm = uair::commit(&witness);
    let t_hash = t0.elapsed().as_secs_f64();
    let stmt = Statement { a: public.a.clone(), cm };

    // ---- statement-derived public data (verifier recomputes this) ----
    let t0 = Instant::now();
    let wts = uair::weights_for(&stmt);
    let t_weights = t0.elapsed().as_secs_f64();

    // ---- trace ----
    let t0 = Instant::now();
    let built = uair::build_trace(&stmt, &wts, &witness);
    let mut trace = built;
    let t_trace = t0.elapsed().as_secs_f64();
    eprintln!(
        "witness: sample+pk {t_sample:.3} s | Poseidon commitments {t_hash:.3} s | FS weights (verifier-side) {t_weights:.3} s | trace build {t_trace:.3} s"
    );

    if tamper {
        let mode = std::env::var("TAMPER").unwrap();
        let mut int = trace.int.to_vec();
        let (col, row, newv, what) = match mode.as_str() {
            "u" => (NUM_PUB + p.u[0][0], 5, None, "u limb of acc 0 at row 5 (+1)"),
            "range" => (NUM_PUB + p.eek[0].0, 5, Some(64i64), "eek cell out of range (64)"),
            "eek41" => (NUM_PUB + p.eek[0].0, 5, Some(41i64), "eek + 20 = 41 (passes Word{6}, violates the two-sided bound)"),
            "pk" => (NUM_PUB + p.pk[0][0].limbs[3], 2, None, "pk0 top limb (+1): pk0 leaves [−c, c]"),
            "crt" => (NUM_PUB + p.quot[0][1].limbs[0], 4, None, "e_sm quotient limb (+1): breaks lifted = res + quot·q"),
            "out" => (NUM_PUB + p.st[0][0].0, 3, None, "ST lane 0 lo of sponge S at row 3 (+1)"),
            "pub" => (PUB_LAST, 7, Some(1i64), "public 'last' selector at row 7"),
            _ => (NUM_PUB + p.t[0], 0, None, "s_0 ternary cell (±1)"),
        };
        let v = int[col].evaluations[row];
        int[col].evaluations[row] = newv.unwrap_or(if v == 2 && mode == "t" { 1 } else { v + 1 });
        trace.int = int.into();
        eprintln!("TAMPER({mode}): {what}");
    }

    // ---- Zip+ parameters ----
    let row_len = ROWS;
    let pp = (
        ZipPlus::<<Zt as ZincTypes<DEGREE_PLUS_ONE>>::BinaryZt, _>::setup(
            row_len,
            IprsCode::new_with_optimal_depth(row_len).unwrap(),
        ),
        ZipPlus::<<Zt as ZincTypes<DEGREE_PLUS_ONE>>::ArbitraryZt, _>::setup(
            row_len,
            IprsCode::new_with_optimal_depth(row_len).unwrap(),
        ),
        ZipPlus::<<Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntZt, _>::setup(
            row_len,
            <<Zt as ZincTypes<DEGREE_PLUS_ONE>>::IntLc as BenchCode>::bench_new(row_len),
        ),
    );

    // ---- prove ----
    let mut prove_times = Vec::new();
    let mut proof: Option<Proof<F>> = None;
    for r in 0..reps.max(1) {
        let t0 = Instant::now();
        let res = Piop::prove::<false, PERFORM_CHECKS>(&pp, &trace, NUM_VARS, zinc_protocol::project_scalar_fn);
        let dt = t0.elapsed().as_secs_f64();
        match res {
            Ok(pf) => {
                prove_times.push(dt);
                eprintln!("prove #{r}: {dt:.3} s");
                proof = Some(pf);
            }
            Err(e) => {
                eprintln!("prove #{r} FAILED after {dt:.3} s: {e:?}");
                if tamper {
                    println!("TAMPER RESULT: prover rejected the tampered trace ({e:?})");
                    return;
                }
                std::process::exit(1);
            }
        }
    }
    let proof = proof.expect("no proof");
    eprint_proof_size(format!("interfold-c1-poseidon/rate1-{REP}/sec{SECURITY_BITS}"), &proof);

    // ---- verify (includes recomputing the public columns from the statement) ----
    let proj_ideal = |ideal: &IdealOrZero<<LatticePoseidonUair as Uair>::Ideal>, cfg: &<F as PrimeField>::Config| {
        ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, cfg))
    };
    let verify_once = |pf: Proof<F>| {
        let t0 = Instant::now();
        let wts = uair::weights_for(&stmt);
        let public_trace = uair::public_trace(&stmt, &wts);
        let t_pub = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let r = Piop::verify::<_, PERFORM_CHECKS>(
            &pp,
            pf,
            &public_trace,
            NUM_VARS,
            zinc_protocol::project_scalar_fn,
            proj_ideal,
        );
        (r, t_pub, t1.elapsed().as_secs_f64())
    };
    let mut verify_times = Vec::new();
    let mut pub_times = Vec::new();
    for r in 0..reps.max(1) {
        let (res, t_pub, t_ver) = verify_once(proof.clone());
        match res {
            Ok(()) => {
                eprintln!("verify #{r}: public-column derivation {t_pub:.3} s + Zinc+ verify {t_ver:.3} s");
                verify_times.push(t_ver);
                pub_times.push(t_pub);
                if tamper {
                    println!("TAMPER RESULT: VERIFY ACCEPTED A TAMPERED PROOF — SOUNDNESS BUG");
                    std::process::exit(2);
                }
            }
            Err(e) => {
                if tamper {
                    println!("TAMPER RESULT: verifier rejected ({e:?}) after {t_ver:.3} s");
                    return;
                }
                eprintln!("verify FAILED: {e:?}");
                std::process::exit(1);
            }
        }
    }
    println!(
        "RESULT interfold-C1+Poseidon2 (one Zinc+ proof, {} rows x {} cols, rate 1/{REP}, {SECURITY_BITS}-bit, checks={PERFORM_CHECKS}, {threads} threads): witness sample+pk {t_sample:.3} s | hash {t_hash:.3} s | trace {t_trace:.3} s | prove median {:.3} s ({} reps) | verify median {:.4} s (+ {:.3} s public-column derivation incl. NTT weights)",
        ROWS,
        NUM_PUB + p.num_cells,
        median(prove_times),
        reps,
        median(verify_times),
        median(pub_times),
    );

    if steps {
        let mut lap = Instant::now();
        let mut times: Vec<(&str, f64)> = Vec::new();
        macro_rules! lap {
            ($name:literal) => {{
                times.push(($name, lap.elapsed().as_secs_f64()));
                lap = Instant::now();
            }};
        }
        let s = Piop::step0_commit(&pp, &trace, NUM_VARS).expect("step0");
        lap!("0 commit");
        let s = s.step1_combined(zinc_protocol::project_scalar_fn).expect("step1");
        lap!("1 prime projection");
        let s = s.step2_ideal_check().expect("step2");
        lap!("2 ideal check");
        let s = s.step3_eval_projection().expect("step3");
        lap!("3 eval projection");
        let s = s.step4_sumcheck().expect("step4");
        lap!("4 CPR sumcheck");
        let s = s.step4b_lookup().expect("step4b");
        lap!("4b lookup (GKR-LogUp)");
        let s = s.step5_multipoint_eval().expect("step5");
        lap!("5 multipoint eval");
        let s = s.step6_lift_and_project().expect("step6");
        lap!("6 lift-and-project");
        let s = s.step7_pcs_open::<PERFORM_CHECKS>().expect("step7");
        lap!("7 PCS open (+reducers)");
        let pf: Proof<F> = s.finish().expect("finish");
        lap!("8 assembly");
        black_box(pf);
        let total: f64 = times.iter().map(|(_, t)| t).sum();
        println!("per-step prover times (one run, total {total:.3} s):");
        for (name, t) in &times {
            println!("  {name:<26} {t:8.3} s  ({:5.1}%)", 100.0 * t / total);
        }
    }
}
