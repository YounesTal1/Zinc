//! Prove + verify interfold C2 / C3 / merged C2+C3 (micro committee) with
//! the coprocessor UAIR of `vm.rs`.  Env: `MODE=c2a|c2b|c3|merged`,
//! `N` (ring dimension, multiple of 8; default 64), `SEED`, `REPS`,
//! `TAMPER=<mode>`, `STEPS=1`, `ROW_LEN`.

#![allow(clippy::type_complexity)]

use crypto_primitives::PrimeField;
use std::{hint::black_box, time::Instant};
use zinc_protocol::{Proof, ZincPlusPiop, ZincTypes};
use zinc_uair::{Uair, ideal::DegreeOneIdeal, ideal_collector::IdealOrZero};
use zincbench::vm::{self, D, Mode, NUM_ARB, Track, VmUair};
use zincbench::zinc_types::{BenchCode, F, GenericBenchZincTypes, NarrowIprs, LIMB_M, FIELD_LIMBS, NUM_COL_OPENINGS_FOR_REP, PERFORM_CHECKS, REP, SECURITY_BITS};
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
type Zt = GenericBenchZincTypes<i64, i128, i128, i128, Int<5>, Int<LIMB_M>, Int<LIMB_M>, Uint<FIELD_LIMBS>, zinc_primality::MillerRabin, D, NarrowIprs>;
use zip_plus::{code::iprs::IprsCode, pcs::structs::ZipPlus, utils::eprint_proof_size};

type Piop = ZincPlusPiop<Zt, VmUair, F, D>;
type IntZt = <Zt as ZincTypes<D>>::IntZt;
type IntLc = <Zt as ZincTypes<D>>::IntLc;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}
fn peak_rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|st| st.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1).map(|k| k.parse::<f64>().unwrap_or(0.0) / 1048576.0)))
        .unwrap_or(0.0)
}
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let mode_s = std::env::var("MODE").unwrap_or_else(|_| "c3".into());
    let mode = match mode_s.as_str() {
        "c2a" => Mode::C2(Track::Sk),
        "c2b" => Mode::C2(Track::Esm),
        "c3" => Mode::C3,
        "merged" => Mode::Merged,
        _ => panic!("MODE must be c2a|c2b|c3|merged"),
    };
    let n = env_usize("N", 64);
    assert!(n % 8 == 0 && n.is_power_of_two(), "N must be a power of two ≥ 8");
    let seed = env_usize("SEED", 1) as u64;
    let reps = env_usize("REPS", 1);
    let tamper = std::env::var("TAMPER").ok();
    let steps = std::env::var("STEPS").is_ok();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    // ---- dealer data (sk/e_sm from C1's sampler, shares, keys, encryptions) ----
    let t0 = Instant::now();
    let dealer = vm::make_dealer(n, seed);
    assert!(vm::dealer_holds(&dealer), "dealer data violates the relation");
    let t_sample = t0.elapsed().as_secs_f64();

    // ---- schedule, statement (digests + recipient keys) ----
    let t0 = Instant::now();
    let sched = vm::schedule(&dealer, mode);
    let stmt = vm::statement(&dealer, &sched);
    let t_hash = t0.elapsed().as_secs_f64();
    let layout = vm::build_layout(mode);
    let nv = sched.num_vars;
    let rows = 1usize << nv;
    eprintln!(
        "{mode_s} N={n}: enc rows {} | shr rows {} | hash rows {} ({} sponges) → {rows} rows (2^{nv}); int witness cols {} (pools {}), arb cols {NUM_ARB}, public cols {}, runs/row {}",
        sched.enc_rows,
        sched.shr_rows,
        sched.hash_rows,
        sched.sponges.len(),
        layout.num_wit,
        vm::TBLS.iter().zip(layout.pool_count.iter()).map(|(t, c)| format!("{t:?}:{c}")).collect::<Vec<_>>().join(" "),
        vm::num_pub(&layout),
        layout.max_runs,
    );

    // ---- Zip+ parameters ----
    let row_len = env_usize("ROW_LEN", rows.min(8192));
    let pp = (
        ZipPlus::<<Zt as ZincTypes<D>>::BinaryZt, _>::setup(rows, IprsCode::new_with_optimal_depth(rows.min(2048)).unwrap()),
        ZipPlus::<<Zt as ZincTypes<D>>::ArbitraryZt, _>::setup(rows, IprsCode::new_with_optimal_depth(row_len).unwrap()),
        ZipPlus::<IntZt, _>::setup(rows, <IntLc as BenchCode>::bench_new(row_len)),
    );

    // ---- prover: int witness (tampered first, like a cheating prover), pre-commit, challenges, accumulators ----
    let t0 = Instant::now();
    let mut wit = vm::witness_columns(&layout, &sched, &dealer);
    let t_wit = t0.elapsed().as_secs_f64();
    let mut stmt_v = stmt.clone();
    if let Some(t) = &tamper {
        let l = &layout;
        let bump = |wit: &mut Vec<Vec<i64>>, c: vm::Cell, row: usize, delta: i64| {
            wit[l.phys(c)][row] += delta;
        };
        let what: String = match t.as_str() {
            "share" => {
                let sh = l.shr.as_ref().expect("no shr rows in this mode");
                bump(&mut wit, sh.y[0][2].0[0], sched.row_shr0 + 1, 1);
                "share y[j=0][party 3] low limb +1 at coefficient 1 (breaks parity; C2: also the share hash)".into()
            }
            "ct" => {
                let el = l.enc.as_ref().expect("no enc rows in this mode");
                bump(&mut wit, el.cells[0][0].ct[0][0].0[0], sched.row_enc0 + 2, 1);
                "ct0 low limb +1 at enc row 2 (breaks identity, ct hash and complement)".into()
            }
            "u" => {
                let el = l.enc.as_ref().expect("no enc rows in this mode");
                let c = l.phys(el.cells[0][0].tu);
                let r = sched.row_enc0 + 2;
                wit[c][r] = if wit[c][r] == 2 { 1 } else { wit[c][r] + 1 };
                "u ternary cell changed at enc row 2 (breaks the encryption identity only)".into()
            }
            "e" => {
                let el = l.enc.as_ref().expect("no enc rows in this mode");
                let (a, b) = el.cells[0][0].e0;
                let r = sched.row_enc0 + 2;
                let va = wit[l.phys(a)][r];
                wit[l.phys(a)][r] = if va == 40 { 39 } else { va + 1 };
                wit[l.phys(b)][r] = 40 - wit[l.phys(a)][r];
                "e0 ±1 (complement kept) at enc row 2 (breaks the encryption identity only)".into()
            }
            "e41" => {
                let el = l.enc.as_ref().expect("no enc rows in this mode");
                let (a, b) = el.cells[0][0].e0;
                let r = sched.row_enc0 + 2;
                wit[l.phys(a)][r] = 41;
                wit[l.phys(b)][r] = -1;
                "e0 cell 41 / complement −1 (fails Word{6} lookup)".into()
            }
            "hash" => {
                bump(&mut wit, l.hash.map[l.hash.st[0].0], sched.row_hash0 + 1, 1);
                "sponge state lane 0 at hash row 1 (+1): breaks the permutation".into()
            }
            "absorb" => {
                let sp = sched.sponges.iter().position(|s| !s.public).unwrap();
                let r = sched.row_hash0 + sched.sponges[..sp].iter().map(|s| s.len / 8).sum::<usize>();
                bump(&mut wit, l.hash.map[l.hash.a[0].0], r, 1);
                format!("absorbed chunk of witness sponge '{}' at its first row (+1): bus mismatch", sched.sponges[sp].name)
            }
            "digest" => {
                stmt_v.digests[0].1[0] ^= 1;
                format!("verifier's statement digest '{}' flipped", stmt_v.digests[0].0)
            }
            "pk" => {
                stmt_v.pks[0].pk0[0][0] = (stmt_v.pks[0].pk0[0][0] + 1) % zincbench::dkg::QD[0];
                "verifier's recipient pk[1] coefficient +1".into()
            }
            "secret" => {
                let sh = l.shr.as_ref().expect("no shr rows in this mode");
                let r = sched.row_shr0 + 1;
                if let (Some(skc), true) = (sh.sk, mode != Mode::C2(Track::Esm)) {
                    let c = l.phys(skc);
                    wit[c][r] = if wit[c][r] == 2 { 1 } else { wit[c][r] + 1 };
                    "sk cell changed at shr row 1 (parity + bus mismatch)".into()
                } else {
                    let res = sh.res.as_ref().unwrap();
                    bump(&mut wit, res[0].0[0], r, 1);
                    "e_sm residue low limb +1 at shr row 1 (parity + bus mismatch)".into()
                }
            }
            "k1" => {
                let sh = l.shr.as_ref().expect("merged mode only");
                let k1s = sh.k1.as_ref().expect("merged mode only");
                bump(&mut wit, k1s[0][0].1[0], sched.row_shr0 + 1, 1);
                "k1 low limb +1 on a share row (breaks k1 = q_mod_t·y − t·z)".into()
            }
            other => panic!("unknown TAMPER mode {other}"),
        };
        eprintln!("TAMPER({t}): {what}");
    }
    let t0 = Instant::now();
    let wit_mles = vm::int_mles(wit.clone());
    let (_hint, pre) = ZipPlus::<IntZt, IntLc>::commit(&pp.2, &wit_mles).expect("pre-commit");
    let t_precommit = t0.elapsed().as_secs_f64();
    let root: Vec<u8> = pre.root.to_vec();
    let ch = vm::challenges(&root, &stmt, sched.enc_list.len());
    let g = vm::install(mode, ch.gammas);
    let t0 = Instant::now();
    let pubcols = vm::public_columns(&g.layout, &sched, &stmt, &ch);
    let t_pub = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let arb = vm::arb_columns_opt(&g.layout, &sched, &pubcols, &wit, &ch, tamper.is_none());
    let t_arb = t0.elapsed().as_secs_f64();
    let np = vm::num_pub(&g.layout);
    let trace = vm::full_trace(pubcols, wit, arb);
    eprintln!(
        "prover prep: dealer data {t_sample:.3} s | sponges {t_hash:.3} s | int witness {t_wit:.3} s | pre-commit {t_precommit:.3} s | public cols {t_pub:.3} s | accumulators {t_arb:.3} s"
    );
    if std::env::var("CHECK").is_ok() {
        let t0 = Instant::now();
        let fails = vm::check::check_trace(&trace);
        eprintln!("integer constraint check: {} failures in {:.2} s", fails.len(), t0.elapsed().as_secs_f64());
        for (r, i, v) in fails.iter().take(20) {
            eprintln!("  row {r} constraint #{i}: {v}");
        }
    }
    let sig = VmUair::signature();
    eprintln!(
        "UAIR: {} lookups, {} shifts, max degree {}, checks={PERFORM_CHECKS}, rate 1/{REP}, {SECURITY_BITS}-bit ({NUM_COL_OPENINGS_FOR_REP} openings), row_len {row_len}, threads={threads}",
        sig.lookup_specs().len(),
        sig.shifts().len(),
        zinc_uair::degree_counter::count_max_degree::<VmUair>(),
    );

    // ---- prove ----
    let mut prove_times = Vec::new();
    let mut proof = None;
    for r in 0..reps.max(1) {
        let t0 = Instant::now();
        let res = Piop::prove::<false, PERFORM_CHECKS>(&pp, &trace, nv, zinc_protocol::project_scalar_fn);
        let dt = t0.elapsed().as_secs_f64();
        match res {
            Ok(p) => {
                prove_times.push(dt);
                eprintln!("prove #{r}: {dt:.3} s (+ {t_precommit:.3} s pre-commit)");
                proof = Some(p);
            }
            Err(e) => {
                eprintln!("prove #{r} FAILED after {dt:.3} s: {e:?}");
                if tamper.is_some() {
                    println!("TAMPER RESULT: prover rejected the tampered trace ({e:?})");
                    return;
                }
                std::process::exit(1);
            }
        }
    }
    let proof = proof.expect("no proof");
    eprint_proof_size(format!("interfold-{mode_s}-N{n}/rate1-{REP}/sec{SECURITY_BITS}"), &proof);

    // ---- verify: challenges from the proof's int commitment, public columns from the statement ----
    let proj_ideal =
        |ideal: &IdealOrZero<<VmUair as Uair>::Ideal>, cfg: &<F as PrimeField>::Config| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, cfg));
    let verify_once = |pf: Proof<F>| {
        let t0 = Instant::now();
        let root_v: Vec<u8> = pf.commitments.2.root.to_vec();
        let ch_v = vm::challenges(&root_v, &stmt_v, sched.enc_list.len());
        if ch_v.gammas != g.scalars_gammas() {
            return (Err(format!("challenge mismatch: proof commitment does not match the installed γ")), t0.elapsed().as_secs_f64(), 0.0);
        }
        let sched_v = vm::verifier_schedule(&stmt_v, mode);
        let pubcols = vm::public_columns(&g.layout, &sched_v, &stmt_v, &ch_v);
        let public_trace = vm::public_trace(pubcols);
        let t_pub = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let r = Piop::verify::<_, PERFORM_CHECKS>(&pp, pf, &public_trace, nv, zinc_protocol::project_scalar_fn, proj_ideal)
            .map_err(|e| format!("{e:?}"));
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
                if tamper.is_some() {
                    println!("TAMPER RESULT: VERIFY ACCEPTED A TAMPERED PROOF — SOUNDNESS BUG");
                    std::process::exit(2);
                }
            }
            Err(e) => {
                if tamper.is_some() {
                    println!("TAMPER RESULT: verifier rejected ({e}) after {t_ver:.3} s");
                    return;
                }
                eprintln!("verify FAILED: {e}");
                std::process::exit(1);
            }
        }
    }
    println!(
        "RESULT interfold-{mode_s} N={n} (one Zinc+ proof, {rows} rows x {} int cols + {NUM_ARB} arb cols, rate 1/{REP}, {SECURITY_BITS}-bit, checks={PERFORM_CHECKS}, {threads} threads): dealer data {t_sample:.3} s | sponges {t_hash:.3} s | witness+accumulators {:.3} s | pre-commit {t_precommit:.3} s | prove median {:.3} s ({} reps) | verify median {:.4} s (+ {:.3} s public columns incl. NTT weights)",
        np + g.layout.num_wit,
        t_wit + t_arb,
        median(prove_times),
        reps,
        median(verify_times),
        median(pub_times),
    );
    println!("peak RSS {:.2} GB; cells (rows x int witness cols) {:.1} M", peak_rss_gb(), (rows * g.layout.num_wit) as f64 / 1e6);

    if steps {
        let mut lap = Instant::now();
        let mut times: Vec<(&str, f64)> = Vec::new();
        #[allow(unused_assignments)]
        macro_rules! lap {
            ($name:literal) => {{
                times.push(($name, lap.elapsed().as_secs_f64()));
                lap = Instant::now();
            }};
        }
        let s = Piop::step0_commit(&pp, &trace, nv).expect("step0");
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
        println!("per-step prover times (one run, total {total:.3} s; + {t_precommit:.3} s pre-commit):");
        for (name, t) in &times {
            println!("  {name:<26} {t:8.3} s  ({:5.1}%)", 100.0 * t / total);
        }
    }
}
