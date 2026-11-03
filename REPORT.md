# interfold C1 (`pk_generation`, secure preset) in one Zinc+ proof

Measured 2026-11-03 on 2 vCPU, 8 GB, rate 1/8, 100-bit target. Medians of 3 unchecked runs after a verified checked run. Statement taken from
`circuits/lib/src/core/threshold/pk_generation.nr` and `circuits/lib/src/configs/secure/threshold.nr`:
N = 8192, L = 3, QIS = 576460752317792257 / 576460752317399041 / 576460752316776449, EEK_BOUND = 20 (two-sided), SK_BOUND = 1, E_SM_BOUND =
13292279957849975063929256991047181926400000 (~2^143.25), E_SM_LIMB_BOUNDS = (q_i−1)/2,
E_SM_QUOTIENT_BOUNDS = (E_SM_BOUND + (q_i−1)/2)/q_i (~2^84.25).

| | interfold C1, Poseidon2 commitments |
|---|---|
| trace | 1024 rows × 13,500 int cols (168 public), 13,324 lookups, 10 Poseidon2 permutations per row |
| witness gen / hashing / trace | 0.015 s / 0.019 s / 0.31 s |
| **prove** | **11.0 s** (checked: 12.5 s) |
| **verify** | **0.34 s** + 0.028 s public-column derivation |
| proof | 24.1 MB raw / 14.1 MB zstd |
| peak RSS | 4.4 GB |
| tamper tests | 7/7 rejected (sk, quotient, sponge state, eek out of lookup range, eek = 21 inside the lookup but outside [−20, 20], pk0 outside [−c, c], e_sm CRT quotient) |

Per-step: commit 1.93 s (18%), prime projection 0.35, ideal check 0.56, CPR sumcheck 3.24 s (30%),
lookup GKR-LogUp 3.62 s (33%), multipoint 0.29, lift 0.23, open 0.68.

What is enforced, per coefficient, exactly as C1 does (`range_check_2bounds` = offset into
`[0, 2·bound]` plus a bounded complement; here both as 16-bit limb decompositions):
`sk ∈ {−1,0,1}` (cubic); `eek ∈ [−20, 20]`; `pk0[i] ∈ [−(q_i−1)/2, (q_i−1)/2]`;
`e_sm_lifted ∈ [−B, B]` (145-bit offset, 10 + 10 limbs); `e_sm[i] ∈ [−(q_i−1)/2, (q_i−1)/2]`;
`e_sm_quot[i] ∈ [−Qb_i, Qb_i]`; `e_sm_lifted = e_sm[i] + e_sm_quot[i]·q_i` (integer identity, constants
`B − c_i − Qb_i q_i` as 145-bit scalars); and `pk0[i] + a[i]·sk − eek ≡ 0 mod (q_i, X^N + 1)`.
Outputs `commit(sk)`, `commit(pk0)` (3 centered polynomials as 29/30-bit limbs), `commit(e_sm)` (the 3
centered residue polynomials, as C1 — not the lift), plus `cm_aux = commit(eek)`.

Differences from C1 that remain, deliberately:
* **Hash function.** C1's commitments and challenge are a Poseidon/SAFE sponge over BN254 with its
  `v + 2^nibble` digit packing and domain separators; here they are a Poseidon2-KoalaBear-16 sponge over
  the same inputs. The commitment *values* therefore differ and C2a/C2b/C5 could not open them. Emulating
  BN254 Poseidon in Zinc+ means 254-bit modular arithmetic per S-box (x^5 with ~32 16-bit quotient limbs
  and 16 output limbs per multiplication): roughly 8–10× the cells of this circuit, ~20 GB on this design,
  so it was not attempted here.
* **Quotient `r`.** C1 checks `pk0 = −a·sk + eek + q·r` at one Fiat–Shamir point with `|r| ≤ 4096`;
  here the identity is checked modulo q_i directly (random root-of-unity combination, two β per prime,
  ≈ 2^-90 per attempt), which is the same statement without the `r` witness.
* **Fiat–Shamir transcript.** C1 derives `gamma` in-circuit from `(a, eek, commit(sk), commit(pk), r)`;
  here β is derived with blake3 from the four public commitments, so eek enters via `cm_aux` rather than
  being absorbed in-circuit. `a` is a public input here and a compile-time constant in C1.
* `e_sm_lifted` is sampled uniformly in `[−B, B]` (C1's witness comes from fhe.rs' smudging sampler;
  the circuit only sees the bound).

Run: `RUSTFLAGS="-C target-cpu=native" cargo build --release --features unchecked`, then `REPS=3 STEPS=1 ./target/release/bench_c1`. Tamper cases `TAMPER=t|u|out|range|eek41|pk|crt`.
Code: `src/c1_uair.rs`, `src/bin/bench_c1.rs` (shares `circuit.rs`, `ring.rs`, `poseidon2.rs`).
