# interfold C1 (`pk_generation`, secure preset) in one Zinc+ proof

Measured 2026-11-03 on 2 vCPU, 8 GB, rate 1/8, 100-bit target. Medians of 3 unchecked runs after a verified checked run. Statement taken from
`circuits/lib/src/core/threshold/pk_generation.nr` and `circuits/lib/src/configs/secure/threshold.nr`:
N = 8192, L = 3, QIS = 576460752317792257 / 576460752317399041 / 576460752316776449, EEK_BOUND = 20 (two-sided), SK_BOUND = 1, E_SM_BOUND =
13292279957849975063929256991047181926400000 (~2^143.25), E_SM_LIMB_BOUNDS = (q_i−1)/2,
E_SM_QUOTIENT_BOUNDS = (E_SM_BOUND + (q_i−1)/2)/q_i (~2^84.25).

| | interfold C1, Poseidon2 commitments |
|---|---|
| trace | 1024 rows × 18,420 int cols (168 public), 18,244 lookups, 14 Poseidon2 permutations per row (all sponges rate 8) |
| witness gen / hashing / trace | 0.015 s / 0.029 s / 0.37 s |
| **prove** | **13.9 s** (checked: 15.7 s) |
| **verify** | **0.35 s** + 0.028 s public-column derivation |
| proof | 32.9 MB raw / 19.2 MB zstd |
| peak RSS | ≈ 6 GB |
| tamper tests | 7/7 rejected (sk, quotient, sponge state, eek out of lookup range, eek = 21 inside the lookup but outside [−20, 20], pk0 outside [−c, c], e_sm CRT quotient) |

Per-step: commit 2.69 s (19%), prime projection 0.48, ideal check 0.77, CPR sumcheck 4.41 s (32%),
lookup GKR-LogUp 3.93 s (28%), multipoint 0.43, lift 0.31, open 0.76.

(Revision: the `commit(pk0)` and `commit(e_sm)` sponges now absorb 8 elements per permutation instead of 12.
Rate 12 on 31-bit lanes leaves a 124-bit capacity, i.e. only ~62-bit collision resistance; rate 8 gives
124-bit. This is why C1 grew from 13.5k to 18.4k columns and from 11.0 s to 13.9 s. C2b below opens exactly
this `commit(e_sm)`, so the two have to agree.)

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


---

# interfold C2 / C3 and the merged dealer circuit (micro committee) in Zinc+

Code: `src/vm.rs` (the UAIR and all prover/verifier data paths), `src/dkg.rs` (parameters, parity matrix,
Shamir sharing, BFV key generation and encryption in C3's exact form), `src/bin/bench_c2c3.rs`.
Measured on the same 2 vCPU / 8 GB machine, rate 1/8, 100-bit target, medians of 3 unchecked runs after a
verified checked run (`cargo test --release --lib vm` with `VM_TEST_MODE=c2a|c2b|c3|merged` runs the
exact integer constraint checker on honest and tampered traces; `TAMPER=…` runs the Zinc+ prover/verifier
on a cheating prover).

## Statements

Parameters from `circuits/lib/src/configs/secure/` and `configs/micro/`: threshold moduli as for C1,
micro committee `N_PARTIES = 9`, `T = 4`, parity matrix `H_j = [−A | I_5]` (rows = the Lagrange
extrapolation coefficients of a degree-4 polynomial through points 0..4 evaluated at 5..9, mod `q_j`,
checked cell-by-cell against `share_computation.nr`'s constants); DKG BFV `N = 8192`, `L = 2`,
`QD = 2305843009214758913 / 2305843009214988289`, plaintext modulus `t = q_0 = 576460752317792257`,
`Q_MOD_T = 3169161947774985`, `K0 = [493937176208329543, 1434253984466648362]` (`k0·t ≡ −1 mod q'_i`),
`u` ternary, `e0, e1 ∈ [−20, 20]`.

* **C2a / C2b** (`share_computation.nr`, one track each). Opens C1's `commit(sk)` (resp. `commit(e_sm)`,
  the 3 centered residue polynomials as 29/30-bit limbs — the same bytes C1 hashed), proves for every
  coefficient and modulus `j` that the 9 shares lie in `[0, q_j)` and satisfy `H_j · (secret, y_1..y_9) ≡ 0
  (mod q_j)` with `secret = sk` (resp. the residue), and outputs the 27 share commitments
  `commit(y[j][p])` (the share polynomial in `[0, q_j)` as three 24-bit chunks per coefficient).
* **C3** (`share_encryption.nr`, scaled-quotient path, one encryption). Opens `commit(m)` (a C2 share
  commitment) and `commit(pk)`, proves `m ∈ [0, t)`, `k1 = q_mod_t·m − t·z` with `k1 ∈ [0, t)`,
  `u ∈ {−1,0,1}`, `e0, e1 ∈ [−20, 20]`, `ct0, ct1 ∈ [−(q'_i−1)/2, (q'_i−1)/2]` and
  `ct0 ≡ pk0·u + e0 + k0·k1`, `ct1 ≡ pk1·u + e1 (mod q'_i, X^N+1)` for both DKG moduli, and outputs
  `commit(ct)` (4 polynomials, three 24-bit chunks of the centered-offset coefficient each).
* **Merged** (both tracks, all 3 moduli, all 9 recipients = 54 encryptions in one proof). Opens
  `commit(sk)`, `commit(e_sm)` and the 9 recipient `commit(pk)`; proves the shares, the parity rows, the
  message scaling of every share, and the 54 encryption identities; outputs the 54 `commit(ct)`. The
  shares are never hashed (that is the point of merging).

Recipient public keys are *public* inputs here (the verifier needs them for the identity weights), and are
hashed in-circuit as well so that `commit(pk)` is checked as in interfold. Nothing is zero-knowledge.

## How it is laid out: a three-kind "coprocessor" trace

One Zinc+ UAIR with three row kinds selected by public one-hot columns, sharing the physical int columns
(pooled by lookup table, so a row is as wide as the widest kind, ~1600–2060 cells) with every constraint of
a kind multiplied by its selector (max degree 4):

* **hash rows**: one Poseidon2-KoalaBear-16 permutation at rate 8 (~1280 cells); public `start`/`cont`/
  `end` flags chain the state and pin the digest; sponges over public data (`pk`) have their inputs pinned
  to public columns; sponges over witness data receive their inputs over the bus;
* **enc rows**: 8 coefficients of one encryption (C3) or 1 coefficient of 9 encryptions (merged);
* **shr rows**: one coefficient of one track: secret, 27 shares with complements, 15 parity quotients, and
  in merged mode `z, k1` for all 27 shares.

Two things cross row kinds and live in dedicated columns of Zinc+'s *arbitrary-polynomial lane*
(degree-0 polynomials, 31-bit limbs so Zip+'s integer IPRS encoding cannot overflow):

* a **bus** of three accumulators mod `M = 2^61 − 1`: `acc' = acc + Σ_runs base·Σ_k γ^k v_k − M·u`, where
  every hashed element has one producer row (a share / secret / ct / k1 cell) adding `γ^idx·v` and one
  consumer row (the hash row that absorbs it) adding `(M − γ^idx)·v`; `acc` starts at 0 (row 0) and must
  return to 0 (explicit `last` flag — Zinc+ does not constrain the last row). The sum vanishes iff the two
  sides agree, except with probability `≤ 2^23/2^61` per `γ`, three independent `γ` ⇒ `≈ 2^-114`;
* **encryption accumulators**, one per (DKG modulus, β) = 4, folding all encryptions and both legs with
  public `ρ`: the same random root-of-unity trick as C1 (`Σ_m ct_m U_m − u_m V_m(pk) − e_m U_m − k0 k1_m U_m`
  with `U, V` from `ring.rs`), ≈ `2^-116` per modulus.

**Fiat–Shamir.** `γ`, `β`, `ρ` must be unpredictable when the row values are fixed, so the prover commits
the int lane first with `ZipPlus::commit` (the same deterministic commitment Zinc+ computes inside
`prove`), derives the challenges with blake3 from that root and the statement, fills the accumulator
columns, and proves. The verifier reads the int-lane commitment from the proof, derives the same
challenges and builds the public weight columns from them. Everything that enters an identity is therefore
committed before its challenge; the accumulator columns are committed separately (arbitrary lane). The
pre-commit costs ~20 % extra prover time; it would disappear with a two-stage commit inside Zinc+
(which was not modified).

Soundness checks that run: the exact integer checker (`vm::check`) on honest traces (0 failures) and on
tampered ones; Zinc+ rejects (prover lookup failure or verifier ideal-check / resolver failure) for: a
share limb +1 (parity), a changed secret cell, a changed `u`, `e0 ± 1` with its complement kept, `e0 = 41`
(lookup), a `ct` limb, a `k1` limb (merged), a sponge-state lane, an absorbed chunk of a witness sponge
(bus), a wrong statement digest, a wrong recipient key — 20/20 rejected.

## Measurements

All rows counted at the measured ring dimension `N'`; the circuits are row-linear in `N`, so the N = 8192
figures are `×(8192/N')` of the measured ones with the measured utilisation (rows round up to a power of
two: C3 88 %, C2 70–74 %, merged 81 %). Prove times exclude the pre-commit, which is listed separately.

| circuit | N' | rows (used) | int cols | prove | pre-commit | verify | proof (zstd) | peak RSS |
|---|---|---|---|---|---|---|---|---|
| C3 | 256 | 1024 (900) | 1830 | 1.66 s | 0.30 s | 0.079 s | 3.9 / 2.3 MB | 0.80 GB |
| C3 | 512 | 2048 (1796) | 1830 | 3.19 s | 0.73 s | 0.125 s | 3.7 / 2.0 MB | 1.74 GB |
| C3 | 1024 | 4096 (3588) | 1830 | 5.84 s | 1.30 s | 0.20 s | 3.9 / 2.3 MB | 3.33 GB |
| C2a | 256 | 4096 (2884) | 1569 | 5.25 s | 1.13 s | 0.17 s | 3.4 / 2.1 MB | 3.00 GB |
| C2a | 512 | 8192 (5764) | 1569 | 11.8 s (1 run) | 2.74 s | 0.32 s | 3.7 / 2.2 MB | 5.65 GB |
| C2b | 256 | 4096 (3140) | 1574 | 5.45 s | 1.16 s | 0.17 s | 3.5 / 2.1 MB | 2.89 GB |
| C2b | 512 | 8192 (6276) | 1574 | 11.3 s | 2.81 s | 0.31 s | — | 5.67 GB |
| merged | 16 | 2048 (1664) | 2058 | 3.62 s | 0.81 s | 0.15 s | 4.0 / 2.2 MB | 1.72 GB |
| merged | 32 | 4096 (3324) | 2058 | 6.69 s | 1.38 s | 0.27 s | 4.1 / 2.5 MB | 3.52 GB |

Scaling is linear in rows (C3 ×1.92, ×1.83 per doubling; C2 ×2.1; merged ×1.85), so at **N = 8192**:

| circuit | rows | cells | prove (+ pre-commit) | verify | RAM (Zinc+ ≈ 480 B/cell) |
|---|---|---|---|---|---|
| C2a | 2^17 (92,164 used) | 206 M | ≈ 180 s (+ 44 s) | ≈ 0.5 s | ≈ 90 GB |
| C2b | 2^17 (97,284 used) | 206 M | ≈ 180 s (+ 45 s) | ≈ 0.5 s | ≈ 91 GB |
| C3, one encryption | 2^15 (28,676 used) | 60 M | ≈ 47 s (+ 10 s) | ≈ 0.3 s | ≈ 27 GB |
| separate pipeline: C2a + C2b + 54 × C3 | | 3.65 G | **≈ 59 min**, 56 proofs (~4 MB each) | 56 × | 91 GB |
| merged (54 encryptions, 1 proof) | 2^20 (846,852 used) | 2.16 G | **≈ 29 min (+ 6 min)** | ≈ 1–5 s | **≈ 900 GB** |

Where the rows go (N = 8192): C3 — 1,024 enc rows, 3,072 message-hash rows, 12,288 pk-hash rows, 12,288
ct-hash rows (hashing is 96 % of the circuit). Merged — 49,152 enc rows, 16,384 share rows, 1,024 + 6,144
secret-hash rows, 110,592 pk-hash rows, 663,552 ct-hash rows (92 % hashing). The merged circuit is cheaper in
total work (each recipient key is hashed once instead of six times, the shares are not hashed, no message
openings), but it is one monolithic trace, so its memory is the whole thing at once; the separate pipeline
never needs more than one C2 trace (~90 GB) in memory and parallelises over 54 independent C3 proofs.
Per-step split (both modes): commit 20 %, ideal check 11–12 %, CPR sumcheck 30 %, lookups 27 %, PCS open 3 %.

Everything above is dominated by hashing 61/62-bit ring coefficients with a 31-bit-field Poseidon2 at
~6.7 cells per hashed bit (three 24-bit chunks per coefficient, eight per permutation, ~1280 cells per
permutation). interfold hashes the same data as ~254-bit BN254 field elements; in Zinc+ a BN254 Poseidon
would be far more expensive still (see the C1 notes), so the choice of commitment hash is the lever that
matters here — e.g. a commitment to `ct` with fewer, wider absorptions or a cheaper hash would scale all
of these numbers down almost proportionally.

## Differences from interfold, and caveats

* Hash function and serialisations: Poseidon2-KoalaBear-16 (rate 8) over 24-bit chunks (secrets/shares/
  ct/pk) resp. C1's 29/30-bit limbs (`e_sm`); interfold uses BN254 Poseidon over field elements. Digests
  differ; the *statements* (what is proven about the preimages) are the same.
* Recipient keys are public inputs (and hashed in-circuit); interfold keeps `pk` private behind
  `commit(pk)`.
* Fiat–Shamir challenges come from a Zip+ commitment root + blake3 instead of an in-circuit transcript;
  identities are checked modulo `q'_i` by random root-of-unity combination (no quotient witness `r`).
* `z < 2^64` is enforced rather than interfold's `z < 2^52`; `k1 ∈ [0, t)` and `m ∈ [0, t)` already fix
  `z = (q_mod_t·m − k1)/t < q_mod_t < 2^52`.
* The `e_sm` residue offsets reuse C1's 59-bit limb layout (`[0, 2^59)` ⊂ `[0, q_j)`): an honest residue
  offset above `2^59 − 1` (probability ≈ 2.5·10⁻¹¹ per residue) cannot be proven — this incompleteness is
  inherited from the C1 circuit's layout and kept so that C2b opens exactly C1's commitment.
* The layout is installed once per process (`vm::install`), so one process proves one mode; the verifier
  in the harness re-derives the challenges from the proof and would reject any proof whose int commitment
  does not match them (the `digest` tamper case is caught at that step).
* The N = 8192 numbers are extrapolations from N' ≤ 1024 (C3), 512 (C2), 32 (merged) on an 8 GB machine.

Run: `cargo build --release --features unchecked`; `MODE=c3 N=1024 REPS=3 ./target/release/bench_c2c3`
(`MODE=c2a|c2b|c3|merged`, `N` a power of two ≥ 8, `STEPS=1`, `CHECK=1` for the integer checker,
`TAMPER=share|secret|ct|u|e|e41|hash|absorb|digest|pk|k1`).
