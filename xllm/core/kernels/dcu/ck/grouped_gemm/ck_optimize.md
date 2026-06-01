# ck_tile DCU migration and optimization notes

## Current workflow

1. Port coverage before tuning. A path is not a performance candidate until the CLI validation and the C ABI dispatch contract are correct.
2. Add a narrow fast build target for the dtype/layout under investigation. Full `tile_example_grouped_gemm` remains the coverage target, while `*_fast` targets reduce compile latency during iteration.
3. Before each benchmark, check HCU use and memory use with `rocm-smi`, then set `HIP_VISIBLE_DEVICES` to an idle card.
4. Record every measured result in `grouped_gemm_readme.md`: commit/stage, command, GPU id, validation result, latency, throughput, bandwidth, and speedup only when comparing two validated runs.
5. Keep remote scratch scripts, logs, ISA dumps, and profiler outputs under `hygon_tmp`.

## Correctness lessons

- For DCU builds where `ck_tile::half_t` maps to host `_Float16`, avoid assuming host random fill, `type_convert<float>`, or generic `check_err` are reliable for fp16 debugging.
- If fp16 validation reports isolated or impossible mismatches, first inspect raw input/output bits and manually convert half bits to float before blaming the kernel.
- For grouped_gemm fp16, deterministic raw half-bit test data plus a raw-bit CPU reference produced stable validation on `gfx938`.
- When adding a new dtype, first look for the MMAC instruction wrapper and the exact warp dispatcher shape required by the current grouped_gemm tile before changing tile sizes.
- For multiple-D epilogues, validate against the epilogue data path rather than pure fp32 GEMM math. The current fp16 AddAdd path casts the GEMM result to half before adding D tensors, so the checker needs half-rounding tolerance.

## Build lessons

- `tile_example_grouped_gemm` is dominated by one large translation unit, so `-j` cannot parallelize most of the compile.
- Opt-in `-save-temps`, opt-in resource reports, and dtype-specific fast targets are currently the cheapest build-speed wins.
- A stronger follow-up is explicit-instantiation source splitting by dtype/layout so the official full target can use multiple compiler processes.

## Tuning gate

- Do not tune a dtype until its default correctness path passes.
- For each candidate, keep the previous validated champion available through `-config` or a separate target.
- Treat resource reports and ISA checks as diagnostics; only measured benchmark deltas on an idle GPU decide whether a candidate is promoted.

## Stage 15 fp8/bf8 notes

- Do not assume fp8 and bf8 share every MMAC warp shape. On gfx938 in this tree, bf8 has a working `32x64x32` grouped_gemm path, but fp8 `WarpGemmMmacDispatcher<fp8, fp8, float, 32, 64, 32, ...>` is undefined and fails at compile time.
- For default grouped shapes, `128x128x128` with `K_Warp_Tile=64` remains the fp8/bf8 champion. Smaller `N_Tile=64` variants validated but lost badly, so keep them as diagnostic configs only if needed.
- A failed compile is still useful evidence: record the exact missing dispatcher signature before deleting or changing the candidate.
- `hipprof --pmc --pmc-type 3` may complete without a useful table in stdout; store logs under `hygon_tmp` and do not make bottleneck claims unless counters are actually available.

## Stage 16 MMAC table notes

- `mmac.xlsx` confirms gfx938 supports `fp8/fp8`, `bf8/bf8`, `fp8/bf8`, and `bf8/fp8` through `f32_16x16x32_*` builtins. It also confirms int4/u4 integer MMAC support through `i32_16x16x64_i4/u4`.
- The f8/f6/f4 mixed floating builtins and fp4 floating MMAC rows are gfx946/gfx948 only; do not target them for BW1000/gfx938.
- If a `WarpGemmMmacDispatcher` shape is missing but the underlying builtin exists, first add a narrow alias and dispatcher specialization, then compile and benchmark as a separate diagnostic config.
- For fp8, adding `WT32x64x32_MR2NR4MI1NI1` made `-config=fp8_128x128_k32` compile and validate, but it was slower than K64 on default grouped shapes. Do not promote K32 without a shape-specific win.
- A pure B-side `NInterleave=4` pipeline is not enough by itself because `CShuffleEpilogue` also instantiates a warp dispatcher for the same C tile. Any future interleave experiment must update both the main GEMM pipeline and epilogue layout expectations.
