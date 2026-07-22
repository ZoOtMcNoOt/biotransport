# Reproducible performance evidence

BioTransport's native benchmarks are measurement tools, not marketing claims.
They emit workload, correctness, build, platform, threading, and raw timing data
in versioned JSON. A timing result applies only to the recorded executable,
machine, operating conditions, and workload. It does not establish a general
speedup or performance guarantee.

The machine-readable contract is
`cpp/benchmarks/performance_schema.json` and the current schema identifier is
`biotransport.performance.v1`.

## Audited workloads

The previous benchmark sources included three headers that no longer existed:

- `physics/mass_transport/diffusion.hpp`
- `physics/mass_transport/linear_reaction_diffusion.hpp`
- `physics/mass_transport/variable_diffusion.hpp`

The repaired programs use current public native APIs:

| Executable | Native path | Correctness evidence | OpenMP-effective kernel |
|---|---|---|---|
| `bench_diffusion_2d` | `MultiSpeciesSolver`, one species, conservative 2D diffusion | closed-boundary trapezoidal mass conservation plus deterministic weighted checksum | node update in `computeCandidateStep`; candidate validation, setup, and checksum remain serial |
| `bench_linear_reaction_diffusion` | canonical `TransportProblem` and conservative `solve` | uniform field compared with the exact discrete forward-Euler decay sequence | none; reported explicitly as serial even in an OpenMP build |
| `bench_variable_diffusion` | canonical `TransportProblem` with a node field for (D) and harmonic face values | closed-boundary trapezoidal mass conservation plus deterministic weighted checksum | none; reported explicitly as serial even in an OpenMP build |

All workloads are deterministic and bounded by default: 64, 128, and 256 cells
per axis, 50 steps, one warmup, and five timed runs. Command-line limits prevent
accidentally unbounded runs. `--quick` uses one 64-by-64 grid, 10 steps, no
warmup, and three timed runs.

The timed scope is deliberately broad and is stored in every result as
`construct_configure_solve_and_compute_invariant`. It includes solver
construction, initial/boundary configuration, integration, and the correctness
calculation. It is therefore not a kernel-only microbenchmark. Throughput uses
the median elapsed time and is labeled `node_species_steps_per_second`.

## Build and run

A portable serial Release build can be configured with:

```text
cmake -S . -B build-bench-serial -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARKS=ON \
  -DBIOTRANSPORT_OPENMP=OFF \
  -DBIOTRANSPORT_EIGEN=OFF
cmake --build build-bench-serial --target bench_diffusion_2d \
  bench_linear_reaction_diffusion bench_variable_diffusion
```

For an OpenMP build, set `BIOTRANSPORT_OPENMP=ON`. Confirm in the output JSON
that all of the following match the intended experiment:

- build type and configured flags;
- `assertions_enabled` state;
- compiler identity and version;
- `openmp.enabled`, specification date, and compile flags;
- observed thread count and `OMP_NUM_THREADS` value; and
- each result's `openmp_effective_for_workload` field.

Do not compare a Debug or unoptimized executable with an optimized executable.
The benchmark records CMake cache flags and separately labels that implicit
toolchain and target flags may not be exhaustive.

### Controlled serial-versus-OpenMP path

Use the **same OpenMP-enabled executable**, arguments, and environment except
for the thread count. This avoids conflating compiler or build differences with
threading:

```powershell
$env:OMP_DYNAMIC = "FALSE"
$env:OMP_NUM_THREADS = "1"
build-bench-openmp\cpp\benchmarks\Release\bench_diffusion_2d.exe `
  --sizes 128,256,512 --steps 100 --warmup 2 --runs 9 `
  --output diffusion-threads-1.json

$env:OMP_NUM_THREADS = "4"
build-bench-openmp\cpp\benchmarks\Release\bench_diffusion_2d.exe `
  --sizes 128,256,512 --steps 100 --warmup 2 --runs 9 `
  --output diffusion-threads-4.json
```

Compare medians only after checking that the workload records match, both
invariants pass, and checksums agree. A ratio may be calculated for description,
but the benchmark deliberately has no speedup threshold. Small workloads can be
slower with more threads, and even favorable results do not imply scaling on
another CPU or problem.

## JSON evidence contract

Every document includes:

- schema version and UTC generation time;
- Git revision and dirty-tree state when discoverable;
- compiler, C++ standard, project version, build type, configured flags,
  assertions, generator, and Eigen/OpenMP feature metadata;
- operating system, architecture, CPU description, and logical thread count;
- OpenMP compile state, specification date, maximum and observed team sizes,
  dynamic-team state, and relevant environment variables;
- requested sizes, steps, warmups, and timed runs;
- exact cell/node/species counts, time step, final time, and model parameters;
- invariant name, reference and observed values, absolute and relative errors,
  tolerance, checksum definition, and repeated-run checksum agreement;
- every raw elapsed sample plus mean, population standard deviation, minimum,
  maximum, and median; and
- explicitly labeled median-based throughput.

The writer refuses non-finite JSON numbers. Warmup correctness failures stop the
run; timed correctness failures produce a failing status and a nonzero process
exit. Failed or non-repeatable numerical evidence must not be used to support a
timing comparison.

`python/tests/test_benchmark_contract.py` checks schema identity, finite numbers,
passing invariants, repeated-run checksums, timing statistics, and throughput.
When JSON files exist under `build/performance-evidence`, it validates those
actual artifacts too.

## Local controlled observation: 2026-07-22

This implementation was exercised on the current development workstation using
the same MSVC/OpenMP executable at one and four threads. This is a local
observation, not a library baseline.

- Revision: `002d94219759c8bc3fee4d0e7a5f8f04c02bde80`, dirty working tree.
- Compiler: MSVC 19.44.35228, C++17.
- Configured Release flags: `/O2 /Ob2 /DNDEBUG /EHsc`; OpenMP flag: `-openmp`.
- Platform metadata: Windows AMD64, Intel64 Family 6 Model 154 Stepping 3,
  20 logical threads.
- OpenMP metadata: specification date 200203; dynamic teams disabled.
- Workload: conservative one-species diffusion, 100 steps, two warmups, nine
  timed samples. Timed scope includes construction, solve, and invariant.

| Cells per axis | Median, 1 thread (ms) | Median, 4 threads (ms) | Observed median ratio | Cross-thread checksum |
|---:|---:|---:|---:|---|
| 128 | 55.7032 | 43.4658 | 1.282 | identical |
| 256 | 234.6763 | 193.9501 | 1.210 | identical |
| 512 | 986.6108 | 692.8581 | 1.424 | identical |

All six runs passed the mass invariant. The largest recorded relative mass
error was (1.93\times10^{-14}). The ratios differ materially by problem size;
no threshold was asserted and no extrapolation beyond this run is warranted.

## Reproducibility limits

Elapsed time remains sensitive to CPU frequency and temperature, background
load, memory placement and bandwidth, operating-system scheduling, antivirus or
filesystem activity, process affinity, OpenMP runtime policy, compiler updates,
and firmware. The current runner does not pin cores, isolate a NUMA node,
disable frequency scaling, flush caches, or measure energy. The first warmups
reduce some cold-start effects but do not control those variables.

For stronger evidence, run on an otherwise idle machine, retain every JSON
artifact, repeat the complete experiment in multiple process launches, state
thread affinity and power policy, and compare distributions rather than a
single best time. Publication-quality performance conclusions require a
predeclared protocol appropriate to the target hardware and intended workload.
