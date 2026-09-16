# GeoFluidLBM

GeoFluidLBM is a Julia lattice Boltzmann simulation of pressure-driven Poiseuille channel flow,
using the D2Q9 BGK model.

This file explains how to set up the project, run the simulation, run the test, run the benchmarks,
and render a video. It also documents the code's interfaces and output formats. For the assignment
itself, including the tasks, the required parallelisation approach, and the grading scheme, see the
assignment specification.

## Getting started

Follow these steps to confirm that your environment works before you write any code:

1. Unpack the assignment archive and open a terminal in the project root.

2. Install the Julia packages:

   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

3. Start Julia inside the project:

   ```bash
   julia --project=.
   ```

4. Run a short simulation and inspect the result:

   ```julia
   using GeoFluidLBM

   cfg = Config(nx = 128, nz = 128, nt = 500)
   timings, s = run_seq(cfg)

   timings["total"]                        # wall-clock time of the timestep loop, in seconds
   velocity(s)[1, :, div(cfg.nx, 2)]       # u_x across the channel at mid-length
   ```

The final line prints one value per grid row. The values start near zero at the walls, rise to a
maximum in the centre, and fall again. That parabola is the analytical Poiseuille profile, and
reproducing it is what the test checks.

## Setup

### Requirements

The project was tested with Julia 1.11.4, which is the version installed on DAS-5. Local runs need no separate
MPI installation, because MPI.jl installs its own MPI build.

The project contains two Julia environments:

- The root environment, defined by `Project.toml`, runs the simulation, the test, and the
  benchmarks. It contains no plotting packages.
- The `render` environment adds CairoMakie for the optional video.

### Local machine

Install the packages and the MPI launcher:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'
```

`install_mpiexecjl` writes the `mpiexecjl` launcher to `~/.julia/bin`. Add that directory to your
`PATH`. On a UNIX system, this can be done with

```bash
export PATH="$HOME/.julia/bin:$PATH"
```

### DAS-5

Run the setup script from the project root:

```bash
source setup.sh
```

The script loads the cluster modules, installs the packages, binds MPI.jl to the cluster MPI,
builds `mpiexecjl` into `bin/`, and precompiles the package.

> **Run the script with `source`, not `bash`.** The script loads modules and extends your `PATH`.
> Both changes are lost if the script runs in a subshell.

## Run a simulation

A run needs a `Config`, which holds every setting for one simulation. The following table lists the
keyword arguments and their defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `nx` | 128 | Number of grid columns. |
| `nz` | 128 | Number of grid rows. |
| `nt` | 100 | Number of timesteps. |
| `nu_f` | 0.1 | Kinematic viscosity, in lattice units. |
| `dt` | 1.0 | Timestep. |
| `dx` | 1.0 | Cell size. |
| `rho_0` | 1.0 | Reference density. |
| `rho_in` | 1.01 | Inlet density, held on the left column. |
| `rho_out` | 0.99 | Outlet density, held on the right column. |

`State(cfg)` builds the simulation data for that configuration. The state starts with the fluid at
rest and solid walls on the top and bottom rows. Both solvers create their state internally.

The project provides two solvers:

- `run_seq(cfg; callback = nothing)` runs the sequential solver.
- `run_par(cfg, comm)` runs the MPI solver.

> **You may assume that `nx` is a multiple of the number of MPI ranks.** Your `run_par` does not
> have to handle a leftover strip of columns. Every grid used by the test and the benchmark sweep
> has a power-of-two `nx`, and every rank count is a power of two, so the assumption always holds.

Both return `(timings, state)`. Read the fields of the returned state with the accessors rather
than by indexing the `State` directly:

| Accessor | Shape | Contents |
|---|---|---|
| `density(s)` | `(nz, nx)` | Fluid density. |
| `velocity(s)` | `(2, nz, nx)` | Velocity, where component 1 is `x` and component 2 is `z`. |
| `speed(s)` | `(nz, nx)` | Velocity magnitude. |

To observe the simulation as it runs, pass a callback:

```julia
run_seq(cfg; callback = (state, step) -> println(step))
```

The solver calls `callback(state, step)` once at `step = 0` and again after every timestep. Leave
the callback out when you benchmark, because it runs inside the timed loop.

## Test

`test/test_par.jl` checks the conditions listed in the assignment specification. The test runs a 64
by 32 grid for 8000 timesteps, which is long enough to reach steady state. It accepts a relative L2
error below 0.08 against the parabola, and below 1e-10 between the two solvers.

Run the test at both rank counts:

```bash
mpiexecjl --project=. -n 2 julia --project=. test/test_par.jl
mpiexecjl --project=. -n 4 julia --project=. test/test_par.jl
```

The test prints the three measured errors, then `PASS` or `FAIL`.

## Benchmark

### Local runs

`benchmark/benchmark.jl` times the solver at the current rank count. Pass the grid on the command
line as `nx nz nt`.  For instance:

```bash
julia --project=. benchmark/benchmark.jl 256 256 50
mpiexecjl --project=. -n 8 julia --project=. benchmark/benchmark.jl 512 512 50
```
At one rank, the script calls `run_seq`, and at more than one rank it calls
`run_par`:

The script runs a small warmup, repeats the measurement three times, and keeps the fastest result.

Use local runs while you develop. They tell you whether your solver works, but their timings mean
nothing for the report, because your machine is not the cluster.

### The DAS-5 sweep

Produce every number in your report with the job script:

```bash
source setup.sh
bash benchmark/benchmark_job.sh
```

The script sweeps the grids and rank counts fixed by the assignment specification, writing one JSON
record per run.

Three rules apply on DAS-5:

- **Run the job script with `bash`, directly on the head node.** Do not use `sbatch`. The script
  calls `prun`, which makes its own reservation, and a SLURM batch job breaks it with
  `Reserve lib error: no callback connection`.
- **Do not run `benchmark.jl` by hand on the head-node of DAS-5.** Without `prun`, `mpiexecjl` executes on the
  shared head node instead of on reserved compute nodes, so the timings are meaningless.
- **Do not change `GRIDS` in the job script.** The grids match the problem sizes fixed by the
  assignment specification.

## Output formats

Each benchmark run writes one file to `benchmark/results/`, named
`benchmark_P{P}_nx{nx}_nz{nz}_nt{nt}.json`. The contents of the files look like:

```json
{
  "P": 8, "nx": 512, "nz": 512, "nt": 250, "nruns": 3,
  "timings": { "total": 0.83, "boundary": 0.00, "stream": 0.34, "macroscopic": 0.21,
               "equilibrium_collide": 0.25, "compute": 0.80, "comm": 0.03 }
}
```

`P` is the number of MPI ranks, and `nruns` is the number of repeats. The `timings` values come from
the fastest repeat. The following table describes each key, with all values in seconds:

| Key | Meaning |
|---|---|
| `total` | Wall-clock time of the whole timestep loop. |
| `boundary` | Time in the boundary step (inlet/outlet density driving), summed over the run. |
| `stream` | Time in `stream!`, summed over the run. |
| `macroscopic` | Time in `macroscopic!`, summed over the run. |
| `equilibrium_collide` | Time in `equilibrium_collide!`, summed over the run. |
| `compute` | Sum of the four kernel times. |
| `comm` | Time spent communicating. Always `0.0` for `run_seq`. |

Setup, the work distribution, and the final gather all sit outside the timed loop, so they do not
appear in these numbers.

## Visualisation

The renderer records an mp4 heatmap of the flow speed. It uses `run_seq`, so it works before you
implement `run_par`. The video is optional and is not marked.

```bash
julia --project=render -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=render render/render.jl
```

The first command takes several minutes, because CairoMakie precompiles slowly. The second writes
`render/output/poiseuille.mp4`.

## Reference

### Project structure

| Path | Contents |
|---|---|
| `src/GeoFluidLBM.jl` | `Config`, `State`, the accessors, and the physics kernels. |
| `src/run_seq.jl` | The sequential solver `run_seq`. |
| `src/run_par.jl` | The MPI solver `run_par`, which you implement. |
| `test/test_par.jl` | The correctness test, run under MPI. |
| `benchmark/benchmark.jl` | Times the solver and writes one JSON record. |
| `benchmark/benchmark_job.sh` | The DAS-5 strong-scaling sweep. |
| `benchmark/results/` | Where the benchmark writes its JSON records. |
| `render/render.jl` | The heatmap video recorder. |
| `setup.sh` | DAS-5 environment setup. |

### Array conventions

Julia arrays are 1-based and stored in column-major order, so the first index varies fastest.

- Vector fields use `[component, z, x]`, where component 1 is `x` and component 2 is `z`.
- `z` is the vertical axis, which indexes rows. `x` is the horizontal axis, which indexes columns.

The assignment specification describes the layout of the particle distribution array.

### Lattice units and helpers

The model works in lattice units, where `dx` and `dt` are 1.0 by default and the speed of sound
satisfies `cs² = 1/3`.

Streaming finds its source cell with `upstream(i, di, n) = mod1(i - di, n)`, which wraps
periodically. Wall bounce-back and the inlet/outlet boundary step handle the non-periodic physics
separately.
