# benchmark/benchmark.jl
#
# Time the solver at the current MPI rank count and write one JSON record per run to
# benchmark/results/. Grid is taken from the command line as `nx nz nt`, with nx a power of two.
#
# Usage, e.g.,:
#   julia --project=. benchmark/benchmark.jl 256 256 50
#   mpiexecjl --project=. -n 8 julia --project=. benchmark/benchmark.jl 512 512 50

using GeoFluidLBM
using MPI
using JSON

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    P    = MPI.Comm_size(comm)

    nruns = 3   # Repeat each run this many times and keep the fastest

    # Problem size from the command line (nx nz nt), default is (512, 512, 50)
    nx    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
    nz    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 512
    nt    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50

    cfg      = Config(nx = nx, nz = nz, nt = nt, rho_in = 1.01, rho_out = 0.99)
    solve(c) = P == 1 ? run_seq(c) : run_par(c, comm)

    # Warmup
    solve(Config(nx = 64, nz = 64, nt = 20, rho_in = 1.01, rho_out = 0.99))

    # Perform benchmark
    MPI.Barrier(comm)
    best = nothing
    for _ in 1:nruns
        timings, _ = solve(cfg)
        if best === nothing || timings["total"] < best["total"]
            best = timings
        end
        MPI.Barrier(comm)
    end

    # Save results
    if rank == 0
        outdir = joinpath(@__DIR__, "results")
        mkpath(outdir)
        file = joinpath(outdir, "benchmark_P$(P)_nx$(nx)_nz$(nz)_nt$(nt).json")
        open(file, "w") do io
            JSON.print(io, Dict("P" => P, "nx" => nx, "nz" => nz, "nt" => nt,
                                "nruns" => nruns, "timings" => best), 4)
        end
        println("benchmark  P=$P  $(nx)x$(nz)  nt=$nt")
        println("  total=$(round(best["total"]; digits = 4)) s  comm=$(round(best["comm"]; digits = 4)) s")
        println("  -> $file")
    end

end

main()
