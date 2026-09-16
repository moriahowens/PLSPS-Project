# test/test_par.jl
#
# Correctness checks, run under MPI:
#   1. run_seq reproduces the analytical Poiseuille parabola.
#   2. run_par reproduces the analytical Poiseuille parabola.
#   3. run_par matches run_seq to round-off.
#
# Usage:
#   mpiexecjl --project=. -n 2 julia --project=. test/test_par.jl
#   mpiexecjl --project=. -n 4 julia --project=. test/test_par.jl

using GeoFluidLBM
using MPI

rel_l2(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

# Relative L2 distance between the mid-channel velocity profile and the analytical Poiseuille parabola
function parabola_error(s, cfg)
    ux = velocity(s)[1, :, div(cfg.nx, 2)]
    nz = cfg.nz
    parab = [(z - 1.5) * ((nz - 0.5) - z) for z in 1:nz]
    parab .*= maximum(ux) / maximum(parab)
    fluid = 2:(nz - 1)
    return rel_l2(ux[fluid], parab[fluid])
end

function main()
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    P    = MPI.Comm_size(comm)

    # Poiseuille flow, run long enough to reach the steady-state parabola.
    # nx is a power of two, so it divides evenly over any power-of-two rank count.
    cfg = Config(nx = 64, nz = 32, nu_f = 0.1, nt = 8000, rho_in = 1.01, rho_out = 0.99)

    _, s_par = run_par(cfg, comm)   # all ranks; rank 0 receives the gathered full state

    if rank == 0
        _, s_seq = run_seq(cfg)

        seq_parab = parabola_error(s_seq, cfg)
        par_parab = parabola_error(s_par, cfg)
        diff_u    = rel_l2(velocity(s_par), velocity(s_seq))
        diff_rho  = rel_l2(density(s_par),  density(s_seq))

        parab_tol = 0.08    # profile must match the analytical parabola
        match_tol = 1e-10   # run_par must reproduce run_seq to round-off

        println("test_par  P=$P  $(cfg.nx)x$(cfg.nz)  nt=$(cfg.nt)")
        println("  run_seq vs parabola : $(round(seq_parab; digits = 4))   (tol $parab_tol)")
        println("  run_par vs parabola : $(round(par_parab; digits = 4))   (tol $parab_tol)")
        println("  run_par vs run_seq  : velocity=$diff_u  density=$diff_rho   (tol $match_tol)")

        if seq_parab < parab_tol && par_parab < parab_tol && diff_u < match_tol && diff_rho < match_tol
            println("PASS")
        else
            println("FAIL")
        end
    end
end

main()
