# src/run_par.jl
#
# Implement the MPI-parallel solver `run_par`.

"""
    run_par(cfg::Config, comm) -> (timings::Dict, state::State)

MPI-parallel counterpart of [`run_seq`](@ref)

**You must implement this.**
"""
function run_par(cfg::Config, comm)
    error("run_par is not implemented yet.")
end
