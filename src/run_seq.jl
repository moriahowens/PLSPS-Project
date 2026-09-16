# src/run_seq.jl
#
# The sequential solver run_seq.

"""
    run_seq(cfg::Config; callback = nothing) -> (timings::Dict, state::State)

Sequential solver: run `cfg.nt` timesteps and return the per-kernel `timings` and the final
`state`. Read results with [`density`](@ref), [`velocity`](@ref), [`speed`](@ref); pass an
optional `callback(state, step)` to observe each step (called at `step = 0` and after every
step).

The boundary step drives channel (Poiseuille) flow along x at the start of each timestep:
hold the leftmost column at density `rho_in` and the rightmost at `rho_out`. That density
difference pushes fluid through the channel.
"""
function run_seq(cfg::Config; callback = nothing)
    s = State(cfg)
    callback === nothing || callback(s, 0)

    t_boundary            = 0.0
    t_stream              = 0.0
    t_macroscopic         = 0.0
    t_equilibrium_collide = 0.0

    total = @elapsed for step in 1:cfg.nt
        t_boundary            += @elapsed begin
            hold_density!(s, 1, cfg.rho_in)
            hold_density!(s, s.nx, cfg.rho_out)
        end
        t_stream              += @elapsed stream!(s)
        t_macroscopic         += @elapsed macroscopic!(s)
        t_equilibrium_collide += @elapsed equilibrium_collide!(s)
        callback === nothing || callback(s, step)
    end

    timings = Dict{String, Float64}(
        "total"               => total,
        "boundary"            => t_boundary,
        "stream"              => t_stream,
        "macroscopic"         => t_macroscopic,
        "equilibrium_collide" => t_equilibrium_collide,
        "compute"             => t_boundary + t_stream + t_macroscopic + t_equilibrium_collide,
        "comm"                => 0.0,
    )
    return timings, s
end
