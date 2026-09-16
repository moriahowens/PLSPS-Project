# src/GeoFluidLBM.jl
#
# D2Q9 BGK lattice Boltzmann core: Config, State, macroscopic field accessors, and
# the per-timestep physics kernels shared by run_seq and run_par.

module GeoFluidLBM

using MPI

export Config, State,
       run_seq, run_par,
       density, velocity, speed

### D2Q9 lattice definition
const c  = [( 0,  0), ( 1,  0), (-1,  0),
            ( 0,  1), ( 0, -1), ( 1,  1),
            (-1, -1), ( 1, -1), (-1,  1)]                 # Discrete velocity vectors
const ai = [1, 3, 2, 5, 4, 7, 6, 9, 8]                    # Bounce-back opposite indices
const w  = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]       # Lattice weights
const na = 9                                              # Number of lattice velocities
const D  = 2                                              # Spatial dimensions

"""
    upstream(i::Integer, di::Integer, n::Integer)

Upstream cell index for streaming (i - di), with periodic wrap on an axis of length n.
Wall bounce-back and the inlet/outlet boundary step handle non-periodic physics separately.
"""
@inline upstream(i::Integer, di::Integer, n::Integer) = mod1(i - di, n)

"""
    Config(; nx, nz, nu_f, dt, dx, rho_0, nt, rho_in, rho_out)

Simulation settings for a single run. Consists of the grid size (`nx` columns × `nz` rows), physics constants
(viscosity `nu_f`, timestep `dt`, cell size `dx`, reference density `rho_0`), the number of timesteps `nt`,
and scenario-specific constants (rho_in, rho_out).

The parallel solver may assume that `nx` is a multiple of the number of MPI ranks, so always use a
power of two for `nx` together with a power-of-two rank count.
"""
struct Config
    nx::Int
    nz::Int
    nu_f::Float64                    # kinematic viscosity (lattice units)
    dt::Float64
    dx::Float64
    rho_0::Float64                   # reference density
    nt::Int                          # number of timesteps

    # Poiseuille driving: inlet/outlet densities held on the left/right columns.
    rho_in::Float64                  # inlet density (left column)
    rho_out::Float64                 # outlet density (right column)
end

function Config(; nx = 128, nz = 128, nu_f = 0.1, dt = 1.0, dx = 1.0, rho_0 = 1.0,
                nt = 100, rho_in = 1.01, rho_out = 0.99)
    @assert nx > 1 && nz > 1              "nx, nz must be > 1"
    @assert nu_f > 0 && dt > 0 && dx > 0  "nu_f, dt, dx must be > 0"
    @assert nt >= 0                       "nt must be >= 0"
    @assert rho_0 > 0 && rho_in > 0 && rho_out > 0  "rho_0, rho_in, rho_out must be > 0"
    return Config(nx, nz, nu_f, dt, dx, rho_0, nt, rho_in, rho_out)
end

"""
    State(cfg::Config) -> State

Holds simulation data for a run: particle distributions, macroscopic fields (density `rho`, velocity `u`),
and solid wall layout. Initialised at rest with solid walls along the top and bottom rows.
Edit fields after creation if a custom initial condition is needed.
"""
struct State
    nx::Int
    nz::Int
    tau_f::Float64               # BGK relaxation time
    c1::Float64                  # Equilibrium Taylor-expansion coefficients
    c2::Float64
    c3::Float64
    c4::Float64
    f::Array{Float64, 3}         # Particle number densities per direction
    f_stream::Array{Float64, 3}  # Scratch array for streaming
    f_eq::Array{Float64, 3}      # Local equilibrium target
    Delta_f::Array{Float64, 3}   # Collision adjustment term
    rho::Matrix{Float64}         # Fluid density
    u::Array{Float64, 3}         # Fluid velocity
    Pi::Array{Float64, 3}        # Momentum density
    u2::Matrix{Float64}          # Velocity squared (scratch)
    cu::Matrix{Float64}          # Dot product c·u (scratch)
    solid::Matrix{Bool}          # Which cells are solid walls
    solid_src::Array{Bool, 3}    # Whether the upstream source cell is solid, per direction
end

function State(cfg::Config)
    nx, nz = cfg.nx, cfg.nz

    S = cfg.dx / cfg.dt

    # BGK coefficients
    tau_f = cfg.nu_f * 3.0 / (S * cfg.dt) + 0.5
    c1 = 1.0
    c2 = 3.0 / S^2
    c3 = 9.0 / (2.0 * S^4)
    c4 = -3.0 / (2.0 * S^2)

    # Particle distribution
    f        = zeros(Float64, na, nz, nx)
    f_stream = zeros(Float64, na, nz, nx)
    f_eq     = zeros(Float64, na, nz, nx)
    Delta_f  = zeros(Float64, na, nz, nx)

    # Macroscopic fields
    rho = fill(cfg.rho_0, nz, nx)
    u   = zeros(Float64, D, nz, nx)
    Pi  = zeros(Float64, D, nz, nx)
    u2  = zeros(Float64, nz, nx)
    cu  = zeros(Float64, nz, nx)

    # Solid walls on the top and bottom rows
    solid = zeros(Bool, nz, nx)
    solid[1, :]  .= true
    solid[nz, :] .= true

    # For each cell and direction, record if the upstream neighbour is a wall (for bounce-back)
    solid_src = zeros(Bool, na, nz, nx)
    for a in 1:na
        cx, cz = c[a][1], c[a][2]
        for x in 1:nx
            for z in 1:nz
                xa = upstream(x, cx, nx)
                za = upstream(z, cz, nz)
                solid_src[a, z, x] = solid[za, xa]
            end
        end
    end

    # Particle distributions for a fluid at rest
    for a in 1:na
        f[a, :, :] .= rho .* w[a]
    end

    return State(nx, nz, tau_f, c1, c2, c3, c4,
                 f, f_stream, f_eq, Delta_f,
                 rho, u, Pi, u2, cu,
                 solid, solid_src)
end

"""
    hold_density!(s::State, x, rho) -> State

Hold column `x` of the distributions at the rest equilibrium for density `rho`
(`f[:, :, x] .= rho .* w`). Shared by the sequential and parallel solvers.
"""
hold_density!(s::State, x::Integer, rho::Real) = (@views s.f[:, :, x] .= rho .* w; s)

### Macroscopic field accessors
"""
    density(s::State)

Density field accessor `(nz, nx)`
"""
density(s::State) = s.rho

"""
    velocity(s::State)

Velocity field accessor `(component, nz, nx)`; component 1 = x, 2 = z.
"""
velocity(s::State) = s.u

"""
    speed(s::State)

Speed magnitude `|u|` field accessor `(nz, nx)`.
"""
speed(s::State) = sqrt.(s.u[1, :, :].^2 .+ s.u[2, :, :].^2)

### Per time-step physics kernels
"""
    stream!(s::State) -> State

For each direction, move particle distributions to their neighbouring cells. If the upstream cell is a solid wall, particles bounce back.
The results are first stored in a temporary array (`f_stream`) to avoid overwriting needed data, then copied back into the main array (`f`).
"""
function stream!(s::State)
    nx, nz = s.nx, s.nz
    f, f_stream, solid_src = s.f, s.f_stream, s.solid_src

    for a in 1:na
        cx, cz = c[a][1], c[a][2]
        opp = ai[a]
        for x in 1:nx
            for z in 1:nz
                # Bounce-back
                if solid_src[a, z, x]
                    f_stream[a, z, x] = f[opp, z, x]

                # Stream
                else
                    xa = upstream(x, cx, nx)
                    za = upstream(z, cz, nz)
                    f_stream[a, z, x] = f[a, za, xa]
                end
            end
        end
    end
    f .= f_stream
    return s
end

"""
    macroscopic!(s::State) -> State

Update the macroscopic fields density `rho`, momentum density `Pi`, and velocity `u = Pi / rho`
by recomputing them from the current particle distributions `f`.
"""
function macroscopic!(s::State)
    f, rho, Pi, u = s.f, s.rho, s.Pi, s.u

    fill!(rho, 0.0)
    fill!(Pi, 0.0)

    for a in 1:na
        rho .+= f[a, :, :]                    # Density
        Pi[1, :, :] .+= f[a, :, :] .* c[a][1] # X momentum
        Pi[2, :, :] .+= f[a, :, :] .* c[a][2] # Z momentum
    end

    # Velocity (u = Pi / rho)
    u[1, :, :] .= Pi[1, :, :] ./ rho
    u[2, :, :] .= Pi[2, :, :] ./ rho
    return s
end

"""
    equilibrium_collide!(s::State) -> State

BGK collision step. From the current macroscopic fields (`rho`, `u`), build the local equilibrium `f_eq` (Taylor expansion in `c·u` and `|u|²`), then relax
`f` toward it with `f += (f_eq - f) / tau_f`.

Call after [`macroscopic!`](@ref). Updates `f`, `f_eq`, and `Delta_f` in place.
"""
function equilibrium_collide!(s::State)
    f, f_eq, Delta_f = s.f, s.f_eq, s.Delta_f
    rho, u, u2, cu = s.rho, s.u, s.u2, s.cu
    (; tau_f, c1, c2, c3, c4) = s

    u2 .= u[1, :, :].^2 .+ u[2, :, :].^2

    for a in 1:na
        cu .= c[a][1] .* u[1, :, :] .+ c[a][2] .* u[2, :, :]

        # Taylor expansion of the BGK equilibrium state
        f_eq[a, :, :] .= rho .* w[a] .* (c1 .+ c2 .* cu .+ c3 .* cu.^2 .+ c4 .* u2)

        # Collision term. Relax the current state toward equilibrium
        Delta_f[a, :, :] .= (f_eq[a, :, :] .- f[a, :, :]) ./ tau_f

        # Apply the collision change to the particle distributions
        f[a, :, :] .+= Delta_f[a, :, :]
    end
    return s
end

include("run_seq.jl")
include("run_par.jl")

end # module
