# render/render.jl
#
# Record an mp4 heatmap video of the flow speed |u| using the sequential solver.
#
# Usage:
#   julia --project=render -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'   # first time only (CairoMakie precompiles slowly)
#   julia --project=render render/render.jl                                             # -> render/output/poiseuille.mp4

using GeoFluidLBM, CairoMakie

function main()
    # A long, shallow channel (nx ≫ nz) so the flow reads as a "pipe".
    cfg = Config(nx = 256, nz = 64, nt = 3000, rho_in = 1.01, rho_out = 0.99)

    println("render  $(cfg.nx)x$(cfg.nz)  nt=$(cfg.nt)")

    # Run the sequential solver and grab the speed field every few steps.
    frames = Matrix{Float64}[]
    run_seq(cfg; callback = (s, t) -> t % 15 == 0 && push!(frames, speed(s)))

    # Animate the frames to an mp4 (colour scaled to the fastest flow reached).
    cmax = maximum(maximum, frames)
    data = Observable(permutedims(frames[1]))
    fig  = Figure(size = (900, 280))
    ax   = Axis(fig[1, 1]; xlabel = "x", ylabel = "z", aspect = DataAspect())
    heatmap!(ax, data; colormap = :turbo, colorrange = (0.0, cmax))

    outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
    path = joinpath(outdir, "poiseuille.mp4")
    record(fig, path, eachindex(frames); framerate = 30) do i
        data[] = permutedims(frames[i])
    end
    println("  -> $path")
end

main()
