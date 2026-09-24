# src/run_par.jl
#
# Implement the MPI-parallel solver `run_par`.

"""
    run_par(cfg::Config, comm) -> (timings::Dict, state::State)

MPI-parallel counterpart of [`run_seq`](@ref)

**You must implement this.**
"""
function run_par(cfg::Config, comm)

    # get basic info
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)
    # println("rank $rank of $P")

    # split the columns evenly between the ranks
    nx_local = cfg.nx ÷ P # nx local stores how many columns for each rank
    # println("nx_local is $nx_local")
    first_column = (rank * nx_local) + 1 # first column for this rank
    last_column = (first_column + (nx_local-1)) # last column for this rank
    before_first_column = 1 # help manage buffer spaces in the local array for each rank
    after_last_column = nx_local + 2 # help manage buffer spaces in the local array for each rank
    # println("my first column is $first_column and my last column is $last_column")

    # configure the local state according to how it looks in GeoFluidLBM
    cfg_local = Config(nx_local + 2, cfg.nz, cfg.nu_f, cfg.dt, cfg.dx, cfg.rho_0, cfg.nt, cfg.rho_in, cfg.rho_out)
    state = State(cfg_local)
    m = size(state.f)
    # println("(Directions, Grid height (nz), local columns (nx_local+2)): $m")

    # because cells get their values from their neighbors, get the left and right surrounding processes
    left = mod(rank-1, P) # correlates with before_first_column
    right = mod(rank+1, P) # correlates with after_last_column
    #println("my rank is $rank, so to my left is $left and to my right is $right")
    # this also makes a loop (feature of mod)

    #pre-allocate 2 request and 2 send objects
    reqs = [MPI.Request() for _ in 1:4]

    # for each process: 
    # my first column goes to my left neighbor's after_last_column
    # my last column goes to my right neighbor's before_first_column
    # this happens in the separated function exchange_halos

    # local to process:
    first_col = 2
    last_col = nx_local + 1

    #Delete later -- test block for exchange halos
    #state.f .= 0.0
    #first_col = 2
    #last_col = nx_local + 1
    #before_first = 1
    #after_last = last_col + 1
    #state.f[:, :, first_col:last_col] .= rank 
    #exchange_halos!(state, comm, reqs, left, right)
    #println("rank $rank: left ghost = $(state.f[1, 1, before_first]), right ghost = $(state.f[1, 1, after_last])")

    # inlet/outlet of fluid in the simulator
    has_inlet  = (rank == 0) # only true for process 0
    has_outlet = (rank == P - 1) # only true for the last process

    # replicate process from run_seq
    t_boundary            = 0.0
    t_comm                = 0.0 # added
    t_stream              = 0.0
    t_macroscopic         = 0.0
    t_equilibrium_collide = 0.0

    total = @elapsed for step in 1:cfg.nt
        t_boundary            += @elapsed begin
            if has_inlet
                hold_density!(state, first_col, cfg.rho_in)
            end
            if has_outlet
                hold_density!(state, last_col, cfg.rho_out)
            end
        end
        t_comm += @elapsed exchange_halos!(state, comm, reqs, left, right) # added
        t_stream              += @elapsed stream!(state)
        t_macroscopic         += @elapsed macroscopic!(state)
        t_equilibrium_collide += @elapsed equilibrium_collide!(state)
    end

    # build receiving space for state 0
    if rank == 0
        full_state = State(cfg) # buffer to receive everything
    else
        full_state = nothing # not needed because they are only sending to 0
    end

    # each state prepares to send their rho and u 
    send_rho = @view state.rho[:, first_col:last_col]
    send_u   = @view state.u[:, :, first_col:last_col]

    # prepare to receive rho
    if rank == 0 # only receive rho if rank = 0
        rec_rho = full_state.rho
    else
        rec_rho = nothing
    end

    # everyone send/receive rho (as appropriate)
    MPI.Gather!(send_rho, rec_rho, comm; root=0)

    # same as above, with u
    if rank == 0
        rec_u = full_state.u
    else
        rec_u = nothing 
    end

    # everyone send/receive u (as appropriate)
    MPI.Gather!(send_u, rec_u, comm; root=0)

    # define the Dictionary to return 
    timings = Dict{String, Float64}(
        "total" => total, "boundary" => t_boundary, "stream" => t_stream, "macroscopic" => t_macroscopic, 
        "equilibrium_collide" => t_equilibrium_collide, "compute" => t_boundary + t_stream + t_macroscopic + t_equilibrium_collide,
        "comm" => t_comm,
    )

    # return 
    if rank == 0
        return timings, full_state # dictionary and entire result
    else 
        return timings, state # dictionary and state result
    end 
end

function exchange_halos!(state, comm, reqs, left, right)

    nx_local = state.nx - 2

    # column indices (static per process so these can be hard-coded)
    first_col = 2
    last_col = nx_local + 1
    before_first = 1
    after_last = last_col + 1
    #println("nx_local is $nx_local, so my first is $first_col, my last is $last_col, and my after_last is $after_last")

    # create buffers 
    # using view so that it actually edits rather than making a copy
    rec_left  = @view state.f[:, :, before_first] # to receive into your leftmost spot (before first_col)
    rec_right = @view state.f[:, :, after_last] # to receive into your rightmost spot (after last_col)
    send_left  = @view state.f[:, :, first_col] # to send to the process on your left
    send_right = @view state.f[:, :, last_col] # to send to the process on your right

    # use nonblocking I-send/receive to avoid deadlock
    # to specify what messages are being sent:
    # messages sent left have tag 0 and messages sent right have tag 1

    leftrec = MPI.Irecv!(rec_left, comm, reqs[1]; source=left, tag=1) # needs ! since it modifies
    rightrec = MPI.Irecv!(rec_right, comm, reqs[2]; source=right, tag=0) # needs ! since it modifies
    leftsend = MPI.Isend(send_left, comm, reqs[3]; dest=left, tag=0)
    rightsend = MPI.Isend(send_right, comm, reqs[4]; dest=right, tag=1)

    stats = MPI.Waitall(reqs)

end
