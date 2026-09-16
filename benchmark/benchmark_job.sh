#!/bin/bash
#
# benchmark/benchmark_job.sh — DAS-5 strong-scaling sweep: each grid at P = 1, 2, 4, 8, 16.
#
# Usage:
#   bash benchmark/benchmark_job.sh    # run directly on the head node, not with sbatch

cd "$(dirname "$0")/.."     # repo root, so benchmark/ and --project=. resolve
export PATH="$(pwd)/bin:$PATH"   # mpiexecjl from setup.sh; prun script expects it on PATH

# Each grid ("nx nz nt") is kept fixed while P grows (strong scaling).
GRIDS=(
    "256 256 50"
    "512 512 50"
    "1024 1024 50"
)

# Node/core layout per total rank count P (= nodes × cores-per-node); fills a node before adding nodes.
declare -A NODES=([1]=1 [2]=1 [4]=2 [8]=4 [16]=8)
declare -A CORES=([1]=1 [2]=2 [4]=2 [8]=2 [16]=2)

# The compute nodes on DAS-5 have 2 sockets (aka NUMA domains).
# We use at most 2 MPI ranks per node, one rank per socket.
# In this way, MPI ranks do not compete for resources within the socket.
# The OMPI_OPTS are used to place one MPI rank at each socket.
for GRID in "${GRIDS[@]}"; do
    echo "########################  grid: $GRID  ########################"
    for P in 1 2 4 8 16; do
        echo "================  P=$P  ================"
        prun OMPI_OPTS="--map-by numa --bind-to numa" -np "${NODES[$P]}" -"${CORES[$P]}" \
            -script "$PRUN_ETC/prun-openmpi-mpiexecjl" \
            julia --project=. -O3 --check-bounds=no benchmark/benchmark.jl $GRID
    done
done
