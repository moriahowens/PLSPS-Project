#!/bin/bash
#
# setup.sh — DAS-5 environment setup: modules + instantiate + bind MPI + mpiexecjl.
#
# Usage:
#   source setup.sh

# Always go to repo root first, safe even if you source from elsewhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Repo absolute path e.g. /home/abc123/PLSPS2627
cd "$ROOT"

# Load modules on DAS-5
module load slurm
module load gcc
module load openmpi/gcc/64
module load julia/1.11.4
module load prun

mkdir -p "$ROOT/bin"

# Instantiate Julia project
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Bind MPI.jl to the DAS-5 cluster MPI
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary(force=true)'

# Builds `mpiexecjl` launcher into `bin/` and adds it to `PATH`
julia --project=. -e 'using MPI; MPI.install_mpiexecjl(destdir="bin", force=true)'
export PATH="$ROOT/bin:$PATH"

# Precompile the package with the same flags the benchmark job uses
julia --project=. -O3 --check-bounds=no -e 'using Pkg; Pkg.precompile()'

echo
echo "Setup Complete"
