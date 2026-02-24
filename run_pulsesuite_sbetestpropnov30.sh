#!/bin/bash
#PBS -N pulsesuite_nov30
#PBS -A cesm0029
#PBS -l select=1:ncpus=64:mpiprocs=1:mem=235G:ngpus=1
#PBS -l walltime=08:00:00
#PBS -q main
#PBS -j oe

cd $PBS_O_WORKDIR/src/pulsesuite/PSTD3D
export PATH="$HOME/.local/bin:$PATH"

# Match Numba thread count to PBS allocation
export NUMBA_NUM_THREADS=48
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48

# CUDA toggle: set to 1 to use GPU, 0 to force CPU (auto-detect if unset)
export PULSESUITE_USE_CUDA=1

uv run python -u -m pulsesuite.PSTD3D.sbetestpropnov30
