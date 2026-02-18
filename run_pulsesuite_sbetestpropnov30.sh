#!/bin/bash
#PBS -N pulsesuite_nov30
#PBS -A cesm0029
#PBS -l select=1:ncpus=16:mem=32GB
#PBS -l walltime=02:00:00
#PBS -q main
#PBS -j oe

cd $PBS_O_WORKDIR/src/pulsesuite/PSTD3D
export PATH="$HOME/.local/bin:$PATH"

# Match Numba thread count to PBS allocation
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

uv run python -m pulsesuite.PSTD3D.sbetestpropnov30
