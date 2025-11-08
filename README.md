# PulseSuite

PulseSuite is a high-performance computational physics toolkit for simulating ultrafast laser-matter interactions in semiconductor quantum structures. It implements the Semiconductor Bloch Equations (SBEs) coupled with Pseudo-Spectral Time Domain (PSTD) electromagnetic field propagation methods to model quantum wire and quantum well systems under intense optical excitation.

## Core Physics

This codebase enables simulation of coherent light-matter interactions in low-dimensional semiconductors, including:

- **Quantum Well/Wire Optics**: Propagation-to-QW field transformations, polarization calculations, and density matrix evolution
- **Coulomb Interactions**: Many-body electron-hole Coulomb effects with screening
- **Phonon Scattering**: Longitudinal optical phonon interactions with carriers
- **DC Field Transport**: Electric field-induced carrier drift and tunneling
- **Dephasing**: Diagonal and off-diagonal dephasing mechanisms
- **Spontaneous Emission**: Radiative recombination and photoluminescence

The code is a complete Python port of production Fortran simulation tools, optimized for HPC environments using NumPy, accelerated FFTs (scipy/pyFFTW), and optional Numba JIT compilation.

## Quick Start

### Basic Quantum Well Optics Simulation

```python
import numpy as np
import sys
sys.path.append('PSTD3D/src')

from qwopticspythonic import InitializeQWOptics, Prop2QW, QWPolarization3
from constants import eV, hbar

# Define quantum wire parameters
Nr = 64                    # Spatial grid points
Nk = 32                    # Momentum grid points
L = 100e-9                 # Wire length (m)
area = 1e-16               # Cross-sectional area (m^2)
gap = 1.5 * eV             # Band gap energy
dcv = 1.0 + 0.5j           # Dipole matrix element

# Create coordinate grids
RR = np.linspace(-500e-9, 500e-9, Nr)  # Propagation space
R = np.linspace(-L/2, L/2, Nr)         # QW space
kr = np.linspace(-1e7, 1e7, Nk)        # Momentum space
Qr = np.linspace(-2e7, 2e7, Nr)        # QW momentum space

# Energy bands (parabolic approximation)
me = 0.07 * 9.1e-31        # Electron effective mass
mh = 0.45 * 9.1e-31        # Hole effective mass
Ee = (hbar * kr)**2 / (2 * me)
Eh = (hbar * kr)**2 / (2 * mh)

# Initialize QW optics module
InitializeQWOptics(RR, L, dcv, kr, Qr, Ee, Eh, ehint=1.0, area=area, gap=gap)

# Convert propagation fields to QW representation
Exx = np.sin(2*np.pi*RR/L) * np.exp(1j*2*np.pi*RR/L)  # Input field
Edc = np.zeros(1)
Ex = np.zeros(Nr, dtype=np.complex128)
Prop2QW(RR, Exx, np.zeros_like(Exx), np.zeros_like(Exx), np.zeros_like(Exx),
        Edc, R, Ex, np.zeros_like(Ex), np.zeros_like(Ex), np.zeros_like(Ex), 0.0, 0)

print(f"Average DC field: {Edc[0]:.3e} V/m")
print(f"QW field amplitude: {np.max(np.abs(Ex)):.3e}")
```

### Semiconductor Bloch Equations Time Evolution

```python
from SBEspythonic import InitializeSBE, dpdt, dCdt, dDdt
import numpy as np

# Initialize SBE solver
Nk = 32
Nr = 64
q = np.linspace(-1e7, 1e7, Nr)
rr = np.linspace(-200e-9, 200e-9, Nr)

InitializeSBE(q, rr, Edc=0.0, area=1e-16, wavelength=800e-9, Nwire=1, optics=True)

# Initialize density matrices
P = np.zeros((Nk, Nk), dtype=np.complex128)      # Coherence (polarization)
C = np.eye(Nk, dtype=np.complex128) * 1e-6       # Electron occupation
D = np.eye(Nk, dtype=np.complex128) * 1e-6       # Hole occupation

# Hamiltonian matrices (from field coupling and many-body effects)
Heh = np.zeros((Nk, Nk), dtype=np.complex128)    # Electron-hole
Hee = np.zeros((Nk, Nk), dtype=np.complex128)    # Electron-electron
Hhh = np.zeros((Nk, Nk), dtype=np.complex128)    # Hole-hole

# Dephasing rates
GamE = np.ones(Nk) * 1e12                        # Electron (Hz)
GamH = np.ones(Nk) * 1e12                        # Hole (Hz)
OffP = np.zeros((Nk, Nk), dtype=np.complex128)   # Off-diagonal

# Calculate time derivatives
dP = dpdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)
dC = dCdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)
dD = dDdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)

print(f"Polarization evolution rate: {np.max(np.abs(dP)):.3e}")
print(f"Electron density change: {np.max(np.abs(dC)):.3e}")
```

### Accelerated FFT Operations

```python
from fftw import fft_3D, ifft_3D, Transform, HankelTransform, CreateHT
import numpy as np

# Standard 3D FFT (uses pyFFTW if available, scipy.fft otherwise)
field = np.random.randn(64, 64, 256) + 1j*np.random.randn(64, 64, 256)
field = field.astype(np.complex128, order='F')  # Fortran contiguous for speed

fft_3D(field)    # Forward transform (in-place)
ifft_3D(field)   # Inverse transform (in-place)

# Cylindrical symmetry: Hankel transform + FFT
Nr = 64
CreateHT(Nr)  # Precompute Hankel matrix
radial_field = np.random.randn(1, Nr, 256) + 1j*np.random.randn(1, Nr, 256)
radial_field = radial_field.astype(np.complex128, order='F')

Transform(radial_field)   # Hankel(r) + FFT(t)
```

## Installation

```bash
pip install -r requirements.txt
```

## Module Structure

### Core Modules (`src/`)
- **`fftw.py`**: Accelerated FFT library with 1D/2D/3D transforms, centered FFTs, and Hankel transforms for cylindrical symmetry
- **`constants.py`**: Physical constants (e, ħ, c, ε₀, μ₀) and mathematical constants
- **`rungekutta.py`**: Adaptive Runge-Kutta integrators for SBE time evolution
- **`type_*.py`**: Type definitions for plasma, lens, medium, and two-photon absorption media

### PSTD3D Package (`PSTD3D/src/`)

**Quantum Optics**
- **`qwopticspythonic.py`**: Quantum well optics core - field transformations (Prop2QW, QW2Prop), polarization calculation, density matrices

**Many-Body Physics**
- **`coulombpythonic.py`**: Coulomb interaction matrices with screening, exchange, and correlation effects
- **`phononspythonic.py`**: Longitudinal optical phonon scattering rates and matrix elements
- **`dcfieldpythonic.py`**: DC electric field effects, carrier drift, and field-induced tunneling
- **`dephasingpythonic.py`**: T₁ and T₂ dephasing, diagonal and off-diagonal relaxation
- **`emissionpythonic.py`**: Spontaneous emission, photoluminescence spectra

**Semiconductor Bloch Equations**
- **`SBEspythonic.py`**: Complete SBE solver with density matrix evolution (dpdt, dCdt, dDdt)
- **`SBETestpythonic.py`**: Production test harness for SBE simulations

**Utilities**
- **`usefulsubspythonic.py`**: FFT wrappers, array printing, I/O utilities
- **`helperspythonic.py`**: Spatial/momentum grids, array locators, magnitude calculations
- **`splinerpythonic.py`**: Cubic spline interpolation for density/field interpolation
- **`epsrtlpythonic.py`**: Dielectric function calculations with Lorentz oscillators

### Parameter Files (`PSTD3D/params/`)
- **`qw.params`**: Quantum wire/well parameters (length, band gap, effective masses, dephasing rates)
- **`mb.params`**: Many-body interaction parameters (Coulomb, phonon, screening)
- **`pulse.params`**: Laser pulse parameters (wavelength, duration, intensity, chirp)
- **`pstd_abc.params`**: PSTD grid and absorbing boundary conditions

## Performance Optimization

### Threading

The code uses multithreaded FFT and linear algebra operations. Thread counts are controlled via environment variables that must be set **before importing** the modules:

```bash
# Set thread counts for parallel FFT and linear algebra
export OMP_NUM_THREADS=8       # For pyFFTW, OpenBLAS, and FFTW
export MKL_NUM_THREADS=8        # For Intel MKL (if using MKL-backed NumPy/SciPy)
export NUMBA_NUM_THREADS=8      # For Numba JIT compilation (if used)
```

**In Python scripts:**
```python
import os
# Must set BEFORE importing fftw
os.environ['OMP_NUM_THREADS'] = '8'

from fftw import fft_3D, _FFTW_THREADS
print(f"Using {_FFTW_THREADS} threads")  # Should show 8
```

**Test threading performance:**
```bash
python test_threading.py  # Benchmarks different thread counts
```

**Best practices:**
- Start with `OMP_NUM_THREADS = number_of_physical_cores`
- Don't set higher than physical core count (causes slowdown)
- For HPC nodes with 32+ cores, 16 threads often gives best efficiency
- Hyperthreading (2x threads) usually doesn't help FFT performance

### Array Layout
All arrays are Fortran-contiguous (`order='F'`) for cache-friendly access patterns matching the original Fortran code:

```python
# Efficient allocation
field = np.zeros((Nx, Ny, Nt), dtype=np.complex128, order='F')
```

### FFT Acceleration
The code automatically selects the fastest FFT backend available:
1. **pyFFTW** (fastest, multithreaded FFTW)
2. **scipy.fft** (MKL or FFTW backend)
3. **numpy.fft** (fallback)

## Typical Workflow

1. **Set parameters**: Edit `PSTD3D/params/*.params` files for your system
2. **Initialize modules**: Call `Initialize*()` functions for required physics
3. **Define grids**: Set up spatial, momentum, and time grids
4. **Time evolution**: Use Runge-Kutta integrators to evolve SBEs
5. **Extract observables**: Calculate absorption, emission, polarization, carrier densities
6. **Visualize results**: Plot fields, spectra, density matrices

Example parameter sweep:

```python
import numpy as np
from SBEspythonic import InitializeSBE, dpdt

# Sweep over pulse intensities
intensities = np.logspace(10, 14, 20)  # W/m^2
absorption = []

for I0 in intensities:
    # Initialize with new intensity
    E0 = np.sqrt(2 * I0 / (3e8 * 8.854e-12))
    # ... run simulation ...
    # ... extract absorption spectrum ...
    absorption.append(integrated_absorption)

# Plot intensity-dependent absorption saturation
plt.semilogx(intensities, absorption)
plt.xlabel('Intensity (W/m²)')
plt.ylabel('Absorption')
```

## Troubleshooting

**Common issues:**

**Slow FFT performance**
- Ensure pyFFTW is installed: `python -c "import pyfftw; print(pyfftw.__version__)"`
- Use Fortran-contiguous arrays: `np.asfortranarray(arr)`
- Set thread counts: `export OMP_NUM_THREADS=8`

**I/O could be optimized to VELOC-style checkpointing**
- Current implementation uses direct file writes
- Consider implementing asynchronous I/O for large-scale HPC runs
- Use HDF5 or similar for structured output in production runs

**AI review**
- Used AI to write the code with human in the loop, some values might be wrong.
- Constantly reviewing the code and making sure it is correct.

## Physical Units

All quantities in SI units unless specified:
- Length: meters (m)
- Energy: Joules (J) or electron-volts (eV)
- Time: seconds (s)
- Frequency: Hertz (Hz)
- Electric field: V/m
- Momentum: kg⋅m/s or m⁻¹ (wavevector)

Use `constants.py` for unit conversions:
```python
from PSTD3D.src.constants import eV, hbar, c0, me0

gap_eV = 1.5              # Band gap in eV
gap_J = gap_eV * eV       # Convert to Joules
omega = gap_J / hbar      # Corresponding frequency (rad/s)
wavelength = 2*np.pi*c0 / omega  # Wavelength (m)
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-physics-module`
3. Add tests for new functionality in `PSTD3D/tests/`
4. Follow existing code style (NumPy docstrings, type hints)
5. Ensure all tests pass: `pytest PSTD3D/tests/ -v`
6. Submit pull request with clear description

## Contact

For questions, bug reports, or collaboration inquiries:
- Email: f1rahulranjan@gmail.com


