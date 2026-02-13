---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Quickstart: Your First SBE Simulation

:::{abstract}
Run a semiconductor Bloch equation (SBE) simulation of an ultrashort laser pulse
interacting with a GaAs quantum wire array. This example mirrors the working
`sbetestprop.py` test script and produces real physics output in under 3 minutes.
:::

## What you will do

1. Set up a 1D spatial grid along a quantum wire
2. Define an 800 nm Gaussian laser pulse (10 fs pulsewidth)
3. Initialize the full SBE solver (Coulomb, phonons, dephasing, transport, emission)
4. Time-evolve for 3000 steps (30 fs) and watch the polarization response
5. Plot the electric field and induced polarization

## Prerequisites

This example requires the parameter files `params/qw.params` (material properties)
and `params/mb.params` (many-body physics flags) to exist in the working directory.
These ship with the examples and configure a GaAs quantum wire in an AlAs host
with all major physics enabled.

See {doc}`02_architecture` for a detailed description of every parameter.

---

## 1. Setup

```{code-cell} python
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c0_SI

from pulsesuite.PSTD3D.SBEs import InitializeSBE, QWCalculator
from pulsesuite.PSTD3D.typespace import GetKArray, GetSpaceArray

# Ensure output directories exist (InitializeSBE writes diagnostic files here)
for d in ['dataQW/Wire/C', 'dataQW/Wire/D', 'dataQW/Wire/Ee',
          'dataQW/Wire/Eh', 'dataQW/Wire/P', 'dataQW/Wire/Win',
          'dataQW/Wire/Wout', 'dataQW/Wire/Xqw', 'dataQW/Wire/info',
          'dataQW/Wire/ne', 'dataQW/Wire/nh', 'output']:
    os.makedirs(d, exist_ok=True)

print("Setup complete.")
```

## 2. Spatial Grid

The quantum wire lives on a 1D real-space grid. We use 50 pixels at 10 nm spacing
(500 nm total length). `GetSpaceArray` and `GetKArray` create the matched
real-space and momentum-space grids that the SBE solver needs.

```{code-cell} python
Nr = 50             # Number of grid points along the wire
drr = 10e-9         # Pixel size: 10 nm

rr = GetSpaceArray(Nr, (Nr - 1) * drr)   # Real-space grid (m)
qrr = GetKArray(Nr, Nr * drr)            # Momentum-space grid (rad/m)

print(f"Grid: {Nr} points, {drr*1e9:.0f} nm spacing")
print(f"Wire length: {(Nr-1)*drr*1e9:.0f} nm")
print(f"Momentum range: [{qrr.min():.2e}, {qrr.max():.2e}] rad/m")
```

## 3. Laser Pulse Parameters

We drive the quantum wire with a linearly-polarized Gaussian pulse at 800 nm
(near the GaAs band gap at ~1.5 eV). The pulse has a 10 fs full-width and
peaks at 15 fs into the simulation.

The pulse shape includes a super-Gaussian envelope ($e^{-x^{20}}$) that gives
a sharp turn-on/turn-off, preventing long tails from wasting computation.

```{code-cell} python
# Physical constants
c0 = c0_SI
twopi = 2.0 * np.pi

# Pulse parameters
E0x = 1e7           # Peak E-field: 10 MV/m (V/m)
lam = 800e-9        # Wavelength: 800 nm
tw = 10e-15         # Pulsewidth: 10 fs (FWHM of Gaussian envelope)
tp = 15e-15         # Pulse peak time: 15 fs

# Derived quantities
w0 = twopi * c0 / lam   # Angular frequency (rad/s)
Tc = lam / c0            # Optical cycle period (s)

# Time stepping
Nt = 3000           # Number of time steps
dt = 10e-18         # Time step: 10 attoseconds (s)
T_total = Nt * dt   # Total simulation time

print(f"Wavelength: {lam*1e9:.0f} nm")
print(f"Photon energy: {twopi * c0 / lam * 1.055e-34 / 1.6e-19:.3f} eV")
print(f"Optical cycle: {Tc*1e15:.2f} fs")
print(f"Pulse peak: {tp*1e15:.0f} fs, width: {tw*1e15:.0f} fs")
print(f"Simulation: {Nt} steps x {dt*1e18:.0f} as = {T_total*1e15:.0f} fs")
```

## 4. Initialize the SBE Solver

`InitializeSBE` is the central entry point. It:
- Reads material parameters from `params/qw.params` (band gap, masses, dephasing rates)
- Reads physics flags from `params/mb.params` (Coulomb, phonons, transport, etc.)
- Computes the momentum grid and energy bands
- Initializes all subsystem modules (Coulomb matrices, phonon interactions, etc.)
- Sets up initial Fermi-Dirac carrier distributions at 77 K

We configure 2 quantum wire subbands (`Nw=2`).

```{code-cell} python
Emax0 = E0x  # Peak field magnitude for the wireoff guard

# Initialize: this reads params files and sets up everything
InitializeSBE(qrr, rr, 0.0, Emax0, lam, 2, True)

print("SBE solver initialized with all many-body physics.")
```

## 5. Time Evolution

Now we run the simulation. At each time step:

1. Compute the laser E-field at the current time
2. Call `QWCalculator` for each subband -- this solves the SBEs for one time step
3. Average the polarization from both subbands
4. Record the field and polarization at the wire midpoint

```{code-cell} python
# Allocate field and polarization arrays
Exx = np.zeros(Nr, dtype=np.complex128)   # E-field along wire (V/m)
Eyy = np.zeros(Nr, dtype=np.complex128)
Ezz = np.zeros(Nr, dtype=np.complex128)
Vrr_pot = np.zeros(Nr, dtype=np.complex128)  # Potential (V)

# Subband 1 polarizations
Pxx1 = np.zeros(Nr, dtype=np.complex128)
Pyy1 = np.zeros(Nr, dtype=np.complex128)
Pzz1 = np.zeros(Nr, dtype=np.complex128)

# Subband 2 polarizations
Pxx2 = np.zeros(Nr, dtype=np.complex128)
Pyy2 = np.zeros(Nr, dtype=np.complex128)
Pzz2 = np.zeros(Nr, dtype=np.complex128)

Rho = np.zeros(Nr, dtype=np.complex128)   # Charge density

# Boolean flags for QWCalculator (lists for in-place modification)
boolT = [True]
boolF = [False]

# Storage for time-series at wire midpoint
mid = Nr // 2
t_arr = np.zeros(Nt)
Ex_arr = np.zeros(Nt)
Px_arr = np.zeros(Nt)

# Time evolution loop
t = 0.0
for n in range(Nt):
    # Gaussian pulse with super-Gaussian cutoff
    phase = w0 * (t - tp)
    Exx[:] = (E0x
              * np.exp(-(phase**2) / (w0 * tw)**2)
              * np.cos(phase)
              * np.exp(-(phase**20) / (2 * tw * w0)**20))

    # Evolve SBEs for subband 1
    QWCalculator(Exx, Eyy, Ezz, Vrr_pot, rr, qrr, dt, 1,
                 Pxx1, Pyy1, Pzz1, Rho, boolT, boolF)
    # Evolve SBEs for subband 2
    QWCalculator(Exx, Eyy, Ezz, Vrr_pot, rr, qrr, dt, 2,
                 Pxx2, Pyy2, Pzz2, Rho, boolT, boolF)

    # Record midpoint values
    t_arr[n] = t
    Ex_arr[n] = np.real(Exx[mid])
    Px_arr[n] = np.real((Pxx1[mid] + Pxx2[mid]) / 2)

    t += dt

    # Progress indicator
    if (n + 1) % 1000 == 0:
        print(f"  Step {n+1}/{Nt} ({(n+1)*dt*1e15:.0f} fs)")

print(f"Done. Simulated {T_total*1e15:.0f} fs of quantum wire dynamics.")
```

## 6. Results

The top panel shows the driving laser E-field -- a few-cycle pulse at 800 nm.
The bottom panel shows the polarization induced in the quantum wire. The
polarization responds to the field but with a phase lag and amplitude set by
the quantum mechanical response of the electron-hole system.

```{code-cell} python
t_fs = t_arr * 1e15  # Convert to femtoseconds

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# E-field
ax1.plot(t_fs, Ex_arr * 1e-6, 'b-', linewidth=0.8)
ax1.set_ylabel('E-field (MV/m)')
ax1.set_title('Laser Pulse Driving a GaAs Quantum Wire')
ax1.axhline(0, color='gray', linewidth=0.5)

# Polarization
ax2.plot(t_fs, Px_arr, 'r-', linewidth=0.8)
ax2.set_ylabel('Polarization (C/m$^2$)')
ax2.set_xlabel('Time (fs)')
ax2.axhline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.show()

print(f"Peak E-field: {np.max(np.abs(Ex_arr))*1e-6:.1f} MV/m")
print(f"Peak polarization: {np.max(np.abs(Px_arr)):.2e} C/m^2")
```

## What just happened

Behind the scenes, `InitializeSBE` set up and `QWCalculator` solved:

- **Coulomb interactions**: Screened electron-hole, electron-electron, and hole-hole
  scattering matrices (excitonic correlations, band-gap renormalization)
- **Phonon scattering**: LO phonon emission and absorption at 36 meV (GaAs)
- **Dephasing**: Diagonal and off-diagonal dephasing of the density matrix
- **DC transport**: Drift of carriers under any applied DC field
- **Spontaneous emission**: Radiative recombination of electron-hole pairs
- **Optical coupling**: Dipole matrix elements connecting valence and conduction bands

All of these ran self-consistently at every time step, controlled by the flags in
`params/mb.params`. This is the same physics that produces the published results
in Gulley & Huang, *Opt. Express* **27**, 17154 (2019).

## Next steps

- {doc}`02_architecture` -- understand the module structure and parameter files
- {doc}`03_coulomb` -- inspect the Coulomb matrices computed during initialization
- {doc}`04_phonons` -- explore phonon scattering rates and temperature dependence
- {doc}`05_optics` -- see how fields are projected between propagation and QW spaces

---

## References

1. J. R. Gulley and D. Huang, "Self-consistent quantum-kinetic theory for interplay
   between pulsed-laser excitation and nonlinear carrier transport in a quantum-wire
   array," *Opt. Express* **27**, 17154-17185 (2019).
