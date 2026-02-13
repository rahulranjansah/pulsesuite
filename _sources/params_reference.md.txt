# Parameter Files Reference

PulseSuite simulations are configured entirely through plain-text
parameter files. Each file controls a different aspect of the physics,
and all are read at initialization time by `InitializeSBE`. No code
changes are needed to switch between simulation regimes — just edit the
parameter files.

## File Overview

| File | Controls |
|------|----------|
| `qw.params` | Quantum wire material properties and dephasing rates |
| `mb.params` | Many-body physics flags (which interactions are active) |
| `pulse.params` | Optical pulse shape, wavelength, and timing |
| `space.params` | Spatial grid dimensions and resolution |
| `qwarray.params` | Quantum wire array geometry |
| `pstd_abc.params` | Absorbing boundary conditions for the PSTD grid |

All parameter files use the same format: **one value per line**, with
inline comments after `#`. The order of lines is fixed and must be
preserved.

---

## `qw.params` — Quantum Wire Properties

Defines the material and dephasing parameters for the quantum wire
system. These map directly to the physical constants used in the
Semiconductor Bloch Equations.

| Line | Parameter | Units | Description |
|------|-----------|-------|-------------|
| 1 | `L` | m | Length of quantum wire |
| 2 | `Delta0` | m | Z-thickness of quantum wire |
| 3 | `gap` | eV | Band gap energy |
| 4 | `me` | m_e0 | Electron effective mass |
| 5 | `mh` | m_e0 | Hole effective mass |
| 6 | `HO` | eV | Energy level separation |
| 7 | `gam_e` | Hz | Electron dephasing rate (1/lifetime) |
| 8 | `gam_h` | Hz | Hole dephasing rate |
| 9 | `gam_eh` | Hz | Interband dephasing rate |
| 10 | `epsr` | — | Background dielectric constant |
| 11 | `Oph` | eV | LO phonon energy |
| 12 | `Gph` | eV | Phonon damping (inverse lifetime) |
| 13 | `Edc` | V/m | Background DC electric field |
| 14 | `jmax` | steps | Write output every N time steps |
| 15 | `ntmax` | steps | Backup/checkpoint interval |

**Example** (GaAs quantum wire):

```
100e-9     # L      — 100 nm wire
5e-9       # Delta0 — 5 nm thickness
1.5        # gap    — 1.5 eV band gap
0.07       # me     — GaAs electron mass
0.45       # mh     — GaAs hole mass
100e-3     # HO     — 100 meV level separation
1e12       # gam_e  — 1 ps electron lifetime
1e12       # gam_h  — 1 ps hole lifetime
1e12       # gam_eh — 1 ps interband dephasing
9.1        # epsr   — AlAs host dielectric
36e-3      # Oph    — 36 meV LO phonon (GaAs)
3e-3       # Gph    — 3 meV phonon damping
0.0        # Edc    — no DC field
500        # jmax   — output every 500 steps
20000      # ntmax  — checkpoint every 20000 steps
```

---

## `mb.params` — Many-Body Physics Flags

This is the most important configuration file. Each line is a boolean
flag (`1` = enabled, `0` = disabled) that toggles a specific physics
interaction. By flipping these flags, you can run anything from a bare
optical Bloch equation to a full many-body simulation — **without
changing a single line of code**.

| Line | Flag | Physics |
|------|------|---------|
| 1 | `Optics` | Light-matter coupling via dipole operator |
| 2 | `Excitons` | Excitonic correlations and band-gap renormalization (Hartree-Fock) |
| 3 | `EHs` | Electron-hole pair scattering |
| 4 | `Screened` | Screened Coulomb interactions (RPA) |
| 5 | `Phonon` | LO phonon scattering (Frohlich interaction) |
| 6 | `DCTrans` | DC transport (carrier drift under applied field) |
| 7 | `LF` | Longitudinal field / plasmon modes |
| 8 | `FreePot` | Free charge potential |
| 9 | `DiagDph` | Diagonal dephasing (population decay) |
| 10 | `OffDiagDph` | Off-diagonal dephasing (coherence decay) |
| 11 | `Recomb` | Spontaneous emission (radiative recombination) |
| 12 | `PLSpec` | Photoluminescence spectrum output |
| 13 | `Ignorewire` | Single-wire mode (skip inter-wire coupling) |
| 14 | `Xqwparams` | Write susceptibility diagnostic files |
| 15 | `LorentzDelta` | Use Lorentzian broadening for delta functions |

**Example** (full many-body simulation):

```
1          # Optics
1          # Excitons
1          # EHs
1          # Screened
1          # Phonon
1          # DCTrans
1          # LF (Plasmonics)
0          # Free Potential
1          # DiagDph
1          # OffDiagDph
1          # Recomb
0          # PLSpec
0          # Ignorewire
1          # Xqwparams
0          # LorentzDelta
```

See {doc}`building_simulations` for how to combine these flags for
different physics regimes.

---

## `pulse.params` — Optical Pulse Configuration

Defines the laser pulse that drives the simulation.

| Line | Parameter | Units | Description |
|------|-----------|-------|-------------|
| 1 | `dimensions` | — | Spatial dimensionality (1, 2, or 3) |
| 2 | `n0` | — | Background refractive index |
| 3 | `lam` | m | Laser wavelength |
| 4 | `E00` | V/m | Peak electric field amplitude |
| 5 | `tw0` | s | Pulse width |
| 6 | `chirp` | rad/s^2 | Pulse chirp |
| 7 | `xw` | m | Beam width in X (2D/3D) |
| 8 | `zw` | m | Beam width in Z (3D) |
| 9 | `Tpeak1` | s | Pulse peak time (1D/2D) |
| 10 | `Tpeak3` | s | Pulse peak time (3D) |
| 11 | `Ystart` | m | Starting Y position |
| 12 | `Xstart` | m | Starting X position (2D/3D) |
| 13 | `Zstart` | m | Starting Z position (3D) |
| 14 | `Nt/Toc` | — | Time points per optical cycle |
| 15 | `Ny/lambda` | — | Y grid points per wavelength |
| 16 | `Nx/lambda` | — | X grid points per wavelength |
| 17 | `Nz/lambda` | — | Z grid points per wavelength |
| 18 | `Nt` | — | Total optical cycles to simulate |
| 19 | `Ywin` | m | Y window width |
| 20 | `Xwin` | m | X window width (2D/3D) |
| 21 | `Zwin` | m | Z window width (3D) |
| 22 | `jmax` | — | Output interval |
| 23 | `nmax` | — | Maximum time steps |
| 24 | `Restart` | bool | Resume from checkpoint (0/1) |

---

## `space.params` — Spatial Grid

Defines the computational grid for field propagation.

| Line | Parameter | Units | Description |
|------|-----------|-------|-------------|
| 1 | `Dimensions` | — | 1, 2, or 3 |
| 2 | `Nx` | — | Grid points in X |
| 3 | `Ny` | — | Grid points in Y |
| 4 | `Nz` | — | Grid points in Z |
| 5 | `dx` | m | Pixel spacing in X |
| 6 | `dy` | m | Pixel spacing in Y |
| 7 | `dz` | m | Pixel spacing in Z |
| 8 | `epsilon_r` | — | Background dielectric constant |

---

## `qwarray.params` — Quantum Wire Array

Configures the geometry of quantum wire arrays.

| Line | Parameter | Units | Description |
|------|-----------|-------|-------------|
| 1 | `QW` | bool | Include quantum wire calculations (0/1) |
| 2 | `Nw` | — | Number of quantum wires |
| 3 | `y0` | m | Wire placement along X axis |
| 4 | `dxqw` | m | Wire-to-wire spacing |
| 5 | `a0_y` | m | SHO oscillator length in Y |
| 6 | `a0_z` | m | SHO oscillator length in Z |

---

## `pstd_abc.params` — Absorbing Boundary Conditions

Controls the absorbing boundary layer for the PSTD grid to prevent
reflections at domain edges.

| Line | Parameter | Units | Description |
|------|-----------|-------|-------------|
| 1 | Flag | bool | Enable ABC (0/1) |
| 2 | Value 1 | m | Boundary layer parameter |
| 3 | Value 2 | m | Boundary layer parameter |
| 4 | Flag | bool | Secondary ABC flag (0/1) |
| 5 | Value 3 | m | Boundary layer parameter |
| 6 | Value 4 | m | Boundary layer parameter |

---

## How Parameters Flow into the Simulation

All parameter files are read once during `InitializeSBE`. The
initialization sequence is:

```
InitializeSBE(q, rr, r0, Emax, lam, Nw, QW)
  ├── read qw.params   → material constants (gap, masses, dephasing)
  ├── read mb.params    → physics flags (which interactions are ON)
  ├── InitializeQWOptics()     (always)
  ├── InitializeCoulomb()      (always)
  ├── InitializeDephasing()    (always)
  ├── InitializePhonons()      (if Phonon = 1)
  ├── InitializeDC()           (if DCTrans = 1)
  └── InitializeEmission()     (if Recomb = 1)
```

Once initialized, `QWCalculator` runs all enabled physics
self-consistently at every time step. There is no runtime flag
switching — to change the physics, re-initialize.

## Template Files

Default parameter templates are provided in
`src/pulsesuite/PSTD3D/params/templates/`. Copy them as a starting
point for new simulations:

```bash
cp src/pulsesuite/PSTD3D/params/templates/*.params params/
```
