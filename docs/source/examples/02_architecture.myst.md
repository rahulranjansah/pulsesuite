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

# Architecture: How the SBE Solver Works

:::{abstract}
This page explains the module structure, parameter files, and physics behind PulseSuite's
Semiconductor Bloch Equations solver. Read this after completing the {doc}`01_quickstart`
to understand what happened under the hood.
:::

## Theory

The Semiconductor Bloch Equations (SBEs) are coupled differential equations for the
quantum coherence of electrons and holes in a semiconductor under optical excitation.
They extend the optical Bloch equations to include many-body Coulomb interactions,
carrier-carrier scattering, and phonon effects.

### The Three Coupled Equations

**1. Electron-Hole Coherence (Interband Polarization):**

$$
i\hbar \frac{dp_{k_e,k_h}}{dt} = \sum_{k'} H_{k_h,k'}^{hh} p_{k',k_e} + \sum_{k'} H_{k_e,k'}^{ee} p_{k_h,k'} - \sum_{k'} H_{k',k_h}^{eh\dagger} C_{k',k_e} - \sum_{k'} H_{k_e,k'}^{eh} D_{k',k_h} + H_{k_e,k_h}^{eh} - i\hbar(\gamma_e + \gamma_h) p_{k_e,k_h}
$$

where $p_{k_e,k_h}$ is the electron-hole coherence matrix (interband polarization).

**2. Electron-Electron Coherence:**

$$
i\hbar \frac{dC_{k_1,k_2}}{dt} = \sum_{k'} H_{k_2,k'}^{ee} C_{k_1,k'} - \sum_{k'} H_{k',k_1}^{ee} C_{k',k_2} + \sum_{k'} H_{k_2,k'}^{eh} p_{k_1,k'}^\dagger - \sum_{k'} H_{k',k_1}^{eh\dagger} p_{k',k_2}
$$

Diagonal elements $C_{k,k} = n_e(k)$ are electron occupation numbers.

**3. Hole-Hole Coherence:**

$$
i\hbar \frac{dD_{k_1,k_2}}{dt} = \sum_{k'} H_{k_2,k'}^{hh} D_{k_1,k'} - \sum_{k'} H_{k',k_1}^{hh} D_{k',k_2} + \sum_{k'} H_{k',k_2}^{eh\dagger} p_{k',k_1}^\dagger - \sum_{k'} H_{k_1,k'}^{eh\dagger} p_{k_2,k'}
$$

Diagonal elements $D_{k,k} = n_h(k)$ are hole occupation numbers.

### Hamiltonian Construction

The effective Hamiltonians include single-particle energies, dipole-field coupling,
and screened Coulomb interactions:

$$
H_{k_e,k_h}^{eh} = M_{k_e,k_h}^{eh} + \sum_q V_{eh}(q) \, p_{k_e+q,k_h+q}^\dagger
$$

$$
H_{k_1,k_2}^{ee} = E_e(k_1)\delta_{k_1,k_2} - \sum_q V_{ee}(q) \, C_{k_1+q,k_2+q}^\dagger
$$

### Time Evolution

The SBEs use a leapfrog integration scheme with reshuffling for stability:

1. **Leapfrog step:** $X_3 = X_1 + \frac{dX}{dt} \cdot 2\Delta t$ (2nd order accurate)
2. **Reshuffling:** $X_2 = \frac{X_1 + X_3}{2}$ (converts to stable 1st order)

---

## Module Integration

The `SBEs` module is the central orchestrator. `InitializeSBE` sets up all
subsystems, and `QWCalculator` coordinates them at each time step.

```{mermaid}
graph TD
    A[InitializeSBE] --> B[InitializeQWOptics]
    A --> C[InitializeCoulomb]
    A --> D[InitializeDephasing]
    A --> E[InitializePhonons]
    A --> F[InitializeDC]
    A --> G[InitializeEmission]

    B --> H[QW Optics Module]
    C --> I[Coulomb Module]
    D --> J[Dephasing Module]
    E --> K[Phonons Module]
    F --> L[DC Field Module]
    G --> M[Emission Module]

    N[QWCalculator] --> O[Prop2QW]
    N --> P[SBECalculator]
    N --> Q[QW2Prop]

    O --> H
    P --> I
    P --> J
    P --> K
    P --> L
    P --> M
    Q --> H

    P --> R[QWPolarization3]
    P --> S[QWRho5]
    R --> H
    S --> H

    style A fill:#ff6b6b
    style N fill:#4ecdc4
    style H fill:#95e1d3
    style I fill:#95e1d3
    style J fill:#95e1d3
    style K fill:#95e1d3
    style L fill:#95e1d3
    style M fill:#95e1d3
```

**Initialization flow:**

1. `InitializeSBE` reads parameter files (`qw.params`, `mb.params`)
2. Calculates material constants and momentum/spatial grids
3. Initializes subsystems based on flags:
   - Always: `InitializeQWOptics`, `InitializeCoulomb`, `InitializeDephasing`
   - If `Phonon=1`: `InitializePhonons`
   - If `DCTrans=1`: `InitializeDC`
   - If `Recomb=1`: `InitializeEmission`

**Time evolution flow (each call to `QWCalculator`):**

1. `Prop2QW`: Convert propagation fields to QW momentum space
2. `SBECalculator`: Solve SBEs using all initialized modules
3. `QWPolarization3` / `QWRho5`: Calculate polarization and charge density
4. `QW2Prop`: Convert QW fields back to propagation space

---

## Parameter Reference

### Quantum Wire Parameters (`params/qw.params`)

| Line | Parameter | Units | Description | Typical Values |
|------|-----------|-------|-------------|----------------|
| 1 | L | m | Wire length | 50-500 nm |
| 2 | Delta0 | m | Wire thickness (z) | 3-10 nm |
| 3 | gap | eV | Band gap energy | GaAs: 1.42-1.5 |
| 4 | me | m$_0$ | Electron effective mass | GaAs: 0.067 |
| 5 | mh | m$_0$ | Hole effective mass | GaAs: 0.45 (heavy) |
| 6 | HO | eV | Energy level separation | 0.05-0.2 |
| 7 | gam_e | Hz | Electron dephasing rate | 0.1-10 THz |
| 8 | gam_h | Hz | Hole dephasing rate | 0.1-10 THz |
| 9 | gam_eh | Hz | Interband dephasing | (gam_e + gam_h)/2 |
| 10 | epsr | - | Relative permittivity | GaAs: 12-13, AlAs: 9.1 |
| 11 | Oph | eV | LO phonon energy | GaAs: 0.036 |
| 12 | Gph | eV | Phonon damping | 0.001-0.005 |
| 13 | Edc | V/m | DC electric field | 0 (transport studies) |
| 14 | jmax | - | Output interval (steps) | 10-1000 |
| 15 | ntmax | - | Backup interval (steps) | 1000-50000 |

### Many-Body Physics Flags (`params/mb.params`)

| Line | Flag | Description | When to Enable |
|------|------|-------------|----------------|
| 1 | Optics | Light-matter coupling | Always (required for optical response) |
| 2 | Excitons | Excitonic correlations | Exciton physics, bound states |
| 3 | EHs | Carrier-carrier scattering | Many-body collisions, thermalization |
| 4 | Screened | Screened Coulomb | Realistic screening (with Excitons) |
| 5 | Phonon | Phonon scattering | Temperature effects, energy relaxation |
| 6 | DCTrans | DC transport | Drift/diffusion studies |
| 7 | LF | Longitudinal field | Plasmon effects, screening dynamics |
| 8 | FreePot | Free potential | Carrier-induced potential |
| 9 | DiagDph | Diagonal dephasing | Momentum-dependent dephasing rates |
| 10 | OffDiagDph | Off-diagonal dephasing | Correlation effects in scattering |
| 11 | Recomb | Spontaneous recombination | Radiative decay, PL studies |
| 12 | PLSpec | PL spectrum calculation | Photoluminescence output |
| 13 | ignorewire | Single-wire mode | Skip inter-wire coupling |
| 14 | Xqwparams | Write chi params | Output susceptibility files |
| 15 | LorentzDelta | Lorentzian delta | Numerical broadening method |

**Common physics combinations:**

- **Optical Bloch (OBE)**: `Optics=1`, all others `0`
- **Excitons only**: `Optics=1, Excitons=1, Screened=1, DiagDph=1`
- **Full many-body**: `Optics=1, Excitons=1, EHs=1, Screened=1, DiagDph=1`
- **Temperature effects**: Add `Phonon=1` to above
- **Transport**: Add `DCTrans=1, LF=1` for drift and plasmon effects

---

## 1D-to-3D Coupling: SHO Gate Functions

The SBE solver operates in 1D momentum space along the wire axis. But the full
simulation couples to a 3D Maxwell PSTD propagator. The bridge between these
two worlds uses **Simple Harmonic Oscillator (SHO) wavefunctions** as spatial
gate functions.

### Transverse Confinement

The `HO` parameter in `qw.params` (100 meV for GaAs) sets the transverse
confinement energy. From this, the code computes inverse confinement lengths:

$$
\alpha_e = \sqrt{\frac{m_e \cdot \text{HO}}{\hbar}}, \quad
\alpha_h = \sqrt{\frac{m_h \cdot \text{HO}}{\hbar}}
$$

These define **Gaussian envelope functions** -- the ground-state SHO wavefunctions
that describe how the electron and hole probability densities decay in the
transverse (y, z) directions away from the wire axis.

### Derived Quantities

- **Electron-hole overlap integral**: $\text{ehint} = \sqrt{2\alpha_e\alpha_h / (\alpha_e^2 + \alpha_h^2)}$
  -- measures spatial overlap between electron and hole wavefunctions. Closer to 1
  means stronger excitonic effects.
- **Wire cross-section area**: $A = \sqrt{2\pi} \cdot \Delta_0 / \sqrt{\alpha_e^2 + \alpha_h^2}$
  -- effective transverse area for current and field calculations.
- **Critical momentum**: $q_c = 2\alpha_e\alpha_h / (\alpha_e + \alpha_h)$
  -- characteristic momentum scale for Coulomb interactions.

### 3D Field Placement

When coupling to the Maxwell propagator, the 1D polarization $\mathbf{P}(x)$
from the SBE solver is embedded into 3D space using separable Gaussian gates
(from `params/qwarray.params`):

$$
\mathbf{P}_{3D}(x, y, z) = g_x(x) \cdot g_y(y) \cdot g_z(z) \cdot \mathbf{P}_{wire}(x)
$$

where $g_y(y) = \exp\!\left[-(y - y_0)^2 / (2 a_y^2)\right]$ and similarly for $g_z$.
The gate widths $a_y$, $a_z$ are set in `qwarray.params` (typically 5 nm for GaAs).

This means quantum wires are **not mathematical lines** -- they have real 3D
cross-sections with quantum mechanical confinement profiles. The same SHO
wavefunctions enter the Coulomb interaction integrals in `coulomb.py`, where
the transverse overlap determines the effective 1D Coulomb potential.

---

## THz Emission from Current Surge

When an ultrashort laser pulse excites carriers in the quantum wire, the
rapidly changing polarization acts as a current source:

$$
\mathbf{J}(t) = \frac{\partial \mathbf{P}}{\partial t}
$$

This **polarization current surge** emits electromagnetic radiation at
terahertz (THz) frequencies -- the characteristic timescale of carrier dynamics
(femtoseconds to picoseconds corresponds to 0.1-10 THz).

The code computes this current in `rhoPJ.py` via finite differences:
$J_x = (P_x^{new} - P_x^{old}) / \Delta t$. An additional free current
component from charge density continuity ($\mathbf{J}_{free} = -\int \partial\rho/\partial t \, dx$)
contributes to the total THz emission.

THz emission spectroscopy is a major experimental technique for probing
ultrafast carrier dynamics in semiconductors. The infrastructure for tracking
THz fields exists in the codebase (`ETHz.t.dat` output), with full THz
field analysis planned for future propagator examples.

---

## Inspect an Initialized Solver

Let's initialize the solver and inspect what was set up.

```{code-cell} python
import os
import numpy as np
import matplotlib.pyplot as plt

from pulsesuite.PSTD3D.SBEs import InitializeSBE, QWCalculator
from pulsesuite.PSTD3D import SBEs as SBEs_module
from pulsesuite.PSTD3D.typespace import GetKArray, GetSpaceArray

# Ensure output directories exist
for d in ['dataQW/Wire/C', 'dataQW/Wire/D', 'dataQW/Wire/Ee',
          'dataQW/Wire/Eh', 'dataQW/Wire/P', 'dataQW/Wire/Win',
          'dataQW/Wire/Wout', 'dataQW/Wire/Xqw', 'dataQW/Wire/info',
          'dataQW/Wire/ne', 'dataQW/Wire/nh', 'output']:
    os.makedirs(d, exist_ok=True)

Nr = 50
drr = 10e-9
rr = GetSpaceArray(Nr, (Nr - 1) * drr)
qrr = GetKArray(Nr, Nr * drr)

InitializeSBE(qrr, rr, 0.0, 1e7, 800e-9, 2, True)
solver = SBEs_module._default_solver

print("=== Grid ===")
print(f"  Nk (momentum points): {solver.Nk}")
print(f"  Nr (QW spatial points): {solver.Nr}")

print("\n=== Material (from qw.params) ===")
print(f"  Band gap: {solver.gap / 1.6e-19:.3f} eV")
print(f"  Electron mass: {solver.me / 9.109e-31:.4f} m0")
print(f"  Hole mass: {solver.mh / 9.109e-31:.4f} m0")

print("\n=== Derived Constants ===")
print(f"  Dipole moment: {solver.dcv:.3e} C*m")
print(f"  Wire cross-section: {solver.area:.3e} m^2")

print("\n=== Physics Flags (from mb.params) ===")
flags = ['Optics', 'Excitons', 'EHs', 'Screened', 'Phonon',
         'DCTrans', 'LF', 'DiagDph', 'OffDiagDph', 'Recomb']
for flag in flags:
    val = getattr(solver, flag)
    status = "ON" if val else "OFF"
    print(f"  {flag:15s}: {status}")
```

### Energy Bands

After initialization, the solver has computed the electron and hole energy bands
$E_e(k)$ and $E_h(k)$ on the momentum grid.

```{code-cell} python
fig, ax = plt.subplots(figsize=(8, 5))

k_nm = solver.kr * 1e-9  # Convert to 1/nm
Ee_eV = solver.Ee * 6.242e18  # Convert J to eV
Eh_eV = solver.Eh * 6.242e18

ax.plot(k_nm, Ee_eV, 'b-', linewidth=2, label='Electrons $E_e(k)$')
ax.plot(k_nm, -Eh_eV, 'r-', linewidth=2, label='Holes $-E_h(k)$')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

ax.set_xlabel('Momentum $k$ (nm$^{-1}$)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Quantum Wire Energy Bands')
ax.legend()
plt.tight_layout()
plt.show()

print(f"Momentum grid: {solver.Nk} points from "
      f"{solver.kr.min()*1e-9:.2f} to {solver.kr.max()*1e-9:.2f} nm^-1")
```

### Initial Carrier Distributions

The carriers start in thermal equilibrium (Fermi-Dirac at 77 K). The diagonal
of the electron coherence matrix $C_{k,k} = n_e(k)$ gives the electron occupation.

```{code-cell} python
# Extract initial carrier distributions (diagonal of CC2 for wire 1)
ne = np.real(np.diag(solver.CC2[:, :, 0]))  # Electron occupation
nh = np.real(np.diag(solver.DD2[:, :, 0]))  # Hole occupation

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_nm, ne, 'b-', linewidth=2, label='$n_e(k)$')
ax.plot(k_nm, nh, 'r-', linewidth=2, label='$n_h(k)$')

ax.set_xlabel('Momentum $k$ (nm$^{-1}$)')
ax.set_ylabel('Occupation')
ax.set_title('Initial Carrier Distributions (T = 77 K)')
ax.legend()
plt.tight_layout()
plt.show()

print(f"Total electron density: {ne.sum():.4e}")
print(f"Total hole density: {nh.sum():.4e}")
```

---

## References

1. J. R. Gulley and D. Huang, "Self-consistent quantum-kinetic theory for interplay
   between pulsed-laser excitation and nonlinear carrier transport in a quantum-wire
   array," *Opt. Express* **27**, 17154-17185 (2019).

2. J. R. Gulley and D. Huang, "Ultrafast transverse and longitudinal response of
   laser-excited quantum wires," *Opt. Express* **30**(6), 9348-9359 (2022).

3. M. Lindberg and S. W. Koch, "Effective Bloch equations for semiconductors,"
   *Phys. Rev. B* **38**, 3342 (1988).
