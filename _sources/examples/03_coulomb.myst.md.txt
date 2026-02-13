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

# Deep Dive: Coulomb Interactions

:::{abstract}
Inspect the Coulomb matrices computed by `InitializeSBE` -- electron-hole,
electron-electron, and hole-hole interactions in a GaAs quantum wire.
This example uses **real computed data**, not fabricated arrays.
:::

## Theory

The Coulomb module calculates carrier-carrier interactions in the quantum wire.
These enter the SBE Hamiltonians as screened interaction matrices.

### Coulomb Interaction Potential

The interaction between carriers confined in a quantum wire is:

$$
V(k, q) = \frac{e^2}{2\pi \epsilon_0 \epsilon_r L}
\int dy_1 \, dy_2 \; |\phi_e(y_1)|^2 |\phi_h(y_2)|^2 \, K_0(|k-q| \, r)
$$

where $K_0$ is the modified Bessel function of the second kind, $r = \sqrt{(y_1-y_2)^2 + \Delta_0^2}$
accounts for the wire thickness, and $\phi_{e/h}$ are the confinement wavefunctions.

The Bessel function $K_0(x) \sim -\ln(x/2)$ for small $x$, giving the characteristic
logarithmic divergence of the 1D Coulomb potential at $k=q$ (same momentum).

### Energy Renormalization

Many-body Coulomb interactions renormalize the single-particle energies:

**Band Gap Renormalization (BGR):**

$$
\Delta E_{gap} = -\sum_k [n_e(k) V_{ee}(k,k) + n_h(k) V_{hh}(k,k)]
$$

**Electron/Hole Energy Renormalization:**

$$
\Delta E_e(k) = \sum_{k'} n_e(k') [V_{ee}(k,k) - V_{ee}(k,k')]
$$

### Screening

Free carriers screen the bare Coulomb interaction:

$$
V_{screened}(q) = \frac{V_{bare}(q)}{\epsilon(q)}
$$

where $\epsilon(q)$ is the static dielectric function computed from the 1D Lindhard
susceptibility.

---

## Initialize and Inspect

We call `InitializeSBE` which internally calls `InitializeCoulomb` with real
material parameters. Then we access the computed Coulomb arrays.

```{code-cell} python
import os
import numpy as np
import matplotlib.pyplot as plt

from pulsesuite.PSTD3D.SBEs import InitializeSBE
from pulsesuite.PSTD3D import SBEs as SBEs_module
from pulsesuite.PSTD3D import coulomb
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
cm = coulomb._instance  # CoulombModule singleton

print(f"Coulomb module initialized for Nk={solver.Nk} momentum points")
print(f"Unscreened Coulomb matrices: Veh0 {cm.Veh0.shape}, Vee0 {cm.Vee0.shape}, Vhh0 {cm.Vhh0.shape}")
```

## Unscreened Coulomb Matrices

The three interaction matrices $V_{eh}$, $V_{ee}$, $V_{hh}$ in momentum space.
The diagonal ($k=k'$) is strongest because carriers at the same momentum have
maximum spatial overlap. The off-diagonal elements mediate scattering.

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

eV = 1.6e-19
k_nm = solver.kr * 1e-9

for ax, V, title in zip(axes,
                         [cm.Veh0, cm.Vee0, cm.Vhh0],
                         ['$V_{eh}$ (e-h)', '$V_{ee}$ (e-e)', '$V_{hh}$ (h-h)']):
    im = ax.pcolormesh(k_nm, k_nm, np.real(V) / eV,
                        shading='auto', cmap='RdBu_r')
    ax.set_xlabel('$k$ (nm$^{-1}$)')
    ax.set_ylabel("$k'$ (nm$^{-1}$)")
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='eV')

plt.suptitle('Unscreened Coulomb Interaction Matrices', y=1.02)
plt.tight_layout()
plt.show()

print(f"Veh diagonal range: {np.real(cm.Veh0).diagonal().min()/eV:.4f} to "
      f"{np.real(cm.Veh0).diagonal().max()/eV:.4f} eV")
print(f"Vee diagonal range: {np.real(cm.Vee0).diagonal().min()/eV:.4f} to "
      f"{np.real(cm.Vee0).diagonal().max()/eV:.4f} eV")
```

## Momentum-Space Cuts

A cut along the diagonal ($k=k'$) shows the self-energy contribution.
A cut at fixed $k$ shows how the interaction decays with momentum transfer.

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Diagonal cut: V(k, k)
ax1.plot(k_nm, np.real(cm.Veh0).diagonal() / eV, 'g-', linewidth=2, label='$V_{eh}(k,k)$')
ax1.plot(k_nm, np.real(cm.Vee0).diagonal() / eV, 'b-', linewidth=2, label='$V_{ee}(k,k)$')
ax1.plot(k_nm, np.real(cm.Vhh0).diagonal() / eV, 'r-', linewidth=2, label='$V_{hh}(k,k)$')
ax1.set_xlabel('$k$ (nm$^{-1}$)')
ax1.set_ylabel('Coulomb energy (eV)')
ax1.set_title('Diagonal: $V(k, k)$ (self-energy)')
ax1.legend()

# Off-diagonal cut at k=0 (midpoint of grid)
mid_k = solver.Nk // 2
ax2.plot(k_nm, np.real(cm.Veh0[mid_k, :]) / eV, 'g-', linewidth=2, label='$V_{eh}(k_0, k\')$')
ax2.plot(k_nm, np.real(cm.Vee0[mid_k, :]) / eV, 'b-', linewidth=2, label='$V_{ee}(k_0, k\')$')
ax2.plot(k_nm, np.real(cm.Vhh0[mid_k, :]) / eV, 'r-', linewidth=2, label='$V_{hh}(k_0, k\')$')
ax2.set_xlabel("$k'$ (nm$^{-1}$)")
ax2.set_ylabel('Coulomb energy (eV)')
ax2.set_title(f'Row cut: $V(k_0, k\')$ at $k_0$ = {k_nm[mid_k]:.2f} nm$^{{-1}}$')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Many-Body Broadening Arrays

The matrices $C_{eh}$, $C_{ee}$, $C_{hh}$ control the energy-conservation broadening
in many-body collision integrals. They determine which scattering channels are
energetically allowed.

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, C, title in zip(axes,
                         [cm.Ceh, cm.Cee, cm.Chh],
                         ['$C_{eh}$', '$C_{ee}$', '$C_{hh}$']):
    im = ax.pcolormesh(k_nm, k_nm, np.real(C),
                        shading='auto', cmap='viridis')
    ax.set_xlabel('$k$ (nm$^{-1}$)')
    ax.set_ylabel("$k'$ (nm$^{-1}$)")
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

plt.suptitle('Many-Body Broadening Matrices', y=1.02)
plt.tight_layout()
plt.show()

print(f"Ceh range: {np.real(cm.Ceh).min():.3e} to {np.real(cm.Ceh).max():.3e}")
print(f"Cee range: {np.real(cm.Cee).min():.3e} to {np.real(cm.Cee).max():.3e}")
```

## 1D Susceptibility Arrays

The Lindhard susceptibility arrays $\chi^{1D}_e$ and $\chi^{1D}_h$ are used
to compute the dielectric function for screening.

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.pcolormesh(k_nm, k_nm, np.real(cm.Chi1De), shading='auto', cmap='coolwarm')
ax1.set_xlabel('$k$ (nm$^{-1}$)')
ax1.set_ylabel("$k'$ (nm$^{-1}$)")
ax1.set_title('$\\chi^{1D}_e$ (electron susceptibility)')
ax1.set_aspect('equal')

ax2.pcolormesh(k_nm, k_nm, np.real(cm.Chi1Dh), shading='auto', cmap='coolwarm')
ax2.set_xlabel('$k$ (nm$^{-1}$)')
ax2.set_ylabel("$k'$ (nm$^{-1}$)")
ax2.set_title('$\\chi^{1D}_h$ (hole susceptibility)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()
```

---

## References

1. J. R. Gulley and D. Huang, *Opt. Express* **27**, 17154-17185 (2019).
   -- Coulomb interactions and self-consistent SBE framework.
