# Building Simulations

PulseSuite simulations are driven by `sbetestprop` — a script that
time-evolves the Semiconductor Bloch Equations under a pulsed laser
field. The physics that runs at each time step is entirely controlled
by the flags in `params/mb.params`. By toggling these flags between
`0` and `1`, you can build up from a bare two-level system to a
full many-body simulation **without changing any code**.

This page walks through the physics regimes from simplest to most
complete.

## The Simulation Driver

The simulation loop in `sbetestprop` is straightforward:

1. Read parameter files (`qw.params`, `mb.params`)
2. Initialize all enabled physics modules
3. Time-evolve:
   - Compute the driving electric field at each time step
   - Call `QWCalculator` to evolve the SBEs (all enabled interactions
     run self-consistently)
   - Write output fields and polarizations

```bash
# Run a simulation (reads params/ from the working directory)
uv run python -m pulsesuite.PSTD3D.sbetestprop
```

Output goes to `fields/` (E-fields, polarizations) and `dataQW/`
(density matrices, carrier distributions).

See {doc}`params_reference` for the full specification of every
parameter file.

---

## Physics Regimes

### Level 0: Optical Bloch Equations (OBE)

The simplest case — non-interacting two-level systems driven by light.
No Coulomb, no phonons, no screening.

```
1          # Optics        ← light-matter coupling
0          # Excitons
0          # EHs
0          # Screened
0          # Phonon
0          # DCTrans
0          # LF
0          # FreePot
0          # DiagDph
0          # OffDiagDph
0          # Recomb
0          # PLSpec
0          # Ignorewire
0          # Xqwparams
0          # LorentzDelta
```

**What you get:** Free-carrier optical response. The polarization
follows the Rabi oscillation of independent electron-hole pairs. No
bound states, no scattering, no decay.

**Use when:** Testing the basic setup, verifying pulse shapes, or
comparing against analytic two-level solutions.

---

### Level 1: Excitonic Response

Add Coulomb interactions to get bound exciton states and band-gap
renormalization.

```
1          # Optics        ← light-matter coupling
1          # Excitons      ← Hartree-Fock Coulomb (excitonic correlations)
0          # EHs
1          # Screened      ← RPA screening of Coulomb potential
0          # Phonon
0          # DCTrans
0          # LF
0          # FreePot
1          # DiagDph       ← population decay
0          # OffDiagDph
0          # Recomb
0          # PLSpec
0          # Ignorewire
0          # Xqwparams
0          # LorentzDelta
```

**What you get:** Exciton peaks appear below the band edge in the
absorption spectrum. The band gap shifts with carrier density
(renormalization). Dephasing gives finite linewidths.

**Use when:** Studying excitonic absorption, band-gap shifts, or linear
optical properties.

```{note}
`Screened` should generally be enabled together with `Excitons`.
Unscreened Coulomb diverges at high carrier densities.
```

---

### Level 2: Full Coulomb with Scattering

Enable electron-hole scattering and off-diagonal dephasing for
many-body carrier dynamics.

```
1          # Optics
1          # Excitons
1          # EHs           ← e-h pair scattering
1          # Screened
0          # Phonon
0          # DCTrans
0          # LF
0          # FreePot
1          # DiagDph
1          # OffDiagDph    ← coherence decay from scattering
0          # Recomb
0          # PLSpec
0          # Ignorewire
0          # Xqwparams
0          # LorentzDelta
```

**What you get:** Carrier thermalization, excitation-induced dephasing,
phase-space filling. The polarization decays faster at higher
excitation because carrier-carrier scattering destroys coherence.

**Use when:** Studying nonlinear optical response, carrier dynamics, or
density-dependent absorption.

---

### Level 3: Add Temperature (Phonons)

Phonon scattering introduces energy relaxation and
temperature-dependent effects.

```
1          # Optics
1          # Excitons
1          # EHs
1          # Screened
1          # Phonon        ← LO phonon scattering
0          # DCTrans
0          # LF
0          # FreePot
1          # DiagDph
1          # OffDiagDph
0          # Recomb
0          # PLSpec
0          # Ignorewire
0          # Xqwparams
0          # LorentzDelta
```

**What you get:** Carriers lose energy to the lattice via LO phonon
emission. The phonon energy (36 meV for GaAs) and lattice temperature
(set in `qw.params`) determine the relaxation dynamics.

**Use when:** Studying hot-carrier relaxation, temperature-dependent
spectra, or energy transfer to the lattice.

---

### Level 4: Transport and Plasmonics

Add drift under DC fields and longitudinal plasmon modes.

```
1          # Optics
1          # Excitons
1          # EHs
1          # Screened
1          # Phonon
1          # DCTrans       ← carrier drift in DC field
1          # LF            ← plasmon/longitudinal modes
0          # FreePot
1          # DiagDph
1          # OffDiagDph
0          # Recomb
0          # PLSpec
0          # Ignorewire
0          # Xqwparams
0          # LorentzDelta
```

**What you get:** Carriers accelerate under the DC field set by `Edc`
in `qw.params`. The longitudinal field captures collective plasmon
oscillations. Combined with phonons, you get mobility and conductivity.

**Use when:** Studying photocurrent generation, carrier transport, or
THz emission from ultrafast excitation.

```{note}
Set `Edc` in `qw.params` to a nonzero value (e.g., `1e5` V/m) for
transport to have an effect.
```

---

### Level 5: Full Simulation (All Interactions)

Everything enabled — the configuration used in the published results
(Gulley & Huang, *Opt. Express* 2019, 2022).

```
1          # Optics
1          # Excitons
1          # EHs
1          # Screened
1          # Phonon
1          # DCTrans
1          # LF
0          # FreePot
1          # DiagDph
1          # OffDiagDph
1          # Recomb         ← spontaneous emission
0          # PLSpec
0          # Ignorewire
1          # Xqwparams      ← write diagnostic output
0          # LorentzDelta
```

**What you get:** Self-consistent coupling of all interactions at
every time step:
- Coulomb screening feeds into phonon scattering rates
- Phonon relaxation modifies carrier distributions
- Modified distributions change the screening
- Transport redistributes carriers spatially
- Spontaneous emission depletes the excited state

This is the full physics that produces the published results.

**Use when:** Production simulations for publication, or when you need
all interactions to be coupled self-consistently.

---

## Optional Output Flags

The last few flags in `mb.params` control diagnostics rather than
physics:

| Flag | Purpose |
|------|---------|
| `PLSpec` | Enable to write frequency-resolved photoluminescence data (requires `Recomb = 1`) |
| `Ignorewire` | Set to `1` for single-wire isolation (disables inter-wire coupling in arrays) |
| `Xqwparams` | Write susceptibility and dipole diagnostic files |
| `LorentzDelta` | Switch from Gaussian to Lorentzian broadening for energy-conserving delta functions |

---

## Combining with Different Materials

The physics flags in `mb.params` are material-independent. To simulate
a different semiconductor system, change `qw.params`:

| Material | `gap` (eV) | `me` (m_e0) | `mh` (m_e0) | `epsr` | `Oph` (eV) |
|----------|------------|-------------|-------------|--------|------------|
| GaAs | 1.42 | 0.067 | 0.45 | 12.9 | 0.036 |
| GaAs QW in AlAs host | 1.5 | 0.07 | 0.45 | 9.1 | 0.036 |
| InGaAs | 0.75 | 0.041 | 0.45 | 13.9 | 0.033 |
| GaN | 3.4 | 0.20 | 1.4 | 8.9 | 0.091 |

The same `mb.params` flags work regardless of material — the physics
modules adapt to whatever material parameters are loaded.

---

## Running Your First Custom Simulation

1. Copy the template parameter files:
   ```bash
   mkdir -p params
   cp src/pulsesuite/PSTD3D/params/templates/*.params params/
   ```

2. Edit `params/mb.params` to choose your physics level (see above)

3. Edit `params/qw.params` for your material and dephasing rates

4. Run:
   ```bash
   uv run python -m pulsesuite.PSTD3D.sbetestprop
   ```

5. Output appears in `fields/` (time-domain) and `dataQW/` (k-space)
