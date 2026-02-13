# Integrations

PulseSuite modules are designed to work together but can also be used
independently. This page describes how the subsystems connect and
important considerations when coupling them.

## Module Coupling

### Emissions Module

When integrating the emissions module with other subsystems, keep the
following in mind:

- **Coulomb interactions**: The emissions module uses simplified diagonal
  matrices internally. For realistic calculations, couple it with the
  full Coulomb module ({py:mod}`pulsesuite.PSTD3D.coulomb`).
- **Carrier evolution**: Carrier distributions are static within the
  emissions module. Full simulations require coupling with the SBE
  solver ({py:mod}`pulsesuite.PSTD3D.SBEs`).
- **Material parameters**: Default values correspond to bulk GaAs.
  Confined systems (quantum wells/wires) may require adjusted
  parameters.
- **Temperature**: Module-level temperature is set directly for
  demonstration purposes. In production, initialize with the correct
  lattice temperature from your simulation setup.

## External Tools

PulseSuite interoperates with the standard scientific Python stack:

- **NumPy / SciPy** -- array operations and linear algebra
- **Matplotlib** -- visualization of fields, spectra, and band structures
- **pyFFTW** -- accelerated FFTs (drop-in replacement for NumPy FFT)
- **Numba / CUDA** -- JIT compilation and GPU acceleration
