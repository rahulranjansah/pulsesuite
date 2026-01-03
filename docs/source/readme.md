# PulseSuite

```{image} _static/PulseSuitenobg.png
:alt: PulseSuite logo
:align: center
:width: 100%
```

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://pulsesuite.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green)](COPYING)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

PulseSuite is a high-performance computational physics toolkit for simulating ultrafast laser-matter interactions in semiconductor quantum structures. It implements the **Semiconductor Bloch Equations (SBEs)** coupled with **Pseudo-Spectral Time Domain (PSTD)** electromagnetic field propagation methods to model quantum wire and quantum well systems under intense optical excitation.

This codebase is a Python port of production Fortran simulation tools using NumPy, accelerated FFTs (pyFFTW), Numba JIT and CUDA compilation.

## Features

- **Quantum Well/Wire Optics**: Bloch Equations, Field transformations, polarization calculations, density matrix evolution
- **Many-Body Physics**: Coulomb interactions with screening, phonon scattering, carrier dynamics
- **Electromagnetic Field Propagation**: Pseudo-Spectral Time Domain (PSTD)
- **High Performance**: JIT compilation, CUDA, Vectorization, Parallelization

## Installation

```bash
# Clone the repository
git clone https://github.com/rahulranjansah/pulsesuite.git
cd pulsesuite

# Install with pip
pip install -e .

# Or install with dependencies
pip install -e ".[doc]"
```

## Documentation

📖 **Full documentation is available at:** [pulsesuite.readthedocs.io](https://pulsesuite.readthedocs.io)

The documentation includes:
- **Theory and Background**: Physical models and equations
- **Examples Gallery**: Interactive tutorials with executable code
- **API Reference**: Complete function and class documentation
- **Integration Guides**: How to use PulseSuite with other tools

## Package Structure

```
pulsesuite/
├── core/           # Core utilities (FFT, constants, integrators)
├── PSTD3D/         # Quantum wire/well physics modules
│   ├── coulomb.py  # Coulomb interactions
│   ├── dcfield.py  # DC field transport
│   ├── emission.py # Spontaneous emission
│   ├── phonons.py  # Phonon scattering
│   ├── qwoptics.py # Quantum well optics
│   └── SBEs.py     # Semiconductor Bloch Equations
└── libpulsesuite/  # Low-level utilities and integrators
```

## Requirements

- Python ≥3.10
- NumPy ≥1.26.4
- SciPy ≥1.15.2
- Matplotlib ≥3.10.0
- pyFFTW ≥0.15.0 (recommended for performance)
- Numba ≥0.61.2 (optional, for JIT acceleration)
- Numba-CUDA==0.23.0 (optional, for CUDA acceleration)

## Quick Example

```python
import numpy as np
from pulsesuite.PSTD3D.coulomb import InitializeCoulomb
from scipy.constants import e as e0, hbar

# Initialize Coulomb module for quantum wire simulations
# See documentation for complete examples
```

For detailed examples and tutorials, see the [Examples Gallery](https://pulsesuite.readthedocs.io/en/latest/examples/gallery.html) in the documentation.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

## Citation

If you use PulseSuite in your research, please cite:

```bibtex
@software{pulsesuite2025,
  title = {PulseSuite: Simulation suite for ultrafast laser-matter interactions},
  author = {Sah, Rahul R. and Gulley, Jeremy R.},
  year = {2025},
  url = {https://github.com/rahulranjansah/pulsesuite}
}
```

See [CITATION.md](CITATION.md) for more details.

## Authors

See [AUTHORS.md](AUTHORS.md) for the complete list of contributors.

## License

This project is licensed under the MIT License - see [LICENSE](COPYING) file for details.

## Links

- [Documentation](https://pulsesuite.readthedocs.io)
- [Issue Tracker](https://github.com/rahulranjansah/pulsesuite/issues)
- [Contact](mailto:f1rahulranjan@gmail.com)

## Acknowledgments

This codebase inherits from the Fortran simulation tools developed by [Jeremy R. Gulley](https://www.furman.edu/people/jeremy-r-gulley/) and collaborators at Furman University.
