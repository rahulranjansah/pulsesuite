```{image} _static/PulseSuitenobg.png
:alt: PulseSuite logo
:align: center
:width: 100%
```

# PulseSuite

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://pulsesuite.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/rahulranjansah/pulsesuite/blob/main/COPYING)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

PulseSuite is a high-performance computational physics toolkit for simulating ultrafast laser-matter interactions in semiconductor quantum structures. It implements the **Semiconductor Bloch Equations (SBEs)** coupled with **Pseudo-Spectral Time Domain (PSTD)** electromagnetic field propagation methods to model quantum wire and quantum well systems under intense optical excitation.

This codebase is a Python port of production Fortran simulation tools using NumPy, accelerated FFTs (pyFFTW), Numba JIT and CUDA compilation.

## Features

- **Quantum Well/Wire Optics**: Bloch Equations, Field transformations, polarization calculations, density matrix evolution
- **Many-Body Physics**: Coulomb interactions with screening, phonon scattering, carrier dynamics
- **Electromagnetic Field Propagation**: Pseudo-Spectral Time Domain (PSTD)
- **High Performance**: JIT compilation, CUDA, Vectorization, Parallelization

## Installation

PulseSuite uses [uv](https://docs.astral.sh/uv/) for dependency management and [just](https://github.com/casey/just) as a command runner.

```bash
# Clone the repository
git clone https://github.com/rahulranjansah/pulsesuite.git
cd pulsesuite

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies (core + test + doc) in one step
uv sync --all-extras

# Or install with pip (still works)
pip install -e .
```

### Development

```bash
just              # run tests + lint + format check
just test         # run test suite (just test -k coulomb to filter)
just --list       # see all available commands
```

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

## Running a Simulation

The SBE test propagation script drives a full quantum wire simulation:

```bash
# Requires params/qw.params and params/mb.params in the working directory
uv run python -m pulsesuite.PSTD3D.sbetestprop
```

Output is written to `fields/` and `dataQW/`.

For detailed examples and tutorials, see the {doc}`gallery` in the documentation.

## Contributing

We welcome contributions! Please see the {doc}`contributing` guide for guidelines.

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
  author = {Sah, Rahul R., Emily S. Hatten, and Gulley, Jeremy R.},
  year = {2025},
  url = {https://github.com/rahulranjansah/pulsesuite}
}
```

See {doc}`citation` for more details.

## Authors

See {doc}`authors` for the complete list of contributors.

## License

This project is licensed under the MIT License - see [LICENSE](https://github.com/rahulranjansah/pulsesuite/blob/main/COPYING) for details.

## Links

- [Documentation](https://pulsesuite.readthedocs.io)
- [Issue Tracker](https://github.com/rahulranjansah/pulsesuite/issues)
- [Contact](mailto:f1rahulranjan@gmail.com)

## Acknowledgments

This codebase inherits from the Fortran simulation tools developed by [Jeremy R. Gulley](https://www.furman.edu/people/jeremy-r-gulley/) and collaborators at Furman University.
