# PulseSuite

```{image} _static/PulseSuitenobg.png
:alt: PulseSuite logo
:align: center
:width: 100%
```

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://pulsesuite.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

PulseSuite is a high-performance computational physics toolkit for simulating ultrafast laser-matter interactions in semiconductor quantum structures. It implements the **Semiconductor Bloch Equations (SBEs)** coupled with **Pseudo-Spectral Time Domain (PSTD)** electromagnetic field propagation methods to model quantum wire and quantum well systems under intense optical excitation.

This codebase is a Python port of production Fortran simulation tools using NumPy, accelerated FFTs (pyFFTW), Numba JIT and CUDA compilation.

```{include} ../../README.md
:start-after: ## Features
```