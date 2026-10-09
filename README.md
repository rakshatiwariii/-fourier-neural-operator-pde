# 1D Fourier Neural Operator (FNO) for Non-Linear PDE Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

An implementation of the Fourier Neural Operator (FNO) architecture designed to learn infinite-dimensional mappings between function spaces. This repository applies FNOs to solve the non-linear viscous Burgers' Equation directly in the frequency domain, bypassing traditional numerical discretization bottlenecks.

**Viscous Burgers' Equation:**
> **∂u/∂t + u · (∂u/∂x) = ν · (∂²u/∂x²)**

---

## Technical Overview

Traditional partial differential equation (PDE) solvers (e.g., Finite Element, Finite Difference) compute solutions on localized discretization grids, scaling poorly with resolution and time horizon requirements.

The **Fourier Neural Operator** parametrizes the integral kernel directly in Fourier space:

> **(K(a)v)(x) = F⁻¹ [ R_φ · (F v) ] (x)**

Where:
* **F** and **F⁻¹** denote the Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform (IFFT).
* **R_φ** represents a parameterized complex-valued weight matrix filtering truncated lower-frequency modes.

---

## Directory Architecture

```text
fno-pde-solver/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   └── train.py
└── notebooks/
    └── Fourier_Neural_Operator_Colab.ipynb
