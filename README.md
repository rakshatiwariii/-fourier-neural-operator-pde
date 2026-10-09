# -fourier-neural-operator-pde

# 1D Fourier Neural Operator (FNO) for Non-Linear PDE Resolution

An implementation of the Fourier Neural Operator (FNO) architecture designed to learn infinite-dimensional mappings between function spaces. This repository applies FNOs to solve the non-linear viscous Burgers' Equation directly in the frequency domain, bypassing traditional numerical discretization bottlenecks.

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

---

## Technical Overview

Traditional partial differential equation (PDE) solvers (e.g., Finite Element, Finite Difference) compute solutions on localized discretization grids, scaling poorly with resolution and time horizon requirements.

The **Fourier Neural Operator** parametrizes the integral kernel directly in Fourier space:

$$(\mathcal{K}(a)v)(x) = \mathcal{F}^{-1} \left( R_{\phi} \cdot (\mathcal{F} v) \right)(x)$$

Where:
* $\mathcal{F}$ and $\mathcal{F}^{-1}$ denote the Fast Fourier Transform (FFT) and its inverse.
* $R_{\phi}$ represents a parameterized complex-valued weight matrix filtering truncated lower-frequency modes.

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
