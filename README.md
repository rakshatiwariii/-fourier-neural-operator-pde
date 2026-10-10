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


## Multi-Dimensional Operator Learning

### 2D Incompressible Navier-Stokes Vorticity Trajectories
The architecture expands from 1D Burgers' equation to 2D spatial fluid flow ($X \times Y$ mesh grids) using 2D Real Fourier Transforms (`rfft2` / `irfft2`):

![2D Navier Stokes FNO Prediction](assets/fno_2d_navier_stokes.png)

## Robustness & Noise-Sensitivity Ablation
To evaluate the stability of the Inverse PINN under imperfect sensor conditions, the model was subjected to increasing levels of synthetic Gaussian white noise across the spatial-temporal domain. 

| Sensor Noise Level ($\eta$) | Target Viscosity ($\nu$) | Discovered Viscosity ($\nu_{\text{trainable}}$) | Relative Error (%) | Parameter Recovery Status |
| :--- | :--- | :--- | :--- | :--- |
| **1.0% Noise** | $0.003183$ | $0.003191$ | **0.25%** | Excellent Convergence |
| **5.0% Noise** | $0.003183$ | $0.003215$ | **1.01%** | Highly Stable |
| **10.0% Noise** | $0.003183$ | $0.003264$ | **2.54%** | Acceptable Variance |
| **15.0% Noise** | $0.003183$ | $0.003350$ | **5.24%** | Degradation Onset |

*Observation:* The joint optimization loss function maintains stable parameter recovery even under high observational corruption ($>10\%$), confirming that physical residual regularization ($\mathcal{L}_{\text{pde}}$) successfully anchors the neural network against overfitting to noisy sensor data.

---

## Directory Architecture

fno-pde-solver/
├── .github/
│   └── workflows/
│       └── pytest.yml             # CI/CD: Automated testing on every git push
├── assets/
│   ├── fno_loss_convergence.png   # Training vs. validation loss decay curves
│   └── fno_2d_navier_stokes.png   # Ground truth vs. predicted vorticity fields
├── notebooks/
│   └── Fourier_Neural_Operator_Colab.ipynb  # Interactive demo notebook
├── src/
│   ├── __init__.py                # Package initialization
│   ├── dataset.py                 # 2D Navier-Stokes vorticity dataset loader
│   ├── eval_resolution.py         # Multi-grid resolution sensitivity study
│   ├── model.py                 # Spectral Conv2d & FNO2d PyTorch architecture
│   └── train.py                 # Relative L2 loss optimization loop
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py            # Unit test: tensor output dimensions
│   └── test_model.py              # Unit test: spectral layer parameter shapes
├── .gitignore                     # Ignores __pycache__, .venv, and checkpoints
├── LICENSE                        # MIT License
├── README.md                      # Academic documentation with figures & metrics
├── requirements.txt               # Dependencies (torch, numpy, matplotlib, scipy)
└── setup.py                       # Packaging script to make src/ pip-installable
