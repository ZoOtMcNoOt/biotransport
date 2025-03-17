# BioTransport Library

A high-performance C++ library with Python bindings for modeling biotransport phenomena in biological systems.

## Overview

BioTransport is designed to simulate various transport processes that occur in biological systems, such as diffusion, convection, and reaction-diffusion. It provides a solid foundation for implementing custom biotransport models with a focus on performance and ease of use.

## Features

- C++ core for high performance computations
- Python bindings for accessibility and easy visualization
- Structured mesh generation for 1D and 2D domains
- Solvers for diffusion and reaction-diffusion equations
- Boundary condition handling (Dirichlet, Neumann)
- Visualization tools for solution analysis

## Installation

### Prerequisites

- CMake (>= 3.13)
- C++ compiler with C++14 support
- Python 3.6 or newer
- NumPy
- Matplotlib

### Building from Source

```bash
# Clone the repository
git clone https://github.com/ZoOtMcNoOt/biotransport.git
cd biotransport

# Build and install with pip
pip install -e .