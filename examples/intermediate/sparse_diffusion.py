#!/usr/bin/env python3
"""
Sparse Matrix Implicit Diffusion Example
=========================================

Demonstrates the sparse matrix interface and fully implicit diffusion solver.
Compares computational approaches:
1. Explicit method (stability-limited)
2. ADI method (unconditionally stable, O(N))
3. Fully implicit with sparse LU (unconditionally stable)

The fully implicit solver is slower than ADI for regular grids but supports:
- Spatially-varying diffusivity
- Complex boundary conditions
- Non-rectangular domains (with extensions)
"""

import numpy as np
import matplotlib.pyplot as plt
import biotransport as bt
import time

# =============================================================================
# Problem Setup
# =============================================================================

# Domain
Lx, Ly = 1.0, 1.0
nx, ny = 50, 50

# Physical parameters
D = 0.01  # Diffusion coefficient [m²/s]
T_final = 0.5  # Final time [s]

# Create mesh
mesh = bt.StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)
dx = Lx / nx
dy = Ly / ny

# Initial condition: Gaussian pulse at center
x = np.linspace(0, Lx, nx + 1)
y = np.linspace(0, Ly, ny + 1)
X, Y = np.meshgrid(x, y)

x0, y0 = 0.5, 0.5
sigma = 0.1
initial_condition = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
ic_flat = initial_condition.flatten()

# =============================================================================
# Method 1: Fully Implicit with Sparse Matrix
# =============================================================================

print("=" * 60)
print("Sparse Matrix Implicit Diffusion Solver")
print("=" * 60)

# Create implicit solver
implicit_solver = bt.ImplicitDiffusion2D(mesh, D)
implicit_solver.set_initial_condition(ic_flat.tolist())

# Use different solver types
solver_types = [
    (bt.SparseSolverType.SparseLU, "SparseLU (direct)"),
    (bt.SparseSolverType.ConjugateGradient, "Conjugate Gradient (iterative)"),
]

# Take large time step (stable regardless of CFL)
dt_implicit = 0.05
num_steps_implicit = int(T_final / dt_implicit)

for solver_type, name in solver_types:
    # Reset solver
    implicit_solver = bt.ImplicitDiffusion2D(mesh, D)
    implicit_solver.set_initial_condition(ic_flat.tolist())
    implicit_solver.set_solver_type(solver_type)

    start = time.perf_counter()
    result = implicit_solver.solve(dt_implicit, num_steps_implicit)
    elapsed = time.perf_counter() - start

    print(f"\n{name}:")
    print(
        f"  Time step: {dt_implicit:.4f} s (dt/dt_explicit = {dt_implicit / (dx**2 / (4 * D)):.1f})"
    )
    print(f"  Steps: {result.steps}")
    print(f"  Wall time: {elapsed:.3f} s")
    if result.steps > 0:
        print(f"  Time per step: {1000 * elapsed / result.steps:.2f} ms")
    print(f"  Success: {result.success}")
    if not result.success:
        print("  Note: Iterative solver may need more iterations for larger grids")

# =============================================================================
# Method 2: ADI Method (for comparison)
# =============================================================================

print("\n" + "-" * 60)
print("ADI Comparison")
print("-" * 60)

adi_solver = bt.ADIDiffusion2D(mesh, D)
adi_solver.set_initial_condition(ic_flat.tolist())

start = time.perf_counter()
adi_result = adi_solver.solve(dt_implicit, num_steps_implicit)
elapsed_adi = time.perf_counter() - start

print("\nADI (Thomas algorithm):")
print(f"  Time step: {dt_implicit:.4f} s")
print(f"  Steps: {adi_result.steps}")
print(f"  Wall time: {elapsed_adi:.3f} s")
print(f"  Time per step: {1000 * elapsed_adi / adi_result.steps:.2f} ms")

# =============================================================================
# Direct Sparse Matrix Usage Example
# =============================================================================

print("\n" + "=" * 60)
print("Direct Sparse Matrix Usage")
print("=" * 60)

# Build a 2D Laplacian matrix directly
n = 20
A = bt.build_2d_laplacian(n, n, dx, dy)
print("\n2D Laplacian matrix:")
print(f"  Size: {A.rows} x {A.cols}")
print(f"  Non-zeros: {A.nnz}")
print(f"  Fill ratio: {100 * A.nnz / (A.rows * A.cols):.2f}%")

# Create RHS (source term)
b = [1.0] * A.rows

# Solve with different methods
for solver_type in [
    bt.SparseSolverType.SparseLU,
    bt.SparseSolverType.ConjugateGradient,
    bt.SparseSolverType.BiCGSTAB,
]:
    x = A.solve(b, solver_type)
    residual = A.multiply(x)
    residual = [residual[i] - b[i] for i in range(len(b))]
    res_norm = sum(r**2 for r in residual) ** 0.5
    print(f"  {solver_type.name}: residual = {res_norm:.2e}")

# =============================================================================
# Build Custom Implicit Matrix
# =============================================================================

print("\n" + "=" * 60)
print("Custom Implicit Diffusion Matrix")
print("=" * 60)

alpha = D
dt = 0.01
A_impl = bt.build_implicit_diffusion_2d(n, n, dx, dy, alpha, dt)
print("\nImplicit diffusion matrix (I - alpha*dt*∇²):")
print(f"  Size: {A_impl.rows} x {A_impl.cols}")
print(f"  Non-zeros: {A_impl.nnz}")
print("  Diagonal dominance guaranteed: yes")

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Initial condition
ax = axes[0]
im = ax.pcolormesh(X, Y, initial_condition, shading="auto", cmap="hot")
ax.set_title("Initial Condition")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax)

# Implicit solution
implicit_final = np.array(implicit_solver.solution()).reshape((ny + 1, nx + 1))
ax = axes[1]
im = ax.pcolormesh(X, Y, implicit_final, shading="auto", cmap="hot")
ax.set_title(f"Implicit (t = {T_final} s)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax)

# ADI solution
adi_final = np.array(adi_solver.solution()).reshape((ny + 1, nx + 1))
ax = axes[2]
im = ax.pcolormesh(X, Y, adi_final, shading="auto", cmap="hot")
ax.set_title(f"ADI (t = {T_final} s)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("results/sparse_diffusion.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# Convergence Study with Grid Refinement
# =============================================================================

print("\n" + "=" * 60)
print("Grid Convergence Study")
print("=" * 60)

grid_sizes = [10, 20, 40, 80]
errors = []

for n in grid_sizes:
    mesh_n = bt.StructuredMesh(n, n, 0.0, Lx, 0.0, Ly)
    dx_n = Lx / n

    # Smaller problem for analytical comparison
    t_test = 0.1
    dt_n = t_test  # Single large step

    # Initial condition
    x_n = np.linspace(0, Lx, n + 1)
    y_n = np.linspace(0, Ly, n + 1)
    X_n, Y_n = np.meshgrid(x_n, y_n)
    ic_n = np.exp(-((X_n - 0.5) ** 2 + (Y_n - 0.5) ** 2) / (2 * sigma**2))

    # Solve
    solver = bt.ImplicitDiffusion2D(mesh_n, D)
    solver.set_initial_condition(ic_n.flatten().tolist())
    solver.set_solver_type(bt.SparseSolverType.SparseLU)
    solver.solve(dt_n, 1)

    # Analytical (approximate for Gaussian diffusion)
    sigma_t = np.sqrt(sigma**2 + 2 * D * t_test)
    analytical = (sigma**2 / sigma_t**2) * np.exp(
        -((X_n - 0.5) ** 2 + (Y_n - 0.5) ** 2) / (2 * sigma_t**2)
    )

    numerical = np.array(solver.solution()).reshape((n + 1, n + 1))
    error = np.max(np.abs(numerical - analytical))
    errors.append(error)
    print(f"  n = {n:3d}: error = {error:.4e}")

# Estimate convergence order
orders = []
for i in range(1, len(grid_sizes)):
    order = np.log(errors[i - 1] / errors[i]) / np.log(
        grid_sizes[i] / grid_sizes[i - 1]
    )
    orders.append(order)
print(f"\n  Convergence orders: {[f'{o:.2f}' for o in orders]}")

print("\n" + "=" * 60)
print("Example complete!")
print("=" * 60)
