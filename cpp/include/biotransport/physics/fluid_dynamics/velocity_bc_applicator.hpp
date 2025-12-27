/**
 * @file velocity_bc_applicator.hpp
 * @brief Utility for applying velocity boundary conditions.
 *
 * Extracts common boundary condition application logic used by
 * StokesSolver and NavierStokesSolver to avoid code duplication.
 */

#ifndef BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_VELOCITY_BC_APPLICATOR_HPP
#define BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_VELOCITY_BC_APPLICATOR_HPP

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <functional>
#include <vector>

namespace biotransport {

/**
 * @brief Applies velocity boundary conditions to u and v fields.
 *
 * This utility extracts the common BC application pattern shared by
 * Stokes and Navier-Stokes solvers. Supports:
 * - NOSLIP: u = v = 0
 * - DIRICHLET: u = u_value, v = v_value
 * - INFLOW: use profile functions or constant values
 * - OUTFLOW/NEUMANN: zero gradient (extrapolate from interior)
 *
 * @param mesh The structured mesh
 * @param velocity_bcs Array of VelocityBC for [Left, Right, Bottom, Top]
 * @param u_inlet Optional inlet profile functions for u
 * @param v_inlet Optional inlet profile functions for v
 * @param u Velocity field in x (modified)
 * @param v Velocity field in y (modified)
 */
inline void applyVelocityBoundaryConditions(
    const StructuredMesh& mesh, const std::array<VelocityBC, 4>& velocity_bcs,
    const std::array<std::function<double(double, double)>, 4>& u_inlet,
    const std::array<std::function<double(double, double)>, 4>& v_inlet, std::vector<double>& u,
    std::vector<double>& v) {
    int nx = mesh.nx();
    int ny = mesh.ny();
    int stride = nx + 1;

    // Lambda to apply BC for a single node
    auto apply_bc = [&](int idx, int bc_idx, double x, double y, int interior_offset) {
        const auto& bc = velocity_bcs[bc_idx];
        switch (bc.type) {
            case VelocityBCType::NOSLIP:
                u[idx] = 0.0;
                v[idx] = 0.0;
                break;
            case VelocityBCType::DIRICHLET:
                u[idx] = bc.u_value;
                v[idx] = bc.v_value;
                break;
            case VelocityBCType::INFLOW:
                if (u_inlet[bc_idx]) {
                    u[idx] = u_inlet[bc_idx](x, y);
                    v[idx] = v_inlet[bc_idx] ? v_inlet[bc_idx](x, y) : bc.v_value;
                } else {
                    u[idx] = bc.u_value;
                    v[idx] = bc.v_value;
                }
                break;
            case VelocityBCType::OUTFLOW:
            case VelocityBCType::NEUMANN:
                // Zero gradient: copy from interior
                u[idx] = u[idx + interior_offset];
                v[idx] = v[idx + interior_offset];
                break;
        }
    };

    // Left boundary (i = 0)
    for (int j = 0; j <= ny; ++j) {
        int idx = j * stride;
        double x = mesh.x(0);
        double y = mesh.y(0, j);
        apply_bc(idx, to_index(Boundary::Left), x, y, +1);
    }

    // Right boundary (i = nx)
    for (int j = 0; j <= ny; ++j) {
        int idx = j * stride + nx;
        double x = mesh.x(nx);
        double y = mesh.y(nx, j);
        apply_bc(idx, to_index(Boundary::Right), x, y, -1);
    }

    // Bottom boundary (j = 0)
    for (int i = 0; i <= nx; ++i) {
        int idx = i;
        double x = mesh.x(i);
        double y = mesh.y(i, 0);
        apply_bc(idx, to_index(Boundary::Bottom), x, y, +stride);
    }

    // Top boundary (j = ny)
    for (int i = 0; i <= nx; ++i) {
        int idx = ny * stride + i;
        double x = mesh.x(i);
        double y = mesh.y(i, ny);
        apply_bc(idx, to_index(Boundary::Top), x, y, -stride);
    }
}

/**
 * @brief Simplified version without inlet profile functions.
 *
 * Uses constant values from VelocityBC for all boundary types.
 */
inline void applyVelocityBoundaryConditions(const StructuredMesh& mesh,
                                            const std::array<VelocityBC, 4>& velocity_bcs,
                                            std::vector<double>& u, std::vector<double>& v) {
    std::array<std::function<double(double, double)>, 4> no_inlet{};
    applyVelocityBoundaryConditions(mesh, velocity_bcs, no_inlet, no_inlet, u, v);
}

/**
 * @brief Apply Neumann (zero-gradient) pressure boundary conditions.
 *
 * Common pattern for incompressible flow solvers.
 *
 * @param mesh The structured mesh
 * @param p Pressure field (modified)
 */
inline void applyPressureNeumannBCs(const StructuredMesh& mesh, std::vector<double>& p) {
    int nx = mesh.nx();
    int ny = mesh.ny();
    int stride = nx + 1;

    // Left/Right boundaries
    for (int j = 0; j <= ny; ++j) {
        p[j * stride] = p[j * stride + 1];            // Left: p(0,j) = p(1,j)
        p[j * stride + nx] = p[j * stride + nx - 1];  // Right: p(nx,j) = p(nx-1,j)
    }

    // Bottom/Top boundaries
    for (int i = 0; i <= nx; ++i) {
        p[i] = p[stride + i];                           // Bottom: p(i,0) = p(i,1)
        p[ny * stride + i] = p[(ny - 1) * stride + i];  // Top: p(i,ny) = p(i,ny-1)
    }
}

/**
 * @brief Subtract mean pressure to avoid drift.
 *
 * Required when using pure Neumann BCs for pressure.
 *
 * @param mesh The structured mesh
 * @param p Pressure field (modified)
 */
inline void subtractMeanPressure(const StructuredMesh& mesh, std::vector<double>& p) {
    int nx = mesh.nx();
    int ny = mesh.ny();
    int stride = nx + 1;

    double p_mean = 0.0;
    int count = 0;
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            p_mean += p[j * stride + i];
            ++count;
        }
    }

    if (count > 0) {
        p_mean /= count;
        for (size_t i = 0; i < p.size(); ++i) {
            p[i] -= p_mean;
        }
    }
}

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_VELOCITY_BC_APPLICATOR_HPP
