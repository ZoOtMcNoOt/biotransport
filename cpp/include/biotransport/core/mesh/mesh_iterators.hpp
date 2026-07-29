#ifndef BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_HPP
#define BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_HPP

/**
 * @file mesh_iterators.hpp
 * @brief Unified iteration patterns for structured meshes.
 *
 * Provides abstractions to eliminate duplicated 1D/2D branching logic
 * throughout the solver codebase. All iterators work transparently
 * for both 1D and 2D meshes.
 *
 * When compiled with BIOTRANSPORT_ENABLE_OPENMP, the parallel versions
 * of the iterators use OpenMP for multi-threaded execution.
 */

#include <algorithm>
#include <array>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#include <omp.h>
#endif

namespace biotransport {

/**
 * @brief Information about a node during iteration.
 */
struct NodeInfo {
    int idx;   ///< Global linear index
    int i;     ///< x-index
    int j;     ///< y-index (0 for 1D)
    double x;  ///< x-coordinate
    double y;  ///< y-coordinate (0 for 1D)
};

/**
 * @brief Stencil indices for a node (neighbors for finite differences).
 */
struct Stencil {
    int center;  ///< Center node index
    int west;    ///< West neighbor (i-1)
    int east;    ///< East neighbor (i+1)
    int south;   ///< South neighbor (j-1), -1 for 1D
    int north;   ///< North neighbor (j+1), -1 for 1D
};

/**
 * @brief Helper class for iterating over mesh nodes with unified 1D/2D handling.
 *
 * Eliminates the repeated pattern:
 * @code
 *   if (is_1d) {
 *       for (int i = 1; i < nx; ++i) { ... }
 *   } else {
 *       for (int j = 1; j < ny; ++j) {
 *           for (int i = 1; i < nx; ++i) { ... }
 *       }
 *   }
 * @endcode
 */
class MeshIterator {
public:
    explicit MeshIterator(const StructuredMesh& mesh) : mesh_(mesh) { stride_ = mesh_.nx() + 1; }

    /**
     * @brief Iterate over interior nodes (excludes boundaries).
     *
     * @param callback Function called for each interior node with (idx, i, j)
     *
     * When BIOTRANSPORT_ENABLE_OPENMP is defined, the 2D loop is parallelized
     * using OpenMP with static scheduling. The 1D case is not parallelized
     * due to low overhead.
     */
    template <typename Func>
    void forEachInterior(Func&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();

        if (mesh_.is1D()) {
            for (int i = 1; i < nx; ++i) {
                callback(i, i, 0);
            }
        } else {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 1; j < ny; ++j) {
                const int row = j * stride_;
                for (int i = 1; i < nx; ++i) {
                    callback(row + i, i, j);
                }
            }
        }
    }

    /**
     * @brief Iterate over interior nodes with full node info.
     *
     * @param callback Function called for each interior node with NodeInfo
     *
     * When BIOTRANSPORT_ENABLE_OPENMP is defined, the 2D loop is parallelized.
     */
    template <typename Func>
    void forEachInteriorWithCoords(Func&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();

        if (mesh_.is1D()) {
            for (int i = 1; i < nx; ++i) {
                NodeInfo info{i, i, 0, mesh_.x(i), 0.0};
                callback(info);
            }
        } else {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 1; j < ny; ++j) {
                const double y = mesh_.y(0, j);
                const int row = j * stride_;
                for (int i = 1; i < nx; ++i) {
                    NodeInfo info{row + i, i, j, mesh_.x(i), y};
                    callback(info);
                }
            }
        }
    }

    /**
     * @brief Iterate over all nodes (including boundaries).
     *
     * @param callback Function called for each node with (idx, i, j)
     *
     * When BIOTRANSPORT_ENABLE_OPENMP is defined, the 2D loop is parallelized.
     */
    template <typename Func>
    void forEachNode(Func&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();

        if (mesh_.is1D()) {
            for (int i = 0; i <= nx; ++i) {
                callback(i, i, 0);
            }
        } else {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j <= ny; ++j) {
                const int row = j * stride_;
                for (int i = 0; i <= nx; ++i) {
                    callback(row + i, i, j);
                }
            }
        }
    }

    /**
     * @brief Iterate over boundary nodes on a specific side.
     *
     * @param side Boundary side (Left, Right, Bottom, Top)
     * @param callback Function called for each boundary node with (idx, i, j)
     */
    template <typename Func>
    void forEachBoundary(Boundary side, Func&& callback) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();

        switch (side) {
            case Boundary::Left:
                if (mesh_.is1D()) {
                    callback(0, 0, 0);
                } else {
                    for (int j = 0; j <= ny; ++j) {
                        callback(j * stride_, 0, j);
                    }
                }
                break;

            case Boundary::Right:
                if (mesh_.is1D()) {
                    callback(nx, nx, 0);
                } else {
                    for (int j = 0; j <= ny; ++j) {
                        callback(j * stride_ + nx, nx, j);
                    }
                }
                break;

            case Boundary::Bottom:
                if (!mesh_.is1D()) {
                    for (int i = 0; i <= nx; ++i) {
                        callback(i, i, 0);
                    }
                }
                break;

            case Boundary::Top:
                if (!mesh_.is1D()) {
                    for (int i = 0; i <= nx; ++i) {
                        callback(ny * stride_ + i, i, ny);
                    }
                }
                break;
        }
    }

    /**
     * @brief Get stencil indices for a node.
     *
     * @param idx Global linear index
     * @return Stencil with neighbor indices
     */
    Stencil getStencil(int idx) const {
        Stencil s;
        s.center = idx;
        s.west = idx - 1;
        s.east = idx + 1;
        if (mesh_.is1D()) {
            s.south = -1;
            s.north = -1;
        } else {
            s.south = idx - stride_;
            s.north = idx + stride_;
        }
        return s;
    }

    /**
     * @brief Get the row stride for 2D indexing.
     */
    int stride() const { return stride_; }

private:
    const StructuredMesh& mesh_;
    int stride_;
};

/**
 * @brief Compute finite difference stencil operations on solution vectors.
 *
 * Provides unified Laplacian, gradient, and other stencil operations
 * that work for both 1D and 2D meshes.
 *
 * Supports 2nd-order, 4th-order, and 6th-order accurate discretizations:
 * - 2nd-order: 3-point stencil, O(dx²) error
 * - 4th-order: 5-point stencil, O(dx⁴) error
 * - 6th-order: 7-point stencil, O(dx⁶) error (1D only)
 */
class StencilOps {
public:
    explicit StencilOps(const StructuredMesh& mesh)
        : mesh_(mesh),
          inv_dx2_(1.0 / (mesh.dx() * mesh.dx())),
          inv_dy2_(mesh.is1D() ? 0.0 : 1.0 / (mesh.dy() * mesh.dy())),
          inv_2dx_(1.0 / (2.0 * mesh.dx())),
          inv_2dy_(mesh.is1D() ? 0.0 : 1.0 / (2.0 * mesh.dy())),
          inv_12dx2_(1.0 / (12.0 * mesh.dx() * mesh.dx())),
          inv_12dy2_(mesh.is1D() ? 0.0 : 1.0 / (12.0 * mesh.dy() * mesh.dy())),
          inv_180dx2_(1.0 / (180.0 * mesh.dx() * mesh.dx())),
          inv_12dx_(1.0 / (12.0 * mesh.dx())),
          inv_12dy_(mesh.is1D() ? 0.0 : 1.0 / (12.0 * mesh.dy())),
          stride_(mesh.nx() + 1) {}

    /**
     * @brief Compute the discrete Laplacian at a node.
     *
     * ∇²u ≈ (u[i+1] - 2*u[i] + u[i-1])/dx² + (u[j+1] - 2*u[j] + u[j-1])/dy²
     *
     * @param u Solution vector
     * @param idx Node index
     * @return Laplacian value
     */
    double laplacian(const std::vector<double>& u, int idx) const {
        double lap = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) * inv_dx2_;
        if (!mesh_.is1D()) {
            lap += (u[idx + stride_] - 2.0 * u[idx] + u[idx - stride_]) * inv_dy2_;
        }
        return lap;
    }

    /**
     * @brief Compute the 4th-order accurate discrete Laplacian at a node.
     *
     * d²u/dx² ≈ (-u[i+2] + 16*u[i+1] - 30*u[i] + 16*u[i-1] - u[i-2]) / (12*dx²)
     *
     * Truncation error: O(dx⁴)
     *
     * Note: Requires at least 2 nodes of padding on each side.
     *
     * @param u Solution vector
     * @param idx Node index
     * @return 4th-order Laplacian value
     */
    double laplacian4thOrder(const std::vector<double>& u, int idx) const {
        double lap =
            (-u[idx + 2] + 16.0 * u[idx + 1] - 30.0 * u[idx] + 16.0 * u[idx - 1] - u[idx - 2]) *
            inv_12dx2_;
        if (!mesh_.is1D()) {
            lap += (-u[idx + 2 * stride_] + 16.0 * u[idx + stride_] - 30.0 * u[idx] +
                    16.0 * u[idx - stride_] - u[idx - 2 * stride_]) *
                   inv_12dy2_;
        }
        return lap;
    }

    /**
     * @brief Compute the 6th-order accurate discrete Laplacian at a node (1D only).
     *
     * d²u/dx² ≈ (2*u[i+3] - 27*u[i+2] + 270*u[i+1] - 490*u[i]
     *           + 270*u[i-1] - 27*u[i-2] + 2*u[i-3]) / (180*dx²)
     *
     * Truncation error: O(dx⁶)
     *
     * Note: Only implemented for 1D meshes. Requires at least 3 nodes padding.
     *
     * @param u Solution vector
     * @param idx Node index
     * @return 6th-order Laplacian value
     */
    double laplacian6thOrder(const std::vector<double>& u, int idx) const {
        return (2.0 * u[idx + 3] - 27.0 * u[idx + 2] + 270.0 * u[idx + 1] - 490.0 * u[idx] +
                270.0 * u[idx - 1] - 27.0 * u[idx - 2] + 2.0 * u[idx - 3]) *
               inv_180dx2_;
    }

    /**
     * @brief Compute diffusion term contribution: D * ∇²u * dt
     *
     * @param u Solution vector
     * @param idx Node index
     * @param D Diffusion coefficient
     * @param dt Time step
     * @return Diffusion contribution to update
     */
    double diffusionTerm(const std::vector<double>& u, int idx, double D, double dt) const {
        const double x_contribution =
            scaledStencilContribution(std::array<double, 3>{u[idx - 1], u[idx], u[idx + 1]},
                                      std::array<double, 3>{1.0, -2.0, 1.0}, D, dt, mesh_.dx());
        if (mesh_.is1D()) {
            return x_contribution;
        }
        const double y_contribution = scaledStencilContribution(
            std::array<double, 3>{u[idx - stride_], u[idx], u[idx + stride_]},
            std::array<double, 3>{1.0, -2.0, 1.0}, D, dt, mesh_.dy());
        return x_contribution + y_contribution;
    }

    /**
     * @brief Apply one Forward Euler diffusion update without false overflow.
     *
     * The usual center-plus-increment form is retained on ordinary scales.
     * If its intermediate increment overflows even though the CFL-stable
     * weighted state is representable, the equivalent convex stencil is
     * evaluated after normalizing the state.
     */
    double diffusionStep(const std::vector<double>& u, int idx, double D, double dt) const {
        const double candidate = u[idx] + diffusionTerm(u, idx, D, dt);
        if (std::isfinite(candidate)) {
            return candidate;
        }

        const double lambda_x = diffusionNumber(D, dt, mesh_.dx());
        const double lambda_y = mesh_.is1D() ? 0.0 : diffusionNumber(D, dt, mesh_.dy());
        const double total_lambda = lambda_x + lambda_y;
        if (!std::isfinite(total_lambda) || total_lambda < 0.0 || total_lambda > 0.5) {
            return candidate;
        }

        double value_scale =
            std::max({std::abs(u[idx - 1]), std::abs(u[idx]), std::abs(u[idx + 1])});
        if (!mesh_.is1D()) {
            value_scale =
                std::max({value_scale, std::abs(u[idx - stride_]), std::abs(u[idx + stride_])});
        }
        if (value_scale == 0.0) {
            return 0.0;
        }

        double normalized = (1.0 - 2.0 * total_lambda) * (u[idx] / value_scale);
        normalized = std::fma(lambda_x, u[idx - 1] / value_scale, normalized);
        normalized = std::fma(lambda_x, u[idx + 1] / value_scale, normalized);
        if (!mesh_.is1D()) {
            normalized = std::fma(lambda_y, u[idx - stride_] / value_scale, normalized);
            normalized = std::fma(lambda_y, u[idx + stride_] / value_scale, normalized);
        }
        return scaledProduct(value_scale, normalized);
    }

    /**
     * @brief Compute the dimensionless diffusion number D*dt/h² safely.
     *
     * Binary exponent decomposition avoids overflowing h² or underflowing
     * D*dt when the final ratio is representable.
     */
    static double diffusionNumber(double D, double dt, double spacing) {
        if (!std::isfinite(D) || !std::isfinite(dt) || !std::isfinite(spacing) || D < 0.0 ||
            dt < 0.0 || spacing <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (D == 0.0 || dt == 0.0) {
            return 0.0;
        }

        int diffusivity_exponent = 0;
        int time_exponent = 0;
        int spacing_exponent = 0;
        const double diffusivity_mantissa = std::frexp(D, &diffusivity_exponent);
        const double time_mantissa = std::frexp(dt, &time_exponent);
        const double spacing_mantissa = std::frexp(spacing, &spacing_exponent);
        const double mantissa =
            diffusivity_mantissa * time_mantissa / (spacing_mantissa * spacing_mantissa);
        return std::scalbn(mantissa, diffusivity_exponent + time_exponent - 2 * spacing_exponent);
    }

    /**
     * @brief Compute diffusion term with spatially-varying diffusivity D(x).
     *
     * Uses flux-form discretization with face-averaged diffusivity:
     *   ∂/∂x(D(x) ∂u/∂x) ≈ (D_{i+1/2}(u_{i+1}-u_i) - D_{i-1/2}(u_i-u_{i-1})) / dx²
     *
     * where D_{i+1/2} = 0.5 * (D[i] + D[i+1]) is the harmonic-style average at faces.
     *
     * @param u Solution vector
     * @param D Spatially-varying diffusion coefficient field
     * @param idx Node index
     * @param dt Time step
     * @return Diffusion contribution to update
     */
    double variableDiffusionTerm(const std::vector<double>& u, const std::vector<double>& D,
                                 int idx, double dt) const {
        // Face-averaged diffusivities (arithmetic mean)
        double D_plus = 0.5 * (D[idx] + D[idx + 1]);
        double D_minus = 0.5 * (D[idx - 1] + D[idx]);

        // Flux-form: d/dx(D du/dx) = (D_{i+1/2}(u_{i+1}-u_i) - D_{i-1/2}(u_i-u_{i-1})) / dx²
        double flux_x =
            (D_plus * (u[idx + 1] - u[idx]) - D_minus * (u[idx] - u[idx - 1])) * inv_dx2_;

        if (!mesh_.is1D()) {
            // Same for y-direction
            double D_top = 0.5 * (D[idx] + D[idx + stride_]);
            double D_bottom = 0.5 * (D[idx - stride_] + D[idx]);
            double flux_y =
                (D_top * (u[idx + stride_] - u[idx]) - D_bottom * (u[idx] - u[idx - stride_])) *
                inv_dy2_;
            flux_x += flux_y;
        }

        return dt * flux_x;
    }

    /**
     * @brief Compute x-gradient at a node using central differences.
     *
     * @param u Solution vector
     * @param idx Node index
     * @return du/dx
     */
    double gradX(const std::vector<double>& u, int idx) const {
        return (u[idx + 1] - u[idx - 1]) * inv_2dx_;
    }

    /**
     * @brief Compute y-gradient at a node using central differences.
     *
     * @param u Solution vector
     * @param idx Node index
     * @return du/dy (0 for 1D)
     */
    double gradY(const std::vector<double>& u, int idx) const {
        if (mesh_.is1D())
            return 0.0;
        return (u[idx + stride_] - u[idx - stride_]) * inv_2dy_;
    }

    /**
     * @brief Compute 4th-order x-gradient at a node.
     *
     * du/dx ≈ (-u[i+2] + 8*u[i+1] - 8*u[i-1] + u[i-2]) / (12*dx)
     *
     * Truncation error: O(dx⁴)
     *
     * @param u Solution vector
     * @param idx Node index
     * @return 4th-order du/dx
     */
    double gradX4thOrder(const std::vector<double>& u, int idx) const {
        return (-u[idx + 2] + 8.0 * u[idx + 1] - 8.0 * u[idx - 1] + u[idx - 2]) * inv_12dx_;
    }

    /**
     * @brief Compute 4th-order y-gradient at a node.
     *
     * du/dy ≈ (-u[j+2] + 8*u[j+1] - 8*u[j-1] + u[j-2]) / (12*dy)
     *
     * Truncation error: O(dx⁴)
     *
     * @param u Solution vector
     * @param idx Node index
     * @return 4th-order du/dy (0 for 1D)
     */
    double gradY4thOrder(const std::vector<double>& u, int idx) const {
        if (mesh_.is1D())
            return 0.0;
        return (-u[idx + 2 * stride_] + 8.0 * u[idx + stride_] - 8.0 * u[idx - stride_] +
                u[idx - 2 * stride_]) *
               inv_12dy_;
    }

    /**
     * @brief Compute upwind derivative in x-direction.
     *
     * @param u Solution vector
     * @param idx Node index
     * @param velocity x-velocity at node
     * @return Upwind du/dx
     */
    double upwindGradX(const std::vector<double>& u, int idx, double velocity) const {
        if (velocity > 0.0) {
            return (u[idx] - u[idx - 1]) / mesh_.dx();
        } else {
            return (u[idx + 1] - u[idx]) / mesh_.dx();
        }
    }

    /**
     * @brief Compute upwind derivative in y-direction.
     *
     * @param u Solution vector
     * @param idx Node index
     * @param velocity y-velocity at node
     * @return Upwind du/dy (0 for 1D)
     */
    double upwindGradY(const std::vector<double>& u, int idx, double velocity) const {
        if (mesh_.is1D())
            return 0.0;
        if (velocity > 0.0) {
            return (u[idx] - u[idx - stride_]) / mesh_.dy();
        } else {
            return (u[idx + stride_] - u[idx]) / mesh_.dy();
        }
    }

    // =========================================================================
    // Bulk Array Operations
    // =========================================================================

    /**
     * @brief Apply 4th-order Laplacian to entire 1D array.
     *
     * Uses 4th-order stencil in interior, falls back to 2nd-order at boundaries.
     *
     * @param u Input array
     * @return Laplacian output array
     */
    std::vector<double> laplacian4thOrderBulk1D(const std::vector<double>& u) const {
        const int n = static_cast<int>(u.size());
        std::vector<double> lap(n, 0.0);

        // 4th-order interior (needs i-2 to i+2)
        for (int i = 2; i < n - 2; ++i) {
            lap[i] = (-u[i + 2] + 16.0 * u[i + 1] - 30.0 * u[i] + 16.0 * u[i - 1] - u[i - 2]) *
                     inv_12dx2_;
        }

        // 2nd-order at boundaries
        if (n > 2) {
            lap[1] = (u[2] - 2.0 * u[1] + u[0]) * inv_dx2_;
            lap[n - 2] = (u[n - 1] - 2.0 * u[n - 2] + u[n - 3]) * inv_dx2_;
        }

        return lap;
    }

    /**
     * @brief Apply 6th-order Laplacian to entire 1D array.
     *
     * Uses 6th-order in deep interior, 4th-order in transition, 2nd-order near boundary.
     *
     * @param u Input array
     * @return Laplacian output array
     */
    std::vector<double> laplacian6thOrderBulk1D(const std::vector<double>& u) const {
        const int n = static_cast<int>(u.size());
        std::vector<double> lap(n, 0.0);

        // 6th-order deep interior (needs i-3 to i+3)
        for (int i = 3; i < n - 3; ++i) {
            lap[i] = (2.0 * u[i + 3] - 27.0 * u[i + 2] + 270.0 * u[i + 1] - 490.0 * u[i] +
                      270.0 * u[i - 1] - 27.0 * u[i - 2] + 2.0 * u[i - 3]) *
                     inv_180dx2_;
        }

        // 4th-order at i=2 and i=n-3
        if (n > 4) {
            for (int i : {2, n - 3}) {
                if (i >= 2 && i <= n - 3) {
                    lap[i] =
                        (-u[i + 2] + 16.0 * u[i + 1] - 30.0 * u[i] + 16.0 * u[i - 1] - u[i - 2]) *
                        inv_12dx2_;
                }
            }
        }

        // 2nd-order at i=1 and i=n-2
        if (n > 2) {
            lap[1] = (u[2] - 2.0 * u[1] + u[0]) * inv_dx2_;
            lap[n - 2] = (u[n - 1] - 2.0 * u[n - 2] + u[n - 3]) * inv_dx2_;
        }

        return lap;
    }

    /**
     * @brief Apply 4th-order Laplacian to entire 2D array (row-major).
     *
     * Uses 4th-order stencil in interior, falls back to 2nd-order near boundaries.
     *
     * @param u Input array (row-major, size = (ny+1) * (nx+1))
     * @return Laplacian output array
     */
    std::vector<double> laplacian4thOrderBulk2D(const std::vector<double>& u) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        std::vector<double> lap(u.size(), 0.0);

        // 4th-order interior (needs 2-cell padding)
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 2; j < ny - 1; ++j) {
            for (int i = 2; i < nx - 1; ++i) {
                const int idx = j * stride_ + i;
                const double lap_x = (-u[idx + 2] + 16.0 * u[idx + 1] - 30.0 * u[idx] +
                                      16.0 * u[idx - 1] - u[idx - 2]) *
                                     inv_12dx2_;
                const double lap_y =
                    (-u[idx + 2 * stride_] + 16.0 * u[idx + stride_] - 30.0 * u[idx] +
                     16.0 * u[idx - stride_] - u[idx - 2 * stride_]) *
                    inv_12dy2_;
                lap[idx] = lap_x + lap_y;
            }
        }

        // 2nd-order in transition zone (1 cell from boundary)
        // j=1 and j=ny-1 rows
        for (int j : {1, ny - 1}) {
            for (int i = 1; i < nx; ++i) {
                const int idx = j * stride_ + i;
                lap[idx] = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) * inv_dx2_ +
                           (u[idx + stride_] - 2.0 * u[idx] + u[idx - stride_]) * inv_dy2_;
            }
        }
        // i=1 and i=nx-1 columns (excluding corners already done)
        for (int i : {1, nx - 1}) {
            for (int j = 2; j < ny - 1; ++j) {
                const int idx = j * stride_ + i;
                lap[idx] = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) * inv_dx2_ +
                           (u[idx + stride_] - 2.0 * u[idx] + u[idx - stride_]) * inv_dy2_;
            }
        }

        return lap;
    }

    /**
     * @brief Apply 4th-order gradient to entire 1D array.
     *
     * Uses 4th-order stencil in interior, falls back to 2nd-order at boundaries.
     *
     * @param u Input array
     * @return Gradient output array
     */
    std::vector<double> gradient4thOrderBulk1D(const std::vector<double>& u) const {
        const int n = static_cast<int>(u.size());
        std::vector<double> grad(n, 0.0);

        // 4th-order interior (needs i-2 to i+2)
        for (int i = 2; i < n - 2; ++i) {
            grad[i] = (-u[i + 2] + 8.0 * u[i + 1] - 8.0 * u[i - 1] + u[i - 2]) * inv_12dx_;
        }

        // 2nd-order at boundaries
        if (n > 2) {
            grad[1] = (u[2] - u[0]) * inv_2dx_;
            grad[n - 2] = (u[n - 1] - u[n - 3]) * inv_2dx_;
        }

        return grad;
    }

    // Accessor for precomputed values
    double invDx2() const { return inv_dx2_; }
    double invDy2() const { return inv_dy2_; }
    double inv12Dx2() const { return inv_12dx2_; }
    double inv12Dy2() const { return inv_12dy2_; }
    double inv180Dx2() const { return inv_180dx2_; }
    int stride() const { return stride_; }

private:
    static double scaledProduct(double first, double second) {
        if (first == 0.0 || second == 0.0) {
            return 0.0;
        }
        int first_exponent = 0;
        int second_exponent = 0;
        const double mantissa =
            std::frexp(first, &first_exponent) * std::frexp(second, &second_exponent);
        return std::scalbn(mantissa, first_exponent + second_exponent);
    }

    template <std::size_t N>
    static double scaledStencilContribution(const std::array<double, N>& values,
                                            const std::array<double, N>& coefficients, double D,
                                            double dt, double spacing) {
        double value_scale = 0.0;
        for (double value : values) {
            value_scale = std::max(value_scale, std::abs(value));
        }
        if (value_scale == 0.0 || std::all_of(values.begin() + 1, values.end(), [&](double value) {
                return value == values.front();
            })) {
            return 0.0;
        }

        int scale_exponent = 0;
        (void)std::frexp(value_scale, &scale_exponent);
        double normalized_sum = 0.0;
        for (std::size_t index = 0; index < N; ++index) {
            normalized_sum = std::fma(coefficients[index],
                                      std::scalbn(values[index], -scale_exponent), normalized_sum);
        }
        if (normalized_sum == 0.0 || D == 0.0 || dt == 0.0) {
            return 0.0;
        }

        int sum_exponent = 0;
        int diffusivity_exponent = 0;
        int time_exponent = 0;
        int spacing_exponent = 0;
        const double sum_mantissa = std::frexp(normalized_sum, &sum_exponent);
        const double diffusivity_mantissa = std::frexp(D, &diffusivity_exponent);
        const double time_mantissa = std::frexp(dt, &time_exponent);
        const double spacing_mantissa = std::frexp(spacing, &spacing_exponent);
        const double mantissa = sum_mantissa * diffusivity_mantissa * time_mantissa /
                                (spacing_mantissa * spacing_mantissa);
        return std::scalbn(mantissa, sum_exponent + scale_exponent + diffusivity_exponent +
                                         time_exponent - 2 * spacing_exponent);
    }

    const StructuredMesh& mesh_;
    double inv_dx2_;
    double inv_dy2_;
    double inv_2dx_;
    double inv_2dy_;
    double inv_12dx2_;   ///< 1/(12*dx²) for 4th-order Laplacian
    double inv_12dy2_;   ///< 1/(12*dy²) for 4th-order Laplacian
    double inv_180dx2_;  ///< 1/(180*dx²) for 6th-order Laplacian
    double inv_12dx_;    ///< 1/(12*dx) for 4th-order gradient
    double inv_12dy_;    ///< 1/(12*dy) for 4th-order gradient
    int stride_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_MESH_ITERATORS_HPP
