#ifndef BIOTRANSPORT_SOLVERS_SOLVER_BASE_HPP
#define BIOTRANSPORT_SOLVERS_SOLVER_BASE_HPP

/**
 * @file solver_base.hpp
 * @brief CRTP base class for finite difference solvers.
 *
 * This header provides a unified template-based framework for time-stepping
 * solvers, eliminating the massive code duplication across diffusion,
 * reaction-diffusion, and advection-diffusion solvers.
 *
 * The key insight is that all these solvers share the same structure:
 *   1. Setup (check stability, resize scratch buffers)
 *   2. Time loop
 *      a. Compute spatial terms (diffusion, advection)
 *      b. Add physics-specific terms (reaction, source)
 *      c. Apply boundary conditions
 *      d. Swap buffers
 *   3. Cleanup
 *
 * Using CRTP, derived classes only need to implement their specific physics
 * via a `computeUpdate()` method, while all the boilerplate is handled here.
 */

#include <algorithm>
#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/mesh_iterators.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace biotransport {

/**
 * @brief CRTP base class for explicit time-stepping solvers.
 *
 * @tparam Derived The derived solver class (CRTP pattern)
 *
 * Derived classes must implement:
 * - `void computeNodeUpdate(int idx, const StencilOps& ops, double dt)`
 *   Sets scratch_[idx] to the new value for that node.
 *
 * Optionally override:
 * - `bool checkStabilityDerived(double dt) const` for custom stability checks
 * - `void preStep(int step, double dt)` called before each time step
 * - `void postStep(int step, double dt)` called after each time step
 */
template <typename Derived>
class ExplicitSolverBase {
public:
    ExplicitSolverBase(const StructuredMesh& mesh, double diffusivity)
        : mesh_(mesh), diffusivity_(diffusivity), iterator_(mesh), stencil_ops_(mesh) {
        if (!std::isfinite(diffusivity) || diffusivity <= 0.0) {
            throw std::invalid_argument("Diffusivity must be finite and positive");
        }
        solution_.resize(mesh.numNodes(), 0.0);
        scratch_.resize(solution_.size(), 0.0);

        // Default boundary conditions (Dirichlet, value = 0)
        for (int i = 0; i < 4; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }
    }

    /**
     * @brief Set the initial condition.
     */
    void setInitialCondition(const std::vector<double>& values) {
        if (values.size() != solution_.size()) {
            throw std::invalid_argument("Initial condition size doesn't match mesh");
        }
        requireFiniteVector(values, "Initial condition");
        solution_ = values;
    }

    /**
     * @brief Set a Dirichlet boundary condition.
     */
    void setDirichletBoundary(Boundary boundary, double value) {
        requireFinite(value, "Dirichlet boundary value");
        boundary_conditions_[checkedBoundaryIndex(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        setDirichletBoundary(static_cast<Boundary>(boundary_id), value);
    }

    /**
     * @brief Set a Neumann boundary condition.
     */
    void setNeumannBoundary(Boundary boundary, double normal_derivative) {
        requireFinite(normal_derivative, "Neumann outward-normal derivative");
        boundary_conditions_[checkedBoundaryIndex(boundary)] =
            BoundaryCondition::Neumann(normal_derivative);
    }

    void setNeumannBoundary(int boundary_id, double flux) {
        setNeumannBoundary(static_cast<Boundary>(boundary_id), flux);
    }

    /**
     * @brief Set a boundary condition.
     */
    void setBoundaryCondition(Boundary boundary, const BoundaryCondition& bc) {
        validateBoundaryCondition(bc);
        boundary_conditions_[checkedBoundaryIndex(boundary)] = bc;
    }

    void setBoundaryCondition(int boundary_id, const BoundaryCondition& bc) {
        setBoundaryCondition(static_cast<Boundary>(boundary_id), bc);
    }

    /**
     * @brief Run the solver for the specified number of steps.
     */
    void solve(double dt, int num_steps) {
        if (!std::isfinite(dt) || dt <= 0.0 || num_steps <= 0) {
            throw std::invalid_argument(
                "Time step must be finite and positive and number of steps must be positive");
        }
        if (!std::isfinite(dt * static_cast<double>(num_steps))) {
            throw std::invalid_argument("Requested integration interval must be finite");
        }

        if (!checkStability(dt)) {
            throw std::runtime_error(
                "Time step may be too large for stability. "
                "Use checkStability(dt) to verify before calling solve().");
        }

        scratch_.resize(solution_.size());

        for (int step = 0; step < num_steps; ++step) {
            derived().preStep(step, dt);

            // Compute updates for all interior nodes
            iterator_.forEachInterior([this, dt](int idx, int i, int j) {
                derived().computeNodeUpdate(idx, i, j, stencil_ops_, dt);
            });

            // Apply boundary conditions
            applyBoundaryConditions(scratch_);

            requireFiniteVector(scratch_, "Updated solution");

            // Swap buffers
            solution_.swap(scratch_);

            derived().postStep(step, dt);
        }
    }

    /**
     * @brief Get the current solution.
     */
    const std::vector<double>& solution() const { return solution_; }

    /**
     * @brief Get the mesh.
     */
    const StructuredMesh& mesh() const { return mesh_; }

    /**
     * @brief Get diffusivity.
     */
    double diffusivity() const { return diffusivity_; }

protected:
    const StructuredMesh& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    std::vector<double> scratch_;
    std::array<BoundaryCondition, 4> boundary_conditions_;

    MeshIterator iterator_;
    StencilOps stencil_ops_;

    // Time tracking for derived classes
    double time_ = 0.0;

    /**
     * @brief Compute the diffusion update term for a node.
     *
     * This is the common diffusion contribution: D * ∇²u * dt
     */
    double diffusionUpdate(int idx, double dt) const {
        return stencil_ops_.diffusionTerm(solution_, idx, diffusivity_, dt);
    }

    /**
     * @brief Check CFL stability condition.
     */
    bool checkStability(double dt) const {
        if (!std::isfinite(dt) || dt <= 0.0) {
            return false;
        }
        double dx = mesh_.dx();
        double max_dt = dx * dx / (2.0 * diffusivity_);

        if (!mesh_.is1D()) {
            double dy = mesh_.dy();
            max_dt = 1.0 / (2.0 * diffusivity_ * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        }

        bool stable = dt <= max_dt;

        // Let derived class add its own stability checks
        if (stable) {
            stable = derived().checkStabilityDerived(dt);
        }

        return stable;
    }

    /**
     * @brief Apply boundary conditions to a solution vector.
     */
    void applyBoundaryConditions(std::vector<double>& u) {
        if (mesh_.is1D()) {
            applyBoundaryConditions1D(u);
        } else {
            applyBoundaryConditions2D(u);
        }
    }

    // Default implementations for derived class hooks
    bool checkStabilityDerived(double /*dt*/) const { return true; }
    void preStep(int /*step*/, double /*dt*/) {}
    void postStep(int /*step*/, double dt) { time_ += dt; }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    static void requireFinite(double value, const char* name) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
    }

    static void requireFiniteVector(const std::vector<double>& values, const char* name) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i])) {
                throw std::invalid_argument(std::string(name) + " contains a non-finite value at " +
                                            std::to_string(i));
            }
        }
    }

    int checkedBoundaryIndex(Boundary boundary) const {
        const int index = to_index(boundary);
        if (index < to_index(Boundary::Left) || index > to_index(Boundary::Top)) {
            throw std::invalid_argument("Boundary identifier is outside [0, 3]");
        }
        if (mesh_.is1D() && (boundary == Boundary::Bottom || boundary == Boundary::Top)) {
            throw std::invalid_argument("Bottom and Top boundaries do not exist on a 1D mesh");
        }
        return index;
    }

    static void validateBoundaryCondition(const BoundaryCondition& bc) {
        switch (bc.type) {
            case BoundaryType::DIRICHLET:
                requireFinite(bc.value, "Dirichlet boundary value");
                return;
            case BoundaryType::NEUMANN:
                requireFinite(bc.value, "Neumann outward-normal derivative");
                return;
            case BoundaryType::ROBIN:
                requireFinite(bc.a, "Robin coefficient a");
                requireFinite(bc.b, "Robin coefficient b");
                requireFinite(bc.c, "Robin right-hand side c");
                if (bc.a == 0.0 && bc.b == 0.0) {
                    throw std::invalid_argument("Robin coefficients a and b cannot both be zero");
                }
                return;
        }
        throw std::invalid_argument("Unsupported boundary-condition type");
    }

    static double boundaryValueFromInterior(const BoundaryCondition& bc, double interior,
                                            double spacing) {
        switch (bc.type) {
            case BoundaryType::DIRICHLET:
                return bc.value;
            case BoundaryType::NEUMANN:
                // At either side, the one-sided outward derivative is
                // (u_boundary - u_interior) / spacing.
                return interior + bc.value * spacing;
            case BoundaryType::ROBIN: {
                // a*u_b + b*(u_b-u_i)/h = c.
                const double denominator = bc.a + bc.b / spacing;
                const double scale = std::max({1.0, std::abs(bc.a), std::abs(bc.b / spacing)});
                if (!std::isfinite(denominator) ||
                    std::abs(denominator) <= std::numeric_limits<double>::epsilon() * scale) {
                    throw std::invalid_argument(
                        "Robin boundary is singular for the selected mesh spacing");
                }
                return (bc.c + (bc.b / spacing) * interior) / denominator;
            }
        }
        throw std::invalid_argument("Unsupported boundary-condition type");
    }

    static double compatibleCornerValue(const BoundaryCondition& first, double first_interior,
                                        double first_spacing, const BoundaryCondition& second,
                                        double second_interior, double second_spacing) {
        if (first.type == BoundaryType::DIRICHLET && second.type == BoundaryType::DIRICHLET) {
            const double scale = std::max({1.0, std::abs(first.value), std::abs(second.value)});
            if (std::abs(first.value - second.value) >
                64.0 * std::numeric_limits<double>::epsilon() * scale) {
                throw std::invalid_argument(
                    "Conflicting Dirichlet values meet at a two-dimensional corner");
            }
            return 0.5 * (first.value + second.value);
        }
        if (first.type == BoundaryType::DIRICHLET) {
            return first.value;
        }
        if (second.type == BoundaryType::DIRICHLET) {
            return second.value;
        }
        return 0.5 * (boundaryValueFromInterior(first, first_interior, first_spacing) +
                      boundaryValueFromInterior(second, second_interior, second_spacing));
    }

    void applyBoundaryConditions1D(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const double dx = mesh_.dx();

        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];

        u[0] = boundaryValueFromInterior(left_bc, u[1], dx);
        u[nx] = boundaryValueFromInterior(right_bc, u[nx - 1], dx);
    }

    void applyBoundaryConditions2D(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = iterator_.stride();
        const double dx = mesh_.dx();
        const double dy = mesh_.dy();

        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];
        const auto& bottom_bc = boundary_conditions_[to_index(Boundary::Bottom)];
        const auto& top_bc = boundary_conditions_[to_index(Boundary::Top)];

        // Left boundary (exclude corners)
        for (int j = 1; j < ny; ++j) {
            int idx = j * stride;
            u[idx] = boundaryValueFromInterior(left_bc, u[idx + 1], dx);
        }

        // Right boundary (exclude corners)
        for (int j = 1; j < ny; ++j) {
            int idx = j * stride + nx;
            u[idx] = boundaryValueFromInterior(right_bc, u[idx - 1], dx);
        }

        // Bottom boundary (exclude corners)
        for (int i = 1; i < nx; ++i) {
            int idx = i;
            u[idx] = boundaryValueFromInterior(bottom_bc, u[idx + stride], dy);
        }

        // Top boundary (exclude corners)
        for (int i = 1; i < nx; ++i) {
            int idx = ny * stride + i;
            u[idx] = boundaryValueFromInterior(top_bc, u[idx - stride], dy);
        }

        u[0] = compatibleCornerValue(left_bc, u[1], dx, bottom_bc, u[stride], dy);
        u[nx] = compatibleCornerValue(right_bc, u[nx - 1], dx, bottom_bc, u[nx + stride], dy);
        u[ny * stride] = compatibleCornerValue(left_bc, u[ny * stride + 1], dx, top_bc,
                                               u[(ny - 1) * stride], dy);
        u[ny * stride + nx] = compatibleCornerValue(right_bc, u[ny * stride + nx - 1], dx, top_bc,
                                                    u[(ny - 1) * stride + nx], dy);
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_SOLVER_BASE_HPP
