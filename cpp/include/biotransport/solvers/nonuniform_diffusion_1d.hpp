#ifndef BIOTRANSPORT_SOLVERS_NONUNIFORM_DIFFUSION_1D_HPP
#define BIOTRANSPORT_SOLVERS_NONUNIFORM_DIFFUSION_1D_HPP

/**
 * @file nonuniform_diffusion_1d.hpp
 * @brief Conservative diffusion on a fitted nonuniform one-dimensional mesh.
 */

#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/nonuniform_mesh_1d.hpp>
#include <cstddef>
#include <vector>

namespace biotransport {

/** Diagnostics for a conservative nonuniform diffusion solve. */
struct NonuniformDiffusionDiagnostics {
    std::size_t steps = 0;
    double reference_time = 0.0;
    double time = 0.0;
    double stability_limit = 0.0;
    double reference_mass = 0.0;
    double total_mass = 0.0;
    double cumulative_boundary_input = 0.0;
    double mass_balance_error = 0.0;
    double minimum_concentration = 0.0;
    double maximum_concentration = 0.0;
    double left_outward_flux = 0.0;
    double right_outward_flux = 0.0;
};

/**
 * @brief Node-centred finite-volume solver for dc/dt = d/dx(D dc/dx).
 *
 * Diffusivity is specified at nodes and combined at faces with the harmonic
 * mean. This preserves a single numerical flux on both sides of each face and
 * handles sharp material contrasts without arithmetic-mean leakage.
 *
 * Neumann values are outward-normal derivatives dc/dn. Therefore the outward
 * Fickian flux is -D dc/dn, and a positive Neumann value adds mass to the
 * domain. Dirichlet boundary nodes are held exactly at their prescribed value.
 */
class NonuniformDiffusion1D {
public:
    NonuniformDiffusion1D(NonuniformMesh1D mesh, double diffusivity);
    NonuniformDiffusion1D(NonuniformMesh1D mesh, std::vector<double> nodal_diffusivity);

    const NonuniformMesh1D& mesh() const noexcept { return mesh_; }
    const std::vector<double>& diffusivity() const noexcept { return diffusivity_; }
    const std::vector<double>& solution() const noexcept { return concentration_; }
    double time() const noexcept { return time_; }
    std::size_t steps() const noexcept { return steps_; }

    NonuniformDiffusion1D& setInitialCondition(std::vector<double> concentration);
    NonuniformDiffusion1D& setUniformInitialCondition(double concentration);

    NonuniformDiffusion1D& setBoundaryCondition(Boundary boundary,
                                                const BoundaryCondition& condition);
    NonuniformDiffusion1D& setDirichletBoundary(Boundary boundary, double concentration);
    NonuniformDiffusion1D& setNeumannBoundary(Boundary boundary, double outward_normal_derivative);

    const BoundaryCondition& boundaryCondition(Boundary boundary) const;

    /** Exact monotonic Forward Euler bound from local conductance/CV ratios. */
    double maxStableTimeStep() const;
    bool checkStability(double dt) const;

    /** Advance one Forward Euler step. The state is unchanged if validation fails. */
    void step(double dt);
    void solve(double dt, std::size_t num_steps);

    /** Advance to an absolute final time using steps no larger than maximum_dt. */
    void solveUntil(double final_time, double maximum_dt);

    /** Harmonic diffusivity at each interior face. */
    const std::vector<double>& faceDiffusivities() const noexcept { return face_diffusivity_; }

    /** Fickian face flux, positive toward increasing x. */
    std::vector<double> faceFluxes() const;

    /** Domain-integrated concentration, sum_i c_i V_i. */
    double totalMass() const;

    /** Physical Fickian flux leaving the requested boundary. */
    double boundaryOutwardFlux(Boundary boundary) const;

    /** Start a fresh mass-balance accounting interval at the current state. */
    void resetBalanceReference();
    NonuniformDiffusionDiagnostics diagnostics() const;

private:
    NonuniformMesh1D mesh_;
    std::vector<double> diffusivity_;
    std::vector<double> face_diffusivity_;
    std::vector<double> conductance_;
    std::vector<double> concentration_;
    std::vector<double> scratch_;
    BoundaryCondition left_boundary_ = BoundaryCondition::Neumann(0.0);
    BoundaryCondition right_boundary_ = BoundaryCondition::Neumann(0.0);
    double time_ = 0.0;
    std::size_t steps_ = 0;
    double reference_time_ = 0.0;
    double reference_mass_ = 0.0;
    double cumulative_boundary_input_ = 0.0;

    static double harmonicMean(double left, double right);
    static void validateBoundary(Boundary boundary);
    static void validateConcentration(double value, const char* quantity);

    void initializeMaterialData();
    void applyDirichletBoundaries() noexcept;
    double boundaryInputRate(Boundary boundary) const;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_NONUNIFORM_DIFFUSION_1D_HPP
