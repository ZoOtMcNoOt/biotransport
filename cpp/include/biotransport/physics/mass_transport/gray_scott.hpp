/**
 * @file gray_scott.hpp
 * @brief Gray-Scott reaction-diffusion system for pattern formation.
 *
 * Implements the Gray-Scott model, a classic two-species reaction-diffusion
 * system that exhibits Turing patterns:
 *
 *   ∂u/∂t = Du ∇²u - u*v² + F*(1-u)
 *   ∂v/∂t = Dv ∇²v + u*v² - (F+k)*v
 *
 * Where:
 *   - u, v: Dimensionless concentrations
 *   - Du, Dv: Diffusion coefficients in mesh_length^2/model_time
 *   - F: Feed rate in 1/model_time (replenishes u)
 *   - k: Kill rate in 1/model_time (removes v)
 *
 * Parameter space includes spots, stripes, waves, and chaos. This is an
 * illustrative pattern-formation model, not a calibrated biological tissue
 * model.
 */

#ifndef BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_GRAY_SCOTT_HPP
#define BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_GRAY_SCOTT_HPP

#include <biotransport/core/mesh/indexing.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace biotransport {

/**
 * @brief Output data structure from Gray-Scott simulation.
 *
 * Contains time-series snapshots of the two species concentrations.
 * Arrays are packed in row-major order: [frame][j][i].
 * Uses float for memory efficiency with large pattern simulations.
 */
struct GrayScottRunResult {
    int nx = 0;               ///< Number of periodic cell-centred samples in x
    int ny = 0;               ///< Number of periodic cell-centred samples in y
    int frames = 0;           ///< Number of saved frames
    int steps_run = 0;        ///< Total simulation steps completed
    double final_time = 0.0;  ///< Model time reached (steps_run * dt)

    std::vector<int> frame_steps;  ///< Step number at each saved frame

    /// Species u concentration at each frame, packed as [frame][j][i]
    std::vector<float> u_frames;
    /// Species v concentration at each frame, packed as [frame][j][i]
    std::vector<float> v_frames;
};

/**
 * @brief Gray-Scott two-species reaction-diffusion solver.
 *
 * Solves the Gray-Scott model on a 2D periodic grid:
 *   ∂u/∂t = Du ∇²u - u·v² + F·(1-u)
 *   ∂v/∂t = Dv ∇²v + u·v² - (F+k)·v
 *
 * The field has one cell-centred degree of freedom per mesh cell, so u0 and v0
 * must each contain mesh.nx()*mesh.ny() values in row-major order. Periodic
 * boundary conditions are applied in both directions, and the actual mesh dx
 * and dy are used in the Laplacian. Traditional Gray-Scott parameter tables
 * commonly assume dx=dy=1 and must be rescaled for another mesh spacing.
 */
class GrayScottSolver {
public:
    /**
     * @brief Construct a Gray-Scott solver.
     *
     * @param mesh 2D structured mesh; fields contain one value per mesh cell
     * @param Du   Diffusion coefficient for species u
     * @param Dv   Diffusion coefficient for species v (typically Dv < Du)
     * @param f    Feed rate (replenishes u from reservoir)
     * @param k    Kill rate (removes v from system)
     */
    GrayScottSolver(const StructuredMesh& mesh, double Du, double Dv, double f, double k);

    /**
     * @brief Run the Gray-Scott simulation.
     *
     * @param u0                       Initial concentration of species u
     * @param v0                       Initial concentration of species v
     * @param total_steps              Maximum number of time steps
     * @param dt                       Time step in model-time units
     * @param steps_between_frames     Steps between saved snapshots
     * @param check_interval           Steps between steady-state checks
     * @param stable_tol               Maximum field change since the previous
     *                                 check; zero disables early termination
     * @param min_frames_before_early_stop Minimum frames before allowing early
     * termination
     * @return GrayScottRunResult Simulation results with u and v concentration fields
     */
    [[nodiscard]] GrayScottRunResult simulate(const std::vector<float>& u0,
                                              const std::vector<float>& v0, int total_steps,
                                              double dt, int steps_between_frames,
                                              int check_interval, double stable_tol,
                                              int min_frames_before_early_stop);

private:
    const StructuredMesh& mesh_;
    int nx_;
    int ny_;

    float Du_;
    float Dv_;
    float f_;
    float k_;
    float inv_dx2_;
    float inv_dy2_;

    // Use shared wrap_index() and idx() from indexing.hpp
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_GRAY_SCOTT_HPP
