#ifndef BIOTRANSPORT_SOLVERS_EXPLICIT_FD_HPP
#define BIOTRANSPORT_SOLVERS_EXPLICIT_FD_HPP

/**
 * @file explicit_fd.hpp
 * @brief Explicit finite difference solver with automatic time-stepping.
 *
 * The ExplicitFD class provides a unified interface for running explicit
 * time-stepping simulations with automatic CFL-based time step selection.
 */

#include <algorithm>
#include <biotransport/core/problems/transport_problem.hpp>
#include <biotransport/physics/reactions.hpp>
#include <biotransport/solvers/advection_diffusion_solver.hpp>
#include <biotransport/solvers/diffusion_solvers.hpp>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace biotransport {

/**
 * @brief Statistics from an ExplicitFD run.
 *
 * Tracks time-stepping information, solution bounds, mass conservation,
 * and wall-clock timing for performance analysis.
 */
struct SolverStats {
    double dt = 0.0;     ///< Time step size used [s]
    int steps = 0;       ///< Total number of time steps taken
    double t_end = 0.0;  ///< Final simulation time reached [s]

    // Summary metrics (useful for validation/monitoring)
    double u_min_initial = 0.0;  ///< Minimum solution value at t=0
    double u_max_initial = 0.0;  ///< Maximum solution value at t=0
    double u_min_final = 0.0;    ///< Minimum solution value at t_end
    double u_max_final = 0.0;    ///< Maximum solution value at t_end

    double mass_initial = 0.0;    ///< Total mass (integral of u) at t=0
    double mass_final = 0.0;      ///< Total mass at t_end
    double mass_abs_drift = 0.0;  ///< Absolute mass change
    double mass_rel_drift = 0.0;  ///< Relative mass change (%)

    double wall_time_s = 0.0;  ///< Wall-clock time for simulation [s]
};

/**
 * @brief Result of an ExplicitFD simulation run.
 */
struct RunResult {
    std::vector<double> solution;  ///< Final solution field
    SolverStats stats;             ///< Simulation statistics
};

/**
 * @brief Unified explicit finite difference solver facade.
 *
 * Provides a simple interface for running transport simulations with
 * automatic solver selection and time step calculation.
 *
 * @code
 *   auto problem = TransportProblem(mesh)
 *       .setDiffusivity(1e-9)
 *       .setInitialCondition(ic)
 *       .setBoundary(Boundary::Left, BoundaryCondition::Dirichlet(1.0));
 *
 *   auto result = ExplicitFD().run(problem, 100.0);  // Run for 100s
 * @endcode
 */
class ExplicitFD {
public:
    ExplicitFD& safetyFactor(double factor) {
        safety_factor_ = factor;
        return *this;
    }

    /**
     * @brief Run a transport problem simulation.
     *
     * Automatically selects the appropriate solver based on problem configuration:
     * - Pure diffusion if no reaction and no advection
     * - Reaction-diffusion if reaction is set
     * - Advection-diffusion if velocity is set
     *
     * @param problem The transport problem specification
     * @param t_end End time of simulation
     * @return RunResult with solution and statistics
     */
    RunResult run(const TransportProblem& problem, double t_end) const {
        if (t_end <= 0.0) {
            throw std::invalid_argument("t_end must be positive");
        }

        // Validate diffusivity based on type
        if (problem.hasUniformDiffusivity()) {
            if (problem.diffusivity() <= 0.0) {
                throw std::invalid_argument("Uniform diffusivity must be > 0");
            }
        } else {
            if (problem.diffusivityField().empty()) {
                throw std::invalid_argument("Diffusivity field must not be empty");
            }
        }

        const auto& mesh = problem.mesh();

        // Choose solver based on problem configuration
        if (problem.hasAdvection()) {
            return runAdvectionDiffusion(problem, t_end);
        } else {
            return runReactionDiffusion(problem, t_end);
        }
    }

private:
    double safety_factor_ = 0.9;

    struct StepsAndDt {
        int steps = 0;
        double dt_eff = 0.0;
    };

    static StepsAndDt chooseStepsAndDt(double t_end, double dt) {
        const int steps = std::max(1, static_cast<int>(std::ceil(t_end / dt)));
        return StepsAndDt{steps, t_end / static_cast<double>(steps)};
    }

    RunResult runReactionDiffusion(const TransportProblem& problem, double t_end) const {
        const auto& mesh = problem.mesh();

        // Check for variable diffusivity - use specialized solver
        if (!problem.hasUniformDiffusivity()) {
            return runVariableDiffusion(problem, t_end);
        }

        // Check if this is a linear decay problem - use specialized solver with implicit treatment
        double linear_rate = problem.linearReactionRate();
        if (linear_rate > 0.0) {
            return runLinearDecay(problem, t_end, linear_rate);
        }

        // Generic reaction-diffusion solver with the problem's reaction function
        ReactionDiffusionSolver solver(mesh, problem.diffusivity(), problem.reaction());

        // Configure initial and boundary conditions
        if (!problem.initial().empty()) {
            solver.setInitialCondition(problem.initial());
        }
        for (int i = 0; i < 4; ++i) {
            solver.setBoundaryCondition(i, problem.boundaries()[i]);
        }

        // Choose stable time step
        const double dt = chooseStableDt(mesh, problem.diffusivity(), 0.0);
        const auto steps_dt = chooseStepsAndDt(t_end, dt);

        return runConfiguredSolver(solver, mesh, t_end, steps_dt.steps, steps_dt.dt_eff);
    }

    RunResult runVariableDiffusion(const TransportProblem& problem, double t_end) const {
        const auto& mesh = problem.mesh();

        // Use variable diffusion solver with spatially-varying D(x)
        VariableDiffusionSolver solver(mesh, problem.diffusivityField());

        // Configure initial and boundary conditions
        if (!problem.initial().empty()) {
            solver.setInitialCondition(problem.initial());
        }
        for (int i = 0; i < 4; ++i) {
            solver.setBoundaryCondition(i, problem.boundaries()[i]);
        }

        // Use max diffusivity for stability calculation
        const double dt = chooseStableDt(mesh, solver.maxDiffusivity(), 0.0);
        const auto steps_dt = chooseStepsAndDt(t_end, dt);

        return runConfiguredSolver(solver, mesh, t_end, steps_dt.steps, steps_dt.dt_eff);
    }

    RunResult runLinearDecay(const TransportProblem& problem, double t_end,
                             double decay_rate) const {
        const auto& mesh = problem.mesh();

        // Use specialized linear decay solver with implicit treatment
        LinearReactionDiffusionSolver solver(mesh, problem.diffusivity(), decay_rate);

        // Configure initial and boundary conditions
        if (!problem.initial().empty()) {
            solver.setInitialCondition(problem.initial());
        }
        for (int i = 0; i < 4; ++i) {
            solver.setBoundaryCondition(i, problem.boundaries()[i]);
        }

        // Only need diffusion stability (implicit decay is unconditionally stable)
        const double dt = chooseStableDt(mesh, problem.diffusivity(), 0.0);
        const auto steps_dt = chooseStepsAndDt(t_end, dt);

        return runConfiguredSolver(solver, mesh, t_end, steps_dt.steps, steps_dt.dt_eff);
    }

    RunResult runAdvectionDiffusion(const TransportProblem& problem, double t_end) const {
        const auto& mesh = problem.mesh();

        // Create advection-diffusion solver
        std::unique_ptr<AdvectionDiffusionSolver> solver;
        if (problem.hasUniformVelocity()) {
            solver = std::make_unique<AdvectionDiffusionSolver>(
                mesh, problem.diffusivity(), problem.vxUniform(), problem.vyUniform(),
                problem.scheme());
        } else {
            solver = std::make_unique<AdvectionDiffusionSolver>(
                mesh, problem.diffusivity(), problem.vxField(), problem.vyField(),
                problem.scheme());
        }

        // Configure initial and boundary conditions
        if (!problem.initial().empty()) {
            solver->setInitialCondition(problem.initial());
        }
        for (int i = 0; i < 4; ++i) {
            solver->setBoundaryCondition(i, problem.boundaries()[i]);
        }

        const double dt = solver->maxTimeStep(safety_factor_);
        const auto steps_dt = chooseStepsAndDt(t_end, dt);
        return runConfiguredSolver(*solver, mesh, t_end, steps_dt.steps, steps_dt.dt_eff);
    }

    template <typename SolverT>
    RunResult runConfiguredSolver(SolverT& solver, const StructuredMesh& mesh, double t_end,
                                  int steps, double dt_eff) const {
        const auto initial_minmax = minmax(solver.solution());
        const double mass0 = computeMass(mesh, solver.solution());

        const auto start = std::chrono::steady_clock::now();
        solver.solve(dt_eff, steps);
        const auto stop = std::chrono::steady_clock::now();

        const auto final_minmax = minmax(solver.solution());
        const double mass1 = computeMass(mesh, solver.solution());

        RunResult result;
        result.solution = solver.solution();
        result.stats.dt = dt_eff;
        result.stats.steps = steps;
        result.stats.t_end = t_end;

        result.stats.u_min_initial = initial_minmax.first;
        result.stats.u_max_initial = initial_minmax.second;
        result.stats.u_min_final = final_minmax.first;
        result.stats.u_max_final = final_minmax.second;

        result.stats.mass_initial = mass0;
        result.stats.mass_final = mass1;
        result.stats.mass_abs_drift = mass1 - mass0;
        result.stats.mass_rel_drift = std::abs(mass1 - mass0) / std::max(1e-12, std::abs(mass0));

        result.stats.wall_time_s = std::chrono::duration<double>(stop - start).count();
        return result;
    }

    static std::pair<double, double> minmax(const std::vector<double>& values) {
        if (values.empty()) {
            return {0.0, 0.0};
        }
        const auto mm = std::minmax_element(values.begin(), values.end());
        return {*mm.first, *mm.second};
    }

    static double computeMass(const StructuredMesh& mesh, const std::vector<double>& u) {
        const int nx = mesh.nx();
        const int ny = mesh.ny();

        if (mesh.is1D()) {
            double mass = 0.0;
            const double dx = mesh.dx();
            for (int i = 0; i <= nx; ++i) {
                mass += u[mesh.index(i)] * dx;
            }
            return mass;
        }

        const double cell_area = mesh.dx() * mesh.dy();
        double mass = 0.0;
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                mass += u[mesh.index(i, j)] * cell_area;
            }
        }
        return mass;
    }

    double chooseStableDt(const StructuredMesh& mesh, double diffusivity,
                          double linear_reaction_rate = 0.0) const {
        // Diffusion stability: dt <= dx^2/(2D) in 1D, dt <= min(dx^2,dy^2)/(4D) in 2D
        const double dx2 = mesh.dx() * mesh.dx();
        double dt_diffusion;
        if (mesh.is1D()) {
            dt_diffusion = safety_factor_ * (dx2 / (2.0 * diffusivity));
        } else {
            const double dy2 = mesh.dy() * mesh.dy();
            const double min_h2 = std::min(dx2, dy2);
            dt_diffusion = safety_factor_ * (min_h2 / (4.0 * diffusivity));
        }

        // Reaction stability for linear decay: |1 - k*dt| <= 1 => dt <= 2/k
        // Use a tighter bound (1.8/k) for safety
        double dt_reaction = std::numeric_limits<double>::max();
        if (linear_reaction_rate > 0.0) {
            dt_reaction = 1.8 / linear_reaction_rate;
        }

        return std::min(dt_diffusion, dt_reaction);
    }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_EXPLICIT_FD_HPP
