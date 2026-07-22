#ifndef BIOTRANSPORT_SOLVERS_MULTI_SPECIES_SOLVER_HPP
#define BIOTRANSPORT_SOLVERS_MULTI_SPECIES_SOLVER_HPP

/**
 * @file multi_species_solver.hpp
 * @brief Generic N-species reaction-diffusion solver framework.
 *
 * This solver extends the biotransport library to handle arbitrary numbers
 * of interacting chemical species with coupled reaction kinetics. It supports:
 *
 * - N species with individual diffusion coefficients
 * - User-defined reaction kinetics via callable function
 * - Per-species boundary conditions
 * - Common reaction models (Lotka-Volterra, enzyme cascades, SIR)
 *
 * The governing equations are:
 *   ∂u_i/∂t = D_i ∇²u_i + R_i(u_1, u_2, ..., u_N, x, y, t)
 *
 * where:
 *   u_i = concentration of species i
 *   D_i = diffusion coefficient of species i
 *   R_i = reaction rate for species i (function of all concentrations)
 *
 * Example usage:
 * @code
 *   // 2-species Lotka-Volterra system
 *   MultiSpeciesSolver solver(mesh, {D_prey, D_predator});
 *   solver.setReactionModel(LotkaVolterraReaction(alpha, beta, gamma, delta));
 *   solver.setInitialCondition(0, prey_ic);  // Species 0: prey
 *   solver.setInitialCondition(1, pred_ic);  // Species 1: predator
 *   solver.solve(dt, num_steps);
 * @endcode
 */

#include <algorithm>
#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#include <omp.h>
#endif

namespace biotransport {

// =============================================================================
// Common Reaction Models
// =============================================================================

/**
 * @brief Lotka-Volterra predator-prey with carrying capacity.
 *
 * For 2 species (prey u, predator v):
 *   du/dt = α·u·(1 - u/K) - β·u·v   (logistic prey growth, consumed by predator)
 *   dv/dt = δ·u·v - γ·v             (predator grows from prey, dies naturally)
 *
 * The carrying capacity K prevents unbounded prey growth.
 *
 * Parameters:
 *   α (alpha) = prey growth rate [1/T]
 *   β (beta)  = predation coefficient [1/(predator concentration·T)]
 *   γ (gamma) = predator death rate [1/T]
 *   δ (delta) = predator reproduction coefficient [1/(prey concentration·T)]
 *   K         = prey carrying capacity [prey concentration] (default = 100)
 */
class LotkaVolterraReaction {
public:
    LotkaVolterraReaction(double alpha, double beta, double gamma, double delta,
                          double carrying_capacity = 100.0)
        : alpha_(alpha), beta_(beta), gamma_(gamma), delta_(delta), K_(carrying_capacity) {
        if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(gamma) ||
            !std::isfinite(delta) || alpha < 0 || beta < 0 || gamma < 0 || delta < 0) {
            throw std::invalid_argument("All Lotka-Volterra parameters must be non-negative");
        }
        if (!std::isfinite(K_) || K_ <= 0) {
            throw std::invalid_argument("Carrying capacity must be positive");
        }
    }

    void operator()(std::vector<double>& rates, const std::vector<double>& u, double /*x*/,
                    double /*y*/, double /*t*/) const {
        if (u.size() < 2 || rates.size() < 2) {
            throw std::runtime_error("Lotka-Volterra requires at least 2 species");
        }
        const double prey = u[0];
        const double pred = u[1];

        // Logistic prey growth with carrying capacity
        rates[0] = alpha_ * prey * (1.0 - prey / K_) - beta_ * prey * pred;
        rates[1] = delta_ * prey * pred - gamma_ * pred;
    }

    double alpha() const { return alpha_; }
    double beta() const { return beta_; }
    double gamma() const { return gamma_; }
    double delta() const { return delta_; }
    double carrying_capacity() const { return K_; }

private:
    double alpha_, beta_, gamma_, delta_, K_;
};

/**
 * @brief SIR (Susceptible-Infected-Recovered) epidemiological model.
 *
 * For 3 species (S, I, R):
 *   dS/dt = -β·S·I / N         (susceptible become infected)
 *   dI/dt = β·S·I / N - γ·I    (infected from S, recover)
 *   dR/dt = γ·I                (recovered from infected)
 *
 * Parameters:
 *   β (beta)  = transmission rate [1/T]
 *   γ (gamma) = recovery rate [1/T]
 *   N         = reference population in the same units as S, I, and R
 *
 * For spatial density fields, N is a reference local density rather than the
 * domain-integrated population. R₀ = β/γ assumes S ≈ N.
 */
class SIRReaction {
public:
    SIRReaction(double beta, double gamma, double total_population)
        : beta_(beta), gamma_(gamma), N_(total_population) {
        if (!std::isfinite(beta) || !std::isfinite(gamma) || beta < 0 || gamma < 0) {
            throw std::invalid_argument("SIR parameters must be non-negative");
        }
        if (!std::isfinite(N_) || N_ <= 0) {
            throw std::invalid_argument("Total population must be positive");
        }
    }

    void operator()(std::vector<double>& rates, const std::vector<double>& u, double /*x*/,
                    double /*y*/, double /*t*/) const {
        if (u.size() < 3 || rates.size() < 3) {
            throw std::runtime_error("SIR model requires 3 species");
        }
        double S = u[0];  // Susceptible
        double I = u[1];  // Infected
        // R = u[2]       // Recovered (not needed for rate calculation)

        double infection_rate = beta_ * S * I / N_;

        rates[0] = -infection_rate;              // dS/dt
        rates[1] = infection_rate - gamma_ * I;  // dI/dt
        rates[2] = gamma_ * I;                   // dR/dt
    }

    double beta() const { return beta_; }
    double gamma() const { return gamma_; }
    double N() const { return N_; }
    double R0() const {
        return gamma_ == 0.0 ? std::numeric_limits<double>::infinity() : beta_ / gamma_;
    }

private:
    double beta_, gamma_, N_;
};

/**
 * @brief SEIR (Susceptible-Exposed-Infected-Recovered) epidemiological model.
 *
 * Extension of SIR with an exposed (latent) period:
 *   dS/dt = -β·S·I / N
 *   dE/dt = β·S·I / N - σ·E    (exposed become infected after incubation)
 *   dI/dt = σ·E - γ·I
 *   dR/dt = γ·I
 *
 * Parameters:
 *   β (beta)  = transmission rate [1/T]
 *   σ (sigma) = rate of becoming infectious [1/T]
 *   γ (gamma) = recovery rate [1/T]
 *   N         = reference population in the same units as S, E, I, and R
 */
class SEIRReaction {
public:
    SEIRReaction(double beta, double sigma, double gamma, double total_population)
        : beta_(beta), sigma_(sigma), gamma_(gamma), N_(total_population) {
        if (!std::isfinite(beta) || !std::isfinite(sigma) || !std::isfinite(gamma) || beta < 0 ||
            sigma < 0 || gamma < 0) {
            throw std::invalid_argument("SEIR parameters must be non-negative");
        }
        if (!std::isfinite(N_) || N_ <= 0) {
            throw std::invalid_argument("Total population must be positive");
        }
    }

    void operator()(std::vector<double>& rates, const std::vector<double>& u, double /*x*/,
                    double /*y*/, double /*t*/) const {
        if (u.size() < 4 || rates.size() < 4) {
            throw std::runtime_error("SEIR model requires 4 species");
        }
        double S = u[0];  // Susceptible
        double E = u[1];  // Exposed
        double I = u[2];  // Infected
        // R = u[3]       // Recovered

        double infection_rate = beta_ * S * I / N_;

        rates[0] = -infection_rate;              // dS/dt
        rates[1] = infection_rate - sigma_ * E;  // dE/dt
        rates[2] = sigma_ * E - gamma_ * I;      // dI/dt
        rates[3] = gamma_ * I;                   // dR/dt
    }

    double beta() const { return beta_; }
    double sigma() const { return sigma_; }
    double gamma() const { return gamma_; }
    double N() const { return N_; }

private:
    double beta_, sigma_, gamma_, N_;
};

/**
 * @brief Enzyme cascade reaction kinetics.
 *
 * Models a linear cascade of enzyme activations:
 *   E₀ → E₁ → E₂ → ... → Eₙ
 *
 * Each enzyme is activated by the previous one with Michaelis-Menten kinetics:
 *   dE_i/dt = (V_max,i · E_{i-1}) / (K_m,i + E_{i-1}) - k_deg,i · E_i
 *
 * The first enzyme (E₀) is typically a constant input signal.
 *
 * Parameters:
 *   vmax_values = maximum production rates for each step [target concentration/T]
 *   km_values   = Michaelis constants [upstream concentration]
 *   kdeg_values = degradation rates for each species [1/T]
 */
class EnzymeCascadeReaction {
public:
    EnzymeCascadeReaction(const std::vector<double>& vmax_values,
                          const std::vector<double>& km_values,
                          const std::vector<double>& kdeg_values)
        : vmax_(vmax_values), km_(km_values), kdeg_(kdeg_values) {
        if (vmax_.size() != km_.size()) {
            throw std::invalid_argument("vmax and km vectors must have same size");
        }
        if (kdeg_.size() != vmax_.size() + 1) {
            throw std::invalid_argument("kdeg vector must have size = num_enzymes");
        }
        for (size_t i = 0; i < vmax_.size(); ++i) {
            if (!std::isfinite(vmax_[i]) || !std::isfinite(km_[i]) || vmax_[i] < 0 || km_[i] <= 0) {
                throw std::invalid_argument("Invalid enzyme kinetic parameters");
            }
        }
        for (double k : kdeg_) {
            if (!std::isfinite(k) || k < 0) {
                throw std::invalid_argument("Degradation rates must be non-negative");
            }
        }
    }

    void operator()(std::vector<double>& rates, const std::vector<double>& u, double /*x*/,
                    double /*y*/, double /*t*/) const {
        size_t n = u.size();
        if (n != kdeg_.size() || rates.size() < n) {
            throw std::runtime_error("Enzyme cascade species count mismatch");
        }

        // First enzyme: only degradation (or could add external source)
        rates[0] = -kdeg_[0] * u[0];

        // Subsequent enzymes: activation from previous + degradation
        for (size_t i = 1; i < n; ++i) {
            double activation = 0.0;
            if (u[i - 1] > 0 && vmax_[i - 1] > 0) {
                activation = vmax_[i - 1] * u[i - 1] / (km_[i - 1] + u[i - 1]);
            }
            rates[i] = activation - kdeg_[i] * u[i];
        }
    }

    size_t numEnzymes() const { return kdeg_.size(); }

private:
    std::vector<double> vmax_;  // N-1 values
    std::vector<double> km_;    // N-1 values
    std::vector<double> kdeg_;  // N values
};

/**
 * @brief Competitive inhibition reaction model.
 *
 * Models substrate (S) competing with inhibitor (I) for enzyme (E):
 *   dS/dt = -Vmax · S / (Km · (1 + I/Ki) + S)
 *   dI/dt = 0 (inhibitor is not consumed, optional decay)
 *   dP/dt = Vmax · S / (Km · (1 + I/Ki) + S)  (product formation)
 *
 * Parameters:
 *   Vmax = maximum reaction velocity [substrate concentration/T]
 *   Km   = Michaelis constant [substrate concentration]
 *   Ki   = inhibition constant [inhibitor concentration]
 */
class CompetitiveInhibitionReaction {
public:
    CompetitiveInhibitionReaction(double vmax, double km, double ki, double inhibitor_decay = 0.0)
        : vmax_(vmax), km_(km), ki_(ki), inhibitor_decay_(inhibitor_decay) {
        if (!std::isfinite(vmax) || !std::isfinite(km) || !std::isfinite(ki) ||
            !std::isfinite(inhibitor_decay) || vmax < 0 || km <= 0 || ki <= 0 ||
            inhibitor_decay < 0) {
            throw std::invalid_argument("Invalid enzyme kinetic parameters");
        }
    }

    void operator()(std::vector<double>& rates, const std::vector<double>& u, double /*x*/,
                    double /*y*/, double /*t*/) const {
        if (u.size() < 3 || rates.size() < 3) {
            throw std::runtime_error("Competitive inhibition requires 3 species (S, I, P)");
        }
        double S = u[0];  // Substrate
        double I = u[1];  // Inhibitor
        // P = u[2]       // Product

        double apparent_km = km_ * (1.0 + I / ki_);
        double rate = 0.0;
        if (S > 0) {
            rate = vmax_ * S / (apparent_km + S);
        }

        rates[0] = -rate;                  // dS/dt (consumption)
        rates[1] = -inhibitor_decay_ * I;  // dI/dt (optional decay)
        rates[2] = rate;                   // dP/dt (production)
    }

    double vmax() const { return vmax_; }
    double km() const { return km_; }
    double ki() const { return ki_; }

private:
    double vmax_, km_, ki_, inhibitor_decay_;
};

/**
 * @brief Brusselator reaction model (chemical oscillator).
 *
 * Classic 2-species autocatalytic system that exhibits limit cycle oscillations:
 *   dX/dt = A - (B+1)·X + X²·Y
 *   dY/dt = B·X - X²·Y
 *
 * For B > 1 + A², the system exhibits sustained oscillations.
 *
 * This is the conventional nondimensional Brusselator. A and B are positive,
 * dimensionless control parameters, and model time is dimensionless.
 */
class BrusselatorReaction {
public:
    BrusselatorReaction(double A, double B) : A_(A), B_(B) {
        if (!std::isfinite(A) || !std::isfinite(B) || A <= 0 || B <= 0) {
            throw std::invalid_argument("Brusselator parameters must be positive");
        }
    }

    void operator()(std::vector<double>& rates, const std::vector<double>& u, double /*x*/,
                    double /*y*/, double /*t*/) const {
        if (u.size() < 2 || rates.size() < 2) {
            throw std::runtime_error("Brusselator requires 2 species");
        }
        double X = u[0];
        double Y = u[1];

        double X2Y = X * X * Y;
        rates[0] = A_ - (B_ + 1.0) * X + X2Y;  // dX/dt
        rates[1] = B_ * X - X2Y;               // dY/dt
    }

    double A() const { return A_; }
    double B() const { return B_; }

    // Check if parameters lead to oscillations
    bool isOscillatory() const { return B_ > 1.0 + A_ * A_; }

private:
    double A_, B_;
};

// =============================================================================
// Multi-Species Solver
// =============================================================================

/**
 * @brief Generic N-species reaction-diffusion solver.
 *
 * Solves the coupled system:
 *   ∂u_i/∂t = D_i ∇²u_i + R_i(u_1, ..., u_N, x, y, t)
 *
 * for i = 1, ..., N species.
 *
 * The spatial discretization is a conservative node-centred finite-volume
 * scheme. Boundary nodes represent half control volumes in 1D, edge half
 * volumes in 2D, and corner quarter volumes. This makes zero outward-normal
 * derivative boundaries conserve the trapezoidal integral of each species.
 *
 * Time integration is forward Euler. The diffusion CFL limit is reported by
 * maxStableTimeStep(). Reaction kinetics can impose a stricter, state-dependent
 * positivity limit; every candidate step is checked before it is committed. A
 * non-finite or materially negative concentration is rejected rather than
 * silently clipped.
 *
 * Units must be self-consistent: mesh coordinates have units L, time and dt
 * have units T, D_i has units L^2/T, u_i has the caller's concentration units,
 * and R_i must return concentration/T in those same units.
 */
class MultiSpeciesSolver {
public:
    /**
     * @brief Type for reaction function.
     *
     * The function takes:
     *   - rates: output vector, rates[i] = R_i for species i
     *   - concentrations: input vector, u[i] = concentration of species i
     *   - x, y: spatial coordinates
     *   - t: current time
     */
    using ReactionFunction =
        std::function<void(std::vector<double>& rates, const std::vector<double>& concentrations,
                           double x, double y, double t)>;

    /**
     * @brief Construct a multi-species solver.
     *
     * @param mesh The computational mesh
     * @param diffusivities Diffusion coefficient for each species
     * @param num_species Number of species (inferred from diffusivities if 0)
     */
    MultiSpeciesSolver(const StructuredMesh& mesh, const std::vector<double>& diffusivities,
                       size_t num_species = 0)
        : mesh_(mesh),
          diffusivities_(diffusivities),
          num_species_(num_species == 0 ? diffusivities.size() : num_species),
          time_(0.0) {
        if (num_species_ == 0) {
            throw std::invalid_argument("Must have at least 1 species");
        }
        if (diffusivities_.size() != num_species_) {
            throw std::invalid_argument("Diffusivity vector size must match number of species");
        }
        for (double D : diffusivities_) {
            if (!std::isfinite(D) || D < 0.0) {
                throw std::invalid_argument("Diffusivities must be non-negative");
            }
        }

        // Allocate storage for each species
        size_t num_nodes = mesh.numNodes();
        species_.resize(num_species_);
        scratch_.resize(num_species_);
        for (size_t s = 0; s < num_species_; ++s) {
            species_[s].resize(num_nodes, 0.0);
            scratch_[s].resize(num_nodes, 0.0);
        }

        // A closed system is the least surprising default for concentrations.
        // Neumann values are outward-normal derivatives, not physical fluxes.
        boundary_conditions_.resize(num_species_);
        for (size_t s = 0; s < num_species_; ++s) {
            for (int b = 0; b < 4; ++b) {
                boundary_conditions_[s][b] = BoundaryCondition::Neumann(0.0);
            }
        }

        // Pre-cache coordinates for reaction function evaluation
        cacheCoordinates();
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /**
     * @brief Set the reaction function for all species.
     */
    void setReactionFunction(ReactionFunction reaction) {
        reaction_ = std::move(reaction);
        // An arbitrary callback may have mutable state (and Python callbacks
        // require interpreter coordination), so evaluate it serially.
        reaction_thread_safe_ = false;
    }

    /**
     * @brief Set the reaction function from a callable object.
     */
    template <typename Callable>
    void setReactionModel(Callable&& model) {
        using Model = std::decay_t<Callable>;
        validateReactionModelArity<Model>(model);
        reaction_ = [model = std::forward<Callable>(model)](std::vector<double>& rates,
                                                            const std::vector<double>& u, double x,
                                                            double y, double t) mutable {
            model(rates, u, x, y, t);
        };
        reaction_thread_safe_ =
            std::is_same_v<Model, LotkaVolterraReaction> || std::is_same_v<Model, SIRReaction> ||
            std::is_same_v<Model, SEIRReaction> || std::is_same_v<Model, EnzymeCascadeReaction> ||
            std::is_same_v<Model, CompetitiveInhibitionReaction> ||
            std::is_same_v<Model, BrusselatorReaction>;
    }

    /**
     * @brief Set initial condition for a specific species.
     */
    void setInitialCondition(size_t species_idx, const std::vector<double>& values) {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        if (values.size() != species_[species_idx].size()) {
            throw std::invalid_argument("Initial condition size doesn't match mesh");
        }
        validateConcentrations(values, "Initial condition");
        species_[species_idx] = values;
    }

    /**
     * @brief Set initial condition for all species from a single value.
     */
    void setUniformInitialCondition(size_t species_idx, double value) {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("Initial concentration must be finite and non-negative");
        }
        std::fill(species_[species_idx].begin(), species_[species_idx].end(), value);
    }

    /**
     * @brief Set Dirichlet boundary condition for a specific species.
     */
    void setDirichletBoundary(size_t species_idx, Boundary boundary, double value) {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        validateBoundary(boundary);
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("Dirichlet concentration must be finite and non-negative");
        }
        boundary_conditions_[species_idx][to_index(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(size_t species_idx, int boundary_id, double value) {
        setDirichletBoundary(species_idx, checkedBoundary(boundary_id), value);
    }

    /**
     * @brief Set an outward-normal derivative for a specific species.
     *
     * This value is du/dn. The corresponding outward Fickian flux is -D du/dn.
     */
    void setNeumannBoundary(size_t species_idx, Boundary boundary, double normal_derivative) {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        validateBoundary(boundary);
        if (!std::isfinite(normal_derivative)) {
            throw std::invalid_argument("Neumann derivative must be finite");
        }
        boundary_conditions_[species_idx][to_index(boundary)] =
            BoundaryCondition::Neumann(normal_derivative);
    }

    void setNeumannBoundary(size_t species_idx, int boundary_id, double normal_derivative) {
        setNeumannBoundary(species_idx, checkedBoundary(boundary_id), normal_derivative);
    }

    /**
     * @brief Set the same boundary condition for all species on a boundary.
     */
    void setAllSpeciesDirichlet(Boundary boundary, double value) {
        validateBoundary(boundary);
        for (size_t s = 0; s < num_species_; ++s) {
            setDirichletBoundary(s, boundary, value);
        }
    }

    void setAllSpeciesNeumann(Boundary boundary, double normal_derivative) {
        validateBoundary(boundary);
        for (size_t s = 0; s < num_species_; ++s) {
            setNeumannBoundary(s, boundary, normal_derivative);
        }
    }

    // -------------------------------------------------------------------------
    // Solution
    // -------------------------------------------------------------------------

    /**
     * @brief Check the diffusion CFL condition.
     *
     * The mesh spacing and maximum species diffusivity determine the ceiling.
     * Reaction kinetics may impose a stricter state-dependent limit, which is
     * checked during solve().
     */
    bool checkStability(double dt) const {
        return std::isfinite(dt) && dt > 0.0 && dt <= diffusionTimeStepLimit();
    }

    /**
     * @brief Get the exact diffusion-only forward-Euler CFL limit.
     *
     * Use a safety factor below one for production runs. This bound does not
     * include state-dependent reaction kinetics; solve() checks positivity of
     * every candidate step and rejects an inadmissible step atomically.
     */
    double maxStableTimeStep() const { return diffusionTimeStepLimit(); }

    /**
     * @brief Run the solver for the specified number of steps.
     */
    void solve(double dt, int num_steps) {
        if (!std::isfinite(dt) || dt <= 0.0 || num_steps <= 0) {
            throw std::invalid_argument("Time step and number of steps must be positive");
        }

        const double start_time = time_;
        const double requested_end_time = std::fma(dt, static_cast<double>(num_steps), start_time);
        if (!std::isfinite(requested_end_time)) {
            throw std::invalid_argument("Requested end time is not finite");
        }

        if (!checkStability(dt)) {
            const double limit = diffusionTimeStepLimit();
            throw std::runtime_error("Time step " + std::to_string(dt) +
                                     " exceeds the explicit diffusion limit " +
                                     std::to_string(limit));
        }

        validateBoundaryConfiguration();
        validateCurrentState();

        // The OpenMP team size can be configured after construction. Resize
        // outside the parallel region so every worker has exclusive scratch
        // storage during reaction evaluation.
        if (reaction_) {
            prepareReactionWorkspaces();
        }

        for (int step = 0; step < num_steps; ++step) {
            computeCandidateStep(dt);
            validateCandidateStep(dt);

            // Swap buffers
            for (size_t s = 0; s < num_species_; ++s) {
                species_[s].swap(scratch_[s]);
            }

            // Multiplication from the solve-call start avoids accumulation drift
            // and makes the reported final time exactly start + n*dt (up to one
            // floating-point rounding).
            time_ = std::fma(dt, static_cast<double>(step + 1), start_time);
        }

        time_ = requested_end_time;
    }

    /**
     * @brief Advance exactly to an absolute model time.
     *
     * The interval is split into equal forward-Euler steps no larger than the
     * user ceiling or diffusion CFL ceiling, whichever is smaller. Equal
     * subdivision avoids a pathologically tiny remainder step.
     *
     * @param final_time Absolute target time; must not precede time().
     * @param maximum_dt User-requested step ceiling. Reaction admissibility is
     * checked atomically by solve().
     */
    void solveUntil(double final_time, double maximum_dt) {
        if (!std::isfinite(final_time) || !std::isfinite(maximum_dt) || maximum_dt <= 0.0) {
            throw std::invalid_argument(
                "Final time and maximum dt must be finite; dt must be positive");
        }
        if (final_time < time_) {
            throw std::invalid_argument("Final time must not precede the current solver time");
        }
        if (final_time == time_) {
            return;
        }

        const double remaining = final_time - time_;
        const double effective_maximum_dt = std::min(maximum_dt, diffusionTimeStepLimit());
        const double step_count_value = std::ceil(remaining / effective_maximum_dt);
        if (!std::isfinite(step_count_value) ||
            step_count_value > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Requested interval requires too many explicit steps");
        }
        const int step_count = std::max(1, static_cast<int>(step_count_value));
        const double dt =
            std::min(effective_maximum_dt, remaining / static_cast<double>(step_count));
        solve(dt, step_count);
        time_ = final_time;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /**
     * @brief Get the solution for a specific species.
     */
    const std::vector<double>& solution(size_t species_idx) const {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        return species_[species_idx];
    }

    /**
     * @brief Get all species solutions.
     */
    const std::vector<std::vector<double>>& allSolutions() const { return species_; }

    /**
     * @brief Get the mesh.
     */
    const StructuredMesh& mesh() const { return mesh_; }

    /**
     * @brief Get the number of species.
     */
    size_t numSpecies() const { return num_species_; }

    /**
     * @brief Get diffusivity for a species.
     */
    double diffusivity(size_t species_idx) const {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        return diffusivities_[species_idx];
    }

    /**
     * @brief Get current simulation time.
     */
    double time() const { return time_; }

    /**
     * @brief Reset time to zero (without changing solution).
     */
    void resetTime() { time_ = 0.0; }

    /**
     * @brief Get total concentration across all species at a node.
     */
    double totalConcentration(int node_idx) const {
        validateNodeIndex(node_idx);
        double total = 0.0;
        for (size_t s = 0; s < num_species_; ++s) {
            total += species_[s][node_idx];
        }
        return total;
    }

    /**
     * @brief Get concentration of a species at a node.
     */
    double concentration(size_t species_idx, int node_idx) const {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        validateNodeIndex(node_idx);
        return species_[species_idx][node_idx];
    }

    /**
     * @brief Compute L2 norm of a species solution.
     */
    double solutionNorm(size_t species_idx) const {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        return std::sqrt(integrateNodalField(species_[species_idx], true));
    }

    /**
     * @brief Compute total mass (integral) of a species.
     */
    double totalMass(size_t species_idx) const {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        return integrateNodalField(species_[species_idx], false);
    }

private:
    struct ReactionWorkspace {
        std::vector<double> rates;
        std::vector<double> concentrations;
    };

    template <typename Model, typename Callable>
    void validateReactionModelArity(const Callable& model) const {
        if constexpr (std::is_same_v<Model, LotkaVolterraReaction> ||
                      std::is_same_v<Model, BrusselatorReaction>) {
            if (num_species_ < 2) {
                throw std::invalid_argument("Reaction model requires at least 2 species");
            }
        } else if constexpr (std::is_same_v<Model, SIRReaction> ||
                             std::is_same_v<Model, CompetitiveInhibitionReaction>) {
            if (num_species_ < 3) {
                throw std::invalid_argument("Reaction model requires at least 3 species");
            }
        } else if constexpr (std::is_same_v<Model, SEIRReaction>) {
            if (num_species_ < 4) {
                throw std::invalid_argument("SEIR reaction model requires at least 4 species");
            }
        } else if constexpr (std::is_same_v<Model, EnzymeCascadeReaction>) {
            if (num_species_ != model.numEnzymes()) {
                throw std::invalid_argument(
                    "Enzyme cascade species count must match the solver species count");
            }
        }
    }

    void prepareReactionWorkspaces() {
        std::size_t workspace_count = 1;
#ifdef BIOTRANSPORT_ENABLE_OPENMP
        workspace_count = static_cast<std::size_t>(omp_get_max_threads());
#endif
        reaction_workspaces_.resize(workspace_count);
        for (auto& workspace : reaction_workspaces_) {
            workspace.rates.resize(num_species_);
            workspace.concentrations.resize(num_species_);
        }
    }

    static Boundary checkedBoundary(int boundary_id) {
        if (boundary_id < to_index(Boundary::Left) || boundary_id > to_index(Boundary::Top)) {
            throw std::invalid_argument("Boundary ID must be in [0, 3]");
        }
        return static_cast<Boundary>(boundary_id);
    }

    void validateBoundary(Boundary boundary) const {
        const int id = to_index(boundary);
        if (id < to_index(Boundary::Left) || id > to_index(Boundary::Top)) {
            throw std::invalid_argument("Invalid boundary value");
        }
        if (mesh_.is1D() && (boundary == Boundary::Bottom || boundary == Boundary::Top)) {
            throw std::invalid_argument("Bottom and top boundaries do not exist on a 1D mesh");
        }
    }

    static void validateConcentrations(const std::vector<double>& values, const char* label) {
        for (double value : values) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(std::string(label) +
                                            " must contain finite, non-negative concentrations");
            }
        }
    }

    void validateNodeIndex(int node_idx) const {
        if (node_idx < 0 || node_idx >= mesh_.numNodes()) {
            throw std::out_of_range("Node index out of range");
        }
    }

    double diffusionTimeStepLimit() const {
        const double max_D = *std::max_element(diffusivities_.begin(), diffusivities_.end());
        if (max_D == 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        const double inv_dx2 = 1.0 / (mesh_.dx() * mesh_.dx());
        const double inverse_spacing_sum =
            mesh_.is1D() ? inv_dx2 : inv_dx2 + 1.0 / (mesh_.dy() * mesh_.dy());
        return 1.0 / (2.0 * max_D * inverse_spacing_sum);
    }

    static bool fixesValue(const BoundaryCondition& bc) {
        return bc.type == BoundaryType::DIRICHLET ||
               (bc.type == BoundaryType::ROBIN && bc.b == 0.0);
    }

    static double fixedValue(const BoundaryCondition& bc) {
        return bc.type == BoundaryType::DIRICHLET ? bc.value : bc.c / bc.a;
    }

    static bool nearlyEqual(double lhs, double rhs) {
        const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
        return std::abs(lhs - rhs) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
    }

    void validateBoundaryCondition(const BoundaryCondition& bc) const {
        if (bc.type == BoundaryType::DIRICHLET) {
            if (!std::isfinite(bc.value) || bc.value < 0.0) {
                throw std::invalid_argument(
                    "Dirichlet concentration must be finite and non-negative");
            }
        } else if (bc.type == BoundaryType::NEUMANN) {
            if (!std::isfinite(bc.value)) {
                throw std::invalid_argument("Neumann derivative must be finite");
            }
        } else {
            if (!std::isfinite(bc.a) || !std::isfinite(bc.b) || !std::isfinite(bc.c) ||
                (bc.a == 0.0 && bc.b == 0.0)) {
                throw std::invalid_argument("Robin coefficients must be finite and not both zero");
            }
            if (bc.b == 0.0 && fixedValue(bc) < 0.0) {
                throw std::invalid_argument("Robin boundary fixes a negative concentration");
            }
        }
    }

    void validateCorner(size_t species_idx, Boundary first, Boundary second) const {
        const auto& a = boundary_conditions_[species_idx][to_index(first)];
        const auto& b = boundary_conditions_[species_idx][to_index(second)];
        if (fixesValue(a) && fixesValue(b) && !nearlyEqual(fixedValue(a), fixedValue(b))) {
            throw std::invalid_argument("Conflicting fixed concentration values at a 2D corner");
        }
    }

    void validateBoundaryConfiguration() const {
        for (size_t s = 0; s < num_species_; ++s) {
            for (int b = 0; b < 4; ++b) {
                if (!mesh_.is1D() || b < 2) {
                    validateBoundaryCondition(boundary_conditions_[s][b]);
                }
            }
            if (!mesh_.is1D()) {
                validateCorner(s, Boundary::Left, Boundary::Bottom);
                validateCorner(s, Boundary::Left, Boundary::Top);
                validateCorner(s, Boundary::Right, Boundary::Bottom);
                validateCorner(s, Boundary::Right, Boundary::Top);
            }
        }
    }

    bool fixedValueAtNode(size_t species_idx, int i, int j, double& value) const {
        const auto& bcs = boundary_conditions_[species_idx];
        if (i == 0 && fixesValue(bcs[to_index(Boundary::Left)])) {
            value = fixedValue(bcs[to_index(Boundary::Left)]);
            return true;
        }
        if (i == mesh_.nx() && fixesValue(bcs[to_index(Boundary::Right)])) {
            value = fixedValue(bcs[to_index(Boundary::Right)]);
            return true;
        }
        if (!mesh_.is1D() && j == 0 && fixesValue(bcs[to_index(Boundary::Bottom)])) {
            value = fixedValue(bcs[to_index(Boundary::Bottom)]);
            return true;
        }
        if (!mesh_.is1D() && j == mesh_.ny() && fixesValue(bcs[to_index(Boundary::Top)])) {
            value = fixedValue(bcs[to_index(Boundary::Top)]);
            return true;
        }
        return false;
    }

    double currentValueAtNode(size_t species_idx, int i, int j) const {
        double prescribed_value = 0.0;
        if (fixedValueAtNode(species_idx, i, j, prescribed_value)) {
            return prescribed_value;
        }
        const int node = mesh_.is1D() ? i : mesh_.index(i, j);
        return species_[species_idx][static_cast<std::size_t>(node)];
    }

    static double normalDerivative(const BoundaryCondition& bc, double boundary_value) {
        if (bc.type == BoundaryType::NEUMANN) {
            return bc.value;
        }
        if (bc.type == BoundaryType::ROBIN && bc.b != 0.0) {
            return (bc.c - bc.a * boundary_value) / bc.b;
        }
        throw std::logic_error("A fixed-value boundary has no derivative update");
    }

    double diffusionRate(size_t species_idx, int idx, int i, int j) const {
        const double D = diffusivities_[species_idx];
        if (D == 0.0) {
            return 0.0;
        }

        const auto& bcs = boundary_conditions_[species_idx];
        const double center = species_[species_idx][idx];
        const double dx = mesh_.dx();
        double laplacian = 0.0;

        if (i == 0) {
            const double q = normalDerivative(bcs[to_index(Boundary::Left)], center);
            laplacian +=
                2.0 * (currentValueAtNode(species_idx, 1, j) - center) / (dx * dx) + 2.0 * q / dx;
        } else if (i == mesh_.nx()) {
            const double q = normalDerivative(bcs[to_index(Boundary::Right)], center);
            laplacian += 2.0 * (currentValueAtNode(species_idx, i - 1, j) - center) / (dx * dx) +
                         2.0 * q / dx;
        } else {
            laplacian += (currentValueAtNode(species_idx, i + 1, j) - 2.0 * center +
                          currentValueAtNode(species_idx, i - 1, j)) /
                         (dx * dx);
        }

        if (!mesh_.is1D()) {
            const double dy = mesh_.dy();
            if (j == 0) {
                const double q = normalDerivative(bcs[to_index(Boundary::Bottom)], center);
                laplacian += 2.0 * (currentValueAtNode(species_idx, i, 1) - center) / (dy * dy) +
                             2.0 * q / dy;
            } else if (j == mesh_.ny()) {
                const double q = normalDerivative(bcs[to_index(Boundary::Top)], center);
                laplacian +=
                    2.0 * (currentValueAtNode(species_idx, i, j - 1) - center) / (dy * dy) +
                    2.0 * q / dy;
            } else {
                laplacian += (currentValueAtNode(species_idx, i, j + 1) - 2.0 * center +
                              currentValueAtNode(species_idx, i, j - 1)) /
                             (dy * dy);
            }
        }
        return D * laplacian;
    }

    void computeNodeUpdate(int idx, int i, int j, double dt) {
        ReactionWorkspace* workspace = nullptr;
        if (reaction_) {
            std::size_t workspace_index = 0;
#ifdef BIOTRANSPORT_ENABLE_OPENMP
            workspace_index = static_cast<std::size_t>(omp_get_thread_num());
#endif
            workspace = &reaction_workspaces_[workspace_index];

            for (size_t s = 0; s < num_species_; ++s) {
                // Reaction coupling at a fixed-concentration boundary must see
                // the prescribed value even if the caller's initial array did
                // not already satisfy that boundary condition.
                workspace->concentrations[s] = currentValueAtNode(s, i, j);
            }

            std::fill(workspace->rates.begin(), workspace->rates.end(), 0.0);
            reaction_(workspace->rates, workspace->concentrations, x_coords_[i],
                      mesh_.is1D() ? 0.0 : y_coords_[j], time_);
            if (workspace->rates.size() != num_species_) {
                throw std::runtime_error("Reaction callback changed the size of the rate vector");
            }
        }

        for (size_t s = 0; s < num_species_; ++s) {
            double prescribed_value = 0.0;
            if (fixedValueAtNode(s, i, j, prescribed_value)) {
                scratch_[s][idx] = prescribed_value;
            } else {
                const double reaction_rate = workspace == nullptr ? 0.0 : workspace->rates[s];
                scratch_[s][idx] =
                    species_[s][idx] + dt * (diffusionRate(s, idx, i, j) + reaction_rate);
            }
        }
    }

    void computeCandidateStep(double dt) {
        if (mesh_.is1D()) {
            if (!reaction_ || reaction_thread_safe_) {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    computeNodeUpdate(i, i, 0, dt);
                }
            } else {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    computeNodeUpdate(i, i, 0, dt);
                }
            }
            return;
        }

        const int stride = mesh_.nx() + 1;
        if (!reaction_ || reaction_thread_safe_) {
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j <= mesh_.ny(); ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    computeNodeUpdate(j * stride + i, i, j, dt);
                }
            }
        } else {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    computeNodeUpdate(j * stride + i, i, j, dt);
                }
            }
        }
    }

    void validateCurrentState() const {
        for (const auto& field : species_) {
            validateConcentrations(field, "Current state");
        }
    }

    void validateCandidateStep(double dt) {
        constexpr double roundoff_factor = 64.0;
        for (size_t s = 0; s < num_species_; ++s) {
            for (size_t idx = 0; idx < scratch_[s].size(); ++idx) {
                double& candidate = scratch_[s][idx];
                if (!std::isfinite(candidate)) {
                    throw std::runtime_error("Non-finite concentration produced for species " +
                                             std::to_string(s) + " at node " + std::to_string(idx));
                }
                const double scale = std::max(1.0, std::abs(species_[s][idx]));
                const double tolerance =
                    roundoff_factor * std::numeric_limits<double>::epsilon() * scale;
                if (candidate < -tolerance) {
                    const double rate = (candidate - species_[s][idx]) / dt;
                    const double local_limit = rate < 0.0 ? species_[s][idx] / (-rate) : 0.0;
                    throw std::runtime_error(
                        "Forward Euler produced a negative concentration for species " +
                        std::to_string(s) + " at node " + std::to_string(idx) +
                        "; reduce dt below the local positivity limit " +
                        std::to_string(local_limit));
                }
                if (candidate < 0.0) {
                    candidate = 0.0;
                }
            }
        }
    }

    double integrateNodalField(const std::vector<double>& field, bool square_values) const {
        double weighted_sum = 0.0;
        if (mesh_.is1D()) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                const double weight = (i == 0 || i == mesh_.nx()) ? 0.5 : 1.0;
                const double value = field[static_cast<size_t>(i)];
                weighted_sum += weight * (square_values ? value * value : value);
            }
            return weighted_sum * mesh_.dx();
        }

        for (int j = 0; j <= mesh_.ny(); ++j) {
            const double wy = (j == 0 || j == mesh_.ny()) ? 0.5 : 1.0;
            for (int i = 0; i <= mesh_.nx(); ++i) {
                const double wx = (i == 0 || i == mesh_.nx()) ? 0.5 : 1.0;
                const double value = field[static_cast<size_t>(mesh_.index(i, j))];
                weighted_sum += wx * wy * (square_values ? value * value : value);
            }
        }
        return weighted_sum * mesh_.dx() * mesh_.dy();
    }

    void cacheCoordinates() {
        x_coords_.resize(mesh_.nx() + 1);
        for (int i = 0; i <= mesh_.nx(); ++i) {
            x_coords_[i] = mesh_.x(i);
        }
        if (!mesh_.is1D()) {
            y_coords_.resize(mesh_.ny() + 1);
            for (int j = 0; j <= mesh_.ny(); ++j) {
                y_coords_[j] = mesh_.y(0, j);
            }
        }
    }

    // Mesh geometry is borrowed; language bindings keep it alive with the solver.
    const StructuredMesh& mesh_;

    // Species data
    std::vector<double> diffusivities_;
    size_t num_species_;
    std::vector<std::vector<double>> species_;  // species_[s][idx]
    std::vector<std::vector<double>> scratch_;

    // Boundary conditions: boundary_conditions_[species][boundary]
    std::vector<std::array<BoundaryCondition, 4>> boundary_conditions_;

    // Reaction
    ReactionFunction reaction_;
    bool reaction_thread_safe_ = false;

    // Time tracking
    double time_;

    // Cached coordinates
    std::vector<double> x_coords_;
    std::vector<double> y_coords_;

    // One reusable reaction workspace per OpenMP worker. Built-in immutable
    // models may run in parallel; arbitrary callbacks are evaluated serially.
    std::vector<ReactionWorkspace> reaction_workspaces_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_MULTI_SPECIES_SOLVER_HPP
