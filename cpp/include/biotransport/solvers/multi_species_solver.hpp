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
 *   // 3-species Lotka-Volterra system
 *   MultiSpeciesSolver solver(mesh, {D1, D2, D3}, 3);
 *   solver.setReactionFunction(LotkaVolterraReaction(alpha, beta, gamma, delta));
 *   solver.setInitialCondition(0, prey_ic);   // Species 0: prey
 *   solver.setInitialCondition(1, pred_ic);   // Species 1: predator
 *   solver.setInitialCondition(2, super_ic);  // Species 2: super-predator
 *   solver.solve(dt, num_steps);
 * @endcode
 */

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/mesh_iterators.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

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
 *   α (alpha) = prey growth rate
 *   β (beta)  = predation rate
 *   γ (gamma) = predator death rate
 *   δ (delta) = predator reproduction rate from prey
 *   K         = prey carrying capacity (default = 100)
 */
class LotkaVolterraReaction {
public:
    LotkaVolterraReaction(double alpha, double beta, double gamma, double delta,
                          double carrying_capacity = 100.0)
        : alpha_(alpha), beta_(beta), gamma_(gamma), delta_(delta), K_(carrying_capacity) {
        if (alpha < 0 || beta < 0 || gamma < 0 || delta < 0) {
            throw std::invalid_argument("All Lotka-Volterra parameters must be non-negative");
        }
        if (K_ <= 0) {
            throw std::invalid_argument("Carrying capacity must be positive");
        }
    }

    void operator()(std::vector<double>& rates, const std::vector<double>& u, double /*x*/,
                    double /*y*/, double /*t*/) const {
        if (u.size() < 2 || rates.size() < 2) {
            throw std::runtime_error("Lotka-Volterra requires at least 2 species");
        }
        double prey = std::max(0.0, u[0]);
        double pred = std::max(0.0, u[1]);

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
 *   β (beta)  = transmission rate
 *   γ (gamma) = recovery rate
 *   N         = total population (for normalization, typically S+I+R at t=0)
 *
 * Note: R₀ = β/γ is the basic reproduction number.
 */
class SIRReaction {
public:
    SIRReaction(double beta, double gamma, double total_population)
        : beta_(beta), gamma_(gamma), N_(total_population) {
        if (beta < 0 || gamma < 0) {
            throw std::invalid_argument("SIR parameters must be non-negative");
        }
        if (N_ <= 0) {
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
    double R0() const { return beta_ / gamma_; }  // Basic reproduction number

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
 *   β (beta)  = transmission rate
 *   σ (sigma) = rate of becoming infectious (1/incubation period)
 *   γ (gamma) = recovery rate
 *   N         = total population
 */
class SEIRReaction {
public:
    SEIRReaction(double beta, double sigma, double gamma, double total_population)
        : beta_(beta), sigma_(sigma), gamma_(gamma), N_(total_population) {
        if (beta < 0 || sigma < 0 || gamma < 0) {
            throw std::invalid_argument("SEIR parameters must be non-negative");
        }
        if (N_ <= 0) {
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
 *   vmax_values = maximum reaction rates for each step
 *   km_values   = Michaelis constants for each step
 *   kdeg_values = degradation rates for each species
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
            if (vmax_[i] < 0 || km_[i] <= 0) {
                throw std::invalid_argument("Invalid enzyme kinetic parameters");
            }
        }
        for (double k : kdeg_) {
            if (k < 0) {
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
 *   Vmax = maximum reaction velocity
 *   Km   = Michaelis constant for substrate
 *   Ki   = inhibition constant
 */
class CompetitiveInhibitionReaction {
public:
    CompetitiveInhibitionReaction(double vmax, double km, double ki, double inhibitor_decay = 0.0)
        : vmax_(vmax), km_(km), ki_(ki), inhibitor_decay_(inhibitor_decay) {
        if (vmax < 0 || km <= 0 || ki <= 0) {
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
 * Parameters:
 *   A, B = kinetic parameters
 */
class BrusselatorReaction {
public:
    BrusselatorReaction(double A, double B) : A_(A), B_(B) {
        if (A <= 0 || B <= 0) {
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
 * The solver uses explicit time-stepping with the standard CFL stability
 * condition based on the maximum diffusivity.
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
          iterator_(mesh),
          stencil_ops_(mesh),
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
            if (D < 0.0) {
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

        // Default boundary conditions (Dirichlet, value = 0) for all species
        boundary_conditions_.resize(num_species_);
        for (size_t s = 0; s < num_species_; ++s) {
            for (int b = 0; b < 4; ++b) {
                boundary_conditions_[s][b] = BoundaryCondition::Dirichlet(0.0);
            }
        }

        // Pre-cache coordinates for reaction function evaluation
        cacheCoordinates();

        // Allocate temporary vectors for reaction evaluation
        reaction_rates_.resize(num_species_);
        point_concentrations_.resize(num_species_);
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /**
     * @brief Set the reaction function for all species.
     */
    void setReactionFunction(ReactionFunction reaction) { reaction_ = std::move(reaction); }

    /**
     * @brief Set the reaction function from a callable object.
     */
    template <typename Callable>
    void setReactionModel(Callable&& model) {
        reaction_ = [model = std::forward<Callable>(model)](std::vector<double>& rates,
                                                            const std::vector<double>& u, double x,
                                                            double y, double t) {
            model(rates, u, x, y, t);
        };
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
        species_[species_idx] = values;
    }

    /**
     * @brief Set initial condition for all species from a single value.
     */
    void setUniformInitialCondition(size_t species_idx, double value) {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
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
        boundary_conditions_[species_idx][to_index(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(size_t species_idx, int boundary_id, double value) {
        setDirichletBoundary(species_idx, static_cast<Boundary>(boundary_id), value);
    }

    /**
     * @brief Set Neumann boundary condition for a specific species.
     */
    void setNeumannBoundary(size_t species_idx, Boundary boundary, double flux) {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        boundary_conditions_[species_idx][to_index(boundary)] = BoundaryCondition::Neumann(flux);
    }

    void setNeumannBoundary(size_t species_idx, int boundary_id, double flux) {
        setNeumannBoundary(species_idx, static_cast<Boundary>(boundary_id), flux);
    }

    /**
     * @brief Set the same boundary condition for all species on a boundary.
     */
    void setAllSpeciesDirichlet(Boundary boundary, double value) {
        for (size_t s = 0; s < num_species_; ++s) {
            setDirichletBoundary(s, boundary, value);
        }
    }

    void setAllSpeciesNeumann(Boundary boundary, double flux) {
        for (size_t s = 0; s < num_species_; ++s) {
            setNeumannBoundary(s, boundary, flux);
        }
    }

    // -------------------------------------------------------------------------
    // Solution
    // -------------------------------------------------------------------------

    /**
     * @brief Check CFL stability condition.
     *
     * For explicit schemes, we need dt ≤ h²/(2*D*dim) for each species.
     * We use the maximum diffusivity to determine the constraint.
     */
    bool checkStability(double dt) const {
        double max_D = *std::max_element(diffusivities_.begin(), diffusivities_.end());
        if (max_D == 0.0) {
            return true;  // Pure reaction, no diffusion stability constraint
        }

        double dx = mesh_.dx();
        double dx2 = dx * dx;

        if (mesh_.is1D()) {
            return dt <= dx2 / (2.0 * max_D);
        } else {
            double dy = mesh_.dy();
            double dy2 = dy * dy;
            double factor = 2.0 * max_D * (1.0 / dx2 + 1.0 / dy2);
            return dt <= 1.0 / factor;
        }
    }

    /**
     * @brief Get the maximum stable time step.
     */
    double maxStableTimeStep() const {
        double max_D = *std::max_element(diffusivities_.begin(), diffusivities_.end());
        if (max_D == 0.0) {
            return std::numeric_limits<double>::infinity();
        }

        double dx = mesh_.dx();
        double dx2 = dx * dx;

        if (mesh_.is1D()) {
            return 0.4 * dx2 / (2.0 * max_D);  // 0.4 safety factor
        } else {
            double dy = mesh_.dy();
            double dy2 = dy * dy;
            double factor = 2.0 * max_D * (1.0 / dx2 + 1.0 / dy2);
            return 0.4 / factor;
        }
    }

    /**
     * @brief Run the solver for the specified number of steps.
     */
    void solve(double dt, int num_steps) {
        if (dt <= 0.0 || num_steps <= 0) {
            throw std::invalid_argument("Time step and number of steps must be positive");
        }

        if (!checkStability(dt)) {
            double max_dt = maxStableTimeStep();
            throw std::runtime_error("Time step " + std::to_string(dt) +
                                     " exceeds stability limit. "
                                     "Maximum stable dt = " +
                                     std::to_string(max_dt));
        }

        for (int step = 0; step < num_steps; ++step) {
            // Compute updates for all interior nodes
            iterator_.forEachInterior(
                [this, dt](int idx, int i, int j) { computeNodeUpdate(idx, i, j, dt); });

            // Apply boundary conditions for each species
            for (size_t s = 0; s < num_species_; ++s) {
                applyBoundaryConditions(s, scratch_[s]);
            }

            // Swap buffers
            for (size_t s = 0; s < num_species_; ++s) {
                species_[s].swap(scratch_[s]);
            }

            time_ += dt;
        }
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
        return species_[species_idx][node_idx];
    }

    /**
     * @brief Compute L2 norm of a species solution.
     */
    double solutionNorm(size_t species_idx) const {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        double sum_sq = 0.0;
        for (double val : species_[species_idx]) {
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq);
    }

    /**
     * @brief Compute total mass (integral) of a species.
     */
    double totalMass(size_t species_idx) const {
        if (species_idx >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        double sum = 0.0;
        for (double val : species_[species_idx]) {
            sum += val;
        }
        // Multiply by cell area for proper integration
        double cell_area = mesh_.dx() * (mesh_.is1D() ? 1.0 : mesh_.dy());
        return sum * cell_area;
    }

private:
    void computeNodeUpdate(int idx, int i, int j, double dt) {
        double x = x_coords_[i];
        double y = mesh_.is1D() ? 0.0 : y_coords_[j];

        // Gather concentrations at this point
        for (size_t s = 0; s < num_species_; ++s) {
            point_concentrations_[s] = species_[s][idx];
        }

        // Compute reaction rates
        std::fill(reaction_rates_.begin(), reaction_rates_.end(), 0.0);
        if (reaction_) {
            reaction_(reaction_rates_, point_concentrations_, x, y, time_);
        }

        // Update each species: diffusion + reaction
        for (size_t s = 0; s < num_species_; ++s) {
            double u = species_[s][idx];
            double diffusion = 0.0;

            if (diffusivities_[s] > 0.0) {
                diffusion = stencil_ops_.diffusionTerm(species_[s], idx, diffusivities_[s], dt);
            }

            scratch_[s][idx] = u + diffusion + dt * reaction_rates_[s];
        }
    }

    void applyBoundaryConditions(size_t species_idx, std::vector<double>& field) {
        const auto& bcs = boundary_conditions_[species_idx];

        if (mesh_.is1D()) {
            // 1D: only left and right boundaries
            applyBC1D(bcs[to_index(Boundary::Left)], 0, field);
            applyBC1D(bcs[to_index(Boundary::Right)], mesh_.nx(), field);
        } else {
            // 2D: all four boundaries
            // Bottom (j = 0)
            for (int i = 0; i <= mesh_.nx(); ++i) {
                int idx = mesh_.index(i, 0);
                applyBC2D(bcs[to_index(Boundary::Bottom)], idx, i, 0, field);
            }
            // Top (j = ny)
            for (int i = 0; i <= mesh_.nx(); ++i) {
                int idx = mesh_.index(i, mesh_.ny());
                applyBC2D(bcs[to_index(Boundary::Top)], idx, i, mesh_.ny(), field);
            }
            // Left (i = 0)
            for (int j = 0; j <= mesh_.ny(); ++j) {
                int idx = mesh_.index(0, j);
                applyBC2D(bcs[to_index(Boundary::Left)], idx, 0, j, field);
            }
            // Right (i = nx)
            for (int j = 0; j <= mesh_.ny(); ++j) {
                int idx = mesh_.index(mesh_.nx(), j);
                applyBC2D(bcs[to_index(Boundary::Right)], idx, mesh_.nx(), j, field);
            }
        }
    }

    void applyBC1D(const BoundaryCondition& bc, int i, std::vector<double>& field) {
        int idx = i;  // 1D indexing
        switch (bc.type) {
            case BoundaryType::DIRICHLET:
                field[idx] = bc.value;
                break;
            case BoundaryType::NEUMANN:
                if (i == 0) {
                    field[idx] = field[idx + 1] - bc.value * mesh_.dx();
                } else {
                    field[idx] = field[idx - 1] + bc.value * mesh_.dx();
                }
                break;
            case BoundaryType::ROBIN:
                // Robin: a*u + b*du/dn = g
                // For now, treat as Dirichlet with bc.value
                field[idx] = bc.value;
                break;
        }
    }

    void applyBC2D(const BoundaryCondition& bc, int idx, int i, int j, std::vector<double>& field) {
        switch (bc.type) {
            case BoundaryType::DIRICHLET:
                field[idx] = bc.value;
                break;
            case BoundaryType::NEUMANN: {
                // Determine which neighbor to use based on boundary location
                int neighbor_idx = -1;
                double h = 0.0;
                if (j == 0) {  // Bottom
                    neighbor_idx = mesh_.index(i, 1);
                    h = mesh_.dy();
                } else if (j == mesh_.ny()) {  // Top
                    neighbor_idx = mesh_.index(i, mesh_.ny() - 1);
                    h = mesh_.dy();
                } else if (i == 0) {  // Left
                    neighbor_idx = mesh_.index(1, j);
                    h = mesh_.dx();
                } else if (i == mesh_.nx()) {  // Right
                    neighbor_idx = mesh_.index(mesh_.nx() - 1, j);
                    h = mesh_.dx();
                }
                if (neighbor_idx >= 0) {
                    field[idx] = field[neighbor_idx] + bc.value * h;
                }
                break;
            }
            case BoundaryType::ROBIN:
                field[idx] = bc.value;
                break;
        }
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

    // Mesh and iteration helpers
    const StructuredMesh& mesh_;
    MeshIterator iterator_;
    StencilOps stencil_ops_;

    // Species data
    std::vector<double> diffusivities_;
    size_t num_species_;
    std::vector<std::vector<double>> species_;  // species_[s][idx]
    std::vector<std::vector<double>> scratch_;

    // Boundary conditions: boundary_conditions_[species][boundary]
    std::vector<std::array<BoundaryCondition, 4>> boundary_conditions_;

    // Reaction
    ReactionFunction reaction_;

    // Time tracking
    double time_;

    // Cached coordinates
    std::vector<double> x_coords_;
    std::vector<double> y_coords_;

    // Temporary storage for reaction evaluation (avoid allocation in loop)
    mutable std::vector<double> reaction_rates_;
    mutable std::vector<double> point_concentrations_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_MULTI_SPECIES_SOLVER_HPP
