#ifndef BIOTRANSPORT_SOLVERS_NERNST_PLANCK_SOLVER_HPP
#define BIOTRANSPORT_SOLVERS_NERNST_PLANCK_SOLVER_HPP

/**
 * @file nernst_planck_solver.hpp
 * @brief Nernst-Planck solver for electrochemical ion transport.
 *
 * The Nernst-Planck equation describes the transport of charged species
 * under the influence of both concentration gradients and electric fields:
 *
 *   ∂c_i/∂t = ∇·(D_i ∇c_i) + ∇·(z_i F D_i c_i ∇φ / RT)
 *
 * where:
 *   c_i = concentration of species i [mol/m³]
 *   D_i = diffusion coefficient [m²/s]
 *   z_i = ion valence (charge number)
 *   F   = Faraday constant (96485 C/mol)
 *   R   = gas constant (8.314 J/(mol·K))
 *   T   = temperature [K]
 *   φ   = electric potential [V]
 *
 * Applications:
 *   - Ion channels and membrane transport
 *   - Neural action potentials
 *   - Battery electrolytes
 *   - Electrophoresis
 *   - Drug iontophoresis
 *
 * This implementation supports:
 *   1. Single ion transport with prescribed electric field
 *   2. Multi-ion transport with electroneutrality constraint
 *   3. Full Poisson-Nernst-Planck (PNP) coupling (optional)
 *
 * @author BioTransport Development Team
 * @date December 2025
 */

#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/mesh_iterators.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

// Define M_PI if not available (MSVC doesn't define it by default)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace biotransport {

// =============================================================================
// Physical Constants
// =============================================================================

namespace constants {
constexpr double FARADAY = 96485.33212;                   ///< Faraday constant [C/mol]
constexpr double GAS_CONSTANT = 8.31446;                  ///< Gas constant [J/(mol·K)]
constexpr double BOLTZMANN = 1.380649e-23;                ///< Boltzmann constant [J/K]
constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;     ///< [C]
constexpr double VACUUM_PERMITTIVITY = 8.8541878128e-12;  ///< [F/m]
}  // namespace constants

// =============================================================================
// Ion Species Definition
// =============================================================================

/**
 * @brief Represents a single ion species with its transport properties.
 */
struct IonSpecies {
    std::string name;    ///< Species name (e.g., "Na+", "K+", "Cl-")
    int valence;         ///< Ion valence (z): +1 for Na+, -1 for Cl-, etc.
    double diffusivity;  ///< Diffusion coefficient [m²/s]
    double mobility;     ///< Electrical mobility [m²/(V·s)], computed from D

    /**
     * @brief Create an ion species.
     * @param name Species identifier
     * @param valence Ion charge number
     * @param diffusivity Diffusion coefficient [m²/s]
     * @param temperature Temperature [K] for mobility calculation (default 310K body temp)
     */
    IonSpecies(const std::string& name, int valence, double diffusivity, double temperature = 310.0)
        : name(name), valence(valence), diffusivity(diffusivity) {
        if (valence == 0) {
            throw std::invalid_argument("Ion valence cannot be zero - use regular diffusion");
        }
        if (diffusivity <= 0.0) {
            throw std::invalid_argument("Diffusion coefficient must be positive");
        }
        // Einstein relation: μ = |z|eD/(kT) = |z|FD/(RT)
        mobility = std::abs(valence) * constants::FARADAY * diffusivity /
                   (constants::GAS_CONSTANT * temperature);
    }

    /**
     * @brief Get thermal voltage RT/F at given temperature.
     */
    static double thermalVoltage(double temperature) {
        return constants::GAS_CONSTANT * temperature / constants::FARADAY;
    }
};

// =============================================================================
// Common Ion Species (Physiological)
// =============================================================================

namespace ions {
// Diffusion coefficients at 37°C (310K) in aqueous solution [m²/s]
inline IonSpecies sodium() {
    return IonSpecies("Na+", +1, 1.33e-9);
}
inline IonSpecies potassium() {
    return IonSpecies("K+", +1, 1.96e-9);
}
inline IonSpecies chloride() {
    return IonSpecies("Cl-", -1, 2.03e-9);
}
inline IonSpecies calcium() {
    return IonSpecies("Ca2+", +2, 0.79e-9);
}
inline IonSpecies magnesium() {
    return IonSpecies("Mg2+", +2, 0.71e-9);
}
inline IonSpecies hydrogen() {
    return IonSpecies("H+", +1, 9.31e-9);
}
inline IonSpecies hydroxide() {
    return IonSpecies("OH-", -1, 5.27e-9);
}
inline IonSpecies bicarbonate() {
    return IonSpecies("HCO3-", -1, 1.18e-9);
}
}  // namespace ions

// =============================================================================
// Electric Potential Models
// =============================================================================

/**
 * @brief Base class for electric potential specification.
 */
class PotentialField {
public:
    virtual ~PotentialField() = default;

    /**
     * @brief Get potential at position (x, y) at time t.
     */
    virtual double operator()(double x, double y, double t) const = 0;

    /**
     * @brief Get potential gradient in x-direction.
     */
    virtual double gradX(double x, double y, double t) const = 0;

    /**
     * @brief Get potential gradient in y-direction.
     */
    virtual double gradY(double x, double y, double t) const = 0;
};

/**
 * @brief Uniform electric field (constant gradient).
 */
class UniformField : public PotentialField {
public:
    /**
     * @brief Create uniform field with specified gradients.
     * @param Ex Electric field in x-direction [V/m] (negative of potential gradient)
     * @param Ey Electric field in y-direction [V/m]
     */
    UniformField(double Ex, double Ey = 0.0) : Ex_(Ex), Ey_(Ey) {}

    double operator()(double x, double y, double /*t*/) const override {
        return -Ex_ * x - Ey_ * y;
    }

    double gradX(double /*x*/, double /*y*/, double /*t*/) const override {
        return -Ex_;  // ∇φ = -E
    }

    double gradY(double /*x*/, double /*y*/, double /*t*/) const override { return -Ey_; }

private:
    double Ex_, Ey_;
};

/**
 * @brief Time-varying sinusoidal field (e.g., AC stimulation).
 */
class ACField : public PotentialField {
public:
    /**
     * @brief Create AC field.
     * @param amplitude Peak electric field [V/m]
     * @param frequency Frequency [Hz]
     * @param direction 0=x, 1=y
     */
    ACField(double amplitude, double frequency, int direction = 0)
        : amplitude_(amplitude), omega_(2.0 * M_PI * frequency), dir_(direction) {}

    double operator()(double x, double y, double t) const override {
        double coord = (dir_ == 0) ? x : y;
        return -amplitude_ * std::sin(omega_ * t) * coord;
    }

    double gradX(double /*x*/, double /*y*/, double t) const override {
        return (dir_ == 0) ? -amplitude_ * std::sin(omega_ * t) : 0.0;
    }

    double gradY(double /*x*/, double /*y*/, double t) const override {
        return (dir_ == 1) ? -amplitude_ * std::sin(omega_ * t) : 0.0;
    }

private:
    double amplitude_, omega_;
    int dir_;
};

/**
 * @brief User-defined potential field via lambda.
 */
class CustomPotential : public PotentialField {
public:
    using PotentialFunc = std::function<double(double x, double y, double t)>;

    /**
     * @brief Create custom potential with analytical gradients.
     */
    CustomPotential(PotentialFunc phi, PotentialFunc grad_x, PotentialFunc grad_y)
        : phi_(std::move(phi)), grad_x_(std::move(grad_x)), grad_y_(std::move(grad_y)) {}

    /**
     * @brief Create custom potential with numerical gradient.
     */
    explicit CustomPotential(PotentialFunc phi, double eps = 1e-8)
        : phi_(std::move(phi)), eps_(eps), use_numerical_grad_(true) {}

    double operator()(double x, double y, double t) const override { return phi_(x, y, t); }

    double gradX(double x, double y, double t) const override {
        if (use_numerical_grad_) {
            return (phi_(x + eps_, y, t) - phi_(x - eps_, y, t)) / (2.0 * eps_);
        }
        return grad_x_(x, y, t);
    }

    double gradY(double x, double y, double t) const override {
        if (use_numerical_grad_) {
            return (phi_(x, y + eps_, t) - phi_(x, y - eps_, t)) / (2.0 * eps_);
        }
        return grad_y_(x, y, t);
    }

private:
    PotentialFunc phi_;
    PotentialFunc grad_x_ = nullptr;
    PotentialFunc grad_y_ = nullptr;
    double eps_ = 1e-8;
    bool use_numerical_grad_ = false;
};

// =============================================================================
// Nernst-Planck Solver (Single Species)
// =============================================================================

/**
 * @brief Solver for single-ion Nernst-Planck transport.
 *
 * Solves: ∂c/∂t = D∇²c + (zFD/RT) ∇·(c ∇φ)
 *
 * Expanding the electromigration term:
 *   ∇·(c ∇φ) = ∇c · ∇φ + c ∇²φ
 *
 * Using finite differences:
 *   dc/dt = D ∇²c + (zFD/RT) [∇c · ∇φ + c ∇²φ]
 *
 * The electromigration is discretized using upwind differencing for stability.
 */
class NernstPlanckSolver {
public:
    /**
     * @brief Construct solver for a single ion species.
     * @param mesh The computational mesh
     * @param ion The ion species parameters
     * @param temperature Temperature [K] (default 310K = body temperature)
     */
    NernstPlanckSolver(const StructuredMesh& mesh, const IonSpecies& ion,
                       double temperature = 310.0)
        : mesh_(mesh), ion_(ion), temperature_(temperature), iterator_(mesh), stencil_ops_(mesh) {
        solution_.resize(mesh.numNodes(), 0.0);
        scratch_.resize(mesh.numNodes(), 0.0);
        potential_.resize(mesh.numNodes(), 0.0);

        // Precompute thermal factor
        Vt_ = IonSpecies::thermalVoltage(temperature);
        zeta_ = static_cast<double>(ion_.valence) / Vt_;  // z*F/(R*T)

        // Default boundary conditions
        for (int i = 0; i < 4; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Dirichlet(0.0);
        }
    }

    // -------------------------------------------------------------------------
    // Setup Methods
    // -------------------------------------------------------------------------

    /**
     * @brief Set initial concentration field.
     */
    void setInitialCondition(const std::vector<double>& values) {
        if (values.size() != solution_.size()) {
            throw std::invalid_argument("Initial condition size mismatch");
        }
        solution_ = values;
    }

    /**
     * @brief Set electric potential field (static).
     */
    void setPotentialField(const std::vector<double>& phi) {
        if (phi.size() != potential_.size()) {
            throw std::invalid_argument("Potential field size mismatch");
        }
        potential_ = phi;
        use_potential_function_ = false;
    }

    /**
     * @brief Set electric potential from analytical function.
     */
    void setPotentialField(std::shared_ptr<PotentialField> field) {
        potential_func_ = std::move(field);
        use_potential_function_ = true;
        updatePotentialFromFunction(0.0);
    }

    /**
     * @brief Set uniform electric field.
     * @param Ex Electric field in x [V/m]
     * @param Ey Electric field in y [V/m]
     */
    void setUniformField(double Ex, double Ey = 0.0) {
        setPotentialField(std::make_shared<UniformField>(Ex, Ey));
    }

    /**
     * @brief Set Dirichlet (fixed concentration) boundary.
     */
    void setDirichletBoundary(Boundary boundary, double value) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        setDirichletBoundary(static_cast<Boundary>(boundary_id), value);
    }

    /**
     * @brief Set Neumann (flux) boundary.
     */
    void setNeumannBoundary(Boundary boundary, double flux) {
        boundary_conditions_[to_index(boundary)] = BoundaryCondition::Neumann(flux);
    }

    void setNeumannBoundary(int boundary_id, double flux) {
        setNeumannBoundary(static_cast<Boundary>(boundary_id), flux);
    }

    // -------------------------------------------------------------------------
    // Solver
    // -------------------------------------------------------------------------

    /**
     * @brief Run simulation for specified time steps.
     */
    void solve(double dt, int num_steps) {
        if (dt <= 0.0 || num_steps <= 0) {
            throw std::invalid_argument("Time step and steps must be positive");
        }

        if (!checkStability(dt)) {
            throw std::runtime_error(
                "Time step too large for Nernst-Planck stability. "
                "Consider the electromigration CFL condition.");
        }

        for (int step = 0; step < num_steps; ++step) {
            // Update potential if time-varying
            if (use_potential_function_) {
                updatePotentialFromFunction(time_);
            }

            // Compute updates for interior nodes
            iterator_.forEachInterior(
                [this, dt](int idx, int i, int j) { computeNodeUpdate(idx, i, j, dt); });

            // Apply boundary conditions
            applyBoundaryConditions(scratch_);

            // Swap buffers
            solution_.swap(scratch_);
            time_ += dt;
        }
    }

    /**
     * @brief Get current solution (concentration field).
     */
    const std::vector<double>& solution() const { return solution_; }

    /**
     * @brief Get current potential field.
     */
    const std::vector<double>& potential() const { return potential_; }

    /**
     * @brief Get current simulation time.
     */
    double time() const { return time_; }

    /**
     * @brief Get the ion species.
     */
    const IonSpecies& ion() const { return ion_; }

    /**
     * @brief Get the mesh.
     */
    const StructuredMesh& mesh() const { return mesh_; }

    /**
     * @brief Get thermal voltage V_T = RT/F.
     */
    double thermalVoltage() const { return Vt_; }

    /**
     * @brief Check stability condition for given dt.
     *
     * For Nernst-Planck, we need to consider both diffusion and drift:
     *   dt < min(dx²/(2D), dx/(|z|μ|E|))
     */
    bool checkStability(double dt) const {
        double dx = mesh_.dx();
        double D = ion_.diffusivity;

        // Diffusion stability
        double dt_diff = dx * dx / (2.0 * D);
        if (!mesh_.is1D()) {
            double dy = mesh_.dy();
            dt_diff = 1.0 / (2.0 * D * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        }

        if (dt > dt_diff)
            return false;

        // Electromigration stability (estimate max E field)
        double max_grad_phi = estimateMaxGradPhi();
        if (max_grad_phi > 1e-12) {
            double velocity = ion_.mobility * max_grad_phi;
            double dt_drift = dx / velocity;
            if (dt > dt_drift)
                return false;
        }

        return true;
    }

    /**
     * @brief Compute the total ionic current density [A/m²].
     *
     * J = -D ∇c - (zFD/RT) c ∇φ
     *   = F * z * (-D ∇c - D*z*c*∇φ/Vt)
     */
    std::vector<double> computeCurrentDensity() const {
        std::vector<double> current(mesh_.numNodes() * 2, 0.0);  // Jx, Jy pairs
        double D = ion_.diffusivity;
        double z = ion_.valence;
        double dx = mesh_.dx();
        double dy = mesh_.is1D() ? 1.0 : mesh_.dy();

        iterator_.forEachInterior([&](int idx, int i, int j) {
            // Concentration gradients
            double dc_dx, dc_dy;
            int nx = mesh_.nx();
            int stride = mesh_.is1D() ? 1 : (nx + 1);

            dc_dx = (solution_[idx + 1] - solution_[idx - 1]) / (2.0 * dx);
            if (!mesh_.is1D()) {
                dc_dy = (solution_[idx + stride] - solution_[idx - stride]) / (2.0 * dy);
            } else {
                dc_dy = 0.0;
            }

            // Potential gradients
            double dphi_dx = (potential_[idx + 1] - potential_[idx - 1]) / (2.0 * dx);
            double dphi_dy =
                mesh_.is1D() ? 0.0
                             : (potential_[idx + stride] - potential_[idx - stride]) / (2.0 * dy);

            double c = solution_[idx];

            // Current density: J = F*z*(-D*∇c - D*z*c*∇φ/Vt)
            double Jx = constants::FARADAY * z * (-D * dc_dx - D * zeta_ * c * dphi_dx);
            double Jy = constants::FARADAY * z * (-D * dc_dy - D * zeta_ * c * dphi_dy);

            current[2 * idx] = Jx;
            current[2 * idx + 1] = Jy;
        });

        return current;
    }

private:
    const StructuredMesh& mesh_;
    IonSpecies ion_;
    double temperature_;
    double Vt_;    // Thermal voltage RT/F
    double zeta_;  // z*F/(R*T) = z/Vt
    double time_ = 0.0;

    std::vector<double> solution_;
    std::vector<double> scratch_;
    std::vector<double> potential_;

    std::shared_ptr<PotentialField> potential_func_;
    bool use_potential_function_ = false;

    std::array<BoundaryCondition, 4> boundary_conditions_;
    MeshIterator iterator_;
    StencilOps stencil_ops_;

    /**
     * @brief Compute the Nernst-Planck update for a single node.
     */
    void computeNodeUpdate(int idx, int i, int j, double dt) {
        double D = ion_.diffusivity;
        double dx = mesh_.dx();
        double dy = mesh_.is1D() ? 1.0 : mesh_.dy();
        int nx = mesh_.nx();
        int stride = mesh_.is1D() ? 1 : (nx + 1);

        double c = solution_[idx];

        // 1. Diffusion term: D ∇²c
        double laplacian_c;
        if (mesh_.is1D()) {
            laplacian_c = (solution_[idx - 1] - 2.0 * c + solution_[idx + 1]) / (dx * dx);
        } else {
            double d2c_dx2 = (solution_[idx - 1] - 2.0 * c + solution_[idx + 1]) / (dx * dx);
            double d2c_dy2 =
                (solution_[idx - stride] - 2.0 * c + solution_[idx + stride]) / (dy * dy);
            laplacian_c = d2c_dx2 + d2c_dy2;
        }
        double diffusion = D * laplacian_c;

        // 2. Electromigration term: ∇·(c * mobility * ∇φ)
        //    = mobility * (∇c · ∇φ + c * ∇²φ)
        //    Using upwind differencing for stability

        // Potential gradients (central difference)
        double dphi_dx = (potential_[idx + 1] - potential_[idx - 1]) / (2.0 * dx);
        double dphi_dy =
            mesh_.is1D() ? 0.0 : (potential_[idx + stride] - potential_[idx - stride]) / (2.0 * dy);

        // Electric field components (E = -∇φ)
        double Ex = -dphi_dx;
        double Ey = -dphi_dy;

        // Drift velocity: v = z*D/Vt * E = zeta * D * E
        double vx = zeta_ * D * Ex;
        double vy = zeta_ * D * Ey;

        // Upwind scheme for advection: ∇·(c*v) ≈ v · ∇c (if ∇·v = 0)
        double dc_dx, dc_dy;

        // Upwind in x
        if (vx > 0) {
            dc_dx = (c - solution_[idx - 1]) / dx;  // backward difference
        } else {
            dc_dx = (solution_[idx + 1] - c) / dx;  // forward difference
        }

        // Upwind in y
        if (mesh_.is1D()) {
            dc_dy = 0.0;
        } else if (vy > 0) {
            dc_dy = (c - solution_[idx - stride]) / dy;
        } else {
            dc_dy = (solution_[idx + stride] - c) / dy;
        }

        // Advection term (negative because we're computing dc/dt, not flux divergence)
        double advection = -(vx * dc_dx + vy * dc_dy);

        // 3. Full update
        scratch_[idx] = c + dt * (diffusion + advection);

        // Enforce positivity (concentration can't be negative)
        if (scratch_[idx] < 0.0) {
            scratch_[idx] = 0.0;
        }
    }

    /**
     * @brief Update potential array from analytical function.
     */
    void updatePotentialFromFunction(double t) {
        if (!potential_func_)
            return;

        if (mesh_.is1D()) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                potential_[i] = (*potential_func_)(mesh_.x(i), 0.0, t);
            }
        } else {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    int idx = j * (mesh_.nx() + 1) + i;
                    potential_[idx] = (*potential_func_)(mesh_.x(i), mesh_.y(i, j), t);
                }
            }
        }
    }

    /**
     * @brief Estimate maximum potential gradient for stability check.
     */
    double estimateMaxGradPhi() const {
        double max_grad = 0.0;
        double dx = mesh_.dx();

        for (size_t i = 1; i < potential_.size(); ++i) {
            double grad = std::abs(potential_[i] - potential_[i - 1]) / dx;
            max_grad = std::max(max_grad, grad);
        }

        return max_grad;
    }

    /**
     * @brief Apply boundary conditions.
     */
    void applyBoundaryConditions(std::vector<double>& u) {
        if (mesh_.is1D()) {
            applyBoundaryConditions1D(u);
        } else {
            applyBoundaryConditions2D(u);
        }
    }

    void applyBoundaryConditions1D(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const double dx = mesh_.dx();

        const auto& left_bc = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right_bc = boundary_conditions_[to_index(Boundary::Right)];

        if (left_bc.type == BoundaryType::DIRICHLET) {
            u[0] = left_bc.value;
        } else {
            u[0] = u[1] - left_bc.value * dx;
        }

        if (right_bc.type == BoundaryType::DIRICHLET) {
            u[nx] = right_bc.value;
        } else {
            u[nx] = u[nx - 1] + right_bc.value * dx;
        }
    }

    void applyBoundaryConditions2D(std::vector<double>& u) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = nx + 1;
        const double dx = mesh_.dx();
        const double dy = mesh_.dy();

        const auto& left = boundary_conditions_[to_index(Boundary::Left)];
        const auto& right = boundary_conditions_[to_index(Boundary::Right)];
        const auto& bottom = boundary_conditions_[to_index(Boundary::Bottom)];
        const auto& top = boundary_conditions_[to_index(Boundary::Top)];

        // Left/Right boundaries
        for (int j = 0; j <= ny; ++j) {
            int left_idx = j * stride;
            int right_idx = j * stride + nx;

            if (left.type == BoundaryType::DIRICHLET) {
                u[left_idx] = left.value;
            } else {
                u[left_idx] = u[left_idx + 1] - left.value * dx;
            }

            if (right.type == BoundaryType::DIRICHLET) {
                u[right_idx] = right.value;
            } else {
                u[right_idx] = u[right_idx - 1] + right.value * dx;
            }
        }

        // Bottom/Top boundaries
        for (int i = 0; i <= nx; ++i) {
            int bottom_idx = i;
            int top_idx = ny * stride + i;

            if (bottom.type == BoundaryType::DIRICHLET) {
                u[bottom_idx] = bottom.value;
            } else {
                u[bottom_idx] = u[bottom_idx + stride] - bottom.value * dy;
            }

            if (top.type == BoundaryType::DIRICHLET) {
                u[top_idx] = top.value;
            } else {
                u[top_idx] = u[top_idx - stride] + top.value * dy;
            }
        }
    }
};

// =============================================================================
// Multi-Ion Nernst-Planck Solver
// =============================================================================

/**
 * @brief Solver for multiple ion species with electroneutrality or Poisson coupling.
 *
 * This solver handles N ion species simultaneously, with options for:
 * 1. Prescribed potential field (decoupled)
 * 2. Electroneutrality constraint (local charge balance)
 * 3. Full Poisson-Nernst-Planck coupling (future)
 *
 * The governing equations are:
 *   ∂c_i/∂t = D_i ∇²c_i + (z_i F D_i / RT) ∇·(c_i ∇φ)
 *
 * With electroneutrality: Σ z_i c_i = 0 (or fixed background charge)
 */
class MultiIonSolver {
public:
    /**
     * @brief Construct multi-ion solver.
     * @param mesh The computational mesh
     * @param ions Vector of ion species
     * @param temperature Temperature [K]
     */
    MultiIonSolver(const StructuredMesh& mesh, std::vector<IonSpecies> ions,
                   double temperature = 310.0)
        : mesh_(mesh),
          ions_(std::move(ions)),
          temperature_(temperature),
          num_species_(ions_.size()),
          iterator_(mesh) {
        if (num_species_ == 0) {
            throw std::invalid_argument("Must provide at least one ion species");
        }

        int n_nodes = mesh.numNodes();

        concentrations_.resize(num_species_);
        scratch_.resize(num_species_);
        for (size_t s = 0; s < num_species_; ++s) {
            concentrations_[s].resize(n_nodes, 0.0);
            scratch_[s].resize(n_nodes, 0.0);
        }

        potential_.resize(n_nodes, 0.0);

        Vt_ = IonSpecies::thermalVoltage(temperature);

        // Default: all Dirichlet zero
        boundary_conditions_.resize(num_species_);
        for (size_t s = 0; s < num_species_; ++s) {
            for (int b = 0; b < 4; ++b) {
                boundary_conditions_[s][b] = BoundaryCondition::Dirichlet(0.0);
            }
        }
    }

    /**
     * @brief Set initial condition for a species.
     */
    void setInitialCondition(size_t species, const std::vector<double>& values) {
        if (species >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        if (values.size() != concentrations_[species].size()) {
            throw std::invalid_argument("Initial condition size mismatch");
        }
        concentrations_[species] = values;
    }

    /**
     * @brief Set Dirichlet boundary for a species.
     */
    void setDirichletBoundary(size_t species, Boundary boundary, double value) {
        if (species >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        boundary_conditions_[species][to_index(boundary)] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(size_t species, int boundary_id, double value) {
        setDirichletBoundary(species, static_cast<Boundary>(boundary_id), value);
    }

    /**
     * @brief Set Neumann boundary for a species.
     */
    void setNeumannBoundary(size_t species, Boundary boundary, double flux) {
        if (species >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        boundary_conditions_[species][to_index(boundary)] = BoundaryCondition::Neumann(flux);
    }

    /**
     * @brief Set electric potential field.
     */
    void setPotentialField(const std::vector<double>& phi) {
        if (phi.size() != potential_.size()) {
            throw std::invalid_argument("Potential size mismatch");
        }
        potential_ = phi;
        use_potential_func_ = false;
    }

    /**
     * @brief Set uniform electric field.
     */
    void setUniformField(double Ex, double Ey = 0.0) {
        potential_func_ = std::make_shared<UniformField>(Ex, Ey);
        use_potential_func_ = true;
        updatePotentialFromFunction(0.0);
    }

    /**
     * @brief Enable electroneutrality mode.
     *
     * When enabled, the solver computes the potential that satisfies
     * local electroneutrality: Σ z_i c_i = background_charge
     */
    void setElectroneutralityMode(bool enable, double background_charge = 0.0) {
        electroneutrality_mode_ = enable;
        background_charge_ = background_charge;
    }

    /**
     * @brief Run simulation.
     */
    void solve(double dt, int num_steps) {
        if (dt <= 0.0 || num_steps <= 0) {
            throw std::invalid_argument("Time step and steps must be positive");
        }

        for (int step = 0; step < num_steps; ++step) {
            if (use_potential_func_) {
                updatePotentialFromFunction(time_);
            }

            // Update all species
            for (size_t s = 0; s < num_species_; ++s) {
                updateSpecies(s, dt);
            }

            // Apply boundary conditions and swap
            for (size_t s = 0; s < num_species_; ++s) {
                applyBoundaryConditions(s, scratch_[s]);
                concentrations_[s].swap(scratch_[s]);
            }

            time_ += dt;
        }
    }

    /**
     * @brief Get concentration for a species.
     */
    const std::vector<double>& concentration(size_t species) const {
        if (species >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        return concentrations_[species];
    }

    /**
     * @brief Get current potential field.
     */
    const std::vector<double>& potential() const { return potential_; }

    /**
     * @brief Get current time.
     */
    double time() const { return time_; }

    /**
     * @brief Get number of species.
     */
    size_t numSpecies() const { return num_species_; }

    /**
     * @brief Get ion species.
     */
    const IonSpecies& ion(size_t i) const { return ions_[i]; }

    /**
     * @brief Compute total charge density at each node.
     */
    std::vector<double> chargeDensity() const {
        std::vector<double> rho(mesh_.numNodes(), 0.0);
        for (size_t s = 0; s < num_species_; ++s) {
            double z = ions_[s].valence;
            for (size_t i = 0; i < rho.size(); ++i) {
                rho[i] += z * concentrations_[s][i];
            }
        }
        // Convert to actual charge density: F * Σ z_i c_i
        for (double& r : rho) {
            r *= constants::FARADAY;
        }
        return rho;
    }

    /**
     * @brief Get mesh.
     */
    const StructuredMesh& mesh() const { return mesh_; }

private:
    const StructuredMesh& mesh_;
    std::vector<IonSpecies> ions_;
    double temperature_;
    size_t num_species_;
    double Vt_;
    double time_ = 0.0;

    std::vector<std::vector<double>> concentrations_;
    std::vector<std::vector<double>> scratch_;
    std::vector<double> potential_;

    std::shared_ptr<PotentialField> potential_func_;
    bool use_potential_func_ = false;
    bool electroneutrality_mode_ = false;
    double background_charge_ = 0.0;

    std::vector<std::array<BoundaryCondition, 4>> boundary_conditions_;
    MeshIterator iterator_;

    void updateSpecies(size_t species, double dt) {
        const auto& ion = ions_[species];
        double D = ion.diffusivity;
        double zeta = static_cast<double>(ion.valence) / Vt_;
        double dx = mesh_.dx();
        double dy = mesh_.is1D() ? 1.0 : mesh_.dy();
        int nx = mesh_.nx();
        int stride = mesh_.is1D() ? 1 : (nx + 1);

        const auto& c = concentrations_[species];
        auto& s = scratch_[species];

        iterator_.forEachInterior([&](int idx, int i, int j) {
            double c_center = c[idx];

            // Diffusion
            double laplacian_c;
            if (mesh_.is1D()) {
                laplacian_c = (c[idx - 1] - 2.0 * c_center + c[idx + 1]) / (dx * dx);
            } else {
                double d2c_dx2 = (c[idx - 1] - 2.0 * c_center + c[idx + 1]) / (dx * dx);
                double d2c_dy2 = (c[idx - stride] - 2.0 * c_center + c[idx + stride]) / (dy * dy);
                laplacian_c = d2c_dx2 + d2c_dy2;
            }

            // Electromigration
            double dphi_dx = (potential_[idx + 1] - potential_[idx - 1]) / (2.0 * dx);
            double dphi_dy =
                mesh_.is1D() ? 0.0
                             : (potential_[idx + stride] - potential_[idx - stride]) / (2.0 * dy);

            double vx = -zeta * D * dphi_dx;
            double vy = -zeta * D * dphi_dy;

            double dc_dx = (vx > 0) ? (c_center - c[idx - 1]) / dx : (c[idx + 1] - c_center) / dx;
            double dc_dy = mesh_.is1D() ? 0.0
                                        : ((vy > 0) ? (c_center - c[idx - stride]) / dy
                                                    : (c[idx + stride] - c_center) / dy);

            double advection = -(vx * dc_dx + vy * dc_dy);

            s[idx] = std::max(0.0, c_center + dt * (D * laplacian_c + advection));
        });
    }

    void updatePotentialFromFunction(double t) {
        if (!potential_func_)
            return;

        if (mesh_.is1D()) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                potential_[i] = (*potential_func_)(mesh_.x(i), 0.0, t);
            }
        } else {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    int idx = j * (mesh_.nx() + 1) + i;
                    potential_[idx] = (*potential_func_)(mesh_.x(i), mesh_.y(i, j), t);
                }
            }
        }
    }

    void applyBoundaryConditions(size_t species, std::vector<double>& u) {
        if (mesh_.is1D()) {
            applyBC1D(species, u);
        } else {
            applyBC2D(species, u);
        }
    }

    void applyBC1D(size_t species, std::vector<double>& u) {
        const int nx = mesh_.nx();
        const double dx = mesh_.dx();
        const auto& bcs = boundary_conditions_[species];

        if (bcs[to_index(Boundary::Left)].type == BoundaryType::DIRICHLET) {
            u[0] = bcs[to_index(Boundary::Left)].value;
        } else {
            u[0] = u[1] - bcs[to_index(Boundary::Left)].value * dx;
        }

        if (bcs[to_index(Boundary::Right)].type == BoundaryType::DIRICHLET) {
            u[nx] = bcs[to_index(Boundary::Right)].value;
        } else {
            u[nx] = u[nx - 1] + bcs[to_index(Boundary::Right)].value * dx;
        }
    }

    void applyBC2D(size_t species, std::vector<double>& u) {
        const int nx = mesh_.nx();
        const int ny = mesh_.ny();
        const int stride = nx + 1;
        const double dx = mesh_.dx();
        const double dy = mesh_.dy();
        const auto& bcs = boundary_conditions_[species];

        // Left/Right
        for (int j = 0; j <= ny; ++j) {
            int left = j * stride;
            int right = j * stride + nx;

            if (bcs[to_index(Boundary::Left)].type == BoundaryType::DIRICHLET) {
                u[left] = bcs[to_index(Boundary::Left)].value;
            } else {
                u[left] = u[left + 1] - bcs[to_index(Boundary::Left)].value * dx;
            }

            if (bcs[to_index(Boundary::Right)].type == BoundaryType::DIRICHLET) {
                u[right] = bcs[to_index(Boundary::Right)].value;
            } else {
                u[right] = u[right - 1] + bcs[to_index(Boundary::Right)].value * dx;
            }
        }

        // Bottom/Top
        for (int i = 0; i <= nx; ++i) {
            int bottom = i;
            int top = ny * stride + i;

            if (bcs[to_index(Boundary::Bottom)].type == BoundaryType::DIRICHLET) {
                u[bottom] = bcs[to_index(Boundary::Bottom)].value;
            } else {
                u[bottom] = u[bottom + stride] - bcs[to_index(Boundary::Bottom)].value * dy;
            }

            if (bcs[to_index(Boundary::Top)].type == BoundaryType::DIRICHLET) {
                u[top] = bcs[to_index(Boundary::Top)].value;
            } else {
                u[top] = u[top - stride] + bcs[to_index(Boundary::Top)].value * dy;
            }
        }
    }
};

// =============================================================================
// Goldman-Hodgkin-Katz Utilities
// =============================================================================

namespace ghk {

/**
 * @brief Compute Nernst equilibrium potential for an ion.
 *
 * E = (RT/zF) * ln(c_out / c_in)
 *
 * @param z Ion valence
 * @param c_in Intracellular concentration [mol/m³]
 * @param c_out Extracellular concentration [mol/m³]
 * @param temperature Temperature [K]
 * @return Equilibrium potential [V]
 */
inline double nernstPotential(int z, double c_in, double c_out, double temperature = 310.0) {
    if (z == 0) {
        throw std::invalid_argument("Ion valence cannot be zero");
    }
    if (c_in <= 0 || c_out <= 0) {
        throw std::invalid_argument("Concentrations must be positive");
    }
    double Vt = constants::GAS_CONSTANT * temperature / constants::FARADAY;
    return (Vt / z) * std::log(c_out / c_in);
}

/**
 * @brief Goldman-Hodgkin-Katz voltage equation for membrane potential.
 *
 * For monovalent ions (Na+, K+, Cl-):
 * V_m = (RT/F) * ln((P_K[K]_o + P_Na[Na]_o + P_Cl[Cl]_i) /
 *                   (P_K[K]_i + P_Na[Na]_i + P_Cl[Cl]_o))
 *
 * @param P_K Potassium permeability
 * @param K_in Intracellular [K+]
 * @param K_out Extracellular [K+]
 * @param P_Na Sodium permeability
 * @param Na_in Intracellular [Na+]
 * @param Na_out Extracellular [Na+]
 * @param P_Cl Chloride permeability
 * @param Cl_in Intracellular [Cl-]
 * @param Cl_out Extracellular [Cl-]
 * @param temperature Temperature [K]
 * @return Membrane potential [V]
 */
inline double ghkVoltage(double P_K, double K_in, double K_out, double P_Na, double Na_in,
                         double Na_out, double P_Cl, double Cl_in, double Cl_out,
                         double temperature = 310.0) {
    double Vt = constants::GAS_CONSTANT * temperature / constants::FARADAY;

    double numerator = P_K * K_out + P_Na * Na_out + P_Cl * Cl_in;
    double denominator = P_K * K_in + P_Na * Na_in + P_Cl * Cl_out;

    if (denominator <= 0 || numerator <= 0) {
        throw std::invalid_argument("Invalid concentration or permeability values");
    }

    return Vt * std::log(numerator / denominator);
}

}  // namespace ghk

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_NERNST_PLANCK_SOLVER_HPP
