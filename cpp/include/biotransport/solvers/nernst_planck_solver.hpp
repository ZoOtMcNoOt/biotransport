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
 * This implementation advances one or more dilute, non-reacting ionic
 * species in a prescribed electric-potential field. It is suitable for
 * electrodiffusion and electrophoresis studies when that potential is known.
 * It does not compute the potential from charge density, enforce
 * electroneutrality, model ion-channel kinetics, or solve a full
 * Poisson-Nernst-Planck system. Requests for the legacy electroneutrality mode
 * therefore fail explicitly instead of silently running a different model.
 *
 * @author BioTransport Development Team
 * @date December 2025
 */

#include <algorithm>
#include <array>
#include <biotransport/core/boundary.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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

namespace electrochem_detail {

inline void requireFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

inline void requirePositive(double value, const char* name) {
    requireFinite(value, name);
    if (value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

inline void requireNonnegative(double value, const char* name) {
    requireFinite(value, name);
    if (value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be non-negative");
    }
}

/** Bernoulli function B(x)=x/(exp(x)-1), evaluated without cancellation/overflow. */
inline double bernoulli(double x) {
    requireFinite(x, "dimensionless electrochemical potential difference");
    const double ax = std::abs(x);
    if (ax < 1.0e-6) {
        const double x2 = x * x;
        return 1.0 - 0.5 * x + x2 / 12.0 - x2 * x2 / 720.0;
    }
    if (x > 50.0) {
        const double em = std::exp(-x);
        return x * em / (1.0 - em);
    }
    if (x < -50.0) {
        return -x / (1.0 - std::exp(x));
    }
    return x / std::expm1(x);
}

struct FaceFluxCoefficients {
    double from_left;
    double from_right;
};

/**
 * Scharfetter-Gummel face flux for
 * N = -D (grad(c) + z*c*grad(phi)/V_T).
 *
 * The returned coefficients give N = from_left*c_left - from_right*c_right.
 * Both coefficients are non-negative, which makes the fitted flux
 * positivity-compatible and exactly preserves discrete Boltzmann equilibrium.
 */
inline FaceFluxCoefficients faceFluxCoefficients(double diffusivity, double z_over_vt,
                                                 double phi_left, double phi_right,
                                                 double spacing) {
    const double delta_psi = z_over_vt * (phi_right - phi_left);
    const double scale = diffusivity / spacing;
    return {scale * bernoulli(delta_psi), scale * bernoulli(-delta_psi)};
}

inline double faceMolarFlux(double diffusivity, double z_over_vt, double c_left, double c_right,
                            double phi_left, double phi_right, double spacing) {
    const auto coeff = faceFluxCoefficients(diffusivity, z_over_vt, phi_left, phi_right, spacing);
    return coeff.from_left * c_left - coeff.from_right * c_right;
}

inline int checkedBoundaryIndex(Boundary boundary, bool is_1d) {
    const int index = to_index(boundary);
    if (index < to_index(Boundary::Left) || index > to_index(Boundary::Top)) {
        throw std::invalid_argument("Invalid boundary identifier");
    }
    if (is_1d && (boundary == Boundary::Bottom || boundary == Boundary::Top)) {
        throw std::invalid_argument("Bottom and Top boundaries are not defined for a 1D mesh");
    }
    return index;
}

inline void validateConcentrationField(const std::vector<double>& values, const char* name) {
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(std::string(name) +
                                        " must contain finite, non-negative concentrations");
        }
    }
}

inline void validatePotentialField(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Potential field must contain only finite values");
        }
    }
}

}  // namespace electrochem_detail

// =============================================================================
// Ion Species Definition
// =============================================================================

/**
 * @brief Represents a single ion species with its transport properties.
 */
struct IonSpecies {
    std::string name;             ///< Species name (e.g., "Na+", "K+", "Cl-")
    int valence;                  ///< Ion valence (z): +1 for Na+, -1 for Cl-, etc.
    double diffusivity;           ///< Diffusion coefficient [m²/s]
    double mobility;              ///< Mobility magnitude at mobility_temperature [m²/(V·s)]
    double mobility_temperature;  ///< Absolute temperature used for mobility [K]

    /**
     * @brief Create an ion species.
     * @param name Species identifier
     * @param valence Ion charge number
     * @param diffusivity Diffusion coefficient [m²/s]
     * @param temperature Temperature [K] for mobility calculation (default 310K body temp)
     */
    IonSpecies(const std::string& name, int valence, double diffusivity, double temperature = 310.0)
        : name(name),
          valence(valence),
          diffusivity(diffusivity),
          mobility(0.0),
          mobility_temperature(temperature) {
        if (name.empty()) {
            throw std::invalid_argument("Ion species name cannot be empty");
        }
        if (valence == 0) {
            throw std::invalid_argument("Ion valence cannot be zero - use regular diffusion");
        }
        electrochem_detail::requirePositive(diffusivity, "Diffusion coefficient");
        electrochem_detail::requirePositive(temperature, "Temperature");
        // Einstein relation: μ = |z|eD/(kT) = |z|FD/(RT)
        mobility = std::abs(static_cast<double>(valence)) * constants::FARADAY * diffusivity /
                   (constants::GAS_CONSTANT * temperature);
    }

    /**
     * @brief Get thermal voltage RT/F at given temperature.
     */
    static double thermalVoltage(double temperature) {
        electrochem_detail::requirePositive(temperature, "Temperature");
        return constants::GAS_CONSTANT * temperature / constants::FARADAY;
    }

    /** @brief Mobility magnitude evaluated at a requested absolute temperature. */
    double mobilityAt(double temperature) const {
        return std::abs(static_cast<double>(valence)) * diffusivity / thermalVoltage(temperature);
    }
};

// =============================================================================
// Common Ion Species (Physiological)
// =============================================================================

namespace ions {
// Representative aqueous infinite-dilution diffusivities [m²/s]. These
// convenience values are approximate; quantitative studies should provide a
// coefficient measured or corrected at the model temperature.
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
    UniformField(double Ex, double Ey = 0.0) : Ex_(Ex), Ey_(Ey) {
        electrochem_detail::requireFinite(Ex, "Electric field Ex");
        electrochem_detail::requireFinite(Ey, "Electric field Ey");
    }

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
        : amplitude_(amplitude), omega_(2.0 * M_PI * frequency), dir_(direction) {
        electrochem_detail::requireFinite(amplitude, "AC field amplitude");
        electrochem_detail::requireNonnegative(frequency, "AC field frequency");
        if (direction != 0 && direction != 1) {
            throw std::invalid_argument("AC field direction must be 0 (x) or 1 (y)");
        }
    }

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
        : phi_(std::move(phi)), grad_x_(std::move(grad_x)), grad_y_(std::move(grad_y)) {
        if (!phi_ || !grad_x_ || !grad_y_) {
            throw std::invalid_argument("Potential and analytical gradient functions are required");
        }
    }

    /**
     * @brief Create custom potential with numerical gradient.
     */
    explicit CustomPotential(PotentialFunc phi, double eps = 1e-8)
        : phi_(std::move(phi)), eps_(eps), use_numerical_grad_(true) {
        if (!phi_) {
            throw std::invalid_argument("Potential function is required");
        }
        electrochem_detail::requirePositive(eps, "Numerical-gradient spacing");
    }

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
 * The conservative molar flux N = -D(grad(c) + z*c*grad(phi)/V_T) is
 * discretized with exponentially fitted Scharfetter-Gummel face fluxes. This
 * preserves a discrete Boltzmann equilibrium exactly and remains robust when
 * electrical drift dominates ordinary diffusion.
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
        : mesh_(mesh), ion_(ion) {
        electrochem_detail::requirePositive(temperature, "Temperature");
        solution_.resize(mesh.numNodes(), 0.0);
        scratch_.resize(mesh.numNodes(), 0.0);
        potential_.resize(mesh.numNodes(), 0.0);

        // Precompute thermal factor
        Vt_ = IonSpecies::thermalVoltage(temperature);
        zeta_ = static_cast<double>(ion_.valence) / Vt_;  // z*F/(R*T)

        // An isolated domain is the least surprising and conservative default.
        for (int i = 0; i < 4; ++i) {
            boundary_conditions_[i] = BoundaryCondition::Neumann(0.0);
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
        electrochem_detail::validateConcentrationField(values, "Initial condition");
        solution_ = values;
    }

    /**
     * @brief Set electric potential field (static).
     */
    void setPotentialField(const std::vector<double>& phi) {
        if (phi.size() != potential_.size()) {
            throw std::invalid_argument("Potential field size mismatch");
        }
        electrochem_detail::validatePotentialField(phi);
        potential_ = phi;
        use_potential_function_ = false;
    }

    /**
     * @brief Set electric potential from analytical function.
     */
    void setPotentialField(std::shared_ptr<PotentialField> field) {
        if (!field) {
            throw std::invalid_argument("Potential field cannot be null");
        }
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
        electrochem_detail::requireNonnegative(value, "Boundary concentration");
        const int index = electrochem_detail::checkedBoundaryIndex(boundary, mesh_.is1D());
        boundary_conditions_[index] = BoundaryCondition::Dirichlet(value);
    }

    void setDirichletBoundary(int boundary_id, double value) {
        setDirichletBoundary(static_cast<Boundary>(boundary_id), value);
    }

    /**
     * @brief Set prescribed outward total molar flux boundary.
     *
     * The value is N dot n [mol/(m^2 s)], where
     * N = -D(grad(c) + z*c*grad(phi)/V_T). Positive values leave the domain.
     * This is intentionally a physical flux, not merely dc/dn.
     */
    void setNeumannBoundary(Boundary boundary, double flux) {
        electrochem_detail::requireFinite(flux, "Boundary molar flux");
        const int index = electrochem_detail::checkedBoundaryIndex(boundary, mesh_.is1D());
        boundary_conditions_[index] = BoundaryCondition::Neumann(flux);
    }

    void setNeumannBoundary(int boundary_id, double flux) {
        setNeumannBoundary(static_cast<Boundary>(boundary_id), flux);
    }

    /**
     * @brief Prescribe the outward total molar flux N dot n [mol/(m^2 s)].
     *
     * This is the unambiguous spelling of setNeumannBoundary(): the value is a
     * physical flux, positive when ions leave the domain, not a concentration
     * derivative.  Both names install the same condition.
     */
    void setOutwardFluxBoundary(Boundary boundary, double outward_molar_flux) {
        setNeumannBoundary(boundary, outward_molar_flux);
    }

    // -------------------------------------------------------------------------
    // Solver
    // -------------------------------------------------------------------------

    /**
     * @brief Run simulation for specified time steps.
     */
    void solve(double dt, int num_steps) {
        if (!std::isfinite(dt) || dt <= 0.0 || num_steps <= 0) {
            throw std::invalid_argument("Time step and steps must be positive");
        }

        // Validate before mutating: a rejected step must leave the exposed
        // state untouched, so the stability bound at the current time is
        // checked before any Dirichlet trace is written into the field.
        if (use_potential_function_) {
            updatePotentialFromFunction(time_);
        }
        if (!checkStability(dt)) {
            throw std::invalid_argument(
                "Time step is too large for the positivity-preserving "
                "Scharfetter-Gummel Nernst-Planck update");
        }

        for (int step = 0; step < num_steps; ++step) {
            if (use_potential_function_) {
                updatePotentialFromFunction(time_);
            }
            applyDirichletBoundaryValues(solution_);

            if (!checkStability(dt)) {
                throw std::runtime_error(
                    "Time step is too large for the positivity-preserving "
                    "Scharfetter-Gummel Nernst-Planck update");
            }
            computeStep(dt);

            solution_.swap(scratch_);
            time_ += dt;
        }

        // Keep the exposed potential synchronized with the exposed final time.
        if (use_potential_function_) {
            updatePotentialFromFunction(time_);
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

    /** @brief Mobility magnitude at this solver's temperature [m²/(V·s)]. */
    double electricalMobility() const {
        return std::abs(static_cast<double>(ion_.valence)) * ion_.diffusivity / Vt_;
    }

    /**
     * @brief Check stability condition for given dt.
     *
     * The bound is assembled from the actual outgoing coefficients of the
     * exponentially fitted face-flux operator, including half control volumes
     * at non-Dirichlet boundaries.
     */
    bool checkStability(double dt) const {
        if (!std::isfinite(dt) || dt <= 0.0) {
            return false;
        }
        const double max_rate = maximumDepletionRate();
        return std::isfinite(max_rate) && (max_rate == 0.0 || dt * max_rate <= 1.0 + 1.0e-14);
    }

    /**
     * @brief Largest explicit step allowed by the fitted transport operator.
     *
     * This is the positivity bound for the homogeneous face-flux operator.
     * A prescribed outward boundary flux is a concentration-independent sink
     * and can require a smaller step; solve() rejects a step if that sink
     * would make a concentration negative.
     */
    double maximumStableTimeStep() const {
        const double max_rate = maximumDepletionRate();
        if (!std::isfinite(max_rate)) {
            throw std::runtime_error(
                "Unable to determine a finite electrochemical stability bound");
        }
        return max_rate == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / max_rate;
    }

    /** @brief Conservative explicit step suggestion as a fraction of the bound. */
    double recommendedTimeStep(double safety = 0.9) const {
        electrochem_detail::requirePositive(safety, "Stability safety factor");
        if (safety > 1.0) {
            throw std::invalid_argument("Stability safety factor must not exceed one");
        }
        return safety * maximumStableTimeStep();
    }

    /**
     * @brief Compute the total ionic current density [A/m²].
     *
     * N = -D (∇c + z*c*∇φ/Vt) is molar flux [mol/(m² s)], and
     * i = z*F*N is electrical current density [A/m²].
     */
    std::vector<double> computeCurrentDensity() const {
        std::vector<double> current(mesh_.numNodes() * 2, 0.0);  // Jx, Jy pairs
        const double charge_per_mole = constants::FARADAY * static_cast<double>(ion_.valence);
        const int nx = mesh_.nx();
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        const int stride = nx + 1;

        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const int idx = j * stride + i;
                double flux_x = 0.0;
                if (i == 0) {
                    const auto& bc = boundary_conditions_[to_index(Boundary::Left)];
                    flux_x = bc.type == BoundaryType::NEUMANN ? -bc.value : xFaceFlux(idx, idx + 1);
                } else if (i == nx) {
                    const auto& bc = boundary_conditions_[to_index(Boundary::Right)];
                    flux_x = bc.type == BoundaryType::NEUMANN ? bc.value : xFaceFlux(idx - 1, idx);
                } else {
                    flux_x = 0.5 * (xFaceFlux(idx - 1, idx) + xFaceFlux(idx, idx + 1));
                }

                double flux_y = 0.0;
                if (!mesh_.is1D()) {
                    if (j == 0) {
                        const auto& bc = boundary_conditions_[to_index(Boundary::Bottom)];
                        flux_y = bc.type == BoundaryType::NEUMANN ? -bc.value
                                                                  : yFaceFlux(idx, idx + stride);
                    } else if (j == ny) {
                        const auto& bc = boundary_conditions_[to_index(Boundary::Top)];
                        flux_y = bc.type == BoundaryType::NEUMANN ? bc.value
                                                                  : yFaceFlux(idx - stride, idx);
                    } else {
                        flux_y =
                            0.5 * (yFaceFlux(idx - stride, idx) + yFaceFlux(idx, idx + stride));
                    }
                }

                current[2 * idx] = charge_per_mole * flux_x;
                current[2 * idx + 1] = charge_per_mole * flux_y;
            }
        }

        return current;
    }

private:
    const StructuredMesh& mesh_;
    IonSpecies ion_;
    double Vt_;    // Thermal voltage RT/F
    double zeta_;  // z*F/(R*T) = z/Vt
    double time_ = 0.0;

    std::vector<double> solution_;
    std::vector<double> scratch_;
    std::vector<double> potential_;

    std::shared_ptr<PotentialField> potential_func_;
    bool use_potential_function_ = false;

    std::array<BoundaryCondition, 4> boundary_conditions_;

    double xFaceFlux(int left_idx, int right_idx) const {
        return electrochem_detail::faceMolarFlux(ion_.diffusivity, zeta_, solution_[left_idx],
                                                 solution_[right_idx], potential_[left_idx],
                                                 potential_[right_idx], mesh_.dx());
    }

    double yFaceFlux(int bottom_idx, int top_idx) const {
        return electrochem_detail::faceMolarFlux(ion_.diffusivity, zeta_, solution_[bottom_idx],
                                                 solution_[top_idx], potential_[bottom_idx],
                                                 potential_[top_idx], mesh_.dy());
    }

    std::optional<double> dirichletValueAt(int i, int j) const {
        std::optional<double> value;
        auto consider = [&](Boundary boundary) {
            const auto& bc = boundary_conditions_[to_index(boundary)];
            if (bc.type != BoundaryType::DIRICHLET) {
                return;
            }
            if (value && std::abs(*value - bc.value) >
                             64.0 * std::numeric_limits<double>::epsilon() *
                                 std::max({1.0, std::abs(*value), std::abs(bc.value)})) {
                throw std::invalid_argument(
                    "Conflicting Dirichlet concentrations meet at a corner");
            }
            value = bc.value;
        };

        if (i == 0)
            consider(Boundary::Left);
        if (i == mesh_.nx())
            consider(Boundary::Right);
        if (!mesh_.is1D()) {
            if (j == 0)
                consider(Boundary::Bottom);
            if (j == mesh_.ny())
                consider(Boundary::Top);
        }
        return value;
    }

    void applyDirichletBoundaryValues(std::vector<double>& field) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        const int stride = nx + 1;
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                if (const auto value = dirichletValueAt(i, j)) {
                    field[j * stride + i] = *value;
                }
            }
        }
    }

    double nodeDepletionRate(int i, int j) const {
        if (dirichletValueAt(i, j)) {
            return 0.0;
        }
        const int nx = mesh_.nx();
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        const int stride = nx + 1;
        const int idx = j * stride + i;
        const double width_x = mesh_.dx() * ((i == 0 || i == nx) ? 0.5 : 1.0);
        double rate = 0.0;

        if (i > 0) {
            const auto coeff = electrochem_detail::faceFluxCoefficients(
                ion_.diffusivity, zeta_, potential_[idx - 1], potential_[idx], mesh_.dx());
            rate += coeff.from_right / width_x;
        }
        if (i < nx) {
            const auto coeff = electrochem_detail::faceFluxCoefficients(
                ion_.diffusivity, zeta_, potential_[idx], potential_[idx + 1], mesh_.dx());
            rate += coeff.from_left / width_x;
        }

        if (!mesh_.is1D()) {
            const double width_y = mesh_.dy() * ((j == 0 || j == ny) ? 0.5 : 1.0);
            if (j > 0) {
                const auto coeff = electrochem_detail::faceFluxCoefficients(
                    ion_.diffusivity, zeta_, potential_[idx - stride], potential_[idx], mesh_.dy());
                rate += coeff.from_right / width_y;
            }
            if (j < ny) {
                const auto coeff = electrochem_detail::faceFluxCoefficients(
                    ion_.diffusivity, zeta_, potential_[idx], potential_[idx + stride], mesh_.dy());
                rate += coeff.from_left / width_y;
            }
        }
        return rate;
    }

    double maximumDepletionRate() const {
        double maximum = 0.0;
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                maximum = std::max(maximum, nodeDepletionRate(i, j));
            }
        }
        return maximum;
    }

    void computeStep(double dt) {
        const int nx = mesh_.nx();
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        const int stride = nx + 1;

        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const int idx = j * stride + i;
                if (const auto value = dirichletValueAt(i, j)) {
                    scratch_[idx] = *value;
                    continue;
                }

                const double width_x = mesh_.dx() * ((i == 0 || i == nx) ? 0.5 : 1.0);
                const double flux_left = i == 0
                                             ? -boundary_conditions_[to_index(Boundary::Left)].value
                                             : xFaceFlux(idx - 1, idx);
                const double flux_right =
                    i == nx ? boundary_conditions_[to_index(Boundary::Right)].value
                            : xFaceFlux(idx, idx + 1);
                double derivative = -(flux_right - flux_left) / width_x;

                if (!mesh_.is1D()) {
                    const double width_y = mesh_.dy() * ((j == 0 || j == ny) ? 0.5 : 1.0);
                    const double flux_bottom =
                        j == 0 ? -boundary_conditions_[to_index(Boundary::Bottom)].value
                               : yFaceFlux(idx - stride, idx);
                    const double flux_top =
                        j == ny ? boundary_conditions_[to_index(Boundary::Top)].value
                                : yFaceFlux(idx, idx + stride);
                    derivative -= (flux_top - flux_bottom) / width_y;
                }

                double candidate = solution_[idx] + dt * derivative;
                const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
                                         std::max({1.0, solution_[idx], dt * std::abs(derivative)});
                if (!std::isfinite(candidate)) {
                    throw std::runtime_error("Nernst-Planck update produced a non-finite value");
                }
                if (candidate < -tolerance) {
                    throw std::runtime_error(
                        "Nernst-Planck update produced a negative concentration; reduce the "
                        "time step or prescribed outward boundary flux");
                }
                scratch_[idx] = std::max(0.0, candidate);
            }
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
        electrochem_detail::validatePotentialField(potential_);
    }
};

// =============================================================================
// Multi-Ion Nernst-Planck Solver
// =============================================================================

/**
 * @brief Solver for multiple independent ion species in a prescribed potential.
 *
 * This solver advances N species against one prescribed potential field. It
 * does not solve Poisson's equation and does not enforce electroneutrality.
 * Requests for those unsupported couplings are rejected explicitly.
 *
 * The governing equations are:
 *   ∂c_i/∂t = D_i ∇²c_i + (z_i F D_i / RT) ∇·(c_i ∇φ)
 *
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
        : mesh_(mesh), ions_(std::move(ions)), num_species_(ions_.size()) {
        electrochem_detail::requirePositive(temperature, "Temperature");
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

        // Default to an isolated, zero-total-flux domain for every species.
        boundary_conditions_.resize(num_species_);
        for (size_t s = 0; s < num_species_; ++s) {
            for (int b = 0; b < 4; ++b) {
                boundary_conditions_[s][b] = BoundaryCondition::Neumann(0.0);
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
        electrochem_detail::validateConcentrationField(values, "Initial condition");
        concentrations_[species] = values;
    }

    /**
     * @brief Set Dirichlet boundary for a species.
     */
    void setDirichletBoundary(size_t species, Boundary boundary, double value) {
        if (species >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        electrochem_detail::requireNonnegative(value, "Boundary concentration");
        const int index = electrochem_detail::checkedBoundaryIndex(boundary, mesh_.is1D());
        boundary_conditions_[species][index] = BoundaryCondition::Dirichlet(value);
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
        electrochem_detail::requireFinite(flux, "Boundary molar flux");
        const int index = electrochem_detail::checkedBoundaryIndex(boundary, mesh_.is1D());
        boundary_conditions_[species][index] = BoundaryCondition::Neumann(flux);
    }

    /**
     * @brief Prescribe the outward total molar flux of one species [mol/(m^2 s)].
     *
     * Unambiguous spelling of setNeumannBoundary(): a physical flux, positive
     * when ions leave the domain.
     */
    void setOutwardFluxBoundary(size_t species, Boundary boundary, double outward_molar_flux) {
        setNeumannBoundary(species, boundary, outward_molar_flux);
    }

    /**
     * @brief Set electric potential field.
     */
    void setPotentialField(const std::vector<double>& phi) {
        if (phi.size() != potential_.size()) {
            throw std::invalid_argument("Potential size mismatch");
        }
        electrochem_detail::validatePotentialField(phi);
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
     * @brief Reject the not-yet-implemented electroneutrality coupling.
     *
     * Passing false with zero background charge is accepted for
     * backward-compatible explicit opt-out. Any requested coupling data is
     * rejected because it would otherwise be ignored.
     */
    void setElectroneutralityMode(bool enable, double background_charge = 0.0) {
        electrochem_detail::requireFinite(background_charge, "Background charge concentration");
        if (enable || background_charge != 0.0) {
            throw std::logic_error(
                "Electroneutrality coupling is not implemented. Supply a prescribed potential "
                "field, or use a validated Poisson-Nernst-Planck solver.");
        }
    }

    /**
     * @brief Run simulation.
     */
    void solve(double dt, int num_steps) {
        if (!std::isfinite(dt) || dt <= 0.0 || num_steps <= 0) {
            throw std::invalid_argument("Time step and steps must be positive");
        }

        // Validate before mutating: reject an unstable step before any species
        // field receives its Dirichlet traces.
        if (use_potential_func_) {
            updatePotentialFromFunction(time_);
        }
        for (size_t s = 0; s < num_species_; ++s) {
            if (!checkSpeciesStability(s, dt)) {
                throw std::invalid_argument(
                    "Time step is too large for the positivity-preserving multi-ion "
                    "Nernst-Planck update");
            }
        }

        for (int step = 0; step < num_steps; ++step) {
            if (use_potential_func_) {
                updatePotentialFromFunction(time_);
            }

            for (size_t s = 0; s < num_species_; ++s) {
                applyDirichletBoundaryValues(s, concentrations_[s]);
                if (!checkSpeciesStability(s, dt)) {
                    throw std::runtime_error(
                        "Time step is too large for the positivity-preserving multi-ion "
                        "Nernst-Planck update");
                }
                updateSpecies(s, dt);
            }

            for (size_t s = 0; s < num_species_; ++s) {
                concentrations_[s].swap(scratch_[s]);
            }

            time_ += dt;
        }
        if (use_potential_func_) {
            updatePotentialFromFunction(time_);
        }
    }

    /** @brief Test the explicit positivity bound for every species. */
    bool checkStability(double dt) const {
        if (!std::isfinite(dt) || dt <= 0.0) {
            return false;
        }
        for (size_t species = 0; species < num_species_; ++species) {
            if (!checkSpeciesStability(species, dt)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Largest explicit operator-limited step over all species.
     *
     * As in the single-species solver, a prescribed outward flux can impose a
     * smaller concentration-dependent limit that solve() checks at runtime.
     */
    double maximumStableTimeStep() const {
        double maximum_rate = 0.0;
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        for (size_t species = 0; species < num_species_; ++species) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= mesh_.nx(); ++i) {
                    maximum_rate = std::max(maximum_rate, nodeDepletionRate(species, i, j));
                }
            }
        }
        if (!std::isfinite(maximum_rate)) {
            throw std::runtime_error("Unable to determine a finite multi-ion stability bound");
        }
        return maximum_rate == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / maximum_rate;
    }

    /** @brief Conservative explicit step suggestion as a fraction of the bound. */
    double recommendedTimeStep(double safety = 0.9) const {
        electrochem_detail::requirePositive(safety, "Stability safety factor");
        if (safety > 1.0) {
            throw std::invalid_argument("Stability safety factor must not exceed one");
        }
        return safety * maximumStableTimeStep();
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
    const IonSpecies& ion(size_t i) const {
        if (i >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        return ions_[i];
    }

    /** @brief Mobility magnitude for a species at this solver's temperature. */
    double electricalMobility(size_t species) const {
        if (species >= num_species_) {
            throw std::out_of_range("Species index out of range");
        }
        return std::abs(static_cast<double>(ions_[species].valence)) * ions_[species].diffusivity /
               Vt_;
    }

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
    size_t num_species_;
    double Vt_;
    double time_ = 0.0;

    std::vector<std::vector<double>> concentrations_;
    std::vector<std::vector<double>> scratch_;
    std::vector<double> potential_;

    std::shared_ptr<PotentialField> potential_func_;
    bool use_potential_func_ = false;
    std::vector<std::array<BoundaryCondition, 4>> boundary_conditions_;

    double speciesZeta(size_t species) const {
        return static_cast<double>(ions_[species].valence) / Vt_;
    }

    double xFaceFlux(size_t species, int left_idx, int right_idx) const {
        const auto& c = concentrations_[species];
        return electrochem_detail::faceMolarFlux(ions_[species].diffusivity, speciesZeta(species),
                                                 c[left_idx], c[right_idx], potential_[left_idx],
                                                 potential_[right_idx], mesh_.dx());
    }

    double yFaceFlux(size_t species, int bottom_idx, int top_idx) const {
        const auto& c = concentrations_[species];
        return electrochem_detail::faceMolarFlux(ions_[species].diffusivity, speciesZeta(species),
                                                 c[bottom_idx], c[top_idx], potential_[bottom_idx],
                                                 potential_[top_idx], mesh_.dy());
    }

    std::optional<double> dirichletValueAt(size_t species, int i, int j) const {
        const auto& bcs = boundary_conditions_[species];
        std::optional<double> value;
        auto consider = [&](Boundary boundary) {
            const auto& bc = bcs[to_index(boundary)];
            if (bc.type != BoundaryType::DIRICHLET) {
                return;
            }
            if (value && std::abs(*value - bc.value) >
                             64.0 * std::numeric_limits<double>::epsilon() *
                                 std::max({1.0, std::abs(*value), std::abs(bc.value)})) {
                throw std::invalid_argument(
                    "Conflicting Dirichlet concentrations meet at a corner");
            }
            value = bc.value;
        };
        if (i == 0)
            consider(Boundary::Left);
        if (i == mesh_.nx())
            consider(Boundary::Right);
        if (!mesh_.is1D()) {
            if (j == 0)
                consider(Boundary::Bottom);
            if (j == mesh_.ny())
                consider(Boundary::Top);
        }
        return value;
    }

    void applyDirichletBoundaryValues(size_t species, std::vector<double>& field) const {
        const int nx = mesh_.nx();
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        const int stride = nx + 1;
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                if (const auto value = dirichletValueAt(species, i, j)) {
                    field[j * stride + i] = *value;
                }
            }
        }
    }

    double nodeDepletionRate(size_t species, int i, int j) const {
        if (dirichletValueAt(species, i, j)) {
            return 0.0;
        }
        const int nx = mesh_.nx();
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        const int stride = nx + 1;
        const int idx = j * stride + i;
        const double D = ions_[species].diffusivity;
        const double zeta = speciesZeta(species);
        const double width_x = mesh_.dx() * ((i == 0 || i == nx) ? 0.5 : 1.0);
        double rate = 0.0;
        if (i > 0) {
            const auto coeff = electrochem_detail::faceFluxCoefficients(
                D, zeta, potential_[idx - 1], potential_[idx], mesh_.dx());
            rate += coeff.from_right / width_x;
        }
        if (i < nx) {
            const auto coeff = electrochem_detail::faceFluxCoefficients(
                D, zeta, potential_[idx], potential_[idx + 1], mesh_.dx());
            rate += coeff.from_left / width_x;
        }
        if (!mesh_.is1D()) {
            const double width_y = mesh_.dy() * ((j == 0 || j == ny) ? 0.5 : 1.0);
            if (j > 0) {
                const auto coeff = electrochem_detail::faceFluxCoefficients(
                    D, zeta, potential_[idx - stride], potential_[idx], mesh_.dy());
                rate += coeff.from_right / width_y;
            }
            if (j < ny) {
                const auto coeff = electrochem_detail::faceFluxCoefficients(
                    D, zeta, potential_[idx], potential_[idx + stride], mesh_.dy());
                rate += coeff.from_left / width_y;
            }
        }
        return rate;
    }

    bool checkSpeciesStability(size_t species, double dt) const {
        double maximum = 0.0;
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                maximum = std::max(maximum, nodeDepletionRate(species, i, j));
            }
        }
        return std::isfinite(maximum) && (maximum == 0.0 || dt * maximum <= 1.0 + 1.0e-14);
    }

    void updateSpecies(size_t species, double dt) {
        const int nx = mesh_.nx();
        const int ny = mesh_.is1D() ? 0 : mesh_.ny();
        const int stride = nx + 1;
        const auto& bcs = boundary_conditions_[species];
        const auto& c = concentrations_[species];
        auto& next = scratch_[species];

        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const int idx = j * stride + i;
                if (const auto value = dirichletValueAt(species, i, j)) {
                    next[idx] = *value;
                    continue;
                }
                const double width_x = mesh_.dx() * ((i == 0 || i == nx) ? 0.5 : 1.0);
                const double flux_left = i == 0 ? -bcs[to_index(Boundary::Left)].value
                                                : xFaceFlux(species, idx - 1, idx);
                const double flux_right = i == nx ? bcs[to_index(Boundary::Right)].value
                                                  : xFaceFlux(species, idx, idx + 1);
                double derivative = -(flux_right - flux_left) / width_x;
                if (!mesh_.is1D()) {
                    const double width_y = mesh_.dy() * ((j == 0 || j == ny) ? 0.5 : 1.0);
                    const double flux_bottom = j == 0 ? -bcs[to_index(Boundary::Bottom)].value
                                                      : yFaceFlux(species, idx - stride, idx);
                    const double flux_top = j == ny ? bcs[to_index(Boundary::Top)].value
                                                    : yFaceFlux(species, idx, idx + stride);
                    derivative -= (flux_top - flux_bottom) / width_y;
                }
                double candidate = c[idx] + dt * derivative;
                const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
                                         std::max({1.0, c[idx], dt * std::abs(derivative)});
                if (!std::isfinite(candidate)) {
                    throw std::runtime_error("Multi-ion update produced a non-finite value");
                }
                if (candidate < -tolerance) {
                    throw std::runtime_error(
                        "Multi-ion update produced a negative concentration; reduce the time "
                        "step or prescribed outward boundary flux");
                }
                next[idx] = std::max(0.0, candidate);
            }
        }
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
        electrochem_detail::validatePotentialField(potential_);
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
    electrochem_detail::requirePositive(c_in, "Intracellular concentration");
    electrochem_detail::requirePositive(c_out, "Extracellular concentration");
    electrochem_detail::requirePositive(temperature, "Temperature");
    const double Vt = constants::GAS_CONSTANT * temperature / constants::FARADAY;
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
    electrochem_detail::requireNonnegative(P_K, "Potassium permeability");
    electrochem_detail::requireNonnegative(P_Na, "Sodium permeability");
    electrochem_detail::requireNonnegative(P_Cl, "Chloride permeability");
    electrochem_detail::requireNonnegative(K_in, "Intracellular potassium concentration");
    electrochem_detail::requireNonnegative(K_out, "Extracellular potassium concentration");
    electrochem_detail::requireNonnegative(Na_in, "Intracellular sodium concentration");
    electrochem_detail::requireNonnegative(Na_out, "Extracellular sodium concentration");
    electrochem_detail::requireNonnegative(Cl_in, "Intracellular chloride concentration");
    electrochem_detail::requireNonnegative(Cl_out, "Extracellular chloride concentration");
    electrochem_detail::requirePositive(temperature, "Temperature");
    if (P_K == 0.0 && P_Na == 0.0 && P_Cl == 0.0) {
        throw std::invalid_argument("At least one permeability must be positive");
    }
    const double Vt = constants::GAS_CONSTANT * temperature / constants::FARADAY;

    double numerator = P_K * K_out + P_Na * Na_out + P_Cl * Cl_in;
    double denominator = P_K * K_in + P_Na * Na_in + P_Cl * Cl_out;

    if (denominator <= 0.0 || numerator <= 0.0) {
        throw std::invalid_argument(
            "Permeability-weighted concentrations must be positive on both sides");
    }

    return Vt * std::log(numerator / denominator);
}

}  // namespace ghk

}  // namespace biotransport

#endif  // BIOTRANSPORT_SOLVERS_NERNST_PLANCK_SOLVER_HPP
