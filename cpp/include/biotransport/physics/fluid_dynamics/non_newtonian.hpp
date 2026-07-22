/**
 * @file non_newtonian.hpp
 * @brief Non-Newtonian fluid constitutive models.
 *
 * Implements various non-Newtonian viscosity models for fluids where
 * viscosity depends on shear rate (generalized Newtonian fluids).
 *
 * Models included:
 * - Newtonian: mu = mu_0 (constant)
 * - Power-law: mu = K * gamma_dot^(n-1)
 * - Carreau: mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda*gamma_dot)^2)^((n-1)/2)
 * - Cross: mu = mu_inf + (mu_0 - mu_inf) / (1 + (K*gamma_dot)^m)
 * - Bingham plastic: tau = tau_y + mu_p * gamma_dot (if |tau| > tau_y)
 * - Herschel-Bulkley: tau = tau_y + K * gamma_dot^n (if |tau| > tau_y)
 * - Casson: sqrt(tau) = sqrt(tau_y) + sqrt(mu_p * gamma_dot)
 *
 * Where:
 *   - mu = apparent viscosity [Pa·s]
 *   - gamma_dot = shear rate [1/s]
 *   - tau = shear stress [Pa]
 *
 * Applications in biotransport:
 *   - Blood (shear-thinning, yield stress)
 *   - Synovial fluid (shear-thinning)
 *   - Mucus (viscoelastic, shear-thinning)
 *   - Polymer solutions in drug delivery
 *   - Cell suspensions
 */

#ifndef BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NON_NEWTONIAN_HPP
#define BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NON_NEWTONIAN_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace biotransport {

namespace non_newtonian_detail {

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

inline void requireNonNegative(double value, const char* name) {
    requireFinite(value, name);
    if (value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be non-negative");
    }
}

inline double shearRateMagnitude(double gamma_dot) {
    requireFinite(gamma_dot, "Shear rate");
    return std::abs(gamma_dot);
}

inline double requirePositiveResult(double value, const char* quantity) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::overflow_error(std::string(quantity) +
                                  " is not finite and positive for these inputs");
    }
    return value;
}

inline double requireFiniteResult(double value, const char* quantity) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(std::string(quantity) + " is not finite for these inputs");
    }
    return value;
}

inline double logOnePlusExp(double value) {
    if (value > 0.0) {
        return value + std::log1p(std::exp(-value));
    }
    return std::log1p(std::exp(value));
}

}  // namespace non_newtonian_detail

/**
 * @brief Fluid model type enumeration.
 */
enum class FluidModel {
    NEWTONIAN,
    POWER_LAW,
    CARREAU,
    CARREAU_YASUDA,
    CROSS,
    BINGHAM,
    HERSCHEL_BULKLEY,
    CASSON
};

/**
 * @brief Abstract base class for non-Newtonian viscosity models.
 *
 * All models compute apparent viscosity as a function of shear rate.
 */
class ViscosityModel {
public:
    virtual ~ViscosityModel() = default;

    /**
     * @brief Compute apparent viscosity at given shear rate.
     *
     * @param gamma_dot Shear rate [1/s]
     * @return Apparent viscosity [Pa·s]
     */
    virtual double viscosity(double gamma_dot) const = 0;

    /**
     * @brief Get the model name.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get the model type.
     */
    virtual FluidModel type() const = 0;

    /**
     * @brief Compute shear stress at given shear rate.
     *
     * For generalized Newtonian: tau = mu(gamma_dot) * gamma_dot
     *
     * @param gamma_dot Shear rate [1/s]
     * @return Shear stress [Pa]
     */
    virtual double shearStress(double gamma_dot) const {
        non_newtonian_detail::requireFinite(gamma_dot, "Shear rate");
        return non_newtonian_detail::requireFiniteResult(viscosity(gamma_dot) * gamma_dot,
                                                         "Shear stress");
    }
};

/**
 * @brief Newtonian fluid model (constant viscosity).
 *
 * mu = mu_0
 */
class NewtonianModel : public ViscosityModel {
public:
    /**
     * @brief Create Newtonian model.
     *
     * @param mu0 Constant viscosity [Pa·s]
     */
    explicit NewtonianModel(double mu0) : mu0_(mu0) {
        non_newtonian_detail::requirePositive(mu0, "Viscosity");
    }

    double viscosity(double gamma_dot) const override {
        non_newtonian_detail::requireFinite(gamma_dot, "Shear rate");
        return mu0_;
    }
    std::string name() const override { return "Newtonian"; }
    FluidModel type() const override { return FluidModel::NEWTONIAN; }

    double mu0() const { return mu0_; }

private:
    double mu0_;
};

/**
 * @brief Power-law (Ostwald-de Waele) fluid model.
 *
 * tau = K * gamma_dot^n
 * mu = K * gamma_dot^(n-1)
 *
 * Where:
 * - K = consistency index [Pa·s^n]
 * - n = flow behavior index (n < 1: shear-thinning, n > 1: shear-thickening)
 *
 * Common values:
 * - Blood: K ~ 0.017, n ~ 0.708
 * - Polymer melts: K ~ 1000-10000, n ~ 0.3-0.7
 */
class PowerLawModel : public ViscosityModel {
public:
    /**
     * @brief Create power-law model.
     *
     * @param K Consistency index [Pa·s^n]
     * @param n Flow behavior index (dimensionless)
     * @param gamma_min Minimum shear rate cutoff to avoid infinite viscosity [1/s]
     */
    PowerLawModel(double K, double n, double gamma_min = 1e-10)
        : K_(K), n_(n), gamma_min_(gamma_min) {
        non_newtonian_detail::requirePositive(K, "Consistency index K");
        non_newtonian_detail::requirePositive(n, "Flow index n");
        non_newtonian_detail::requirePositive(gamma_min, "Minimum shear-rate cutoff");
    }

    double viscosity(double gamma_dot) const override {
        const double gamma =
            std::max(gamma_min_, non_newtonian_detail::shearRateMagnitude(gamma_dot));
        return non_newtonian_detail::requirePositiveResult(K_ * std::pow(gamma, n_ - 1.0),
                                                           "Power-law viscosity");
    }

    std::string name() const override { return "Power-law"; }
    FluidModel type() const override { return FluidModel::POWER_LAW; }

    double K() const { return K_; }
    double n() const { return n_; }

    /**
     * @brief Check if fluid is shear-thinning.
     */
    bool isShearThinning() const { return n_ < 1.0; }

    /**
     * @brief Check if fluid is shear-thickening.
     */
    bool isShearThickening() const { return n_ > 1.0; }

private:
    double K_;
    double n_;
    double gamma_min_;
};

/**
 * @brief Carreau model for shear-thinning fluids.
 *
 * mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * gamma_dot)^2)^((n-1)/2)
 *
 * Where:
 * - mu_0 = zero-shear viscosity [Pa·s]
 * - mu_inf = infinite-shear viscosity [Pa·s]
 * - lambda = relaxation time [s]
 * - n = power-law index (n < 1 for shear-thinning)
 *
 * Advantages over power-law:
 * - Bounded viscosity at low shear rates
 * - Newtonian plateaus at both low and high shear
 */
class CarreauModel : public ViscosityModel {
public:
    /**
     * @brief Create Carreau model.
     *
     * @param mu0 Zero-shear viscosity [Pa·s]
     * @param mu_inf Infinite-shear viscosity [Pa·s]
     * @param lambda Relaxation time [s]
     * @param n Power-law index (n < 1 for shear-thinning)
     */
    CarreauModel(double mu0, double mu_inf, double lambda, double n)
        : mu0_(mu0), mu_inf_(mu_inf), lambda_(lambda), n_(n) {
        non_newtonian_detail::requirePositive(mu0, "mu0");
        non_newtonian_detail::requireNonNegative(mu_inf, "mu_inf");
        if (mu_inf > mu0)
            throw std::invalid_argument("mu_inf must be <= mu0");
        non_newtonian_detail::requirePositive(lambda, "lambda");
        non_newtonian_detail::requirePositive(n, "n");
        if (n > 1.0)
            throw std::invalid_argument("Carreau n must be <= 1 for a bounded shear-thinning law");
    }

    double viscosity(double gamma_dot) const override {
        const double gamma = non_newtonian_detail::shearRateMagnitude(gamma_dot);
        const double factor = std::pow(std::hypot(1.0, lambda_ * gamma), n_ - 1.0);
        return non_newtonian_detail::requirePositiveResult(mu_inf_ + (mu0_ - mu_inf_) * factor,
                                                           "Carreau viscosity");
    }

    std::string name() const override { return "Carreau"; }
    FluidModel type() const override { return FluidModel::CARREAU; }

    double mu0() const { return mu0_; }
    double muInf() const { return mu_inf_; }
    double lambda() const { return lambda_; }
    double n() const { return n_; }

private:
    double mu0_;
    double mu_inf_;
    double lambda_;
    double n_;
};

/**
 * @brief Carreau-Yasuda model (generalized Carreau).
 *
 * mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * gamma_dot)^a)^((n-1)/a)
 *
 * The parameter 'a' provides additional flexibility for fitting.
 * a = 2 recovers the standard Carreau model.
 */
class CarreauYasudaModel : public ViscosityModel {
public:
    /**
     * @brief Create Carreau-Yasuda model.
     *
     * @param mu0 Zero-shear viscosity [Pa·s]
     * @param mu_inf Infinite-shear viscosity [Pa·s]
     * @param lambda Relaxation time [s]
     * @param a Yasuda parameter (a = 2 gives standard Carreau)
     * @param n Power-law index
     */
    CarreauYasudaModel(double mu0, double mu_inf, double lambda, double a, double n)
        : mu0_(mu0), mu_inf_(mu_inf), lambda_(lambda), a_(a), n_(n) {
        non_newtonian_detail::requirePositive(mu0, "mu0");
        non_newtonian_detail::requireNonNegative(mu_inf, "mu_inf");
        if (mu_inf > mu0)
            throw std::invalid_argument("mu_inf must be <= mu0");
        non_newtonian_detail::requirePositive(lambda, "lambda");
        non_newtonian_detail::requirePositive(a, "a");
        non_newtonian_detail::requirePositive(n, "n");
        if (n > 1.0) {
            throw std::invalid_argument(
                "Carreau-Yasuda n must be <= 1 for a bounded shear-thinning law");
        }
    }

    double viscosity(double gamma_dot) const override {
        const double gamma = non_newtonian_detail::shearRateMagnitude(gamma_dot);
        if (gamma == 0.0 || n_ == 1.0) {
            return mu0_;
        }
        const double log_argument = a_ * (std::log(lambda_) + std::log(gamma));
        const double log_factor =
            ((n_ - 1.0) / a_) * non_newtonian_detail::logOnePlusExp(log_argument);
        const double factor = std::exp(log_factor);
        return non_newtonian_detail::requirePositiveResult(mu_inf_ + (mu0_ - mu_inf_) * factor,
                                                           "Carreau-Yasuda viscosity");
    }

    std::string name() const override { return "Carreau-Yasuda"; }
    FluidModel type() const override { return FluidModel::CARREAU_YASUDA; }

    double mu0() const { return mu0_; }
    double muInf() const { return mu_inf_; }
    double lambda() const { return lambda_; }
    double a() const { return a_; }
    double n() const { return n_; }

private:
    double mu0_;
    double mu_inf_;
    double lambda_;
    double a_;
    double n_;
};

/**
 * @brief Cross model for shear-thinning fluids.
 *
 * mu = mu_inf + (mu_0 - mu_inf) / (1 + (K * gamma_dot)^m)
 *
 * Similar to Carreau but with different mathematical form.
 */
class CrossModel : public ViscosityModel {
public:
    /**
     * @brief Create Cross model.
     *
     * @param mu0 Zero-shear viscosity [Pa·s]
     * @param mu_inf Infinite-shear viscosity [Pa·s]
     * @param K Consistency parameter [s]
     * @param m Power-law exponent (typically 0 < m < 1)
     */
    CrossModel(double mu0, double mu_inf, double K, double m)
        : mu0_(mu0), mu_inf_(mu_inf), K_(K), m_(m) {
        non_newtonian_detail::requirePositive(mu0, "mu0");
        non_newtonian_detail::requireNonNegative(mu_inf, "mu_inf");
        if (mu_inf > mu0)
            throw std::invalid_argument("mu_inf must be <= mu0");
        non_newtonian_detail::requirePositive(K, "K");
        non_newtonian_detail::requirePositive(m, "m");
        if (m > 1.0) {
            // For tau(gamma) = gamma * mu(gamma), the minimum constitutive
            // slope is
            //
            //   mu_inf - (mu0 - mu_inf) * (m - 1)^2 / (4m).
            //
            // Checking that slope preserves physically admissible m > 1
            // fits while rejecting parameter combinations whose stress curve
            // folds back on itself and therefore cannot be inverted uniquely.
            const double minimum_stress_slope =
                mu_inf_ - (mu0_ - mu_inf_) * (m_ - 1.0) * (m_ - 1.0) / (4.0 * m_);
            if (minimum_stress_slope < 0.0) {
                throw std::invalid_argument(
                    "Cross parameters must define a monotone shear-stress curve");
            }
        }
    }

    double viscosity(double gamma_dot) const override {
        const double gamma = non_newtonian_detail::shearRateMagnitude(gamma_dot);
        if (gamma == 0.0) {
            return mu0_;
        }
        const double log_power = m_ * (std::log(K_) + std::log(gamma));
        const double reciprocal_denominator =
            std::exp(-non_newtonian_detail::logOnePlusExp(log_power));
        return non_newtonian_detail::requirePositiveResult(
            mu_inf_ + (mu0_ - mu_inf_) * reciprocal_denominator, "Cross viscosity");
    }

    std::string name() const override { return "Cross"; }
    FluidModel type() const override { return FluidModel::CROSS; }

    double mu0() const { return mu0_; }
    double muInf() const { return mu_inf_; }
    double K() const { return K_; }
    double m() const { return m_; }

private:
    double mu0_;
    double mu_inf_;
    double K_;
    double m_;
};

/**
 * @brief Bingham plastic model.
 *
 * tau = tau_y + mu_p * gamma_dot  (if |tau| > tau_y)
 * gamma_dot = 0                   (if |tau| <= tau_y)
 *
 * Regularized form for numerical stability:
 * mu = mu_p + tau_y / (|gamma_dot| + epsilon)
 *
 * Applications:
 * - Drilling muds
 * - Toothpaste
 * - Some food products
 */
class BinghamModel : public ViscosityModel {
public:
    /**
     * @brief Create Bingham plastic model.
     *
     * @param tau_y Yield stress [Pa]
     * @param mu_p Plastic viscosity [Pa·s]
     * @param epsilon Regularization parameter [1/s]
     */
    BinghamModel(double tau_y, double mu_p, double epsilon = 1e-6)
        : tau_y_(tau_y), mu_p_(mu_p), epsilon_(epsilon) {
        non_newtonian_detail::requireNonNegative(tau_y, "Yield stress");
        non_newtonian_detail::requirePositive(mu_p, "Plastic viscosity");
        non_newtonian_detail::requirePositive(epsilon, "Epsilon");
    }

    double viscosity(double gamma_dot) const override {
        const double gamma = non_newtonian_detail::shearRateMagnitude(gamma_dot);
        return non_newtonian_detail::requirePositiveResult(mu_p_ + tau_y_ / (gamma + epsilon_),
                                                           "Bingham viscosity");
    }

    std::string name() const override { return "Bingham"; }
    FluidModel type() const override { return FluidModel::BINGHAM; }

    double yieldStress() const { return tau_y_; }
    double plasticViscosity() const { return mu_p_; }

    /**
     * @brief Compute Bingham number.
     *
     * Bn = tau_y * L / (mu_p * U)
     *
     * @param L Characteristic length [m]
     * @param U Characteristic velocity [m/s]
     * @return Bingham number (dimensionless)
     */
    double binghamNumber(double L, double U) const {
        non_newtonian_detail::requirePositive(L, "Characteristic length");
        non_newtonian_detail::requirePositive(U, "Characteristic speed");
        return non_newtonian_detail::requireFiniteResult(tau_y_ * L / (mu_p_ * U),
                                                         "Bingham number");
    }

private:
    double tau_y_;
    double mu_p_;
    double epsilon_;
};

/**
 * @brief Herschel-Bulkley model.
 *
 * tau = tau_y + K * gamma_dot^n  (if |tau| > tau_y)
 * gamma_dot = 0                  (if |tau| <= tau_y)
 *
 * Combines yield stress with power-law behavior.
 *
 * Special cases:
 * - n = 1: Bingham plastic
 * - tau_y = 0: Power-law fluid
 */
class HerschelBulkleyModel : public ViscosityModel {
public:
    /**
     * @brief Create Herschel-Bulkley model.
     *
     * @param tau_y Yield stress [Pa]
     * @param K Consistency index [Pa·s^n]
     * @param n Flow behavior index
     * @param epsilon Regularization parameter [1/s]
     */
    HerschelBulkleyModel(double tau_y, double K, double n, double epsilon = 1e-6)
        : tau_y_(tau_y), K_(K), n_(n), epsilon_(epsilon) {
        non_newtonian_detail::requireNonNegative(tau_y, "Yield stress");
        non_newtonian_detail::requirePositive(K, "K");
        non_newtonian_detail::requirePositive(n, "n");
        non_newtonian_detail::requirePositive(epsilon, "Epsilon");
    }

    double viscosity(double gamma_dot) const override {
        const double gamma = non_newtonian_detail::shearRateMagnitude(gamma_dot) + epsilon_;
        return non_newtonian_detail::requirePositiveResult(
            tau_y_ / gamma + K_ * std::pow(gamma, n_ - 1.0), "Herschel-Bulkley viscosity");
    }

    std::string name() const override { return "Herschel-Bulkley"; }
    FluidModel type() const override { return FluidModel::HERSCHEL_BULKLEY; }

    double yieldStress() const { return tau_y_; }
    double K() const { return K_; }
    double n() const { return n_; }

private:
    double tau_y_;
    double K_;
    double n_;
    double epsilon_;
};

/**
 * @brief Casson model for blood rheology.
 *
 * sqrt(tau) = sqrt(tau_y) + sqrt(mu_p * gamma_dot)
 *
 * Squaring: tau = tau_y + 2*sqrt(tau_y * mu_p * gamma_dot) + mu_p * gamma_dot
 *
 * The apparent viscosity is:
 * mu = tau / gamma_dot = tau_y/gamma_dot + 2*sqrt(tau_y * mu_p / gamma_dot) + mu_p
 *
 * Widely used for blood, especially at low shear rates.
 * Typical blood values:
 * - tau_y ~ 0.005-0.01 Pa (depends on hematocrit)
 * - mu_p ~ 0.003-0.004 Pa·s
 */
class CassonModel : public ViscosityModel {
public:
    /**
     * @brief Create Casson model.
     *
     * @param tau_y Casson yield stress [Pa]
     * @param mu_p Casson plastic viscosity [Pa·s]
     * @param epsilon Regularization parameter [1/s]
     */
    CassonModel(double tau_y, double mu_p, double epsilon = 1e-6)
        : tau_y_(tau_y), mu_p_(mu_p), epsilon_(epsilon) {
        non_newtonian_detail::requireNonNegative(tau_y, "Yield stress");
        non_newtonian_detail::requirePositive(mu_p, "Plastic viscosity");
        non_newtonian_detail::requirePositive(epsilon, "Epsilon");
    }

    double viscosity(double gamma_dot) const override {
        const double gamma = non_newtonian_detail::shearRateMagnitude(gamma_dot) + epsilon_;
        const double sqrt_tau_y = std::sqrt(tau_y_);
        const double sqrt_mu_gamma = std::sqrt(mu_p_ * gamma);
        const double sqrt_tau = sqrt_tau_y + sqrt_mu_gamma;
        return non_newtonian_detail::requirePositiveResult(sqrt_tau * sqrt_tau / gamma,
                                                           "Casson viscosity");
    }

    double shearStress(double gamma_dot) const override {
        // The regularized constitutive response is odd in signed shear rate and
        // passes through the origin. Returning the positive yield stress at
        // gamma_dot == 0 would invent a stress direction where none was given.
        return ViscosityModel::shearStress(gamma_dot);
    }

    std::string name() const override { return "Casson"; }
    FluidModel type() const override { return FluidModel::CASSON; }

    double yieldStress() const { return tau_y_; }
    double plasticViscosity() const { return mu_p_; }

private:
    double tau_y_;
    double mu_p_;
    double epsilon_;
};

// =============================================================================
// Utility functions for blood rheology
// =============================================================================

/**
 * @brief Create a Casson model for blood based on hematocrit.
 *
 * Uses the bounded Merrill hematocrit parameterization reproduced in
 * Mouza et al., Fluids 3 (2018) 75, doi:10.3390/fluids3040075:
 *
 *   mu_p = mu_plasma * (1 + 0.025 H + 7.35e-4 H^2)
 *   tau_y = A * (H - H_c)^3 for H > H_c
 *
 * where H is hematocrit in percent, H_c = 6%, and A = 0.9e-7 Pa.
 * The helper is a population-level constitutive parameterization, not a
 * patient-specific blood model; aggregation, temperature, anticoagulant,
 * plasma proteins, and vessel-scale cell migration are not represented.
 *
 * Mouza et al. studied healthy-blood cases from 35% to 55% hematocrit.
 * Inputs outside that evidence range are formula extrapolations even though
 * the helper supports the wider mathematical interval [0, 0.60].
 *
 * @param hematocrit Volume fraction of red blood cells (0.0 to 0.60)
 * @return CassonModel configured for blood at given hematocrit
 */
inline CassonModel bloodCassonModel(double hematocrit) {
    non_newtonian_detail::requireFinite(hematocrit, "Hematocrit");
    if (hematocrit < 0.0 || hematocrit > 0.60) {
        throw std::invalid_argument(
            "Hematocrit must be in the supported correlation range [0, 0.60]");
    }

    constexpr double critical_hematocrit_percent = 6.0;
    constexpr double yield_coefficient_pa = 0.9e-7;
    constexpr double plasma_viscosity_pa_s = 0.0012;
    const double hematocrit_percent = 100.0 * hematocrit;
    const double excess_percent = std::max(0.0, hematocrit_percent - critical_hematocrit_percent);
    const double tau_y = yield_coefficient_pa * std::pow(excess_percent, 3.0);
    const double mu_p = plasma_viscosity_pa_s * (1.0 + 0.025 * hematocrit_percent +
                                                 7.35e-4 * hematocrit_percent * hematocrit_percent);

    return CassonModel(tau_y, mu_p);
}

/**
 * @brief Create a Carreau model for blood.
 *
 * Uses the commonly reported 45% hematocrit Carreau fit
 * (mu0=0.056 Pa s, mu_inf=0.00345 Pa s, lambda=3.313 s, n=0.3568).
 * Away from 45%, the high-shear viscosity increment is scaled with the shape
 * of the Merrill correlation used by bloodCassonModel, while the low-shear
 * excess is linearly scaled with hematocrit. That interpolation is an explicit
 * educational surrogate rather than a validated hematocrit-dependent fit and
 * should be replaced by sample-specific rheometry for predictive work.
 *
 * @param hematocrit Volume fraction of red blood cells (0.0 to 0.60)
 * @return CarreauModel configured for blood at given hematocrit
 */
inline CarreauModel bloodCarreauModel(double hematocrit) {
    non_newtonian_detail::requireFinite(hematocrit, "Hematocrit");
    if (hematocrit < 0.0 || hematocrit > 0.60) {
        throw std::invalid_argument(
            "Hematocrit must be in the supported surrogate range [0, 0.60]");
    }

    constexpr double plasma_viscosity_pa_s = 0.0012;
    constexpr double reference_hematocrit = 0.45;
    constexpr double reference_mu0_pa_s = 0.056;
    constexpr double reference_mu_inf_pa_s = 0.00345;
    constexpr double lambda = 3.313;
    constexpr double n = 0.3568;
    const auto high_shear_viscosity = [](double volume_fraction) {
        constexpr double plasma = 0.0012;
        const double percent = 100.0 * volume_fraction;
        return plasma * (1.0 + 0.025 * percent + 7.35e-4 * percent * percent);
    };
    const double raw_mu_inf = high_shear_viscosity(hematocrit);
    const double raw_reference_mu_inf = high_shear_viscosity(reference_hematocrit);
    const double relative_rbc_increment =
        (raw_mu_inf - plasma_viscosity_pa_s) / (raw_reference_mu_inf - plasma_viscosity_pa_s);
    const double mu_inf = plasma_viscosity_pa_s +
                          (reference_mu_inf_pa_s - plasma_viscosity_pa_s) * relative_rbc_increment;
    const double mu0 =
        mu_inf + (reference_mu0_pa_s - reference_mu_inf_pa_s) * (hematocrit / reference_hematocrit);

    return CarreauModel(mu0, mu_inf, lambda, n);
}

/**
 * @brief Compute shear rate in pipe flow.
 *
 * For Newtonian Poiseuille flow, the nominal wall shear-rate magnitude is
 * gamma_dot = 4*|Q|/(pi*R^3). For non-Newtonian fluids this is the nominal
 * (uncorrected) rate, not the true wall shear rate.
 *
 * @param Q Volume flow rate [m^3/s]
 * @param R Pipe radius [m]
 * @return Wall shear rate [1/s]
 */
inline double pipeWallShearRate(double Q, double R) {
    non_newtonian_detail::requireFinite(Q, "Volume flow rate Q");
    non_newtonian_detail::requirePositive(R, "Pipe radius R");
    constexpr double pi = 3.141592653589793238462643383279502884;
    return non_newtonian_detail::requireFiniteResult(4.0 * std::abs(Q) / (pi * R * R * R),
                                                     "Nominal wall shear rate");
}

namespace non_newtonian_detail {

inline double positiveShearStress(const ViscosityModel& model, double gamma_dot) {
    try {
        const double stress = model.shearStress(gamma_dot);
        if (!std::isfinite(stress) || stress < 0.0) {
            throw std::domain_error(
                "Viscosity model must produce finite non-negative stress for positive shear");
        }
        return stress;
    } catch (const std::overflow_error&) {
        return std::numeric_limits<double>::infinity();
    }
}

inline double shearRateAtStress(const ViscosityModel& model, double target_stress) {
    if (target_stress == 0.0) {
        return 0.0;
    }

    double lower = 0.0;
    double upper = std::max(1e-12, target_stress / model.viscosity(0.0));
    bool bracketed = false;
    for (int iteration = 0; iteration < 256; ++iteration) {
        if (positiveShearStress(model, upper) >= target_stress) {
            bracketed = true;
            break;
        }
        if (upper > std::numeric_limits<double>::max() / 2.0) {
            break;
        }
        upper *= 2.0;
    }
    if (!bracketed) {
        throw std::domain_error(
            "Viscosity model stress is not invertible over the finite shear-rate domain");
    }

    for (int iteration = 0; iteration < 100; ++iteration) {
        const double midpoint = lower + 0.5 * (upper - lower);
        if (positiveShearStress(model, midpoint) < target_stress) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    return lower + 0.5 * (upper - lower);
}

inline double dimensionlessPipeFlux(const ViscosityModel& model, double wall_stress) {
    // Q/(pi R^3) = integral_0^1 gamma(tau_w*s) s^2 ds.
    constexpr int panels = 64;
    constexpr double spacing = 1.0 / static_cast<double>(panels);
    double weighted_sum = 0.0;
    for (int index = 0; index <= panels; ++index) {
        const double radius_fraction = spacing * static_cast<double>(index);
        const double integrand = shearRateAtStress(model, wall_stress * radius_fraction) *
                                 radius_fraction * radius_fraction;
        const double weight = (index == 0 || index == panels) ? 1.0 : (index % 2 == 0 ? 2.0 : 4.0);
        weighted_sum += weight * integrand;
    }
    return requirePositiveResult(weighted_sum * spacing / 3.0, "Model pipe-flow integral");
}

}  // namespace non_newtonian_detail

/**
 * @brief Compute apparent viscosity for pipe flow.
 *
 * Uses the Rabinowitsch-Mooney relation
 *
 *   gamma_w = Q/(pi R^3) * (3 + d ln(Q_model)/d ln(tau_w))
 *
 * where Q_model(tau_w) is obtained by integrating the selected monotone
 * generalized-Newtonian constitutive law across a circular pipe. This makes
 * the selected model determine the flow-curve slope instead of silently using
 * the Newtonian factor four. The magnitudes of Q and dP/dz are used, so reverse
 * flow produces the same positive apparent viscosity.
 *
 * @param model Viscosity model
 * @param Q Volume flow rate [m^3/s]
 * @param R Pipe radius [m]
 * @param dP_dz Pressure gradient [Pa/m]
 * @return Apparent viscosity [Pa·s]
 */
inline double apparentViscosityPipe(const ViscosityModel& model, double Q, double R, double dP_dz) {
    non_newtonian_detail::requireFinite(Q, "Volume flow rate Q");
    non_newtonian_detail::requirePositive(R, "Pipe radius R");
    non_newtonian_detail::requireFinite(dP_dz, "Pressure gradient dP_dz");
    if (Q == 0.0) {
        throw std::domain_error("Apparent pipe viscosity is undefined for zero measured flow rate");
    }
    if (dP_dz == 0.0) {
        throw std::domain_error("Apparent pipe viscosity is undefined for zero pressure gradient");
    }

    const double tau_w =
        non_newtonian_detail::requirePositiveResult(R * std::abs(dP_dz) / 2.0, "Wall shear stress");
    constexpr double log_step = 1e-4;
    const double lower_stress = tau_w * std::exp(-log_step);
    const double upper_stress = tau_w * std::exp(log_step);
    const double lower_flux = non_newtonian_detail::dimensionlessPipeFlux(model, lower_stress);
    const double upper_flux = non_newtonian_detail::dimensionlessPipeFlux(model, upper_stress);
    const double log_slope = (std::log(upper_flux) - std::log(lower_flux)) / (2.0 * log_step);
    if (!std::isfinite(log_slope) || log_slope <= 0.0) {
        throw std::domain_error("Viscosity model must define a positive monotone pipe-flow slope");
    }

    constexpr double pi = 3.141592653589793238462643383279502884;
    const double gamma_w = non_newtonian_detail::requirePositiveResult(
        std::abs(Q) * (3.0 + log_slope) / (pi * R * R * R), "Rabinowitsch-Mooney wall shear rate");
    return non_newtonian_detail::requirePositiveResult(tau_w / gamma_w, "Apparent pipe viscosity");
}

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NON_NEWTONIAN_HPP
