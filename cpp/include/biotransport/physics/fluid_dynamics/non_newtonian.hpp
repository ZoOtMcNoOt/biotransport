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

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace biotransport {

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
    virtual double shearStress(double gamma_dot) const { return viscosity(gamma_dot) * gamma_dot; }
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
        if (mu0 <= 0)
            throw std::invalid_argument("Viscosity must be positive");
    }

    double viscosity(double /*gamma_dot*/) const override { return mu0_; }
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
        if (K <= 0)
            throw std::invalid_argument("Consistency index K must be positive");
        if (n <= 0)
            throw std::invalid_argument("Flow index n must be positive");
    }

    double viscosity(double gamma_dot) const override {
        double gamma = std::max(gamma_min_, std::abs(gamma_dot));
        return K_ * std::pow(gamma, n_ - 1.0);
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
        if (mu0 <= 0)
            throw std::invalid_argument("mu0 must be positive");
        if (mu_inf < 0)
            throw std::invalid_argument("mu_inf must be non-negative");
        if (mu_inf > mu0)
            throw std::invalid_argument("mu_inf must be <= mu0");
        if (lambda <= 0)
            throw std::invalid_argument("lambda must be positive");
        if (n <= 0)
            throw std::invalid_argument("n must be positive");
    }

    double viscosity(double gamma_dot) const override {
        double gamma = std::abs(gamma_dot);
        double factor = std::pow(1.0 + std::pow(lambda_ * gamma, 2.0), (n_ - 1.0) / 2.0);
        return mu_inf_ + (mu0_ - mu_inf_) * factor;
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
        if (mu0 <= 0)
            throw std::invalid_argument("mu0 must be positive");
        if (mu_inf < 0)
            throw std::invalid_argument("mu_inf must be non-negative");
        if (lambda <= 0)
            throw std::invalid_argument("lambda must be positive");
        if (a <= 0)
            throw std::invalid_argument("a must be positive");
        if (n <= 0)
            throw std::invalid_argument("n must be positive");
    }

    double viscosity(double gamma_dot) const override {
        double gamma = std::abs(gamma_dot);
        double factor = std::pow(1.0 + std::pow(lambda_ * gamma, a_), (n_ - 1.0) / a_);
        return mu_inf_ + (mu0_ - mu_inf_) * factor;
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
        if (mu0 <= 0)
            throw std::invalid_argument("mu0 must be positive");
        if (mu_inf < 0)
            throw std::invalid_argument("mu_inf must be non-negative");
        if (K <= 0)
            throw std::invalid_argument("K must be positive");
        if (m <= 0)
            throw std::invalid_argument("m must be positive");
    }

    double viscosity(double gamma_dot) const override {
        double gamma = std::abs(gamma_dot);
        return mu_inf_ + (mu0_ - mu_inf_) / (1.0 + std::pow(K_ * gamma, m_));
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
        if (tau_y < 0)
            throw std::invalid_argument("Yield stress must be non-negative");
        if (mu_p <= 0)
            throw std::invalid_argument("Plastic viscosity must be positive");
        if (epsilon <= 0)
            throw std::invalid_argument("Epsilon must be positive");
    }

    double viscosity(double gamma_dot) const override {
        double gamma = std::abs(gamma_dot);
        return mu_p_ + tau_y_ / (gamma + epsilon_);
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
    double binghamNumber(double L, double U) const { return tau_y_ * L / (mu_p_ * U); }

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
        if (tau_y < 0)
            throw std::invalid_argument("Yield stress must be non-negative");
        if (K <= 0)
            throw std::invalid_argument("K must be positive");
        if (n <= 0)
            throw std::invalid_argument("n must be positive");
        if (epsilon <= 0)
            throw std::invalid_argument("Epsilon must be positive");
    }

    double viscosity(double gamma_dot) const override {
        double gamma = std::abs(gamma_dot) + epsilon_;
        return tau_y_ / gamma + K_ * std::pow(gamma, n_ - 1.0);
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
        if (tau_y < 0)
            throw std::invalid_argument("Yield stress must be non-negative");
        if (mu_p <= 0)
            throw std::invalid_argument("Plastic viscosity must be positive");
        if (epsilon <= 0)
            throw std::invalid_argument("Epsilon must be positive");
    }

    double viscosity(double gamma_dot) const override {
        double gamma = std::abs(gamma_dot) + epsilon_;
        double sqrt_tau_y = std::sqrt(tau_y_);
        double sqrt_mu_gamma = std::sqrt(mu_p_ * gamma);
        double sqrt_tau = sqrt_tau_y + sqrt_mu_gamma;
        return sqrt_tau * sqrt_tau / gamma;
    }

    double shearStress(double gamma_dot) const override {
        double gamma = std::abs(gamma_dot) + epsilon_;
        double sqrt_tau_y = std::sqrt(tau_y_);
        double sqrt_mu_gamma = std::sqrt(mu_p_ * gamma);
        double sqrt_tau = sqrt_tau_y + sqrt_mu_gamma;
        return sqrt_tau * sqrt_tau;
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
 * Uses empirical correlations from Merrill et al.
 *
 * @param hematocrit Volume fraction of red blood cells (0.0 to 0.6)
 * @return CassonModel configured for blood at given hematocrit
 */
inline CassonModel bloodCassonModel(double hematocrit) {
    if (hematocrit < 0.0 || hematocrit > 0.7) {
        throw std::invalid_argument("Hematocrit must be between 0 and 0.7");
    }

    // Empirical correlations (approximate)
    // tau_y = A * (H - H_c)^3 for H > H_c, where H_c ~ 0.05
    double H_c = 0.05;  // Critical hematocrit
    double tau_y = 0.0;
    if (hematocrit > H_c) {
        tau_y = 0.8 * std::pow(hematocrit - H_c, 3.0);  // Simplified correlation
    }

    // mu_p increases with hematocrit
    // mu_plasma ~ 0.0012 Pa·s
    double mu_plasma = 0.0012;
    double mu_p = mu_plasma * std::pow(1.0 - hematocrit / 0.67, -2.5);  // Einstein-like

    return CassonModel(tau_y, mu_p);
}

/**
 * @brief Create a Carreau model for blood.
 *
 * Uses typical literature values.
 *
 * @param hematocrit Volume fraction of red blood cells (0.0 to 0.6)
 * @return CarreauModel configured for blood at given hematocrit
 */
inline CarreauModel bloodCarreauModel(double hematocrit) {
    if (hematocrit < 0.0 || hematocrit > 0.7) {
        throw std::invalid_argument("Hematocrit must be between 0 and 0.7");
    }

    // Typical values for blood at 37C
    // These vary significantly with hematocrit and conditions
    double mu_plasma = 0.0012;  // Pa·s

    // Zero-shear viscosity increases exponentially with hematocrit
    double mu0 = mu_plasma * std::exp(2.5 * hematocrit / (1.0 - 0.45 * hematocrit));

    // Infinite-shear viscosity (close to plasma)
    double mu_inf = mu_plasma * (1.0 + 2.5 * hematocrit);

    // Relaxation time and power-law index (typical values)
    double lambda = 3.313;  // s
    double n = 0.3568;      // shear-thinning

    return CarreauModel(mu0, mu_inf, lambda, n);
}

/**
 * @brief Compute shear rate in pipe flow.
 *
 * For Poiseuille flow: gamma_dot = 4*Q / (pi*R^3) at wall
 *
 * @param Q Volume flow rate [m^3/s]
 * @param R Pipe radius [m]
 * @return Wall shear rate [1/s]
 */
inline double pipeWallShearRate(double Q, double R) {
    return 4.0 * Q / (M_PI * R * R * R);
}

/**
 * @brief Compute apparent viscosity for pipe flow.
 *
 * Uses the Rabinowitsch-Mooney equation for non-Newtonian fluids.
 *
 * @param model Viscosity model
 * @param Q Volume flow rate [m^3/s]
 * @param R Pipe radius [m]
 * @param dP_dz Pressure gradient [Pa/m]
 * @return Apparent viscosity [Pa·s]
 */
inline double apparentViscosityPipe(const ViscosityModel& model, double Q, double R, double dP_dz) {
    // Wall shear stress: tau_w = R * |dP/dz| / 2
    double tau_w = R * std::abs(dP_dz) / 2.0;

    // Wall shear rate (for Newtonian)
    double gamma_w = pipeWallShearRate(Q, R);

    // Apparent viscosity
    return tau_w / gamma_w;
}

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_FLUID_DYNAMICS_NON_NEWTONIAN_HPP
