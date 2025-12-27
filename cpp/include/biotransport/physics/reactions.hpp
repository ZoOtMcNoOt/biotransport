#ifndef BIOTRANSPORT_PHYSICS_REACTIONS_HPP
#define BIOTRANSPORT_PHYSICS_REACTIONS_HPP

/**
 * @file reactions.hpp
 * @brief Library of common reaction term functors for reaction-diffusion systems.
 *
 * This header provides a collection of well-documented reaction kinetics
 * commonly used in biotransport modeling. Each functor is a callable that
 * computes the reaction rate R(u, x, y, t) for use with ReactionDiffusionSolver.
 *
 * Instead of creating separate solver classes for each reaction type
 * (LinearReactionDiffusionSolver, LogisticReactionDiffusionSolver, etc.),
 * users can compose any reaction with the generic solver:
 *
 * @code
 *   auto solver = ReactionDiffusionSolver(mesh, D, reactions::MichaelisMenten(Vmax, Km));
 *   // or
 *   auto solver = ReactionDiffusionSolver(mesh, D, reactions::logistic(r, K));
 * @endcode
 *
 * Benefits:
 * - Eliminates 8+ nearly-identical solver class implementations
 * - Clear documentation of reaction kinetics in one place
 * - Easy to combine reactions: R = R1 + R2
 * - Easy to extend with custom reactions
 */

#include <cmath>
#include <functional>

namespace biotransport {
namespace reactions {

/**
 * @brief Standard reaction function signature.
 *
 * @param u Concentration/value at the point
 * @param x x-coordinate
 * @param y y-coordinate (0 for 1D)
 * @param t Current time
 * @return Reaction rate R(u, x, y, t)
 */
using ReactionFunc = std::function<double(double u, double x, double y, double t)>;

// =============================================================================
// Zero-order reactions (constant source/sink)
// =============================================================================

/**
 * @brief Constant source/sink term: R = S
 *
 * @param source_rate Constant source rate S [units/s]
 *                    Positive = source, Negative = sink
 */
inline ReactionFunc constantSource(double source_rate) {
    return [source_rate](double /*u*/, double /*x*/, double /*y*/, double /*t*/) {
        return source_rate;
    };
}

/**
 * @brief Time-varying source: R = S(t)
 *
 * @param source_func Function returning source rate at time t
 */
inline ReactionFunc timeVaryingSource(std::function<double(double t)> source_func) {
    return [source_func](double /*u*/, double /*x*/, double /*y*/, double t) {
        return source_func(t);
    };
}

/**
 * @brief Spatially-varying source: R = S(x, y)
 *
 * @param source_func Function returning source rate at position (x, y)
 */
inline ReactionFunc spatialSource(std::function<double(double x, double y)> source_func) {
    return [source_func](double /*u*/, double x, double y, double /*t*/) {
        return source_func(x, y);
    };
}

// =============================================================================
// First-order reactions (linear decay/growth)
// =============================================================================

/**
 * @brief First-order decay: R = -k * u
 *
 * Models exponential decay (radioactive decay, first-order drug elimination).
 *
 * @param decay_rate First-order decay constant k [1/s]
 */
inline ReactionFunc linearDecay(double decay_rate) {
    return [decay_rate](double u, double /*x*/, double /*y*/, double /*t*/) {
        return -decay_rate * u;
    };
}

/**
 * @brief First-order growth: R = k * u
 *
 * Models exponential growth (unlimited population growth).
 *
 * @param growth_rate First-order growth constant k [1/s]
 */
inline ReactionFunc linearGrowth(double growth_rate) {
    return [growth_rate](double u, double /*x*/, double /*y*/, double /*t*/) {
        return growth_rate * u;
    };
}

// =============================================================================
// Saturation kinetics
// =============================================================================

/**
 * @brief Michaelis-Menten consumption: R = -Vmax * u / (Km + u)
 *
 * Standard enzyme kinetics model. Consumption rate saturates at high u.
 * Used for oxygen consumption by tissue, drug metabolism, etc.
 *
 * @param vmax Maximum consumption rate Vmax [units/s]
 * @param km Michaelis constant Km [concentration units]
 */
inline ReactionFunc michaelisMenten(double vmax, double km) {
    return [vmax, km](double u, double /*x*/, double /*y*/, double /*t*/) {
        if (u <= 0.0)
            return 0.0;
        return -vmax * u / (km + u);
    };
}

/**
 * @brief Michaelis-Menten production: R = Vmax * S / (Km + S)
 *
 * Production of product from substrate S following MM kinetics.
 *
 * @param vmax Maximum production rate Vmax [units/s]
 * @param km Michaelis constant Km [concentration units]
 */
inline ReactionFunc michaelisMentenProduction(double vmax, double km) {
    return [vmax, km](double u, double /*x*/, double /*y*/, double /*t*/) {
        if (u <= 0.0)
            return 0.0;
        return vmax * u / (km + u);
    };
}

/**
 * @brief Hill equation: R = -Vmax * u^n / (K^n + u^n)
 *
 * Generalization of Michaelis-Menten with cooperativity.
 * n > 1: positive cooperativity (sigmoidal)
 * n < 1: negative cooperativity
 * n = 1: Michaelis-Menten
 *
 * @param vmax Maximum rate
 * @param k Half-saturation constant
 * @param n Hill coefficient
 */
inline ReactionFunc hillEquation(double vmax, double k, double n) {
    return [vmax, k, n](double u, double /*x*/, double /*y*/, double /*t*/) {
        if (u <= 0.0)
            return 0.0;
        double u_n = std::pow(u, n);
        double k_n = std::pow(k, n);
        return -vmax * u_n / (k_n + u_n);
    };
}

// =============================================================================
// Population dynamics
// =============================================================================

/**
 * @brief Logistic growth: R = r * u * (1 - u/K)
 *
 * Classic population growth model with carrying capacity.
 * Growth is positive when u < K, negative when u > K.
 *
 * @param growth_rate Intrinsic growth rate r [1/s]
 * @param carrying_capacity Carrying capacity K [concentration units]
 */
inline ReactionFunc logistic(double growth_rate, double carrying_capacity) {
    return [growth_rate, carrying_capacity](double u, double /*x*/, double /*y*/, double /*t*/) {
        return growth_rate * u * (1.0 - u / carrying_capacity);
    };
}

/**
 * @brief Gompertz growth: R = r * u * ln(K/u)
 *
 * Alternative to logistic, slower approach to carrying capacity.
 * Common in tumor growth modeling.
 *
 * @param growth_rate Growth rate r [1/s]
 * @param carrying_capacity Carrying capacity K
 */
inline ReactionFunc gompertz(double growth_rate, double carrying_capacity) {
    return [growth_rate, carrying_capacity](double u, double /*x*/, double /*y*/, double /*t*/) {
        if (u <= 0.0)
            return 0.0;
        return growth_rate * u * std::log(carrying_capacity / u);
    };
}

/**
 * @brief Allee effect: R = r * u * (u/A - 1) * (1 - u/K)
 *
 * Strong Allee effect with threshold A below which population declines.
 *
 * @param growth_rate Maximum growth rate r
 * @param allee_threshold Allee threshold A
 * @param carrying_capacity Carrying capacity K
 */
inline ReactionFunc alleeEffect(double growth_rate, double allee_threshold,
                                double carrying_capacity) {
    return [growth_rate, allee_threshold, carrying_capacity](double u, double /*x*/, double /*y*/,
                                                             double /*t*/) {
        return growth_rate * u * (u / allee_threshold - 1.0) * (1.0 - u / carrying_capacity);
    };
}

// =============================================================================
// Chemical kinetics
// =============================================================================

/**
 * @brief Second-order decay: R = -k * u^2
 *
 * @param rate_constant Second-order rate constant k
 */
inline ReactionFunc secondOrderDecay(double rate_constant) {
    return [rate_constant](double u, double /*x*/, double /*y*/, double /*t*/) {
        return -rate_constant * u * u;
    };
}

/**
 * @brief Reversible first-order reaction: A ⇌ B
 *
 * R = -k_f * u + k_r * (u0 - u)
 *
 * @param k_forward Forward rate constant
 * @param k_reverse Reverse rate constant
 * @param initial_total Initial total concentration (conserved)
 */
inline ReactionFunc reversibleFirstOrder(double k_forward, double k_reverse, double initial_total) {
    return
        [k_forward, k_reverse, initial_total](double u, double /*x*/, double /*y*/, double /*t*/) {
            return -k_forward * u + k_reverse * (initial_total - u);
        };
}

// =============================================================================
// Combination utilities
// =============================================================================

/**
 * @brief Combine two reaction terms: R = R1 + R2
 *
 * @param r1 First reaction function
 * @param r2 Second reaction function
 */
inline ReactionFunc combine(ReactionFunc r1, ReactionFunc r2) {
    return [r1, r2](double u, double x, double y, double t) {
        return r1(u, x, y, t) + r2(u, x, y, t);
    };
}

/**
 * @brief Scale a reaction term: R = scale * R_original
 *
 * @param reaction Original reaction function
 * @param scale Scaling factor
 */
inline ReactionFunc scale(ReactionFunc reaction, double scale_factor) {
    return [reaction, scale_factor](double u, double x, double y, double t) {
        return scale_factor * reaction(u, x, y, t);
    };
}

/**
 * @brief No reaction (pure diffusion): R = 0
 */
inline ReactionFunc none() {
    return [](double /*u*/, double /*x*/, double /*y*/, double /*t*/) {
        return 0.0;
    };
}

// =============================================================================
// Biotransport-specific models
// =============================================================================

/**
 * @brief Oxygen consumption with critical threshold
 *
 * R = -Vmax * u / (Km + u)  if u > u_crit
 * R = 0                      if u <= u_crit (tissue death, no consumption)
 *
 * @param vmax Maximum consumption rate
 * @param km Michaelis constant
 * @param u_critical Critical oxygen level below which consumption stops
 */
inline ReactionFunc oxygenConsumption(double vmax, double km, double u_critical) {
    return [vmax, km, u_critical](double u, double /*x*/, double /*y*/, double /*t*/) {
        if (u <= u_critical)
            return 0.0;
        return -vmax * u / (km + u);
    };
}

/**
 * @brief Drug binding with receptor saturation
 *
 * Models reversible binding: Drug + Receptor ⇌ Complex
 * R = -k_on * u * (R_total - bound) + k_off * bound
 *
 * For the free drug concentration where bound is implicit:
 * R ≈ -k_on * u * R_total / (1 + u/Kd)
 *
 * @param k_on Association rate constant
 * @param receptor_total Total receptor concentration
 * @param kd Dissociation constant (Kd = k_off / k_on)
 */
inline ReactionFunc receptorBinding(double k_on, double receptor_total, double kd) {
    return [k_on, receptor_total, kd](double u, double /*x*/, double /*y*/, double /*t*/) {
        if (u <= 0.0)
            return 0.0;
        return -k_on * u * receptor_total / (1.0 + u / kd);
    };
}

}  // namespace reactions
}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_REACTIONS_HPP
