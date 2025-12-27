#ifndef BIOTRANSPORT_CORE_DIMENSIONLESS_HPP
#define BIOTRANSPORT_CORE_DIMENSIONLESS_HPP

/**
 * @file dimensionless.hpp
 * @brief Dimensionless number utilities for biotransport analysis.
 *
 * These are the standard dimensionless groups taught in BMEN 341 and used
 * throughout mass/heat/momentum transport. All functions are header-only
 * and constexpr-friendly.
 */

#include <cmath>
#include <stdexcept>

namespace biotransport {
namespace dimensionless {

/**
 * @brief Reynolds number: ratio of inertial to viscous forces.
 *
 * Re = ρ v L / μ  =  v L / ν
 *
 * @param density       Fluid density ρ [kg/m³]
 * @param velocity      Characteristic velocity v [m/s]
 * @param length        Characteristic length L [m]
 * @param viscosity     Dynamic viscosity μ [Pa·s]
 * @return Reynolds number (dimensionless)
 */
inline double reynolds(double density, double velocity, double length, double viscosity) {
    if (viscosity <= 0.0) {
        throw std::invalid_argument("viscosity must be positive");
    }
    return density * velocity * length / viscosity;
}

/**
 * @brief Reynolds number from kinematic viscosity.
 *
 * Re = v L / ν
 *
 * @param velocity              Characteristic velocity v [m/s]
 * @param length                Characteristic length L [m]
 * @param kinematic_viscosity   Kinematic viscosity ν [m²/s]
 * @return Reynolds number (dimensionless)
 */
inline double reynolds_kinematic(double velocity, double length, double kinematic_viscosity) {
    if (kinematic_viscosity <= 0.0) {
        throw std::invalid_argument("kinematic_viscosity must be positive");
    }
    return velocity * length / kinematic_viscosity;
}

/**
 * @brief Schmidt number: ratio of momentum diffusivity to mass diffusivity.
 *
 * Sc = μ / (ρ D) = ν / D
 *
 * @param viscosity     Dynamic viscosity μ [Pa·s]
 * @param density       Fluid density ρ [kg/m³]
 * @param diffusivity   Mass diffusivity D [m²/s]
 * @return Schmidt number (dimensionless)
 */
inline double schmidt(double viscosity, double density, double diffusivity) {
    if (diffusivity <= 0.0) {
        throw std::invalid_argument("diffusivity must be positive");
    }
    if (density <= 0.0) {
        throw std::invalid_argument("density must be positive");
    }
    return viscosity / (density * diffusivity);
}

/**
 * @brief Schmidt number from kinematic viscosity.
 *
 * Sc = ν / D
 *
 * @param kinematic_viscosity   Kinematic viscosity ν [m²/s]
 * @param diffusivity           Mass diffusivity D [m²/s]
 * @return Schmidt number (dimensionless)
 */
inline double schmidt_kinematic(double kinematic_viscosity, double diffusivity) {
    if (diffusivity <= 0.0) {
        throw std::invalid_argument("diffusivity must be positive");
    }
    return kinematic_viscosity / diffusivity;
}

/**
 * @brief Peclet number: ratio of convective to diffusive transport rate.
 *
 * Pe = v L / D = Re × Sc
 *
 * @param velocity      Characteristic velocity v [m/s]
 * @param length        Characteristic length L [m]
 * @param diffusivity   Mass (or thermal) diffusivity D [m²/s]
 * @return Peclet number (dimensionless)
 */
inline double peclet(double velocity, double length, double diffusivity) {
    if (diffusivity <= 0.0) {
        throw std::invalid_argument("diffusivity must be positive");
    }
    return velocity * length / diffusivity;
}

/**
 * @brief Biot number (mass transfer): ratio of external to internal mass transfer resistance.
 *
 * Bi_m = h_m L / D
 *
 * A small Bi (< 0.1) means internal diffusion is fast compared to external convection,
 * justifying a lumped-parameter (uniform concentration) assumption.
 *
 * @param h_m           Mass transfer coefficient [m/s]
 * @param length        Characteristic length L [m] (often V/A for spheres)
 * @param diffusivity   Internal diffusivity D [m²/s]
 * @return Biot number (dimensionless)
 */
inline double biot(double h_m, double length, double diffusivity) {
    if (diffusivity <= 0.0) {
        throw std::invalid_argument("diffusivity must be positive");
    }
    return h_m * length / diffusivity;
}

/**
 * @brief Fourier number: dimensionless time for diffusion processes.
 *
 * Fo = D t / L²
 *
 * Indicates how far diffusion has penetrated relative to the characteristic length.
 *
 * @param diffusivity   Diffusivity D [m²/s]
 * @param time          Elapsed time t [s]
 * @param length        Characteristic length L [m]
 * @return Fourier number (dimensionless)
 */
inline double fourier(double diffusivity, double time, double length) {
    if (length <= 0.0) {
        throw std::invalid_argument("length must be positive");
    }
    return diffusivity * time / (length * length);
}

/**
 * @brief Sherwood number: ratio of convective to diffusive mass transfer.
 *
 * Sh = h_m L / D
 *
 * Analogous to Nusselt number for heat transfer.
 *
 * @param h_m           Mass transfer coefficient [m/s]
 * @param length        Characteristic length L [m]
 * @param diffusivity   Mass diffusivity D [m²/s]
 * @return Sherwood number (dimensionless)
 */
inline double sherwood(double h_m, double length, double diffusivity) {
    if (diffusivity <= 0.0) {
        throw std::invalid_argument("diffusivity must be positive");
    }
    return h_m * length / diffusivity;
}

/**
 * @brief Check if flow is convection-dominated (Pe > threshold).
 *
 * @param pe        Peclet number
 * @param threshold Threshold for "convection dominated" (default 1.0)
 * @return true if Pe > threshold
 */
inline bool is_convection_dominated(double pe, double threshold = 1.0) {
    return pe > threshold;
}

/**
 * @brief Check if lumped parameter assumption is valid (Bi < threshold).
 *
 * @param bi        Biot number
 * @param threshold Threshold for lumped validity (default 0.1)
 * @return true if Bi < threshold
 */
inline bool is_lumped_valid(double bi, double threshold = 0.1) {
    return bi < threshold;
}

}  // namespace dimensionless
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_DIMENSIONLESS_HPP
