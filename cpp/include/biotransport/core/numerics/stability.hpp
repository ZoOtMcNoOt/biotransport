#ifndef BIOTRANSPORT_CORE_NUMERICS_STABILITY_HPP
#define BIOTRANSPORT_CORE_NUMERICS_STABILITY_HPP

/**
 * @file stability.hpp
 * @brief Time step stability helpers for explicit finite-difference schemes.
 *
 * These functions compute maximum stable time steps (dt) for various PDEs
 * solved with explicit time integration. All assume 2nd-order central
 * differencing in space with forward Euler in time.
 *
 * Usage:
 *   double dt = suggest_diffusion_dt(dx, D, 0.9);  // 90% of CFL limit
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace biotransport {
namespace stability {

/**
 * @brief Maximum stable dt for 1D diffusion equation.
 *
 * Stability condition: dt <= dx² / (2D)
 *
 * @param dx Grid spacing
 * @param D Diffusion coefficient
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_diffusion_dt_1d(double dx, double D, double safety = 0.9) {
    if (dx <= 0.0)
        throw std::invalid_argument("dx must be positive");
    if (D <= 0.0)
        throw std::invalid_argument("D must be positive");
    if (safety <= 0.0 || safety > 1.0)
        throw std::invalid_argument("safety must be in (0, 1]");

    return safety * (dx * dx) / (2.0 * D);
}

/**
 * @brief Maximum stable dt for 2D diffusion equation.
 *
 * Stability condition: dt <= min(dx², dy²) / (4D)
 *
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param D Diffusion coefficient
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_diffusion_dt_2d(double dx, double dy, double D, double safety = 0.9) {
    if (dx <= 0.0)
        throw std::invalid_argument("dx must be positive");
    if (dy <= 0.0)
        throw std::invalid_argument("dy must be positive");
    if (D <= 0.0)
        throw std::invalid_argument("D must be positive");
    if (safety <= 0.0 || safety > 1.0)
        throw std::invalid_argument("safety must be in (0, 1]");

    const double min_h2 = std::min(dx * dx, dy * dy);
    return safety * min_h2 / (4.0 * D);
}

/**
 * @brief Maximum stable dt for 1D advection equation (upwind scheme).
 *
 * CFL condition: dt <= dx / |v|
 *
 * @param dx Grid spacing
 * @param v Advection velocity (magnitude used)
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_advection_dt_1d(double dx, double v, double safety = 0.9) {
    if (dx <= 0.0)
        throw std::invalid_argument("dx must be positive");
    if (safety <= 0.0 || safety > 1.0)
        throw std::invalid_argument("safety must be in (0, 1]");

    const double v_mag = std::abs(v);
    if (v_mag < 1e-15)
        return 1e10;  // No advection limit

    return safety * dx / v_mag;
}

/**
 * @brief Maximum stable dt for 2D advection equation (upwind scheme).
 *
 * CFL condition: dt <= 1 / (|vx|/dx + |vy|/dy)
 *
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param vx Velocity in x direction
 * @param vy Velocity in y direction
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_advection_dt_2d(double dx, double dy, double vx, double vy,
                                      double safety = 0.9) {
    if (dx <= 0.0)
        throw std::invalid_argument("dx must be positive");
    if (dy <= 0.0)
        throw std::invalid_argument("dy must be positive");
    if (safety <= 0.0 || safety > 1.0)
        throw std::invalid_argument("safety must be in (0, 1]");

    const double inv_cfl = std::abs(vx) / dx + std::abs(vy) / dy;
    if (inv_cfl < 1e-15)
        return 1e10;  // No advection limit

    return safety / inv_cfl;
}

/**
 * @brief Maximum stable dt for advection-diffusion (1D).
 *
 * Takes the minimum of diffusion and advection constraints.
 *
 * @param dx Grid spacing
 * @param D Diffusion coefficient
 * @param v Advection velocity
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_advection_diffusion_dt_1d(double dx, double D, double v,
                                                double safety = 0.9) {
    const double dt_diff = suggest_diffusion_dt_1d(dx, D, safety);
    const double dt_adv = suggest_advection_dt_1d(dx, v, safety);
    return std::min(dt_diff, dt_adv);
}

/**
 * @brief Maximum stable dt for advection-diffusion (2D).
 *
 * Takes the minimum of diffusion and advection constraints.
 *
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param D Diffusion coefficient
 * @param vx Velocity in x
 * @param vy Velocity in y
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_advection_diffusion_dt_2d(double dx, double dy, double D, double vx,
                                                double vy, double safety = 0.9) {
    const double dt_diff = suggest_diffusion_dt_2d(dx, dy, D, safety);
    const double dt_adv = suggest_advection_dt_2d(dx, dy, vx, vy, safety);
    return std::min(dt_diff, dt_adv);
}

/**
 * @brief Maximum stable dt for reaction-diffusion with linear decay.
 *
 * Combines diffusion stability with reaction stability: dt <= 1/k
 *
 * @param dx Grid spacing
 * @param D Diffusion coefficient
 * @param k Decay rate constant
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_reaction_diffusion_dt_1d(double dx, double D, double k, double safety = 0.9) {
    const double dt_diff = suggest_diffusion_dt_1d(dx, D, safety);
    if (k <= 0.0)
        return dt_diff;

    const double dt_react = safety / k;
    return std::min(dt_diff, dt_react);
}

/**
 * @brief Maximum stable dt for 2D reaction-diffusion with linear decay.
 *
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param D Diffusion coefficient
 * @param k Decay rate constant
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_reaction_diffusion_dt_2d(double dx, double dy, double D, double k,
                                               double safety = 0.9) {
    const double dt_diff = suggest_diffusion_dt_2d(dx, dy, D, safety);
    if (k <= 0.0)
        return dt_diff;

    const double dt_react = safety / k;
    return std::min(dt_diff, dt_react);
}

/**
 * @brief Maximum stable dt for Michaelis-Menten kinetics.
 *
 * Linearization at u=0 gives f'(0) = -Vmax/Km, so dt <= Km/Vmax
 *
 * @param dx Grid spacing
 * @param D Diffusion coefficient
 * @param Vmax Maximum reaction rate
 * @param Km Michaelis constant
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_michaelis_menten_dt_1d(double dx, double D, double Vmax, double Km,
                                             double safety = 0.9) {
    if (Vmax <= 0.0)
        throw std::invalid_argument("Vmax must be positive");
    if (Km <= 0.0)
        throw std::invalid_argument("Km must be positive");

    const double dt_diff = suggest_diffusion_dt_1d(dx, D, safety);
    const double dt_react = safety * Km / Vmax;
    return std::min(dt_diff, dt_react);
}

/**
 * @brief Compute Péclet number for advection-diffusion.
 *
 * Pe = v*L/D where L is characteristic length (grid spacing)
 * Pe > 2 suggests using upwind schemes; Pe >> 1 is advection-dominated.
 *
 * @param dx Grid spacing
 * @param v Velocity magnitude
 * @param D Diffusion coefficient
 * @return Péclet number (cell Péclet number)
 */
inline double peclet_number(double dx, double v, double D) {
    if (D <= 0.0)
        throw std::invalid_argument("D must be positive");
    return std::abs(v) * dx / D;
}

/**
 * @brief Compute Courant number (CFL number) for advection.
 *
 * Co = v*dt/dx
 * For explicit upwind: Co <= 1 required for stability.
 *
 * @param dt Time step
 * @param dx Grid spacing
 * @param v Velocity magnitude
 * @return Courant number
 */
inline double courant_number(double dt, double dx, double v) {
    if (dx <= 0.0)
        throw std::invalid_argument("dx must be positive");
    return std::abs(v) * dt / dx;
}

/**
 * @brief Compute Fourier number (diffusion number).
 *
 * Fo = D*dt/dx²
 * For explicit 1D diffusion: Fo <= 0.5 required for stability.
 * For explicit 2D diffusion: Fo <= 0.25 required for stability.
 *
 * @param dt Time step
 * @param dx Grid spacing
 * @param D Diffusion coefficient
 * @return Fourier number
 */
inline double fourier_number(double dt, double dx, double D) {
    if (dx <= 0.0)
        throw std::invalid_argument("dx must be positive");
    return D * dt / (dx * dx);
}

}  // namespace stability
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_NUMERICS_STABILITY_HPP
