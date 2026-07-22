#ifndef BIOTRANSPORT_CORE_NUMERICS_STABILITY_HPP
#define BIOTRANSPORT_CORE_NUMERICS_STABILITY_HPP

/**
 * @file stability.hpp
 * @brief Time step stability helpers for explicit finite-difference schemes.
 *
 * These functions compute maximum stable time steps (dt) for various PDEs
 * solved with forward Euler time integration. Diffusion uses second-order
 * centered differences; the advection and combined advection-diffusion bounds
 * are for first-order upwind advection.
 *
 * Usage:
 *   double dt = suggest_diffusion_dt_1d(dx, D, 0.9);  // 90% of the limit
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace biotransport {
namespace stability {

namespace detail {

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

inline void requireSafety(double safety) {
    requireFinite(safety, "safety");
    if (safety <= 0.0 || safety > 1.0) {
        throw std::invalid_argument("safety must be in (0, 1]");
    }
}

inline double stepFromRate(long double rate, double safety, const char* name) {
    if (!std::isfinite(rate) || rate <= 0.0L) {
        throw std::overflow_error(std::string(name) + " rate is non-finite or non-positive");
    }
    const long double step = static_cast<long double>(safety) / rate;
    if (!std::isfinite(step) || step <= 0.0L ||
        step > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::overflow_error(std::string(name) + " time step is not representable");
    }
    return static_cast<double>(step);
}

inline double finiteResult(long double value, const char* name) {
    if (!std::isfinite(value) ||
        std::abs(value) > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::overflow_error(std::string(name) + " is not representable");
    }
    return static_cast<double>(value);
}

}  // namespace detail

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
    detail::requirePositive(dx, "dx");
    detail::requirePositive(D, "D");
    detail::requireSafety(safety);
    const long double spacing = dx;
    const long double rate = 2.0L * D / (spacing * spacing);
    return detail::stepFromRate(rate, safety, "1D diffusion");
}

/**
 * @brief Maximum stable dt for 2D diffusion equation.
 *
 * Stability condition: dt <= 1 / (2D(1/dx² + 1/dy²))
 *
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param D Diffusion coefficient
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_diffusion_dt_2d(double dx, double dy, double D, double safety = 0.9) {
    detail::requirePositive(dx, "dx");
    detail::requirePositive(dy, "dy");
    detail::requirePositive(D, "D");
    detail::requireSafety(safety);
    const long double x_spacing = dx;
    const long double y_spacing = dy;
    const long double rate =
        2.0L * D * (1.0L / (x_spacing * x_spacing) + 1.0L / (y_spacing * y_spacing));
    return detail::stepFromRate(rate, safety, "2D diffusion");
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
    detail::requirePositive(dx, "dx");
    detail::requireFinite(v, "v");
    detail::requireSafety(safety);
    const long double rate = std::abs(static_cast<long double>(v)) / dx;
    if (rate == 0.0L) {
        return std::numeric_limits<double>::max();
    }
    return detail::stepFromRate(rate, safety, "1D advection");
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
    detail::requirePositive(dx, "dx");
    detail::requirePositive(dy, "dy");
    detail::requireFinite(vx, "vx");
    detail::requireFinite(vy, "vy");
    detail::requireSafety(safety);
    const long double rate =
        std::abs(static_cast<long double>(vx)) / dx + std::abs(static_cast<long double>(vy)) / dy;
    if (rate == 0.0L) {
        return std::numeric_limits<double>::max();
    }
    return detail::stepFromRate(rate, safety, "2D advection");
}

/**
 * @brief Maximum stable dt for advection-diffusion (1D).
 *
 * For forward Euler with first-order upwind advection, non-negative stencil
 * coefficients require
 *   dt * (2D/dx² + |v|/dx) <= 1.
 * This combined bound is stricter than taking the minimum of the separate
 * diffusion and advection limits.
 *
 * @param dx Grid spacing
 * @param D Diffusion coefficient
 * @param v Advection velocity
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_advection_diffusion_dt_1d(double dx, double D, double v,
                                                double safety = 0.9) {
    detail::requirePositive(dx, "dx");
    detail::requirePositive(D, "D");
    detail::requireFinite(v, "v");
    detail::requireSafety(safety);
    const long double spacing = dx;
    const long double rate =
        2.0L * D / (spacing * spacing) + std::abs(static_cast<long double>(v)) / spacing;
    return detail::stepFromRate(rate, safety, "1D advection-diffusion");
}

/**
 * @brief Maximum stable dt for advection-diffusion (2D).
 *
 * For forward Euler with dimension-by-dimension first-order upwinding,
 * non-negative stencil coefficients require
 *   dt * (2D(1/dx² + 1/dy²) + |vx|/dx + |vy|/dy) <= 1.
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
    detail::requirePositive(dx, "dx");
    detail::requirePositive(dy, "dy");
    detail::requirePositive(D, "D");
    detail::requireFinite(vx, "vx");
    detail::requireFinite(vy, "vy");
    detail::requireSafety(safety);
    const long double x_spacing = dx;
    const long double y_spacing = dy;
    const long double rate =
        2.0L * D * (1.0L / (x_spacing * x_spacing) + 1.0L / (y_spacing * y_spacing)) +
        std::abs(static_cast<long double>(vx)) / x_spacing +
        std::abs(static_cast<long double>(vy)) / y_spacing;
    return detail::stepFromRate(rate, safety, "2D advection-diffusion");
}

/**
 * @brief Maximum stable dt for reaction-diffusion with linear decay.
 *
 * For forward Euler applied to dc/dt = D*d2c/dx2 - k*c, the
 * positivity-preserving combined bound is dt*(2D/dx^2 + k) <= 1.
 *
 * @param dx Grid spacing
 * @param D Diffusion coefficient
 * @param k Decay rate constant
 * @param safety Safety factor (0 < safety <= 1), default 0.9
 * @return Maximum stable time step
 */
inline double suggest_reaction_diffusion_dt_1d(double dx, double D, double k, double safety = 0.9) {
    detail::requirePositive(dx, "dx");
    detail::requirePositive(D, "D");
    detail::requireNonnegative(k, "k");
    detail::requireSafety(safety);
    const long double spacing = dx;
    const long double rate = 2.0L * D / (spacing * spacing) + k;
    return detail::stepFromRate(rate, safety, "1D reaction-diffusion");
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
    detail::requirePositive(dx, "dx");
    detail::requirePositive(dy, "dy");
    detail::requirePositive(D, "D");
    detail::requireNonnegative(k, "k");
    detail::requireSafety(safety);
    const long double x_spacing = dx;
    const long double y_spacing = dy;
    const long double rate =
        2.0L * D * (1.0L / (x_spacing * x_spacing) + 1.0L / (y_spacing * y_spacing)) + k;
    return detail::stepFromRate(rate, safety, "2D reaction-diffusion");
}

/**
 * @brief Maximum stable dt for Michaelis-Menten kinetics.
 *
 * Linearization at u=0 gives the maximum loss slope Vmax/Km. The returned
 * positivity bound combines that slope with the diffusion depletion rate
 * instead of taking two separate minima.
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
    detail::requirePositive(dx, "dx");
    detail::requirePositive(D, "D");
    detail::requirePositive(Vmax, "Vmax");
    detail::requirePositive(Km, "Km");
    detail::requireSafety(safety);
    const long double spacing = dx;
    const long double rate = 2.0L * D / (spacing * spacing) +
                             static_cast<long double>(Vmax) / static_cast<long double>(Km);
    return detail::stepFromRate(rate, safety, "Michaelis-Menten reaction-diffusion");
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
    detail::requirePositive(dx, "dx");
    detail::requireFinite(v, "v");
    detail::requirePositive(D, "D");
    const long double value =
        std::abs(static_cast<long double>(v)) * static_cast<long double>(dx) / D;
    return detail::finiteResult(value, "Peclet number");
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
    detail::requireNonnegative(dt, "dt");
    detail::requirePositive(dx, "dx");
    detail::requireFinite(v, "v");
    const long double value =
        std::abs(static_cast<long double>(v)) * static_cast<long double>(dt) / dx;
    return detail::finiteResult(value, "Courant number");
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
    detail::requireNonnegative(dt, "dt");
    detail::requirePositive(dx, "dx");
    detail::requireNonnegative(D, "D");
    const long double spacing = dx;
    const long double value =
        static_cast<long double>(D) * static_cast<long double>(dt) / (spacing * spacing);
    return detail::finiteResult(value, "Fourier number");
}

}  // namespace stability
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_NUMERICS_STABILITY_HPP
