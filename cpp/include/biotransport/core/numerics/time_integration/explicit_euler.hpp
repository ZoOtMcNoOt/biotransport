#ifndef BIOTRANSPORT_CORE_NUMERICS_TIME_INTEGRATION_EXPLICIT_EULER_HPP
#define BIOTRANSPORT_CORE_NUMERICS_TIME_INTEGRATION_EXPLICIT_EULER_HPP

/**
 * @file explicit_euler.hpp
 * @brief Explicit (Forward) Euler time integration.
 *
 * The simplest time integration method:
 *   u^{n+1} = u^n + dt * f(u^n, t^n)
 *
 * Properties:
 *   - First-order accurate in time: O(dt)
 *   - Conditionally stable (CFL restriction)
 *   - Explicit: no matrix solve required
 *
 * Best suited for:
 *   - Simple problems where stability limit is acceptable
 *   - Non-stiff ODEs/PDEs
 *   - Teaching/demonstration purposes
 */

#include <cmath>
#include <functional>
#include <vector>

namespace biotransport {
namespace time_integration {

/**
 * @brief Apply one step of Forward Euler.
 *
 * Computes: u^{n+1} = u^n + dt * f(u^n)
 *
 * @tparam RHSFunc Callable type for right-hand side
 * @param u Current state (overwritten with new state)
 * @param rhs Function that computes du/dt given u
 * @param dt Time step size
 */
template <typename RHSFunc>
inline void euler_step(std::vector<double>& u, RHSFunc&& rhs, double dt) {
    std::vector<double> dudt = rhs(u);

    for (size_t i = 0; i < u.size(); ++i) {
        u[i] += dt * dudt[i];
    }
}

/**
 * @brief Apply one step of Forward Euler with time argument.
 *
 * Computes: u^{n+1} = u^n + dt * f(u^n, t^n)
 *
 * @tparam RHSFunc Callable type for right-hand side
 * @param u Current state (overwritten with new state)
 * @param t Current time
 * @param rhs Function that computes du/dt given (u, t)
 * @param dt Time step size
 */
template <typename RHSFunc>
inline void euler_step(std::vector<double>& u, double t, RHSFunc&& rhs, double dt) {
    std::vector<double> dudt = rhs(u, t);

    for (size_t i = 0; i < u.size(); ++i) {
        u[i] += dt * dudt[i];
    }
}

/**
 * @brief Apply multiple Forward Euler steps.
 *
 * @tparam RHSFunc Callable type for right-hand side
 * @param u Current state (overwritten with final state)
 * @param t0 Initial time
 * @param rhs Function that computes du/dt given (u, t)
 * @param dt Time step size
 * @param num_steps Number of steps to take
 * @return Final time
 */
template <typename RHSFunc>
inline double euler_integrate(std::vector<double>& u, double t0, RHSFunc&& rhs, double dt,
                              int num_steps) {
    double t = t0;

    for (int step = 0; step < num_steps; ++step) {
        euler_step(u, t, rhs, dt);
        t += dt;
    }

    return t;
}

/**
 * @brief Heun's method (Improved Euler / RK2).
 *
 * Second-order accurate predictor-corrector:
 *   k1 = f(u^n, t^n)
 *   k2 = f(u^n + dt*k1, t^n + dt)
 *   u^{n+1} = u^n + dt/2 * (k1 + k2)
 *
 * @tparam RHSFunc Callable type for right-hand side
 * @param u Current state (overwritten with new state)
 * @param t Current time
 * @param rhs Function that computes du/dt given (u, t)
 * @param dt Time step size
 */
template <typename RHSFunc>
inline void heun_step(std::vector<double>& u, double t, RHSFunc&& rhs, double dt) {
    const size_t n = u.size();

    // k1 = f(u, t)
    std::vector<double> k1 = rhs(u, t);

    // Predictor: u_pred = u + dt * k1
    std::vector<double> u_pred(n);
    for (size_t i = 0; i < n; ++i) {
        u_pred[i] = u[i] + dt * k1[i];
    }

    // k2 = f(u_pred, t + dt)
    std::vector<double> k2 = rhs(u_pred, t + dt);

    // Corrector: u^{n+1} = u + dt/2 * (k1 + k2)
    for (size_t i = 0; i < n; ++i) {
        u[i] += 0.5 * dt * (k1[i] + k2[i]);
    }
}

/**
 * @brief Classic 4th-order Runge-Kutta (RK4).
 *
 * Fourth-order accurate:
 *   k1 = f(u^n, t^n)
 *   k2 = f(u^n + dt/2*k1, t^n + dt/2)
 *   k3 = f(u^n + dt/2*k2, t^n + dt/2)
 *   k4 = f(u^n + dt*k3, t^n + dt)
 *   u^{n+1} = u^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 *
 * @tparam RHSFunc Callable type for right-hand side
 * @param u Current state (overwritten with new state)
 * @param t Current time
 * @param rhs Function that computes du/dt given (u, t)
 * @param dt Time step size
 */
template <typename RHSFunc>
inline void rk4_step(std::vector<double>& u, double t, RHSFunc&& rhs, double dt) {
    const size_t n = u.size();

    std::vector<double> k1 = rhs(u, t);

    // k2
    std::vector<double> u_temp(n);
    for (size_t i = 0; i < n; ++i) {
        u_temp[i] = u[i] + 0.5 * dt * k1[i];
    }
    std::vector<double> k2 = rhs(u_temp, t + 0.5 * dt);

    // k3
    for (size_t i = 0; i < n; ++i) {
        u_temp[i] = u[i] + 0.5 * dt * k2[i];
    }
    std::vector<double> k3 = rhs(u_temp, t + 0.5 * dt);

    // k4
    for (size_t i = 0; i < n; ++i) {
        u_temp[i] = u[i] + dt * k3[i];
    }
    std::vector<double> k4 = rhs(u_temp, t + dt);

    // Update
    for (size_t i = 0; i < n; ++i) {
        u[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/**
 * @brief Midpoint method (RK2 variant).
 *
 * Second-order accurate:
 *   k1 = f(u^n, t^n)
 *   k2 = f(u^n + dt/2*k1, t^n + dt/2)
 *   u^{n+1} = u^n + dt * k2
 *
 * @tparam RHSFunc Callable type for right-hand side
 * @param u Current state (overwritten with new state)
 * @param t Current time
 * @param rhs Function that computes du/dt given (u, t)
 * @param dt Time step size
 */
template <typename RHSFunc>
inline void midpoint_step(std::vector<double>& u, double t, RHSFunc&& rhs, double dt) {
    const size_t n = u.size();

    // k1 = f(u, t)
    std::vector<double> k1 = rhs(u, t);

    // u_mid = u + dt/2 * k1
    std::vector<double> u_mid(n);
    for (size_t i = 0; i < n; ++i) {
        u_mid[i] = u[i] + 0.5 * dt * k1[i];
    }

    // k2 = f(u_mid, t + dt/2)
    std::vector<double> k2 = rhs(u_mid, t + 0.5 * dt);

    // Update: u^{n+1} = u + dt * k2
    for (size_t i = 0; i < n; ++i) {
        u[i] += dt * k2[i];
    }
}

}  // namespace time_integration
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_NUMERICS_TIME_INTEGRATION_EXPLICIT_EULER_HPP
