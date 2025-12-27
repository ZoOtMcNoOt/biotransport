#ifndef BIOTRANSPORT_CORE_BOUNDARY_HPP
#define BIOTRANSPORT_CORE_BOUNDARY_HPP

/**
 * @file boundary.hpp
 * @brief Common boundary condition types and structures.
 *
 * This header defines the fundamental boundary condition types used
 * throughout the biotransport library. All physics solvers should
 * include this header for consistent BC handling.
 */

namespace biotransport {

/**
 * @brief Identifies which side of a rectangular domain.
 */
enum class Boundary {
    Left = 0,
    Right = 1,
    Bottom = 2,
    Top = 3,
};

/**
 * @brief Convert Boundary enum to integer index.
 *
 * Convenience function to avoid verbose static_cast throughout codebase.
 */
constexpr int to_index(Boundary b) noexcept {
    return static_cast<int>(b);
}

/**
 * @brief Type of boundary condition.
 */
enum class BoundaryType {
    DIRICHLET,  ///< Fixed value (e.g., concentration, temperature)
    NEUMANN,    ///< Fixed flux (e.g., heat flux, mass flux)
    ROBIN       ///< Mixed/Robin: a*u + b*du/dn = c
};

/**
 * @brief Scalar boundary condition (type + value).
 *
 * Used for scalar fields like concentration, temperature, pressure.
 */
struct BoundaryCondition {
    BoundaryType type;
    double value;
    // Robin BC coefficients: a*u + b*du/dn = c
    double a = 0.0;  ///< Coefficient for u (only for Robin)
    double b = 0.0;  ///< Coefficient for du/dn (only for Robin)
    double c = 0.0;  ///< Right-hand side (only for Robin)

    /**
     * @brief Create a Dirichlet (fixed value) boundary condition.
     * @param value The fixed value at the boundary
     */
    static BoundaryCondition Dirichlet(double value) noexcept {
        return BoundaryCondition{BoundaryType::DIRICHLET, value, 0.0, 0.0, 0.0};
    }

    /**
     * @brief Create a Neumann (fixed flux) boundary condition.
     * @param flux The fixed flux at the boundary (positive = outward)
     */
    static BoundaryCondition Neumann(double flux) noexcept {
        return BoundaryCondition{BoundaryType::NEUMANN, flux, 0.0, 0.0, 0.0};
    }

    /**
     * @brief Create a Robin (mixed) boundary condition: a*u + b*du/dn = c
     * @param a Coefficient for u
     * @param b Coefficient for du/dn
     * @param c Right-hand side value
     */
    static BoundaryCondition Robin(double a, double b, double c) noexcept {
        return BoundaryCondition{BoundaryType::ROBIN, 0.0, a, b, c};
    }
};

/**
 * @brief Velocity boundary condition type (for fluid solvers).
 */
enum class VelocityBCType {
    DIRICHLET,  ///< Fixed velocity
    NEUMANN,    ///< Fixed traction (stress-free)
    NOSLIP,     ///< No-slip wall (u = v = 0)
    INFLOW,     ///< Prescribed inflow velocity
    OUTFLOW     ///< Zero normal stress outflow
};

/**
 * @brief Velocity boundary condition specification.
 *
 * Used for vector velocity fields in Stokes/Navier-Stokes solvers.
 */
struct VelocityBC {
    VelocityBCType type;
    double u_value;  ///< x-velocity value (for DIRICHLET, INFLOW)
    double v_value;  ///< y-velocity value (for DIRICHLET, INFLOW)

    static VelocityBC NoSlip() noexcept { return {VelocityBCType::NOSLIP, 0.0, 0.0}; }
    static VelocityBC Inflow(double u, double v = 0.0) noexcept {
        return {VelocityBCType::INFLOW, u, v};
    }
    static VelocityBC Outflow() noexcept { return {VelocityBCType::OUTFLOW, 0.0, 0.0}; }
    static VelocityBC Dirichlet(double u, double v) noexcept {
        return {VelocityBCType::DIRICHLET, u, v};
    }
    static VelocityBC StressFree() noexcept { return {VelocityBCType::NEUMANN, 0.0, 0.0}; }
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_BOUNDARY_HPP
