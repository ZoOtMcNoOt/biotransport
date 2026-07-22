/**
 * @file membrane_diffusion.hpp
 * @brief Steady-state 1D membrane diffusion solver with partition coefficients.
 *
 * Solves steady-state diffusion across a membrane with partition coefficients
 * at interfaces:
 *
 *   j = -D * dC/dx  (Fick's first law)
 *
 * At steady state with constant flux:
 *   j = D * Phi * (C_left - C_right) / L
 *
 * Where:
 *   - j is the steady-state flux [amount/(m²·s)]
 *   - D is the membrane diffusion coefficient [m²/s]
 *   - Phi is the partition coefficient (dimensionless)
 *   - L is the membrane thickness [m]
 *   - C_left, C_right are boundary amount densities [amount/m³]
 *
 * The equations are linear in concentration, so "amount" may be mol, kg, or
 * another consistent amount unit. Inputs must be per cubic metre; the returned
 * flux uses the same amount unit per square metre per second.
 *
 * Optional hindered diffusion for large solutes in pores:
 *   D_eff = D_0 * H(lambda)
 *   lambda = solute_radius / pore_radius
 *   H(lambda) = (1 - lambda)^2
 *               * (1 - 2.104*lambda + 2.09*lambda^3 - 0.95*lambda^5)
 *   (Renkin equation for spherical solutes in cylindrical pores)
 *
 * Model scope: layers are homogeneous and partitioning is instantaneous and
 * ideal-dilute. External film resistances, porosity or tortuosity beyond the
 * supplied effective D and Phi, reactions, swelling, and active transport are
 * not modeled. At steady state, D changes flux but not the linear profile when
 * the two partitioned boundary concentrations are fixed.
 *
 * Applications in biotransport:
 *   - Blood-brain barrier transport
 *   - Cell membrane permeation
 *   - Drug-polymer microsphere release
 *   - Dialysis membranes
 */

#ifndef BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_MEMBRANE_DIFFUSION_HPP
#define BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_MEMBRANE_DIFFUSION_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {

namespace membrane_detail {

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

inline void requireFiniteResult(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(std::string(name) +
                                  " is non-finite; rescale or reduce the input parameters");
    }
}

}  // namespace membrane_detail

/**
 * @brief Result of steady-state membrane diffusion solve.
 */
struct MembraneDiffusionResult {
    std::vector<double> x;              ///< Position coordinates [m]
    std::vector<double> concentration;  ///< Concentration profile [amount/m³]
    double flux;                        ///< Steady-state flux [amount/(m²·s)]
    double permeability;                ///< Membrane permeability P = D*Phi/L [m/s]
    double effective_diffusivity;       ///< Equivalent external-gradient coefficient P*L [m²/s]
};

/**
 * @brief Compute Renkin hindrance factor for spherical solutes in cylindrical pores.
 *
 * @param lambda Ratio of solute radius to pore radius (lambda >= 0)
 * @return Hindrance factor H (0 <= H <= 1)
 */
inline double renkin_hindrance(double lambda) {
    membrane_detail::requireFinite(lambda, "Solute-to-pore radius ratio");
    if (lambda < 0.0) {
        throw std::invalid_argument("Solute-to-pore radius ratio must be non-negative");
    }
    if (lambda == 0.0)
        return 1.0;
    if (lambda >= 1.0)
        return 0.0;

    // Renkin equation (1954)
    // H = (1 - λ)² × (1 - 2.104λ + 2.09λ³ - 0.95λ⁵)
    double one_minus_lambda = 1.0 - lambda;
    double l2 = lambda * lambda;
    double l3 = l2 * lambda;
    double l5 = l3 * l2;

    const double hindrance =
        one_minus_lambda * one_minus_lambda * (1.0 - 2.104 * lambda + 2.09 * l3 - 0.95 * l5);
    return std::clamp(hindrance, 0.0, 1.0);
}

/**
 * @brief Steady-state 1D membrane diffusion solver.
 *
 * Solves for steady-state concentration profile and flux across a membrane
 * with partition coefficients at interfaces.
 *
 * Example usage:
 * @code
 *   MembraneDiffusion1DSolver solver;
 *   solver.setMembraneThickness(100e-6)      // 100 µm membrane
 *         .setDiffusivity(1e-10)             // 10⁻¹⁰ m²/s in membrane
 *         .setPartitionCoefficient(0.1)       // Φ = 0.1
 *         .setLeftConcentration(1.0)          // 1 amount/m³ on left
 *         .setRightConcentration(0.0);         // 0 amount/m³ on right
 *   auto result = solver.solve();
 *   // result.flux is steady-state flux
 *   // result.concentration is profile inside membrane
 * @endcode
 */
class MembraneDiffusion1DSolver {
public:
    /**
     * @brief Default constructor.
     */
    MembraneDiffusion1DSolver() = default;

    /**
     * @brief Set the membrane thickness.
     *
     * @param L Membrane thickness [m]
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setMembraneThickness(double L) {
        membrane_detail::requirePositive(L, "Membrane thickness");
        L_ = L;
        return *this;
    }

    /**
     * @brief Set the diffusion coefficient in the membrane.
     *
     * @param D Diffusion coefficient [m²/s]
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setDiffusivity(double D) {
        membrane_detail::requirePositive(D, "Diffusivity");
        D_ = D;
        return *this;
    }

    /**
     * @brief Set the partition coefficient at membrane interfaces.
     *
     * The partition coefficient Φ = C_membrane / C_solution represents
     * the equilibrium distribution of solute between membrane and solution.
     * For hydrophobic membranes with hydrophilic solutes, Φ < 1.
     * For lipophilic membranes with lipophilic solutes, Φ > 1.
     *
     * @param Phi Partition coefficient (dimensionless)
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setPartitionCoefficient(double Phi) {
        membrane_detail::requirePositive(Phi, "Partition coefficient");
        Phi_ = Phi;
        return *this;
    }

    /**
     * @brief Set the concentration on the left (donor) side.
     *
     * @param C Concentration [amount/m³]
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setLeftConcentration(double C) {
        membrane_detail::requireNonnegative(C, "Left concentration");
        C_left_ = C;
        return *this;
    }

    /**
     * @brief Set the concentration on the right (receiver) side.
     *
     * @param C Concentration [amount/m³]
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setRightConcentration(double C) {
        membrane_detail::requireNonnegative(C, "Right concentration");
        C_right_ = C;
        return *this;
    }

    /**
     * @brief Enable hindered diffusion using Renkin equation.
     *
     * For large solutes in porous membranes, diffusion is hindered by
     * steric and hydrodynamic effects when the solute radius is a
     * significant fraction of the pore radius.
     *
     * @param solute_radius Hydrodynamic radius of solute [m]
     * @param pore_radius Effective pore radius of membrane [m]
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setHinderedDiffusion(double solute_radius, double pore_radius) {
        membrane_detail::requireNonnegative(solute_radius, "Solute radius");
        membrane_detail::requirePositive(pore_radius, "Pore radius");
        if (solute_radius >= pore_radius) {
            throw std::invalid_argument("Solute radius must be less than pore radius");
        }
        use_hindered_ = true;
        lambda_ = solute_radius / pore_radius;
        return *this;
    }

    /**
     * @brief Disable hindered diffusion (use bulk diffusivity).
     *
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& disableHinderedDiffusion() {
        use_hindered_ = false;
        lambda_ = 0.0;
        return *this;
    }

    /**
     * @brief Set the number of nodes for the concentration profile output.
     *
     * @param n Number of nodes (minimum 2)
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setNumNodes(int n) {
        if (n < 2) {
            throw std::invalid_argument("Number of nodes must be at least 2");
        }
        num_nodes_ = n;
        return *this;
    }

    /**
     * @brief Solve for steady-state concentration profile and flux.
     *
     * @return Result containing position, concentration, flux, and permeability
     */
    [[nodiscard]] MembraneDiffusionResult solve() const {
        // Compute effective diffusivity
        double D_eff = D_;
        if (use_hindered_) {
            D_eff *= renkin_hindrance(lambda_);
        }

        // Concentration at membrane boundaries (with partition)
        double C_mem_left = Phi_ * C_left_;
        double C_mem_right = Phi_ * C_right_;
        membrane_detail::requireFiniteResult(C_mem_left, "Left membrane concentration");
        membrane_detail::requireFiniteResult(C_mem_right, "Right membrane concentration");

        // Steady-state flux: j = D * (C_mem_left - C_mem_right) / L
        //                      = D * Phi * (C_left - C_right) / L
        double flux = D_eff * (C_mem_left - C_mem_right) / L_;

        // Permeability: P = D * Phi / L
        double permeability = D_eff * Phi_ / L_;
        membrane_detail::requireFiniteResult(flux, "Membrane flux");
        membrane_detail::requireFiniteResult(permeability, "Membrane permeability");

        // Generate concentration profile (linear at steady state)
        std::vector<double> x(num_nodes_);
        std::vector<double> concentration(num_nodes_);

        double dx = L_ / (num_nodes_ - 1);
        for (int i = 0; i < num_nodes_; ++i) {
            x[i] = i * dx;
            // Linear interpolation inside membrane
            double frac = static_cast<double>(i) / (num_nodes_ - 1);
            concentration[i] = C_mem_left + frac * (C_mem_right - C_mem_left);
        }

        // P*L is the apparent transport coefficient referenced to the external
        // concentration difference. It includes both hindrance and partition.
        const double apparent_diffusivity = permeability * L_;
        return MembraneDiffusionResult{std::move(x), std::move(concentration), flux, permeability,
                                       apparent_diffusivity};
    }

    /**
     * @brief Compute analytical flux for given parameters.
     *
     * Convenience method for quick calculations without full solve.
     *
     * @return Steady-state flux [amount/(m²·s)]
     */
    double computeFlux() const {
        double D_eff = D_;
        if (use_hindered_) {
            D_eff *= renkin_hindrance(lambda_);
        }
        const double flux = D_eff * Phi_ * (C_left_ - C_right_) / L_;
        membrane_detail::requireFiniteResult(flux, "Membrane flux");
        return flux;
    }

    /**
     * @brief Compute membrane permeability.
     *
     * Permeability P = D * Phi / L relates flux to concentration difference:
     *   j = P * (C_left - C_right)
     *
     * @return Permeability [m/s]
     */
    double computePermeability() const {
        double D_eff = D_;
        if (use_hindered_) {
            D_eff *= renkin_hindrance(lambda_);
        }
        const double permeability = D_eff * Phi_ / L_;
        membrane_detail::requireFiniteResult(permeability, "Membrane permeability");
        return permeability;
    }

    // Getters for current settings
    double membraneThickness() const { return L_; }
    double diffusivity() const { return D_; }
    double partitionCoefficient() const { return Phi_; }
    double leftConcentration() const { return C_left_; }
    double rightConcentration() const { return C_right_; }
    bool isHinderedDiffusion() const { return use_hindered_; }
    double lambda() const { return lambda_; }

private:
    double L_ = 100e-6;          ///< Membrane thickness [m] (default 100 µm)
    double D_ = 1e-10;           ///< Diffusion coefficient [m²/s]
    double Phi_ = 1.0;           ///< Partition coefficient (dimensionless)
    double C_left_ = 1.0;        ///< Left (donor) concentration [amount/m³]
    double C_right_ = 0.0;       ///< Right (receiver) concentration [amount/m³]
    bool use_hindered_ = false;  ///< Whether to use hindered diffusion
    double lambda_ = 0.0;        ///< Solute/pore radius ratio
    int num_nodes_ = 101;        ///< Number of output nodes
};

/**
 * @brief Multi-layer membrane solver for composite membranes.
 *
 * Solves steady-state diffusion through a membrane composed of multiple
 * layers with different properties. Useful for modeling:
 *   - Skin with stratum corneum + epidermis + dermis
 *   - Coated drug delivery systems
 *   - Composite separation membranes
 *
 * At steady state, flux is constant through all layers, and total
 * resistance is the sum of individual layer resistances.
 */
class MultiLayerMembraneSolver {
public:
    /**
     * @brief Add a membrane layer.
     *
     * Layers are added from left to right (donor to receiver side).
     *
     * @param thickness Layer thickness [m]
     * @param diffusivity Diffusion coefficient in layer [m²/s]
     * @param partition_coefficient Equilibrium concentration ratio K_i between
     *        this layer and a common reference solution phase
     * @return Reference to this solver for chaining
     */
    MultiLayerMembraneSolver& addLayer(double thickness, double diffusivity,
                                       double partition_coefficient = 1.0) {
        membrane_detail::requirePositive(thickness, "Layer thickness");
        membrane_detail::requirePositive(diffusivity, "Layer diffusivity");
        membrane_detail::requirePositive(partition_coefficient, "Layer partition coefficient");

        layers_.push_back({thickness, diffusivity, partition_coefficient});
        return *this;
    }

    /**
     * @brief Set the concentration on the left (donor) side.
     */
    MultiLayerMembraneSolver& setLeftConcentration(double C) {
        membrane_detail::requireNonnegative(C, "Left concentration");
        C_left_ = C;
        return *this;
    }

    /**
     * @brief Set the concentration on the right (receiver) side.
     */
    MultiLayerMembraneSolver& setRightConcentration(double C) {
        membrane_detail::requireNonnegative(C, "Right concentration");
        C_right_ = C;
        return *this;
    }

    /**
     * @brief Clear all layers.
     */
    MultiLayerMembraneSolver& clearLayers() {
        layers_.clear();
        return *this;
    }

    /**
     * @brief Solve for steady-state flux through composite membrane.
     *
     * Uses a local-equilibrium resistance-in-series model. If q is a
     * reference-phase concentration (equivalently an ideal-dilute activity
     * coordinate), layer i has C_i = K_i*q and therefore:
     *   R_total = Σ (L_i / (D_i * Φ_i))
     *   j = (C_left - C_right) / R_total
     *
     * Adjacent layers can have different K_i, so their membrane-phase
     * concentrations jump at an interface while q remains continuous. Both
     * one-sided interface values are returned at the same x coordinate.
     *
     * @return Result with combined flux and total permeability
     */
    [[nodiscard]] MembraneDiffusionResult solve() const {
        if (layers_.empty()) {
            throw std::runtime_error("No layers added to membrane");
        }

        // Compute total resistance
        double R_total = 0.0;
        double L_total = 0.0;
        for (const auto& layer : layers_) {
            R_total += layer.thickness / (layer.diffusivity * layer.partition);
            L_total += layer.thickness;
        }
        membrane_detail::requireFiniteResult(R_total, "Total membrane resistance");
        membrane_detail::requireFiniteResult(L_total, "Total membrane thickness");
        if (R_total <= 0.0 || L_total <= 0.0) {
            throw std::overflow_error(
                "Membrane resistance or thickness underflowed; rescale the input parameters");
        }

        // Steady-state flux
        double flux = (C_left_ - C_right_) / R_total;

        // Overall permeability
        double permeability = 1.0 / R_total;
        membrane_detail::requireFiniteResult(flux, "Composite membrane flux");
        membrane_detail::requireFiniteResult(permeability, "Composite membrane permeability");

        // Generate concentration profile through all layers
        constexpr std::size_t nodes_per_layer = 21;
        const std::size_t total_nodes = layers_.size() * nodes_per_layer;

        std::vector<double> x;
        std::vector<double> concentration;
        x.reserve(total_nodes);
        concentration.reserve(total_nodes);

        double x_offset = 0.0;
        double q_interface = C_left_;  // Common reference-phase concentration

        for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
            const auto& layer = layers_[layer_idx];

            const double C_entry = layer.partition * q_interface;
            const double q_exit =
                q_interface - flux * layer.thickness / (layer.diffusivity * layer.partition);
            const double C_exit = layer.partition * q_exit;

            // Generate nodes for this layer
            for (std::size_t i = 0; i < nodes_per_layer; ++i) {
                double frac = static_cast<double>(i) / static_cast<double>(nodes_per_layer - 1);
                x.push_back(x_offset + frac * layer.thickness);
                concentration.push_back(C_entry + frac * (C_exit - C_entry));
            }

            x_offset += layer.thickness;

            q_interface = q_exit;
        }

        // Effective diffusivity (for single equivalent layer)
        const double D_eff = L_total * permeability;
        membrane_detail::requireFiniteResult(D_eff, "Equivalent membrane diffusivity");

        return MembraneDiffusionResult{std::move(x), std::move(concentration), flux, permeability,
                                       D_eff};
    }

    /**
     * @brief Get total membrane thickness.
     */
    double totalThickness() const {
        double L = 0.0;
        for (const auto& layer : layers_) {
            L += layer.thickness;
        }
        return L;
    }

    /**
     * @brief Get number of layers.
     */
    size_t numLayers() const { return layers_.size(); }

private:
    struct Layer {
        double thickness;
        double diffusivity;
        double partition;
    };

    std::vector<Layer> layers_;
    double C_left_ = 1.0;
    double C_right_ = 0.0;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_MEMBRANE_DIFFUSION_HPP
