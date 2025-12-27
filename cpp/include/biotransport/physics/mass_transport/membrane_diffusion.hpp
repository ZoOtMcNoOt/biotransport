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
 *   - j is the steady-state flux [mol/(m²·s)]
 *   - D is the membrane diffusion coefficient [m²/s]
 *   - Phi is the partition coefficient (dimensionless)
 *   - L is the membrane thickness [m]
 *   - C_left, C_right are boundary concentrations [mol/m³]
 *
 * Optional hindered diffusion for large solutes in pores:
 *   D_eff = D_0 * H(lambda)
 *   lambda = solute_radius / pore_radius
 *   H(lambda) = (1 - lambda)^2 * (1 - 2.104*lambda + 2.09*lambda^3 - 0.95*lambda^5)
 *   (Renkin equation for spherical solutes in cylindrical pores)
 *
 * Applications in biotransport:
 *   - Blood-brain barrier transport
 *   - Cell membrane permeation
 *   - Drug-polymer microsphere release
 *   - Dialysis membranes
 */

#ifndef BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_MEMBRANE_DIFFUSION_HPP
#define BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_MEMBRANE_DIFFUSION_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

namespace biotransport {

/**
 * @brief Result of steady-state membrane diffusion solve.
 */
struct MembraneDiffusionResult {
    std::vector<double> x;              ///< Position coordinates [m]
    std::vector<double> concentration;  ///< Concentration profile [mol/m³]
    double flux;                        ///< Steady-state flux [mol/(m²·s)]
    double permeability;                ///< Membrane permeability P = D*Phi/L [m/s]
    double effective_diffusivity;       ///< Effective diffusivity (with hindrance) [m²/s]
};

/**
 * @brief Compute Renkin hindrance factor for spherical solutes in cylindrical pores.
 *
 * @param lambda Ratio of solute radius to pore radius (0 < lambda < 1)
 * @return Hindrance factor H (0 < H <= 1)
 */
inline double renkin_hindrance(double lambda) {
    if (lambda <= 0.0)
        return 1.0;
    if (lambda >= 1.0)
        return 0.0;

    // Renkin equation (1954)
    // H = (1 - λ)² × (1 - 2.104λ + 2.09λ³ - 0.95λ⁵)
    double one_minus_lambda = 1.0 - lambda;
    double l2 = lambda * lambda;
    double l3 = l2 * lambda;
    double l5 = l3 * l2;

    return one_minus_lambda * one_minus_lambda * (1.0 - 2.104 * lambda + 2.09 * l3 - 0.95 * l5);
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
 *         .setLeftConcentration(1.0)          // 1 mol/m³ on left
 *         .setRightConcentration(0.0);        // 0 mol/m³ on right
 *
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
        if (L <= 0.0) {
            throw std::invalid_argument("Membrane thickness must be positive");
        }
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
        if (D <= 0.0) {
            throw std::invalid_argument("Diffusivity must be positive");
        }
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
        if (Phi <= 0.0) {
            throw std::invalid_argument("Partition coefficient must be positive");
        }
        Phi_ = Phi;
        return *this;
    }

    /**
     * @brief Set the concentration on the left (donor) side.
     *
     * @param C Concentration [mol/m³]
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setLeftConcentration(double C) {
        C_left_ = C;
        return *this;
    }

    /**
     * @brief Set the concentration on the right (receiver) side.
     *
     * @param C Concentration [mol/m³]
     * @return Reference to this solver for chaining
     */
    MembraneDiffusion1DSolver& setRightConcentration(double C) {
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
        if (solute_radius < 0.0 || pore_radius <= 0.0) {
            throw std::invalid_argument("Radii must be positive");
        }
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

        // Steady-state flux: j = D * (C_mem_left - C_mem_right) / L
        //                      = D * Phi * (C_left - C_right) / L
        double flux = D_eff * (C_mem_left - C_mem_right) / L_;

        // Permeability: P = D * Phi / L
        double permeability = D_eff * Phi_ / L_;

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

        return MembraneDiffusionResult{std::move(x), std::move(concentration), flux, permeability,
                                       D_eff};
    }

    /**
     * @brief Compute analytical flux for given parameters.
     *
     * Convenience method for quick calculations without full solve.
     *
     * @return Steady-state flux [mol/(m²·s)]
     */
    double computeFlux() const {
        double D_eff = D_;
        if (use_hindered_) {
            D_eff *= renkin_hindrance(lambda_);
        }
        return D_eff * Phi_ * (C_left_ - C_right_) / L_;
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
        return D_eff * Phi_ / L_;
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
    double C_left_ = 1.0;        ///< Left (donor) concentration [mol/m³]
    double C_right_ = 0.0;       ///< Right (receiver) concentration [mol/m³]
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
     * @param partition_coefficient Partition coefficient at layer entry
     * @return Reference to this solver for chaining
     */
    MultiLayerMembraneSolver& addLayer(double thickness, double diffusivity,
                                       double partition_coefficient = 1.0) {
        if (thickness <= 0.0) {
            throw std::invalid_argument("Layer thickness must be positive");
        }
        if (diffusivity <= 0.0) {
            throw std::invalid_argument("Diffusivity must be positive");
        }
        if (partition_coefficient <= 0.0) {
            throw std::invalid_argument("Partition coefficient must be positive");
        }

        layers_.push_back({thickness, diffusivity, partition_coefficient});
        return *this;
    }

    /**
     * @brief Set the concentration on the left (donor) side.
     */
    MultiLayerMembraneSolver& setLeftConcentration(double C) {
        C_left_ = C;
        return *this;
    }

    /**
     * @brief Set the concentration on the right (receiver) side.
     */
    MultiLayerMembraneSolver& setRightConcentration(double C) {
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
     * Uses resistance-in-series model:
     *   R_total = Σ (L_i / (D_i * Φ_i))
     *   j = (C_left - C_right) / R_total
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

        // Steady-state flux
        double flux = (C_left_ - C_right_) / R_total;

        // Overall permeability
        double permeability = 1.0 / R_total;

        // Generate concentration profile through all layers
        int nodes_per_layer = 21;
        int total_nodes =
            static_cast<int>(layers_.size()) * nodes_per_layer -
            (static_cast<int>(layers_.size()) - 1);  // Avoid duplicate interface nodes

        std::vector<double> x;
        std::vector<double> concentration;
        x.reserve(total_nodes);
        concentration.reserve(total_nodes);

        double x_offset = 0.0;
        double C_interface = C_left_;  // Concentration in solution at left

        for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
            const auto& layer = layers_[layer_idx];

            // Concentration at layer entry (with partition)
            double C_entry = layer.partition * C_interface;

            // Concentration at layer exit
            double C_exit = C_entry - flux * layer.thickness / layer.diffusivity;

            // Generate nodes for this layer
            int start_i = (layer_idx == 0) ? 0 : 1;  // Skip first node except for first layer
            for (int i = start_i; i < nodes_per_layer; ++i) {
                double frac = static_cast<double>(i) / (nodes_per_layer - 1);
                x.push_back(x_offset + frac * layer.thickness);
                concentration.push_back(C_entry + frac * (C_exit - C_entry));
            }

            x_offset += layer.thickness;

            // Update interface concentration for next layer
            // C in solution at interface = C_exit / partition_current * partition_next
            if (layer_idx + 1 < layers_.size()) {
                C_interface = C_exit / layer.partition;
            }
        }

        // Effective diffusivity (for single equivalent layer)
        double D_eff = L_total * permeability;

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
