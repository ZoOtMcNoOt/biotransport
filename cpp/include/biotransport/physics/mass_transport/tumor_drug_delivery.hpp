/**
 * @file tumor_drug_delivery.hpp
 * @brief Coupled tumor drug delivery model with Darcy flow and transport.
 *
 * Solves a coupled system for drug delivery to solid tumors:
 *   1. Steady-state interstitial fluid pressure (IFP) via Darcy's law
 *   2. Transient drug transport with diffusion, convection, binding, and uptake
 *
 * Physics:
 *   - Darcy velocity: v = -K ∇p
 *   - Drug transport: ∂C/∂t = D∇²C - v·∇C - k_bind*C - k_uptake*C + source
 *   - Vascular source term based on vessel density and permeability
 *
 * Applications: Chemotherapy optimization, nanoparticle delivery, drug screening.
 *
 * @see TumorDrugDeliveryConfig for Python configuration dataclass.
 */

#ifndef BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_TUMOR_DRUG_DELIVERY_HPP
#define BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_TUMOR_DRUG_DELIVERY_HPP

#include <biotransport/core/mesh/indexing.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cstdint>
#include <vector>

namespace biotransport {

/**
 * @brief Output data structure from tumor drug delivery simulation.
 *
 * Contains time-series data of drug concentration fields in different
 * compartments. Arrays are packed in row-major order: [frame][j][i].
 */
struct TumorDrugDeliverySaved {
    int nx = 0;      ///< Number of cells in x direction
    int ny = 0;      ///< Number of cells in y direction
    int frames = 0;  ///< Number of saved time frames

    std::vector<double> times_s;  ///< Time stamps for each frame [s]

    /// Free (unbound) drug concentration, packed as [frame][j][i]
    std::vector<double> free;
    /// Bound drug concentration (reversible binding), packed as [frame][j][i]
    std::vector<double> bound;
    /// Intracellular drug concentration (uptake), packed as [frame][j][i]
    std::vector<double> cellular;
    /// Total drug concentration (free + bound + cellular), packed as [frame][j][i]
    std::vector<double> total;
};

/**
 * @brief Coupled tumor drug delivery solver with Darcy flow and transport.
 *
 * Solves a two-step coupled system:
 * 1. Steady-state pressure: ∇·(K ∇p) = 0 with Dirichlet BCs
 * 2. Transient drug transport: ∂C/∂t = D∇²C - v·∇C + R(C) + S
 *
 * The model accounts for:
 *   - Elevated interstitial fluid pressure (IFP) in tumor
 *   - Convective washout from tumor center
 *   - Spatially varying diffusivity and vessel density
 *   - Drug binding and cellular uptake kinetics
 */
class TumorDrugDeliverySolver {
public:
    /**
     * @brief Construct a tumor drug delivery solver.
     *
     * @param mesh                   2D structured mesh for the tissue domain
     * @param tumor_mask             Binary mask (1 = tumor, 0 = normal tissue)
     * @param hydraulic_conductivity Hydraulic conductivity at each node [m²/(Pa·s)]
     * @param p_boundary             Pressure at domain boundary [Pa]
     * @param p_tumor                Pressure in tumor core [Pa]
     */
    TumorDrugDeliverySolver(const StructuredMesh& mesh, std::vector<std::uint8_t> tumor_mask,
                            std::vector<double> hydraulic_conductivity, double p_boundary,
                            double p_tumor);

    /**
     * @brief Solve the steady-state pressure field using SOR iteration.
     *
     * @param max_iter Maximum number of iterations
     * @param tol      Convergence tolerance (L∞ norm of residual)
     * @param omega    SOR relaxation factor (1.0-2.0, typically 1.5-1.9)
     * @return Pressure field on nodes [Pa] (size = numNodes)
     */
    std::vector<double> solvePressureSOR(int max_iter, double tol, double omega) const;

    /**
     * @brief Run the drug transport simulation.
     *
     * @param pressure        Steady-state pressure field from solvePressureSOR()
     * @param diffusivity     Drug diffusion coefficient at each node [m²/s]
     * @param permeability    Vessel permeability at each node [m/s]
     * @param vessel_density  Microvascular density at each node [1/m²]
     * @param k_binding       Drug binding rate constant [1/s]
     * @param k_uptake        Cellular uptake rate constant [1/s]
     * @param c_plasma        Plasma drug concentration [mol/m³ or normalized]
     * @param dt              Time step size [s]
     * @param num_steps       Total number of time steps
     * @param times_to_save_s Times at which to save snapshots [s]
     * @return TumorDrugDeliverySaved Simulation results with concentration fields
     */
    [[nodiscard]] TumorDrugDeliverySaved simulate(const std::vector<double>& pressure,
                                                  const std::vector<double>& diffusivity,
                                                  const std::vector<double>& permeability,
                                                  const std::vector<double>& vessel_density,
                                                  double k_binding, double k_uptake,
                                                  double c_plasma, double dt, int num_steps,
                                                  const std::vector<double>& times_to_save_s) const;

private:
    const StructuredMesh& mesh_;
    int nx_;
    int ny_;
    int stride_;

    std::vector<std::uint8_t> tumor_mask_;
    std::vector<double> K_;
    double p_boundary_;
    double p_tumor_;

    // Use shared idx() from indexing.hpp
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_TUMOR_DRUG_DELIVERY_HPP
