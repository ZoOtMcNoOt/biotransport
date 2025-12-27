/**
 * @file bioheat_cryotherapy.hpp
 * @brief Bioheat transfer solver with phase change for cryotherapy simulation.
 *
 * Solves the Pennes bioheat equation with:
 *   - Conduction with temperature-dependent thermal properties
 *   - Blood perfusion heat exchange
 *   - Metabolic heat generation
 *   - Phase change (freezing) via effective heat capacity method
 *   - Arrhenius thermal damage accumulation
 *
 * Applications: Cryosurgery, tumor ablation, tissue preservation.
 *
 * @see BioheatCryotherapyConfig for Python configuration dataclass.
 */

#ifndef BIOTRANSPORT_PHYSICS_HEAT_TRANSFER_BIOHEAT_CRYOTHERAPY_HPP
#define BIOTRANSPORT_PHYSICS_HEAT_TRANSFER_BIOHEAT_CRYOTHERAPY_HPP

#include <biotransport/core/mesh/indexing.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cstdint>
#include <vector>

namespace biotransport {

/**
 * @brief Output data structure from bioheat cryotherapy simulation.
 *
 * Contains time-series data of temperature and thermal damage fields.
 * Arrays are packed in row-major order: [frame][j][i].
 */
struct BioheatSaved {
    int nx = 0;      ///< Number of cells in x direction
    int ny = 0;      ///< Number of cells in y direction
    int frames = 0;  ///< Number of saved time frames

    std::vector<double> times_s;  ///< Time stamps for each frame [s]

    /// Temperature field at each frame [K], packed as [frame][j][i]
    std::vector<double> temperature_K;
    /// Arrhenius damage integral Ω at each frame [-], packed as [frame][j][i]
    std::vector<double> damage;
};

/**
 * @brief Solver for Pennes bioheat equation with phase change and damage.
 *
 * Solves the transient bioheat equation:
 *   ρc ∂T/∂t = ∇·(k∇T) + ρ_b c_b w_b (T_a - T) + q_met
 *
 * With modifications for cryotherapy:
 *   - Temperature-dependent thermal conductivity k(T)
 *   - Temperature-dependent specific heat c(T) with latent heat
 *   - Perfusion shutdown in frozen regions
 *   - Arrhenius damage accumulation: dΩ/dt = A·exp(-E_a/RT)
 */
class BioheatCryotherapySolver {
public:
    /**
     * @brief Construct a bioheat cryotherapy solver.
     *
     * @param mesh           2D structured mesh for the tissue domain
     * @param probe_mask     Binary mask (1 = probe location, 0 = tissue)
     * @param perfusion_map  Blood perfusion rate at each node [1/s]
     * @param q_met_map      Metabolic heat generation at each node [W/m³]
     * @param rho_tissue     Tissue density [kg/m³]
     * @param rho_blood      Blood density [kg/m³]
     * @param c_blood        Blood specific heat [J/(kg·K)]
     * @param k_unfrozen     Thermal conductivity of unfrozen tissue [W/(m·K)]
     * @param k_frozen       Thermal conductivity of frozen tissue [W/(m·K)]
     * @param c_unfrozen     Specific heat of unfrozen tissue [J/(kg·K)]
     * @param c_frozen       Specific heat of frozen tissue [J/(kg·K)]
     * @param T_body         Body/arterial temperature [K]
     * @param T_probe        Cryoprobe temperature [K]
     * @param T_freeze       Freezing point temperature [K]
     * @param T_freeze_range Temperature range for phase transition [K]
     * @param L_fusion       Latent heat of fusion [J/kg]
     * @param A              Arrhenius frequency factor [1/s]
     * @param E_a            Arrhenius activation energy [J/mol]
     * @param R_gas          Universal gas constant [J/(mol·K)]
     */
    BioheatCryotherapySolver(const StructuredMesh& mesh, std::vector<std::uint8_t> probe_mask,
                             std::vector<double> perfusion_map, std::vector<double> q_met_map,
                             double rho_tissue, double rho_blood, double c_blood, double k_unfrozen,
                             double k_frozen, double c_unfrozen, double c_frozen, double T_body,
                             double T_probe, double T_freeze, double T_freeze_range,
                             double L_fusion, double A, double E_a, double R_gas);

    /**
     * @brief Run the bioheat simulation.
     *
     * @param dt              Time step size [s]
     * @param num_steps       Total number of time steps to run
     * @param times_to_save_s Times at which to save snapshots [s]
     * @return BioheatSaved   Simulation results with temperature and damage fields
     */
    [[nodiscard]] BioheatSaved simulate(double dt, int num_steps,
                                        const std::vector<double>& times_to_save_s) const;

private:
    const StructuredMesh& mesh_;
    int nx_;
    int ny_;
    int stride_;

    std::vector<std::uint8_t> probe_mask_;
    std::vector<double> perfusion_map_;
    std::vector<double> q_met_map_;

    double rho_tissue_;
    double rho_blood_;
    double c_blood_;

    double k_unfrozen_;
    double k_frozen_;
    double c_unfrozen_;
    double c_frozen_;

    double T_body_;
    double T_probe_;

    double T_freeze_;
    double T_freeze_range_;
    double L_fusion_;

    double A_;
    double E_a_;
    double R_gas_;

    // Use shared idx() from indexing.hpp
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_HEAT_TRANSFER_BIOHEAT_CRYOTHERAPY_HPP
