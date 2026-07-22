/**
 * @file bioheat_cryotherapy.hpp
 * @brief Pennes bioheat solver with a dimensionally consistent phase-change model.
 */

#ifndef BIOTRANSPORT_PHYSICS_HEAT_TRANSFER_BIOHEAT_CRYOTHERAPY_HPP
#define BIOTRANSPORT_PHYSICS_HEAT_TRANSFER_BIOHEAT_CRYOTHERAPY_HPP

#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cstdint>
#include <vector>

namespace biotransport {

/**
 * @brief Saved fields from a two-dimensional bioheat simulation.
 *
 * Field arrays use row-major layout `[frame][j][i]`. Temperatures are absolute
 * temperatures in kelvin. `damage` is the Arrhenius heat-injury integral. It is
 * not a cryogenic cell-death model and must not be interpreted as one.
 */
struct BioheatSaved {
    int nx = 0;      ///< Number of nodes in the x direction
    int ny = 0;      ///< Number of nodes in the y direction
    int frames = 0;  ///< Number of saved frames

    std::vector<double> times_s;                ///< Exact snapshot times [s]
    std::vector<double> temperature_K;          ///< Absolute temperature [K]
    std::vector<double> damage;                 ///< Arrhenius heat-injury integral [-]
    std::vector<double> frozen_fraction;        ///< Apparent frozen fraction [0, 1]
    std::vector<double> minimum_temperature_K;  ///< Spatial minimum for each frame [K]
    std::vector<double> maximum_temperature_K;  ///< Spatial maximum for each frame [K]
    double maximum_stable_dt_s = 0.0;           ///< Conservative explicit-Euler bound [s]
};

/**
 * @brief Explicit finite-volume-style solver for the Pennes bioheat equation.
 *
 * The solver advances
 *
 * \f[
 * \rho_t c_{app}(T) \frac{\partial T}{\partial t}
 * = \nabla\!\cdot(k(T)\nabla T)
 * + \rho_b c_b \omega_b f_l(T)(T_a-T)
 * + f_l(T)q_{met},
 * \f]
 *
 * where \f$f_l=1-f_s\f$ and the apparent mass-specific heat capacity is
 *
 * \f[
 * c_{app}(T)=f_l c_u+f_s c_f+L\left(-\frac{df_s}{dT}\right).
 * \f]
 *
 * `T_freeze_range_K` is the width of a two-standard-deviation Gaussian mushy
 * zone, so its Gaussian standard deviation is `T_freeze_range_K / 2`. All
 * temperatures supplied to this class are in kelvin. The initial, arterial,
 * and fixed outer-boundary temperatures all default to `T_body_K`; they can be
 * separated with the explicit setters below.
 *
 * The Arrhenius integral is retained as a heat-injury diagnostic only. The
 * class deliberately does not invent a low-temperature cell-death law.
 * Probe-mask nodes are embedded fixed-temperature nodes, not a conjugate
 * probe/tissue heat-transfer model: probe heat capacity, contact resistance,
 * and coolant dynamics are outside this model. The perfusion and metabolism
 * shutdown factors are likewise phenomenological extensions of Pennes' model
 * into the freezing range and require application-specific validation.
 */
class BioheatCryotherapySolver {
public:
    /**
     * @brief Construct a solver using SI units and absolute temperatures.
     *
     * `perfusion_map` is volumetric blood perfusion [m^3_blood/(m^3_tissue s)],
     * numerically equivalent to s^-1. `q_met_map` is a volumetric source [W/m^3].
     * The legacy `T_body_K` argument initializes the tissue and is also used for
     * arterial blood and the fixed outer boundary unless explicit setters are
     * called.
     */
    BioheatCryotherapySolver(const StructuredMesh& mesh, std::vector<std::uint8_t> probe_mask,
                             std::vector<double> perfusion_map, std::vector<double> q_met_map,
                             double rho_tissue, double rho_blood, double c_blood, double k_unfrozen,
                             double k_frozen, double c_unfrozen, double c_frozen, double T_body_K,
                             double T_probe_K, double T_freeze_K, double T_freeze_range_K,
                             double L_fusion, double A, double E_a, double R_gas);

    /** Set a spatially uniform initial tissue temperature [K]. */
    BioheatCryotherapySolver& setInitialTemperatureK(double temperature_K);

    /** Set a node-wise initial tissue-temperature field [K]. */
    BioheatCryotherapySolver& setInitialTemperatureFieldK(std::vector<double> temperature_K);

    /** Set the arterial temperature in the Pennes perfusion term [K]. */
    BioheatCryotherapySolver& setArterialTemperatureK(double temperature_K);

    /** Set the fixed Dirichlet temperature on all four outer boundaries [K]. */
    BioheatCryotherapySolver& setBoundaryTemperatureK(double temperature_K);

    /** Apparent frozen mass fraction at an absolute temperature [0, 1]. */
    [[nodiscard]] double frozenFraction(double temperature_K) const;

    /** Temperature-dependent conductivity [W/(m K)]. */
    [[nodiscard]] double thermalConductivity(double temperature_K) const;

    /** Apparent mass-specific heat capacity, including latent heat [J/(kg K)]. */
    [[nodiscard]] double effectiveSpecificHeat(double temperature_K) const;

    /** Arrhenius heat-injury rate [1/s]; this is not a cryoinjury rate. */
    [[nodiscard]] double arrheniusHeatInjuryRate(double temperature_K) const;

    /**
     * @brief Conservative explicit-Euler time-step bound [s].
     *
     * This sufficient bound uses the largest conductivity and perfusion and
     * the smallest sensible heat capacity. It is therefore conservative in the
     * mushy zone, where latent heat increases the apparent heat capacity.
     */
    [[nodiscard]] double maximumStableTimeStep() const;

    /**
     * @brief Simulate for exactly `dt * num_steps` seconds.
     *
     * `dt` is the maximum substep. The solver splits a substep when necessary
     * to land exactly on an off-grid requested save time. Save times must be
     * finite and lie in the closed simulation interval; duplicates are
     * coalesced. An unstable `dt` is rejected before advancing the solution.
     */
    [[nodiscard]] BioheatSaved simulate(double dt, int num_steps,
                                        const std::vector<double>& times_to_save_s) const;

private:
    [[nodiscard]] double frozenFractionUnchecked(double temperature_K) const noexcept;
    [[nodiscard]] double thermalConductivityUnchecked(double temperature_K) const noexcept;
    [[nodiscard]] double effectiveSpecificHeatUnchecked(double temperature_K) const noexcept;
    [[nodiscard]] double arrheniusHeatInjuryRateUnchecked(double temperature_K) const noexcept;

    StructuredMesh mesh_;
    int nx_;
    int ny_;
    int stride_;

    std::vector<std::uint8_t> probe_mask_;
    std::vector<double> perfusion_map_;
    std::vector<double> q_met_map_;
    std::vector<double> initial_temperature_K_;

    double rho_tissue_;
    double rho_blood_;
    double c_blood_;

    double k_unfrozen_;
    double k_frozen_;
    double c_unfrozen_;
    double c_frozen_;

    double T_arterial_;
    double T_boundary_;
    double T_probe_;

    double T_freeze_;
    double T_freeze_range_;
    double L_fusion_;

    double A_;
    double E_a_;
    double R_gas_;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_HEAT_TRANSFER_BIOHEAT_CRYOTHERAPY_HPP
