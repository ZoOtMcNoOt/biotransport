/**
 * @file tumor_drug_delivery.hpp
 * @brief Prescribed-pressure Darcy transport model for tumor drug delivery.
 *
 * The implementation deliberately separates what is solved from what must be
 * supplied.  Interstitial pressure is a prescribed-pressure surrogate: the
 * outer boundary is held at @c p_boundary and nodes selected by
 * @c tumor_mask are held at @c p_tumor.  All remaining nodes solve
 *
 *     div(K grad(p)) = 0,              v = -K grad(p).
 *
 * The mask therefore identifies pressure-clamped nodes, not merely a tissue
 * label.  This class does not solve a Starling fluid-source or lymphatic-drainage
 * model; measured or externally calculated pressures must be supplied when
 * those mechanisms matter.  A pressure clamp implies an unresolved fluid
 * source at the clamp interface.  Conservative transport treats that implied
 * fluid source as solute-free; solvent-drag delivery from vascular filtration
 * is not represented by the permeability source below.
 *
 * Free drug is advanced conservatively on node-centred dual control volumes:
 *
 *   dC_f/dt = -div(v C_f - D grad(C_f))
 *              + P S_v (C_plasma - C_f) - (k_b + k_u) C_f,
 *   dC_b/dt = k_b C_f,
 *   dC_c/dt = k_u C_f.
 *
 * Here P is vessel-wall permeability [m/s] and S_v is perfused vascular
 * surface area per tissue volume [1/m], so P*S_v is an exchange rate [1/s].
 * The outer boundary has zero diffusive flux and permits Darcy outflow.  Inflow
 * is rejected because the API has no external inflow concentration.
 */

#ifndef BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_TUMOR_DRUG_DELIVERY_HPP
#define BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_TUMOR_DRUG_DELIVERY_HPP

#include <biotransport/core/mesh/indexing.hpp>
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cstdint>
#include <vector>

namespace biotransport {

/**
 * @brief Saved concentration fields and integral mass-balance diagnostics.
 *
 * Concentration arrays are packed in row-major order as [frame][j][i].  The
 * amount diagnostics integrate the node-centred dual control volumes.  For a
 * 2D model their units are concentration times area, i.e. amount per unit
 * out-of-plane depth when concentration is volumetric.
 */
struct TumorDrugDeliverySaved {
    int nx = 0;      ///< Number of nodes in x direction
    int ny = 0;      ///< Number of nodes in y direction
    int frames = 0;  ///< Number of saved time frames

    std::vector<double> times_s;  ///< Exact time stamp for each frame [s]

    std::vector<double> free;      ///< Free extracellular concentration
    std::vector<double> bound;     ///< Irreversibly tissue-sequestered concentration
    std::vector<double> cellular;  ///< Irreversibly internalized concentration
    std::vector<double> total;     ///< free + bound + cellular

    /// Integral amounts at each saved time [concentration*m^2].
    std::vector<double> free_amount_per_depth;
    std::vector<double> bound_amount_per_depth;
    std::vector<double> cellular_amount_per_depth;
    std::vector<double> total_amount_per_depth;

    /// Time-integrated net vascular transfer into tissue [concentration*m^2].
    std::vector<double> cumulative_net_vascular_exchange_per_depth;
    /// Time-integrated free-drug loss by Darcy outflow [concentration*m^2].
    std::vector<double> cumulative_boundary_outflow_per_depth;
    /// total - initial - vascular_exchange + boundary_outflow.
    std::vector<double> mass_balance_error_per_depth;

    double final_time_s = 0.0;       ///< Requested simulation end time [s]
    double stability_limit_s = 0.0;  ///< Monotonic explicit-Euler limit [s]
};

/**
 * @brief Prescribed-pressure Darcy flow coupled to conservative drug transport.
 *
 * Important limitations:
 * - pressure-clamped tumor nodes are an empirical surrogate, not a prediction
 *   from Starling filtration or lymphatic drainage;
 * - the fluid source implied by a pressure clamp carries no advected drug;
 * - binding and uptake are irreversible first-order compartments;
 * - vascular exchange is linear and requires S_v [1/m], not a normalized
 *   vessel-count map;
 * - plasma concentration is constant in time and binding saturation,
 *   metabolism, and systemic pharmacokinetics are not represented.
 */
class TumorDrugDeliverySolver {
public:
    /**
     * @param mesh                    2D structured tissue mesh
     * @param tumor_mask              Binary pressure-clamp mask (1 means p_tumor)
     * @param hydraulic_conductivity Darcy mobility K [m^2/(Pa*s)] at every node
     * @param p_boundary              Outer-boundary pressure [Pa]
     * @param p_tumor                 Pressure on masked nodes [Pa]; must be >= p_boundary
     */
    TumorDrugDeliverySolver(const StructuredMesh& mesh, std::vector<std::uint8_t> tumor_mask,
                            std::vector<double> hydraulic_conductivity, double p_boundary,
                            double p_tumor);

    /**
     * @brief Solve div(K grad(p)) = 0 outside the fixed-pressure nodes by SOR.
     *
     * Harmonic face mobilities preserve normal Darcy flux at material
     * interfaces.  Failure to reduce the post-sweep discrete pressure defect
     * below the requested tolerance is reported with an exception instead of
     * returning an unconverged field.  Unlike an update-size test, this
     * criterion cannot be fooled by choosing an extremely small relaxation
     * factor.
     *
     * @param max_iter Maximum number of SOR sweeps
     * @param tol      Absolute maximum discrete pressure-defect tolerance [Pa]
     * @param omega    SOR relaxation factor in (0,2)
     */
    [[nodiscard]] std::vector<double> solvePressureSOR(int max_iter, double tol,
                                                       double omega) const;

    /**
     * @brief Advance conservative drug transport from an initially drug-free tissue.
     *
     * @param pressure        Pressure field consistent with this solver [Pa]
     * @param diffusivity     Effective free-drug diffusivity D [m^2/s], non-negative
     * @param vessel_wall_solute_permeability Vessel-wall solute permeability P [m/s],
     *                                        non-negative
     * @param vascular_surface_area_density Perfused vessel surface area per tissue volume
     *                                      S_v [1/m]
     * @param k_binding       Irreversible tissue-sequestration rate [1/s]
     * @param k_uptake        Irreversible cellular-uptake rate [1/s]
     * @param c_plasma        Constant plasma concentration, in the same units as C
     * @param dt              Maximum explicit time step [s]
     * @param num_steps       Defines final time as dt*num_steps; may be zero
     * @param times_to_save_s Requested exact snapshot times in [0, final_time]
     *
     * Unsorted save times are accepted and duplicates are collapsed.  A step is
     * shortened to land exactly on every requested time.  Inputs that would
     * violate the monotonic explicit-Euler bound are rejected.  Non-negative
     * data therefore remain non-negative; values below zero by roundoff only
     * are reset to zero and any material negativity is an error.
     */
    [[nodiscard]] TumorDrugDeliverySaved simulate(
        const std::vector<double>& pressure, const std::vector<double>& diffusivity,
        const std::vector<double>& vessel_wall_solute_permeability,
        const std::vector<double>& vascular_surface_area_density, double k_binding, double k_uptake,
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
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_PHYSICS_MASS_TRANSPORT_TUMOR_DRUG_DELIVERY_HPP
