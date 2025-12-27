#include <algorithm>
#include <biotransport/physics/heat_transfer/bioheat_cryotherapy.hpp>
#include <cmath>
#include <stdexcept>

namespace biotransport {

BioheatCryotherapySolver::BioheatCryotherapySolver(
    const StructuredMesh& mesh, std::vector<std::uint8_t> probe_mask,
    std::vector<double> perfusion_map, std::vector<double> q_met_map, double rho_tissue,
    double rho_blood, double c_blood, double k_unfrozen, double k_frozen, double c_unfrozen,
    double c_frozen, double T_body, double T_probe, double T_freeze, double T_freeze_range,
    double L_fusion, double A, double E_a, double R_gas)
    : mesh_(mesh),
      nx_(mesh_.nx()),
      ny_(mesh_.ny()),
      stride_(nx_ + 1),
      probe_mask_(std::move(probe_mask)),
      perfusion_map_(std::move(perfusion_map)),
      q_met_map_(std::move(q_met_map)),
      rho_tissue_(rho_tissue),
      rho_blood_(rho_blood),
      c_blood_(c_blood),
      k_unfrozen_(k_unfrozen),
      k_frozen_(k_frozen),
      c_unfrozen_(c_unfrozen),
      c_frozen_(c_frozen),
      T_body_(T_body),
      T_probe_(T_probe),
      T_freeze_(T_freeze),
      T_freeze_range_(T_freeze_range),
      L_fusion_(L_fusion),
      A_(A),
      E_a_(E_a),
      R_gas_(R_gas) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("BioheatCryotherapySolver requires a 2D mesh");
    }

    const int n = mesh_.numNodes();
    if (static_cast<int>(probe_mask_.size()) != n) {
        throw std::invalid_argument("probe_mask size doesn't match mesh");
    }
    if (static_cast<int>(perfusion_map_.size()) != n) {
        throw std::invalid_argument("perfusion_map size doesn't match mesh");
    }
    if (static_cast<int>(q_met_map_.size()) != n) {
        throw std::invalid_argument("q_met_map size doesn't match mesh");
    }

    if (!(rho_tissue_ > 0.0) || !(rho_blood_ > 0.0) || !(c_blood_ > 0.0)) {
        throw std::invalid_argument("rho/c must be positive");
    }
    if (!(k_unfrozen_ > 0.0) || !(k_frozen_ > 0.0) || !(c_unfrozen_ > 0.0) || !(c_frozen_ > 0.0)) {
        throw std::invalid_argument("k/c must be positive");
    }
    if (!(T_freeze_range_ > 0.0) || !(R_gas_ > 0.0)) {
        throw std::invalid_argument("T_freeze_range and R_gas must be positive");
    }
}

BioheatSaved BioheatCryotherapySolver::simulate(double dt, int num_steps,
                                                const std::vector<double>& times_to_save_s) const {
    if (!(dt > 0.0) || num_steps <= 0) {
        throw std::invalid_argument("dt and num_steps must be positive");
    }

    std::vector<double> save_times = times_to_save_s;
    std::sort(save_times.begin(), save_times.end());

    const int n = mesh_.numNodes();

    std::vector<double> T(n, T_body_);
    std::vector<double> damage(n, 0.0);
    std::vector<double> T_new(n, T_body_);

    // Fix probe temperature
    for (int i = 0; i < n; ++i) {
        if (probe_mask_[i] != 0) {
            T[i] = T_probe_;
        }
    }

    BioheatSaved out;
    out.nx = nx_ + 1;
    out.ny = ny_ + 1;

    const std::size_t nodes_per_frame =
        static_cast<std::size_t>(out.nx) * static_cast<std::size_t>(out.ny);

    std::size_t next_save = 0;
    double time = 0.0;

    auto do_save = [&](double t_save) {
        out.times_s.push_back(t_save);
        out.temperature_K.resize(static_cast<std::size_t>(out.times_s.size()) * nodes_per_frame);
        out.damage.resize(static_cast<std::size_t>(out.times_s.size()) * nodes_per_frame);

        double* T_frame = out.temperature_K.data() +
                          (static_cast<std::size_t>(out.times_s.size() - 1) * nodes_per_frame);
        double* D_frame = out.damage.data() +
                          (static_cast<std::size_t>(out.times_s.size() - 1) * nodes_per_frame);

        std::copy(T.begin(), T.end(), T_frame);
        std::copy(damage.begin(), damage.end(), D_frame);
    };

    // Save at t=0 if requested (within half step)
    while (next_save < save_times.size() && time + 0.5 * dt >= save_times[next_save]) {
        do_save(save_times[next_save]);
        next_save += 1;
    }

    const double dx = mesh_.dx();
    const double dy = mesh_.dy();
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;

    const double sigma = (T_freeze_range_ / 2.0);
    const double pi = std::acos(-1.0);
    const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * pi);
    const double inv_sigma_norm = inv_sqrt_2pi / sigma;

    const double erf_scale = (T_freeze_range_ / std::sqrt(2.0));

    for (int step = 1; step <= num_steps; ++step) {
        // Interior updates
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 1; j < ny_; ++j) {
            for (int i = 1; i < nx_; ++i) {
                const std::size_t c = idx(i, j, stride_);

                if (probe_mask_[c] != 0) {
                    T_new[c] = T_probe_;
                    continue;
                }

                const std::size_t e = idx(i + 1, j, stride_);
                const std::size_t w = idx(i - 1, j, stride_);
                const std::size_t nidx = idx(i, j + 1, stride_);
                const std::size_t sidx = idx(i, j - 1, stride_);

                const double Tc = T[c];

                const double phase_fraction = 0.5 * (1.0 + std::erf((T_freeze_ - Tc) / erf_scale));

                const double k = k_unfrozen_ * (1.0 - phase_fraction) + k_frozen_ * phase_fraction;
                const double c_base =
                    c_unfrozen_ * (1.0 - phase_fraction) + c_frozen_ * phase_fraction;

                const double z = (Tc - T_freeze_) / sigma;
                const double latent_peak = std::exp(-0.5 * z * z);
                const double c_effective =
                    c_base + (L_fusion_ * rho_tissue_) * latent_peak * inv_sigma_norm;

                const double w_b_factor = 1.0 - phase_fraction;

                const double lap_T =
                    (T[e] - 2.0 * Tc + T[w]) / dx2 + (T[nidx] - 2.0 * Tc + T[sidx]) / dy2;

                const double diffusion = k * lap_T;
                const double perfusion =
                    rho_blood_ * c_blood_ * perfusion_map_[c] * w_b_factor * (T_body_ - Tc);
                const double metabolism = q_met_map_[c] * w_b_factor;

                const double dT_dt =
                    (diffusion + perfusion + metabolism) / (rho_tissue_ * c_effective);

                const double T_next = Tc + dt * dT_dt;
                T_new[c] = T_next;

                // Damage update (skip probe)
                const double rate = A_ * std::exp(-E_a_ / (R_gas_ * Tc));
                const double freezing_factor =
                    (Tc < T_freeze_) ? (10.0 * (1.0 - Tc / T_freeze_)) : 0.0;
                const double damage_inc = dt * rate * (1.0 + freezing_factor);
                damage[c] += damage_inc;
            }
        }

        // Dirichlet boundary at all edges
        for (int i = 0; i <= nx_; ++i) {
            T_new[idx(i, 0, stride_)] = T_body_;
            T_new[idx(i, ny_, stride_)] = T_body_;
        }
        for (int j = 0; j <= ny_; ++j) {
            T_new[idx(0, j, stride_)] = T_body_;
            T_new[idx(nx_, j, stride_)] = T_body_;
        }

        // Fix probe temperature on full mask
        for (int i = 0; i < n; ++i) {
            if (probe_mask_[i] != 0) {
                T_new[i] = T_probe_;
            }
        }

        T.swap(T_new);
        time = step * dt;

        while (next_save < save_times.size() && time + 0.5 * dt >= save_times[next_save]) {
            do_save(save_times[next_save]);
            next_save += 1;
        }
    }

    out.frames = static_cast<int>(out.times_s.size());
    return out;
}

}  // namespace biotransport
