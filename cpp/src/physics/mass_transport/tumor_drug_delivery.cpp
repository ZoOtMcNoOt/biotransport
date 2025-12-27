#include <algorithm>
#include <biotransport/physics/mass_transport/tumor_drug_delivery.hpp>
#include <cmath>
#include <stdexcept>

namespace biotransport {

TumorDrugDeliverySolver::TumorDrugDeliverySolver(const StructuredMesh& mesh,
                                                 std::vector<std::uint8_t> tumor_mask,
                                                 std::vector<double> hydraulic_conductivity,
                                                 double p_boundary, double p_tumor)
    : mesh_(mesh),
      tumor_mask_(std::move(tumor_mask)),
      K_(std::move(hydraulic_conductivity)),
      p_boundary_(p_boundary),
      p_tumor_(p_tumor) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("TumorDrugDeliverySolver requires a 2D mesh");
    }

    nx_ = mesh_.nx();
    ny_ = mesh_.ny();
    stride_ = nx_ + 1;

    const int n = mesh_.numNodes();
    if (static_cast<int>(tumor_mask_.size()) != n) {
        throw std::invalid_argument("tumor_mask size doesn't match mesh");
    }
    if (static_cast<int>(K_.size()) != n) {
        throw std::invalid_argument("hydraulic_conductivity size doesn't match mesh");
    }
    for (double v : K_) {
        if (!(v > 0.0) || !std::isfinite(v)) {
            throw std::invalid_argument("hydraulic_conductivity must be positive and finite");
        }
    }
}

std::vector<double> TumorDrugDeliverySolver::solvePressureSOR(int max_iter, double tol,
                                                              double omega) const {
    if (max_iter <= 0) {
        throw std::invalid_argument("max_iter must be positive");
    }
    if (!(tol > 0.0)) {
        throw std::invalid_argument("tol must be positive");
    }
    if (!(omega > 0.0) || !(omega < 2.0)) {
        throw std::invalid_argument("omega should be in (0,2)");
    }

    std::vector<double> p(mesh_.numNodes(), p_boundary_);

    // Pin tumor nodes
    for (std::size_t t = 0; t < tumor_mask_.size(); ++t) {
        if (tumor_mask_[t] != 0) {
            p[t] = p_tumor_;
        }
    }

    const double dx = mesh_.dx();
    const double dy = mesh_.dy();
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_delta = 0.0;

// MSVC OpenMP doesn't support max reduction, use critical section
#ifdef _MSC_VER
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 1; j < ny_; ++j) {
            double local_max = 0.0;
            for (int i = 1; i < nx_; ++i) {
                const std::size_t c = idx(i, j, stride_);

                // Skip pinned tumor nodes
                if (tumor_mask_[c] != 0) {
                    continue;
                }

                const std::size_t e = idx(i + 1, j, stride_);
                const std::size_t w = idx(i - 1, j, stride_);
                const std::size_t n = idx(i, j + 1, stride_);
                const std::size_t s = idx(i, j - 1, stride_);

                const double Kc = K_[c];
                const double Ke = 0.5 * (Kc + K_[e]);
                const double Kw = 0.5 * (Kc + K_[w]);
                const double Kn = 0.5 * (Kc + K_[n]);
                const double Ks = 0.5 * (Kc + K_[s]);

                const double a_center = (Ke + Kw) / dx2 + (Kn + Ks) / dy2;

                // East/west/north/south pressures are already pinned on boundary/tumor when needed
                const double rhs = (Ke * p[e] + Kw * p[w]) / dx2 + (Kn * p[n] + Ks * p[s]) / dy2;
                const double p_gs = rhs / a_center;

                const double p_old = p[c];
                const double p_new = (1.0 - omega) * p_old + omega * p_gs;

                const double delta = std::fabs(p_new - p_old);
                local_max = std::max(local_max, delta);
                p[c] = p_new;
            }
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp critical
#endif
            max_delta = std::max(max_delta, local_max);
        }
#else
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static) reduction(max : max_delta)
#endif
        for (int j = 1; j < ny_; ++j) {
            for (int i = 1; i < nx_; ++i) {
                const std::size_t c = idx(i, j, stride_);

                // Skip pinned tumor nodes
                if (tumor_mask_[c] != 0) {
                    continue;
                }

                const std::size_t e = idx(i + 1, j, stride_);
                const std::size_t w = idx(i - 1, j, stride_);
                const std::size_t n = idx(i, j + 1, stride_);
                const std::size_t s = idx(i, j - 1, stride_);

                const double Kc = K_[c];
                const double Ke = 0.5 * (Kc + K_[e]);
                const double Kw = 0.5 * (Kc + K_[w]);
                const double Kn = 0.5 * (Kc + K_[n]);
                const double Ks = 0.5 * (Kc + K_[s]);

                const double a_center = (Ke + Kw) / dx2 + (Kn + Ks) / dy2;

                // East/west/north/south pressures are already pinned on boundary/tumor when needed
                const double rhs = (Ke * p[e] + Kw * p[w]) / dx2 + (Kn * p[n] + Ks * p[s]) / dy2;
                const double p_gs = rhs / a_center;

                const double p_old = p[c];
                const double p_new = (1.0 - omega) * p_old + omega * p_gs;

                const double delta = std::fabs(p_new - p_old);
                max_delta = std::max(max_delta, delta);
                p[c] = p_new;
            }
        }
#endif

        if (max_delta < tol) {
            break;
        }
    }

    // Enforce pinned values exactly
    for (int j = 0; j <= ny_; ++j) {
        p[idx(0, j, stride_)] = p_boundary_;
        p[idx(nx_, j, stride_)] = p_boundary_;
    }
    for (int i = 0; i <= nx_; ++i) {
        p[idx(i, 0, stride_)] = p_boundary_;
        p[idx(i, ny_, stride_)] = p_boundary_;
    }
    for (std::size_t t = 0; t < tumor_mask_.size(); ++t) {
        if (tumor_mask_[t] != 0) {
            p[t] = p_tumor_;
        }
    }

    return p;
}

TumorDrugDeliverySaved TumorDrugDeliverySolver::simulate(
    const std::vector<double>& pressure, const std::vector<double>& diffusivity,
    const std::vector<double>& permeability, const std::vector<double>& vessel_density,
    double k_binding, double k_uptake, double c_plasma, double dt, int num_steps,
    const std::vector<double>& times_to_save_s) const {
    const int n = mesh_.numNodes();
    if (static_cast<int>(pressure.size()) != n) {
        throw std::invalid_argument("pressure size doesn't match mesh");
    }
    if (static_cast<int>(diffusivity.size()) != n) {
        throw std::invalid_argument("diffusivity size doesn't match mesh");
    }
    if (static_cast<int>(permeability.size()) != n) {
        throw std::invalid_argument("permeability size doesn't match mesh");
    }
    if (static_cast<int>(vessel_density.size()) != n) {
        throw std::invalid_argument("vessel_density size doesn't match mesh");
    }
    if (!(dt > 0.0) || num_steps <= 0) {
        throw std::invalid_argument("dt and num_steps must be positive");
    }
    if (!(k_binding >= 0.0) || !(k_uptake >= 0.0)) {
        throw std::invalid_argument("k_binding/k_uptake must be non-negative");
    }

    // Normalize vessel density to get SA/V proxy (match Python)
    const double vd_max = *std::max_element(vessel_density.begin(), vessel_density.end());
    const double inv_vd_max = (vd_max > 0.0) ? (1.0 / vd_max) : 0.0;

    std::vector<double> SA_V(n);
    for (int i = 0; i < n; ++i) {
        SA_V[i] = vessel_density[i] * inv_vd_max;
    }

    // Compute pressure gradient (central difference on interior)
    std::vector<double> grad_x(n, 0.0);
    std::vector<double> grad_y(n, 0.0);

    const double dx = mesh_.dx();
    const double dy = mesh_.dy();
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 1; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const std::size_t c = idx(i, j, stride_);
            grad_x[c] =
                (pressure[idx(i + 1, j, stride_)] - pressure[idx(i - 1, j, stride_)]) * inv_2dx;
            grad_y[c] =
                (pressure[idx(i, j + 1, stride_)] - pressure[idx(i, j - 1, stride_)]) * inv_2dy;
        }
    }

    // Darcy velocity v = -K * grad(p)
    std::vector<double> vx(n, 0.0);
    std::vector<double> vy(n, 0.0);
    for (int i = 0; i < n; ++i) {
        vx[i] = -K_[i] * grad_x[i];
        vy[i] = -K_[i] * grad_y[i];
    }

    // Transport fields
    std::vector<double> C_free(n, 0.0);
    std::vector<double> C_bound(n, 0.0);
    std::vector<double> C_cell(n, 0.0);
    std::vector<double> C_free_new(n, 0.0);

    // Precompute save steps
    std::vector<double> save_times = times_to_save_s;
    std::sort(save_times.begin(), save_times.end());

    TumorDrugDeliverySaved saved;
    saved.nx = nx_ + 1;
    saved.ny = ny_ + 1;

    const std::size_t nodes_per_frame =
        static_cast<std::size_t>(saved.nx) * static_cast<std::size_t>(saved.ny);

    std::size_t next_save = 0;
    double time = 0.0;

    const double dx2 = dx * dx;
    const double dy2 = dy * dy;

    auto maybe_save = [&]() {
        if (next_save >= save_times.size()) {
            return;
        }

        while (next_save < save_times.size() && time + 0.5 * dt >= save_times[next_save]) {
            const double t_save = save_times[next_save];
            saved.times_s.push_back(t_save);

            const std::size_t frame_index = static_cast<std::size_t>(saved.times_s.size() - 1);

            saved.free.resize((static_cast<std::size_t>(saved.times_s.size())) * nodes_per_frame);
            saved.bound.resize((static_cast<std::size_t>(saved.times_s.size())) * nodes_per_frame);
            saved.cellular.resize((static_cast<std::size_t>(saved.times_s.size())) *
                                  nodes_per_frame);
            saved.total.resize((static_cast<std::size_t>(saved.times_s.size())) * nodes_per_frame);

            double* out_free = saved.free.data() + (frame_index * nodes_per_frame);
            double* out_bound = saved.bound.data() + (frame_index * nodes_per_frame);
            double* out_cell = saved.cellular.data() + (frame_index * nodes_per_frame);
            double* out_total = saved.total.data() + (frame_index * nodes_per_frame);

            for (std::size_t i = 0; i < nodes_per_frame; ++i) {
                out_free[i] = C_free[i];
                out_bound[i] = C_bound[i];
                out_cell[i] = C_cell[i];
                out_total[i] = C_free[i] + C_bound[i] + C_cell[i];
            }

            next_save += 1;
        }
    };

    for (int step = 0; step < num_steps; ++step) {
        // Interior update in 2D
#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 1; j < ny_; ++j) {
            for (int i = 1; i < nx_; ++i) {
                const std::size_t c = idx(i, j, stride_);

                const std::size_t e = idx(i + 1, j, stride_);
                const std::size_t w = idx(i - 1, j, stride_);
                const std::size_t nidx = idx(i, j + 1, stride_);
                const std::size_t sidx = idx(i, j - 1, stride_);

                const double Cc = C_free[c];

                const double diffusion =
                    diffusivity[c] * ((C_free[e] - 2.0 * Cc + C_free[w]) / dx2 +
                                      (C_free[nidx] - 2.0 * Cc + C_free[sidx]) / dy2);

                const double vxc = vx[c];
                const double vyc = vy[c];

                const double conv_x =
                    (vxc > 0.0) ? vxc * (Cc - C_free[w]) / dx : vxc * (C_free[e] - Cc) / dx;

                const double conv_y =
                    (vyc > 0.0) ? vyc * (Cc - C_free[sidx]) / dy : vyc * (C_free[nidx] - Cc) / dy;

                const double convection = conv_x + conv_y;

                const double sink = (k_binding + k_uptake) * Cc;
                const double source = permeability[c] * SA_V[c] * (c_plasma - Cc);

                C_free_new[c] = Cc + dt * (diffusion - convection - sink + source);
            }
        }

        // No-flux boundaries (copy neighbor like Python)
        for (int i = 0; i <= nx_; ++i) {
            C_free_new[idx(i, 0, stride_)] = C_free_new[idx(i, 1, stride_)];
            C_free_new[idx(i, ny_, stride_)] = C_free_new[idx(i, ny_ - 1, stride_)];
        }
        for (int j = 0; j <= ny_; ++j) {
            C_free_new[idx(0, j, stride_)] = C_free_new[idx(1, j, stride_)];
            C_free_new[idx(nx_, j, stride_)] = C_free_new[idx(nx_ - 1, j, stride_)];
        }

        // Update bound/cellular using old free (match Python)
        for (int i = 0; i < n; ++i) {
            C_bound[i] += dt * k_binding * C_free[i];
            C_cell[i] += dt * k_uptake * C_free[i];
        }

        C_free.swap(C_free_new);
        time += dt;

        maybe_save();

        // Reset C_free_new so boundaries don't carry stale values at corners
        // (only interior is written each step)
        // We do this cheaply by copying current into new.
        C_free_new = C_free;
    }

    saved.frames = static_cast<int>(saved.times_s.size());
    return saved;
}

}  // namespace biotransport
