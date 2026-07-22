#include <algorithm>
#include <biotransport/physics/mass_transport/gray_scott.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace biotransport {

GrayScottSolver::GrayScottSolver(const StructuredMesh& mesh, double Du, double Dv, double f,
                                 double k)
    : mesh_(mesh) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("GrayScottSolver requires a 2D mesh");
    }

    nx_ = mesh_.nx();
    ny_ = mesh_.ny();

    if (!std::isfinite(Du) || !std::isfinite(Dv) || Du < 0.0 || Dv < 0.0) {
        throw std::invalid_argument("Du and Dv must be finite and non-negative");
    }
    if (!std::isfinite(f) || !std::isfinite(k) || f < 0.0 || k < 0.0) {
        throw std::invalid_argument("f and k must be finite and non-negative");
    }
    const double float_max = static_cast<double>(std::numeric_limits<float>::max());
    if (Du > float_max || Dv > float_max || f > float_max || k > float_max) {
        throw std::invalid_argument("Gray-Scott parameters exceed float range");
    }

    auto checked_float_parameter = [](double value, const char* name) {
        const float converted = static_cast<float>(value);
        if (value > 0.0 && converted == 0.0f) {
            throw std::invalid_argument(std::string(name) +
                                        " is too small for the single-precision kernel");
        }
        return converted;
    };
    Du_ = checked_float_parameter(Du, "Du");
    Dv_ = checked_float_parameter(Dv, "Dv");
    f_ = checked_float_parameter(f, "f");
    k_ = checked_float_parameter(k, "k");
    const double inv_dx2 = 1.0 / (mesh_.dx() * mesh_.dx());
    const double inv_dy2 = 1.0 / (mesh_.dy() * mesh_.dy());
    if (!std::isfinite(inv_dx2) || !std::isfinite(inv_dy2) || inv_dx2 > float_max ||
        inv_dy2 > float_max) {
        throw std::invalid_argument("Mesh spacing is too small for the float Gray-Scott kernel");
    }
    inv_dx2_ = static_cast<float>(inv_dx2);
    inv_dy2_ = static_cast<float>(inv_dy2);
    if (inv_dx2_ == 0.0f || inv_dy2_ == 0.0f) {
        throw std::invalid_argument("Mesh spacing is too large for the float Gray-Scott kernel");
    }
}

GrayScottRunResult GrayScottSolver::simulate(const std::vector<float>& u0,
                                             const std::vector<float>& v0, int total_steps,
                                             double dt, int steps_between_frames,
                                             int check_interval, double stable_tol,
                                             int min_frames_before_early_stop) {
    if (total_steps <= 0) {
        throw std::invalid_argument("total_steps must be positive");
    }
    if (!std::isfinite(dt) || !(dt > 0.0) ||
        dt > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument("dt must be finite, positive, and representable as float");
    }
    if (steps_between_frames <= 0) {
        throw std::invalid_argument("steps_between_frames must be positive");
    }
    if (check_interval <= 0) {
        throw std::invalid_argument("check_interval must be positive");
    }
    if (!std::isfinite(stable_tol) || stable_tol < 0.0) {
        throw std::invalid_argument("stable_tol must be finite and non-negative");
    }
    if (min_frames_before_early_stop <= 0) {
        throw std::invalid_argument("min_frames_before_early_stop must be positive");
    }

    const std::size_t n = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    if (u0.size() != n || v0.size() != n) {
        throw std::invalid_argument("u0/v0 size must be mesh.nx()*mesh.ny()");
    }
    for (std::size_t p = 0; p < n; ++p) {
        if (!std::isfinite(u0[p]) || !std::isfinite(v0[p]) || u0[p] < 0.0f || v0[p] < 0.0f) {
            throw std::invalid_argument(
                "u0 and v0 must contain finite, non-negative dimensionless concentrations");
        }
    }

    std::vector<float> u = u0;
    std::vector<float> v = v0;
    std::vector<float> u_new(n);
    std::vector<float> v_new(n);
    std::vector<float> last_check_u = u;
    std::vector<float> last_check_v = v;

    auto push_frame = [&](GrayScottRunResult& out, int step) {
        out.frame_steps.push_back(step);
        out.u_frames.insert(out.u_frames.end(), u.begin(), u.end());
        out.v_frames.insert(out.v_frames.end(), v.begin(), v.end());
        out.frames = static_cast<int>(out.frame_steps.size());
    };

    GrayScottRunResult out;
    out.nx = nx_;
    out.ny = ny_;

    push_frame(out, 0);

    const float dtf = static_cast<float>(dt);
    if (!(dtf > 0.0f)) {
        throw std::invalid_argument(
            "dt is too small to represent in the single-precision Gray-Scott kernel");
    }

    for (int step = 1; step <= total_steps; ++step) {
        float max_v_squared = 0.0f;
        for (float value : v) {
            max_v_squared = std::max(max_v_squared, value * value);
        }
        const float inverse_spacing_sum = inv_dx2_ + inv_dy2_;
        const float u_loss_coefficient = 2.0f * Du_ * inverse_spacing_sum + max_v_squared + f_;
        const float v_loss_coefficient = 2.0f * Dv_ * inverse_spacing_sum + f_ + k_;
        const float maximum_loss_coefficient = std::max(u_loss_coefficient, v_loss_coefficient);
        const float step_limit = maximum_loss_coefficient == 0.0f
                                     ? std::numeric_limits<float>::infinity()
                                     : 1.0f / maximum_loss_coefficient;
        if (dtf > step_limit) {
            throw std::runtime_error(
                "dt exceeds the current Gray-Scott diffusion/reaction positivity limit; dt=" +
                std::to_string(dt) + ", limit=" + std::to_string(step_limit));
        }

#ifdef BIOTRANSPORT_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < ny_; ++j) {
            const int jn = wrap_index(j + 1, ny_);
            const int js = wrap_index(j - 1, ny_);

            for (int i = 0; i < nx_; ++i) {
                const int ie = wrap_index(i + 1, nx_);
                const int iw = wrap_index(i - 1, nx_);

                const std::size_t c = idx(i, j, nx_);
                const std::size_t e = idx(ie, j, nx_);
                const std::size_t w = idx(iw, j, nx_);
                const std::size_t nidx = idx(i, jn, nx_);
                const std::size_t sidx = idx(i, js, nx_);

                const float uc = u[c];
                const float vc = v[c];

                const float lap_u = (u[e] - 2.0f * uc + u[w]) * inv_dx2_ +
                                    (u[nidx] - 2.0f * uc + u[sidx]) * inv_dy2_;
                const float lap_v = (v[e] - 2.0f * vc + v[w]) * inv_dx2_ +
                                    (v[nidx] - 2.0f * vc + v[sidx]) * inv_dy2_;

                const float uvv = uc * vc * vc;

                u_new[c] = uc + dtf * (Du_ * lap_u - uvv + f_ * (1.0f - uc));
                v_new[c] = vc + dtf * (Dv_ * lap_v + uvv - (f_ + k_) * vc);
            }
        }

        for (std::size_t p = 0; p < n; ++p) {
            if (!std::isfinite(u_new[p]) || !std::isfinite(v_new[p])) {
                throw std::runtime_error("Gray-Scott step produced a non-finite concentration");
            }
            const float u_tolerance =
                64.0f * std::numeric_limits<float>::epsilon() * std::max(1.0f, std::fabs(u[p]));
            const float v_tolerance =
                64.0f * std::numeric_limits<float>::epsilon() * std::max(1.0f, std::fabs(v[p]));
            if (u_new[p] < -u_tolerance || v_new[p] < -v_tolerance) {
                throw std::runtime_error(
                    "Gray-Scott step produced a negative concentration; reduce dt");
            }
            if (u_new[p] < 0.0f) {
                u_new[p] = 0.0f;
            }
            if (v_new[p] < 0.0f) {
                v_new[p] = 0.0f;
            }
        }

        u.swap(u_new);
        v.swap(v_new);

        bool stable_this_check = false;
        if (step % check_interval == 0) {
            float max_diff = 0.0f;
            for (std::size_t p = 0; p < n; ++p) {
                max_diff = std::max(max_diff, std::fabs(u[p] - last_check_u[p]));
                max_diff = std::max(max_diff, std::fabs(v[p] - last_check_v[p]));
            }
            last_check_u = u;
            last_check_v = v;
            stable_this_check = stable_tol > 0.0 && max_diff < stable_tol;
        }

        if (step % steps_between_frames == 0 || step == total_steps) {
            push_frame(out, step);
        }
        if (stable_this_check && out.frames >= min_frames_before_early_stop) {
            if (out.frame_steps.back() != step) {
                push_frame(out, step);
            }
            out.steps_run = step;
            out.final_time = dt * static_cast<double>(step);
            return out;
        }
    }

    out.steps_run = total_steps;
    out.final_time = dt * static_cast<double>(total_steps);
    return out;
}

}  // namespace biotransport
