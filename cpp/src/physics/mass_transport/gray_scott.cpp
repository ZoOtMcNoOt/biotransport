#include <algorithm>
#include <biotransport/physics/mass_transport/gray_scott.hpp>
#include <cmath>
#include <stdexcept>

namespace biotransport {

GrayScottSolver::GrayScottSolver(const StructuredMesh& mesh, double Du, double Dv, double f,
                                 double k)
    : mesh_(mesh) {
    if (mesh_.is1D()) {
        throw std::invalid_argument("GrayScottSolver requires a 2D mesh");
    }

    nx_ = mesh_.nx() + 1;
    ny_ = mesh_.ny() + 1;

    if (!(Du > 0.0) || !(Dv > 0.0)) {
        throw std::invalid_argument("Du and Dv must be positive");
    }
    if (!(f >= 0.0) || !(k >= 0.0)) {
        throw std::invalid_argument("f and k must be non-negative");
    }

    Du_ = static_cast<float>(Du);
    Dv_ = static_cast<float>(Dv);
    f_ = static_cast<float>(f);
    k_ = static_cast<float>(k);
}

GrayScottRunResult GrayScottSolver::simulate(const std::vector<float>& u0,
                                             const std::vector<float>& v0, int total_steps,
                                             double dt, int steps_between_frames,
                                             int check_interval, double stable_tol,
                                             int min_frames_before_early_stop) {
    if (total_steps <= 0) {
        throw std::invalid_argument("total_steps must be positive");
    }
    if (!(dt > 0.0)) {
        throw std::invalid_argument("dt must be positive");
    }
    if (steps_between_frames <= 0) {
        throw std::invalid_argument("steps_between_frames must be positive");
    }
    if (check_interval <= 0) {
        throw std::invalid_argument("check_interval must be positive");
    }

    const std::size_t n = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    if (u0.size() != n || v0.size() != n) {
        throw std::invalid_argument("u0/v0 size must be (nx+1)*(ny+1)");
    }

    std::vector<float> u = u0;
    std::vector<float> v = v0;
    std::vector<float> u_new(n);
    std::vector<float> v_new(n);
    std::vector<float> last_check = v;

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
    const float stable_tolf = static_cast<float>(stable_tol);

    bool stable = false;

    for (int step = 1; step <= total_steps; ++step) {
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

                const float lap_u = -4.0f * uc + u[e] + u[w] + u[nidx] + u[sidx];
                const float lap_v = -4.0f * vc + v[e] + v[w] + v[nidx] + v[sidx];

                const float uvv = uc * vc * vc;

                float un = uc + dtf * (Du_ * lap_u - uvv + f_ * (1.0f - uc));
                float vn = vc + dtf * (Dv_ * lap_v + uvv - (f_ + k_) * vc);

                // Conservative clipping like the Python example
                un = std::min(1.0f, std::max(0.0f, un));
                vn = std::min(2.0f, std::max(0.0f, vn));

                u_new[c] = un;
                v_new[c] = vn;
            }
        }

        u.swap(u_new);
        v.swap(v_new);

        if (step % check_interval == 0) {
            float max_diff = 0.0f;
            for (std::size_t p = 0; p < n; ++p) {
                const float d = std::fabs(v[p] - last_check[p]);
                max_diff = std::max(max_diff, d);
            }
            last_check = v;
            if (max_diff < stable_tolf) {
                stable = true;
            }
        }

        if (step % steps_between_frames == 0 || step == total_steps ||
            (stable && step % (2 * steps_between_frames) == 0)) {
            push_frame(out, step);
            if (stable && out.frames >= min_frames_before_early_stop) {
                out.steps_run = step;
                return out;
            }
        }
    }

    out.steps_run = total_steps;
    return out;
}

}  // namespace biotransport
