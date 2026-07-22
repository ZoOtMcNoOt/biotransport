#include <algorithm>
#include <biotransport/core/mesh/nonuniform_mesh_1d.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace biotransport {

NonuniformMesh1D::NonuniformMesh1D(std::vector<double> nodes) : nodes_(std::move(nodes)) {
    if (nodes_.size() < 2) {
        throw std::invalid_argument("NonuniformMesh1D requires at least two nodes");
    }

    for (double coordinate : nodes_) {
        if (!std::isfinite(coordinate)) {
            throw std::invalid_argument("NonuniformMesh1D node coordinates must be finite");
        }
    }

    spacings_.reserve(nodes_.size() - 1);
    for (std::size_t face = 0; face + 1 < nodes_.size(); ++face) {
        const double width = nodes_[face + 1] - nodes_[face];
        if (!std::isfinite(width) || width <= 0.0) {
            throw std::invalid_argument(
                "NonuniformMesh1D nodes must be finite and strictly increasing");
        }
        spacings_.push_back(width);
    }

    minimum_spacing_ = *std::min_element(spacings_.begin(), spacings_.end());
    const double domain_length = nodes_.back() - nodes_.front();
    if (!std::isfinite(domain_length) || domain_length <= 0.0) {
        throw std::invalid_argument("NonuniformMesh1D domain length must be finite and positive");
    }

    control_volumes_.resize(nodes_.size());
    control_volumes_.front() = 0.5 * spacings_.front();
    for (std::size_t node = 1; node + 1 < nodes_.size(); ++node) {
        control_volumes_[node] = 0.5 * (spacings_[node - 1] + spacings_[node]);
    }
    control_volumes_.back() = 0.5 * spacings_.back();

    for (double volume : control_volumes_) {
        if (!std::isfinite(volume) || volume <= 0.0) {
            throw std::invalid_argument(
                "NonuniformMesh1D control-volume widths must be finite and positive");
        }
    }
}

double NonuniformMesh1D::x(std::size_t node) const {
    if (node >= nodes_.size()) {
        throw std::out_of_range("NonuniformMesh1D node index out of range");
    }
    return nodes_[node];
}

double NonuniformMesh1D::spacing(std::size_t face) const {
    if (face >= spacings_.size()) {
        throw std::out_of_range("NonuniformMesh1D face index out of range");
    }
    return spacings_[face];
}

double NonuniformMesh1D::faceCoordinate(std::size_t face) const {
    if (face >= spacings_.size()) {
        throw std::out_of_range("NonuniformMesh1D face index out of range");
    }
    return nodes_[face] + 0.5 * spacings_[face];
}

double NonuniformMesh1D::controlVolume(std::size_t node) const {
    if (node >= control_volumes_.size()) {
        throw std::out_of_range("NonuniformMesh1D node index out of range");
    }
    return control_volumes_[node];
}

}  // namespace biotransport
