#ifndef BIOTRANSPORT_CORE_MESH_NONUNIFORM_MESH_1D_HPP
#define BIOTRANSPORT_CORE_MESH_NONUNIFORM_MESH_1D_HPP

/**
 * @file nonuniform_mesh_1d.hpp
 * @brief Validated one-dimensional mesh with node-centred control volumes.
 */

#include <cstddef>
#include <vector>

namespace biotransport {

/**
 * @brief A fitted 1D mesh whose nodes need not be uniformly spaced.
 *
 * Nodes must be finite and strictly increasing. The control volume around an
 * interior node extends to the midpoints of its neighbouring faces; boundary
 * nodes own half of the adjacent interval. Consequently, the control-volume
 * widths are positive and sum to the physical domain length.
 */
class NonuniformMesh1D {
public:
    explicit NonuniformMesh1D(std::vector<double> nodes);

    std::size_t numNodes() const noexcept { return nodes_.size(); }
    std::size_t numCells() const noexcept { return spacings_.size(); }

    double x(std::size_t node) const;
    const std::vector<double>& nodes() const noexcept { return nodes_; }

    /** Distance between node @p face and node @p face + 1. */
    double spacing(std::size_t face) const;

    /** Coordinate halfway between node @p face and node @p face + 1. */
    double faceCoordinate(std::size_t face) const;

    /** Width of the node-centred control volume. */
    double controlVolume(std::size_t node) const;
    const std::vector<double>& controlVolumes() const noexcept { return control_volumes_; }

    double xmin() const noexcept { return nodes_.front(); }
    double xmax() const noexcept { return nodes_.back(); }
    double length() const noexcept { return nodes_.back() - nodes_.front(); }
    double minimumSpacing() const noexcept { return minimum_spacing_; }

private:
    std::vector<double> nodes_;
    std::vector<double> spacings_;
    std::vector<double> control_volumes_;
    double minimum_spacing_ = 0.0;
};

}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_MESH_NONUNIFORM_MESH_1D_HPP
