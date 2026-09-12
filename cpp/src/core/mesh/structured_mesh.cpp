#include <biotransport/core/mesh/structured_mesh.hpp>
#include <stdexcept>

namespace biotransport {

StructuredMesh::StructuredMesh(int nx, double xmin, double xmax, Geometry geometry)
    : nx_(nx),
      ny_(1),
      xmin_(xmin),
      xmax_(xmax),
      ymin_(0.0),
      ymax_(0.0),
      is_1d_(true),
      geometry_(geometry) {
    if (nx <= 0) {
        throw std::invalid_argument("Number of cells must be positive");
    }
    if (xmax <= xmin) {
        throw std::invalid_argument("xmax must be greater than xmin");
    }
    if (geometry != Geometry::CARTESIAN && xmin < 0.0) {
        throw std::invalid_argument(
            "a cylindrical or spherical mesh needs a non-negative inner radius");
    }

    dx_ = (xmax - xmin) / nx;
    dy_ = 0.0;
}

StructuredMesh::StructuredMesh(int nx, int ny, double xmin, double xmax, double ymin, double ymax,
                               Geometry geometry)
    : nx_(nx),
      ny_(ny),
      xmin_(xmin),
      xmax_(xmax),
      ymin_(ymin),
      ymax_(ymax),
      is_1d_(false),
      geometry_(geometry) {
    if (nx <= 0 || ny <= 0) {
        throw std::invalid_argument("Number of cells must be positive");
    }
    if (xmax <= xmin || ymax <= ymin) {
        throw std::invalid_argument("Domain bounds are invalid");
    }
    if (geometry == Geometry::SPHERICAL) {
        throw std::invalid_argument(
            "spherical geometry is one-dimensional; a 2D spherical mesh would be "
            "(r, theta), which is a different operator and is not implemented. Use "
            "CYLINDRICAL for an axisymmetric (r, z) mesh.");
    }
    if (geometry != Geometry::CARTESIAN && xmin < 0.0) {
        throw std::invalid_argument(
            "an axisymmetric mesh needs a non-negative inner radius");
    }

    dx_ = (xmax - xmin) / nx;
    dy_ = (ymax - ymin) / ny;
}

int StructuredMesh::numNodes() const {
    return is_1d_ ? (nx_ + 1) : (nx_ + 1) * (ny_ + 1);
}

int StructuredMesh::numCells() const {
    return is_1d_ ? nx_ : nx_ * ny_;
}

double StructuredMesh::x(int i) const {
    if (i < 0 || i > nx_) {
        throw std::out_of_range("Node index out of range");
    }
    return xmin_ + i * dx_;
}

double StructuredMesh::y(int i, int j) const {
    if (is_1d_) {
        return 0.0;
    }

    if (i < 0 || i > nx_ || j < 0 || j > ny_) {
        throw std::out_of_range("Node index out of range");
    }

    return ymin_ + j * dy_;
}

double StructuredMesh::controlVolume(int i) const {
    if (i < 0 || i > nx_) {
        throw std::out_of_range("Node index out of range");
    }
    const double lower = (i == 0) ? xmin_ : xmin_ + (i - 0.5) * dx_;
    const double upper = (i == nx_) ? xmax_ : xmin_ + (i + 0.5) * dx_;
    return measureBetween(lower, upper);
}

double StructuredMesh::lowerFaceArea(int i) const {
    if (i < 0 || i > nx_) {
        throw std::out_of_range("Node index out of range");
    }
    return areaFactor((i == 0) ? xmin_ : xmin_ + (i - 0.5) * dx_);
}

double StructuredMesh::upperFaceArea(int i) const {
    if (i < 0 || i > nx_) {
        throw std::out_of_range("Node index out of range");
    }
    return areaFactor((i == nx_) ? xmax_ : xmin_ + (i + 0.5) * dx_);
}

double StructuredMesh::axialHeight(int j) const {
    if (is_1d_) {
        return 1.0;
    }
    if (j < 0 || j > ny_) {
        throw std::out_of_range("Node index out of range");
    }
    return (j == 0 || j == ny_) ? 0.5 * dy_ : dy_;
}

int StructuredMesh::index(int i, int j) const {
    if (is_1d_) {
        if (i < 0 || i > nx_) {
            throw std::out_of_range("Node index out of range");
        }
        return i;
    } else {
        if (i < 0 || i > nx_ || j < 0 || j > ny_) {
            throw std::out_of_range("Node index out of range");
        }
        return j * (nx_ + 1) + i;
    }
}

}  // namespace biotransport
