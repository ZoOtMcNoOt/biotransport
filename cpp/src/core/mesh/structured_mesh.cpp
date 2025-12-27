#include <biotransport/core/mesh/structured_mesh.hpp>
#include <stdexcept>

namespace biotransport {

StructuredMesh::StructuredMesh(int nx, double xmin, double xmax)
    : nx_(nx), ny_(1), xmin_(xmin), xmax_(xmax), ymin_(0.0), ymax_(0.0), is_1d_(true) {
    if (nx <= 0) {
        throw std::invalid_argument("Number of cells must be positive");
    }
    if (xmax <= xmin) {
        throw std::invalid_argument("xmax must be greater than xmin");
    }

    dx_ = (xmax - xmin) / nx;
    dy_ = 0.0;
}

StructuredMesh::StructuredMesh(int nx, int ny, double xmin, double xmax, double ymin, double ymax)
    : nx_(nx), ny_(ny), xmin_(xmin), xmax_(xmax), ymin_(ymin), ymax_(ymax), is_1d_(false) {
    if (nx <= 0 || ny <= 0) {
        throw std::invalid_argument("Number of cells must be positive");
    }
    if (xmax <= xmin || ymax <= ymin) {
        throw std::invalid_argument("Domain bounds are invalid");
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
