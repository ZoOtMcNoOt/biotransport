#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <stdexcept>

namespace biotransport {

StructuredMesh3D::StructuredMesh3D(int nx, int ny, int nz, double xmin, double xmax, double ymin,
                                   double ymax, double zmin, double zmax)
    : nx_(nx),
      ny_(ny),
      nz_(nz),
      xmin_(xmin),
      xmax_(xmax),
      ymin_(ymin),
      ymax_(ymax),
      zmin_(zmin),
      zmax_(zmax) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Number of cells must be positive in all directions");
    }
    if (xmax <= xmin || ymax <= ymin || zmax <= zmin) {
        throw std::invalid_argument("Domain bounds are invalid (max must be > min)");
    }

    dx_ = (xmax - xmin) / nx;
    dy_ = (ymax - ymin) / ny;
    dz_ = (zmax - zmin) / nz;

    stride_j_ = nx_ + 1;
    stride_k_ = (nx_ + 1) * (ny_ + 1);
    num_nodes_ = (nx_ + 1) * (ny_ + 1) * (nz_ + 1);
}

StructuredMesh3D::StructuredMesh3D(int n, double length)
    : StructuredMesh3D(n, n, n, 0.0, length, 0.0, length, 0.0, length) {}

double StructuredMesh3D::x(int i) const {
    if (i < 0 || i > nx_) {
        throw std::out_of_range("x index out of range");
    }
    return xmin_ + i * dx_;
}

double StructuredMesh3D::y(int j) const {
    if (j < 0 || j > ny_) {
        throw std::out_of_range("y index out of range");
    }
    return ymin_ + j * dy_;
}

double StructuredMesh3D::z(int k) const {
    if (k < 0 || k > nz_) {
        throw std::out_of_range("z index out of range");
    }
    return zmin_ + k * dz_;
}

int StructuredMesh3D::index(int i, int j, int k) const {
    if (i < 0 || i > nx_ || j < 0 || j > ny_ || k < 0 || k > nz_) {
        throw std::out_of_range("Node index out of range");
    }
    return k * stride_k_ + j * stride_j_ + i;
}

std::array<int, 3> StructuredMesh3D::ijk(int idx) const {
    if (idx < 0 || idx >= num_nodes_) {
        throw std::out_of_range("Linear index out of range");
    }
    int k = idx / stride_k_;
    int remainder = idx % stride_k_;
    int j = remainder / stride_j_;
    int i = remainder % stride_j_;
    return {i, j, k};
}

bool StructuredMesh3D::isOnBoundary(int i, int j, int k, Boundary3D boundary) const {
    switch (boundary) {
        case Boundary3D::XMin:
            return i == 0;
        case Boundary3D::XMax:
            return i == nx_;
        case Boundary3D::YMin:
            return j == 0;
        case Boundary3D::YMax:
            return j == ny_;
        case Boundary3D::ZMin:
            return k == 0;
        case Boundary3D::ZMax:
            return k == nz_;
        default:
            return false;
    }
}

bool StructuredMesh3D::isOnAnyBoundary(int i, int j, int k) const {
    return i == 0 || i == nx_ || j == 0 || j == ny_ || k == 0 || k == nz_;
}

}  // namespace biotransport
