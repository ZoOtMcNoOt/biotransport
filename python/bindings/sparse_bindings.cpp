/**
 * @file sparse_bindings.cpp
 * @brief Python bindings for sparse matrix and implicit solvers.
 */

#include "sparse_bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding_helpers.hpp"
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <biotransport/core/mesh/structured_mesh_3d.hpp>
#include <biotransport/core/numerics/linear_algebra/sparse_matrix.hpp>
#include <biotransport/solvers/implicit_diffusion.hpp>

namespace py = pybind11;

namespace biotransport {
namespace bindings {

void register_sparse_bindings(py::module_& m) {
#ifdef BIOTRANSPORT_ENABLE_EIGEN
    m.def(
        "sparse_matrix_available", []() { return true; },
        "Return whether this build includes the Eigen-backed sparse and implicit solvers.");

    // SparseSolverType enum
    py::enum_<linalg::SparseSolverType>(m, "SparseSolverType", "Sparse linear solver types")
        .value("SparseLU", linalg::SparseSolverType::SparseLU,
               "Direct LU decomposition (general matrices)")
        .value("SimplicialLLT", linalg::SparseSolverType::SimplicialLLT,
               "Cholesky LLT (SPD matrices, fastest)")
        .value("SimplicialLDLT", linalg::SparseSolverType::SimplicialLDLT,
               "Cholesky LDLT (symmetric matrices)")
        .value("ConjugateGradient", linalg::SparseSolverType::ConjugateGradient,
               "Iterative CG (SPD matrices, memory efficient)")
        .value("BiCGSTAB", linalg::SparseSolverType::BiCGSTAB,
               "Iterative BiCGSTAB (general matrices)")
        .export_values();

    // SparseSolveResult
    py::class_<linalg::SparseSolveResult>(m, "SparseSolveResult", "Result of a sparse linear solve")
        .def(py::init<>())
        .def_readonly("success", &linalg::SparseSolveResult::success)
        .def_readonly("iterations", &linalg::SparseSolveResult::iterations)
        .def_readonly("residual", &linalg::SparseSolveResult::residual)
        .def_readonly("error_message", &linalg::SparseSolveResult::error_message);

    // Triplet
    py::class_<linalg::Triplet>(m, "Triplet", "Sparse matrix triplet (row, col, value)")
        .def(py::init<int, int, double>(), py::arg("row"), py::arg("col"), py::arg("value"))
        .def_readonly("row", &linalg::Triplet::row)
        .def_readonly("col", &linalg::Triplet::col)
        .def_readonly("value", &linalg::Triplet::value);

    // SparseMatrix
    py::class_<linalg::SparseMatrix>(m, "SparseMatrix",
                                     R"(Sparse matrix class for efficient linear algebra.

This class wraps Eigen's sparse matrix functionality for use in
PDE solvers and other numerical applications.

Example:
    >>> A = bt.SparseMatrix(100, 100)
    >>> A.reserve(500)
    >>> for i in range(100):
    ...     A.add_entry(i, i, 2.0)
    ...     if i > 0: A.add_entry(i, i-1, -1.0)
    ...     if i < 99: A.add_entry(i, i+1, -1.0)
    >>> A.finalize()
    >>> x = A.solve(b, bt.SparseSolverType.SparseLU)
)")
        .def(py::init<>(), "Create empty sparse matrix")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"),
             "Create sparse matrix of given dimensions")
        .def("reserve", &linalg::SparseMatrix::reserve, py::arg("nnz_estimate"),
             "Reserve space for estimated non-zeros")
        .def("add_entry", &linalg::SparseMatrix::addEntry, py::arg("row"), py::arg("col"),
             py::arg("value"), "Add entry (duplicates are summed)")
        .def("finalize", &linalg::SparseMatrix::finalize,
             "Finalize matrix (must call before solving)")
        .def("is_finalized", &linalg::SparseMatrix::isFinalized, "Check if matrix is finalized")
        .def_property_readonly("rows", &linalg::SparseMatrix::rows)
        .def_property_readonly("cols", &linalg::SparseMatrix::cols)
        .def_property_readonly("nnz", &linalg::SparseMatrix::nonZeros, "Number of non-zeros")
        .def(
            "solve",
            [](const linalg::SparseMatrix& A, const std::vector<double>& b,
               linalg::SparseSolverType solver_type, double tol,
               int max_iter) { return A.solve(b, solver_type, tol, max_iter); },
            py::arg("b"), py::arg("solver_type") = linalg::SparseSolverType::SparseLU,
            py::arg("tolerance") = 1e-10, py::arg("max_iterations") = 1000,
            "Solve Ax = b, returns solution vector")
        .def(
            "multiply",
            [](const linalg::SparseMatrix& A, const std::vector<double>& x) {
                return A.multiply(x);
            },
            py::arg("x"), "Compute y = A * x")
        .def("clear", &linalg::SparseMatrix::clear, "Clear matrix for reuse")
        .def("resize", &linalg::SparseMatrix::resize, py::arg("rows"), py::arg("cols"),
             "Resize matrix (clears data)");

    // Helper functions for building common matrices
    m.def("build_2d_laplacian", &linalg::build2DLaplacian, py::arg("nx"), py::arg("ny"),
          py::arg("dx"), py::arg("dy"),
          R"(Build 2D Laplacian matrix for structured mesh.

Creates discretization matrix for -∇²u = f with Dirichlet BCs.

Args:
    nx: Number of cells in x
    ny: Number of cells in y
    dx: Grid spacing in x
    dy: Grid spacing in y

Returns:
    SparseMatrix of size (nx+1)*(ny+1)
)");

    m.def("build_implicit_diffusion_2d", &linalg::buildImplicitDiffusion2D, py::arg("nx"),
          py::arg("ny"), py::arg("dx"), py::arg("dy"), py::arg("alpha"), py::arg("dt"),
          R"(Build 2D implicit diffusion matrix.

Creates (I - α*dt*∇²) for Backward Euler time integration.

Args:
    nx, ny: Number of cells
    dx, dy: Grid spacing
    alpha: Diffusion coefficient
    dt: Time step

Returns:
    Implicit diffusion matrix
)");

    m.def("build_implicit_diffusion_3d", &linalg::buildImplicitDiffusion3D, py::arg("nx"),
          py::arg("ny"), py::arg("nz"), py::arg("dx"), py::arg("dy"), py::arg("dz"),
          py::arg("alpha"), py::arg("dt"), "Build 3D implicit diffusion matrix");

    // ImplicitSolveResult
    py::class_<ImplicitSolveResult>(
        m, "ImplicitSolveResult",
        "Metadata from a completed Backward Euler diffusion solve. Failures raise exceptions.")
        .def(py::init<>())
        .def_readonly("steps", &ImplicitSolveResult::steps, "Number of completed time steps.")
        .def_readonly("total_time", &ImplicitSolveResult::total_time,
                      "Simulation time after the solve [s].")
        .def_readonly("residual", &ImplicitSolveResult::residual,
                      "Scaled infinity norm of the final algebraic residual A*u-b.")
        .def_readonly("success", &ImplicitSolveResult::success,
                      "True for a returned result; failed solves raise an exception.");

    // ImplicitDiffusion2D
    py::class_<ImplicitDiffusion2D>(
        m, "ImplicitDiffusion2D",
        R"(Conservative 2D Backward Euler solver for diffusion with an optional source.

The solved equation is

    du/dt = div(D grad(u)) + source(x, y, t).

Coordinates are measured in metres, time in seconds, and diffusivity D in m²/s.
D may be a positive scalar or one positive value per mesh node. Harmonic face
averaging preserves diffusive flux across material interfaces. Backward Euler is
L-stable for the linear diffusion operator.

Neumann values are outward-normal derivatives du/dn in field-units/m. They are
not Fickian fluxes; the corresponding outward Fickian flux is -D*du/dn.

Example:
    >>> mesh = bt.StructuredMesh(50, 50, 0.0, 1.0, 0.0, 1.0)
    >>> solver = bt.ImplicitDiffusion2D(mesh, diffusivity=0.01)
    >>> solver.set_initial_condition(ic)
    >>> solver.set_neumann_boundary(bt.Boundary.Left, 0.0)
    >>> solver.set_solver_type(bt.SparseSolverType.SparseLU)
    >>> result = solver.solve(dt=0.1, num_steps=10)
)")
        .def(py::init<const StructuredMesh&, double>(), py::arg("mesh"), py::arg("diffusivity"),
             py::keep_alive<1, 2>(),
             "Create a solver with uniform diffusivity [m²/s]. The mesh must be 2D.")
        .def(py::init<const StructuredMesh&, const std::vector<double>&>(), py::arg("mesh"),
             py::arg("diffusivity"), py::keep_alive<1, 2>(),
             "Create a solver with positive nodal diffusivities [m²/s].")
        .def("set_initial_condition", &ImplicitDiffusion2D::setInitialCondition, py::arg("values"),
             "Copy one finite initial field value per mesh node.")
        .def("set_dirichlet_boundary", &ImplicitDiffusion2D::setDirichletBoundary,
             py::arg("boundary"), py::arg("value"), "Set a fixed field value on one boundary.")
        .def("set_neumann_boundary", &ImplicitDiffusion2D::setNeumannBoundary, py::arg("boundary"),
             py::arg("normal_derivative"),
             "Set the outward-normal derivative du/dn [field-units/m], not a flux.")
        .def("set_source_term", &ImplicitDiffusion2D::setSourceTerm, py::arg("source"),
             "Set source(x_m, y_m, time_s), returning field-units/s.")
        .def("clear_source_term", &ImplicitDiffusion2D::clearSourceTerm,
             "Remove the configured source term.")
        .def("set_solver_type", &ImplicitDiffusion2D::setSolverType, py::arg("solver_type"),
             "Select the Eigen sparse linear solver.")
        .def("set_tolerance", &ImplicitDiffusion2D::setTolerance, py::arg("tolerance"),
             "Set the positive iterative-solver tolerance.")
        .def("set_max_iterations", &ImplicitDiffusion2D::setMaxIterations,
             py::arg("max_iterations"), "Set the positive iterative-solver iteration limit.")
        .def("step", &ImplicitDiffusion2D::step, py::arg("dt"),
             "Advance one positive time step dt [s].")
        .def("solve", &ImplicitDiffusion2D::solve, py::arg("dt"), py::arg("num_steps"),
             "Advance num_steps equal time steps of dt seconds.")
        .def(
            "solution",
            [](const ImplicitDiffusion2D& solver) { return copy_to_numpy(solver.solution()); },
            "Return an owned copy of the current flat row-major field.")
        .def(
            "diffusivity",
            [](const ImplicitDiffusion2D& solver) { return copy_to_numpy(solver.diffusivity()); },
            "Return an owned copy of the nodal diffusivity field [m²/s].")
        .def("time", &ImplicitDiffusion2D::time, "Return current simulation time [s].")
        .def("mesh", &ImplicitDiffusion2D::mesh, py::return_value_policy::reference_internal,
             "Return the solver's 2D mesh.");

    // ImplicitDiffusion3D
    py::class_<ImplicitDiffusion3D>(
        m, "ImplicitDiffusion3D",
        R"(Conservative 3D Backward Euler solver for diffusion with an optional source.

Solves du/dt = div(D grad(u)) + source(x, y, z, t) on a uniform Cartesian
mesh. Coordinates are in metres, time in seconds, and diffusivity in m²/s.
D may be uniform or specified at every node; harmonic face averaging preserves
interface fluxes. Neumann data mean the outward derivative du/dn, not Fickian
flux. Backward Euler is L-stable for the linear diffusion operator.
)")
        .def(py::init<const StructuredMesh3D&, double>(), py::arg("mesh"), py::arg("diffusivity"),
             py::keep_alive<1, 2>(), "Create a solver with uniform diffusivity [m²/s].")
        .def(py::init<const StructuredMesh3D&, const std::vector<double>&>(), py::arg("mesh"),
             py::arg("diffusivity"), py::keep_alive<1, 2>(),
             "Create a solver with positive nodal diffusivities [m²/s].")
        .def("set_initial_condition", &ImplicitDiffusion3D::setInitialCondition, py::arg("values"),
             "Copy one finite initial field value per mesh node.")
        .def("set_dirichlet_boundary", &ImplicitDiffusion3D::setDirichletBoundary,
             py::arg("boundary"), py::arg("value"), "Set a fixed field value on one boundary face.")
        .def("set_neumann_boundary", &ImplicitDiffusion3D::setNeumannBoundary, py::arg("boundary"),
             py::arg("normal_derivative"),
             "Set outward-normal derivative du/dn [field-units/m], not a flux.")
        .def("set_source_term", &ImplicitDiffusion3D::setSourceTerm, py::arg("source"),
             "Set source(x_m, y_m, z_m, time_s), returning field-units/s.")
        .def("clear_source_term", &ImplicitDiffusion3D::clearSourceTerm,
             "Remove the configured source term.")
        .def("set_solver_type", &ImplicitDiffusion3D::setSolverType, py::arg("solver_type"),
             "Select the Eigen sparse linear solver.")
        .def("set_tolerance", &ImplicitDiffusion3D::setTolerance, py::arg("tolerance"),
             "Set the positive iterative-solver tolerance.")
        .def("set_max_iterations", &ImplicitDiffusion3D::setMaxIterations,
             py::arg("max_iterations"), "Set the positive iterative-solver iteration limit.")
        .def("step", &ImplicitDiffusion3D::step, py::arg("dt"),
             "Advance one positive time step dt [s].")
        .def("solve", &ImplicitDiffusion3D::solve, py::arg("dt"), py::arg("num_steps"),
             "Advance num_steps equal time steps of dt seconds.")
        .def(
            "solution",
            [](const ImplicitDiffusion3D& solver) { return copy_to_numpy(solver.solution()); },
            "Return an owned copy of the current flat row-major field.")
        .def(
            "diffusivity",
            [](const ImplicitDiffusion3D& solver) { return copy_to_numpy(solver.diffusivity()); },
            "Return an owned copy of the nodal diffusivity field [m²/s].")
        .def("time", &ImplicitDiffusion3D::time, "Return current simulation time [s].")
        .def("mesh", &ImplicitDiffusion3D::mesh, py::return_value_policy::reference_internal,
             "Return the solver's 3D mesh.");

#else
    // Stub when Eigen not available
    m.def(
        "sparse_matrix_available", []() { return false; },
        "Check if sparse matrix support is available");
#endif
}

}  // namespace bindings
}  // namespace biotransport
