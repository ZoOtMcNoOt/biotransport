#include "transport_bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <biotransport/solvers/transport_solver.hpp>
#include <cstring>

namespace py = pybind11;

namespace biotransport::bindings {
namespace {

py::array_t<double> copyToNumpy(const std::vector<double>& values) {
    py::array_t<double> array(values.size());
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(double));
    }
    return array;
}

}  // namespace

void register_transport_bindings(py::module_& module) {
    py::class_<SolveOptions>(module, "SolveOptions")
        .def(py::init<>())
        .def_readwrite("final_time", &SolveOptions::final_time)
        .def_readwrite("time_step", &SolveOptions::time_step)
        .def_readwrite("safety_factor", &SolveOptions::safety_factor)
        .def_readwrite("reaction_step_fraction", &SolveOptions::reaction_step_fraction)
        .def_readwrite("max_steps", &SolveOptions::max_steps)
        .def_readwrite("check_finite", &SolveOptions::check_finite)
        .def_static("until", &SolveOptions::until, py::arg("final_time"));

    py::class_<SolveDiagnostics>(module, "SolveDiagnostics")
        .def_readonly("steps", &SolveDiagnostics::steps)
        .def_readonly("requested_final_time", &SolveDiagnostics::requested_final_time)
        .def_readonly("final_time", &SolveDiagnostics::final_time)
        .def_readonly("requested_time_step", &SolveDiagnostics::requested_time_step)
        .def_readonly("minimum_time_step", &SolveDiagnostics::minimum_time_step)
        .def_readonly("maximum_time_step", &SolveDiagnostics::maximum_time_step)
        .def_readonly("transport_stable_time_step", &SolveDiagnostics::transport_stable_time_step)
        .def_readonly("certified_stable_time_step", &SolveDiagnostics::certified_stable_time_step)
        .def_readonly("maximum_transport_loss_rate", &SolveDiagnostics::maximum_transport_loss_rate)
        .def_readonly("reaction_rate_bound", &SolveDiagnostics::reaction_rate_bound)
        .def_readonly("automatic_time_step", &SolveDiagnostics::automatic_time_step)
        .def_readonly("reaction_stability_bound_known",
                      &SolveDiagnostics::reaction_stability_bound_known)
        .def_readonly("initial_mass", &SolveDiagnostics::initial_mass)
        .def_readonly("final_mass", &SolveDiagnostics::final_mass)
        .def_readonly("mass_change", &SolveDiagnostics::mass_change)
        .def_readonly("initial_minimum", &SolveDiagnostics::initial_minimum)
        .def_readonly("initial_maximum", &SolveDiagnostics::initial_maximum)
        .def_readonly("final_minimum", &SolveDiagnostics::final_minimum)
        .def_readonly("final_maximum", &SolveDiagnostics::final_maximum);

    py::class_<TransportResult>(module, "TransportResult")
        .def_property_readonly(
            "concentration",
            [](const TransportResult& result) { return copyToNumpy(result.concentration); })
        .def_property_readonly(
            "solution",
            [](const TransportResult& result) { return copyToNumpy(result.concentration); })
        .def_readonly("time", &TransportResult::time)
        .def_readonly("diagnostics", &TransportResult::diagnostics);

    module.def(
        "solve_transport",
        [](const TransportProblem& problem, const SolveOptions& options) {
            return solve(problem, options);
        },
        py::arg("problem"), py::arg("options"),
        R"doc(Solve every configured scalar-transport term in the C++ core.

The returned time is exactly ``options.final_time``. Unsupported physics and
uncertified automatic reaction stepping raise before integration begins.)doc");
}

}  // namespace biotransport::bindings
