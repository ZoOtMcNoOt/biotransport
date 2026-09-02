#include "transport_bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "binding_helpers.hpp"
#include <biotransport/solvers/transport_solver.hpp>

namespace py = pybind11;

namespace biotransport::bindings {

void register_transport_bindings(py::module_& module) {
    py::class_<SolveOptions>(module, "SolveOptions")
        .def(py::init<>())
        .def_readwrite("final_time", &SolveOptions::final_time)
        .def_readwrite("time_step", &SolveOptions::time_step)
        .def_readwrite("safety_factor", &SolveOptions::safety_factor)
        .def_readwrite("reaction_step_fraction", &SolveOptions::reaction_step_fraction)
        .def_readwrite("max_steps", &SolveOptions::max_steps)
        .def_readwrite("check_finite", &SolveOptions::check_finite)
        .def_readwrite("save_times", &SolveOptions::save_times,
                       "Absolute times in [0, final_time] at which the field is recorded; "
                       "strictly increasing. Each save time partitions the step schedule so the "
                       "field is captured exactly at that clock.")
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
            [](const TransportResult& result) { return to_numpy(result.concentration); },
            "Owned copy of the final nodal concentration field")
        .def_property_readonly(
            "solution",
            [](const TransportResult& result) {
                warn_deprecated("TransportResult.solution", "TransportResult.concentration",
                                "both names returned the same field; concentration is the "
                                "single spelling used by every canonical result");
                return to_numpy(result.concentration);
            },
            "Deprecated alias of ``concentration``")
        .def_readonly("time", &TransportResult::time)
        .def_readonly("diagnostics", &TransportResult::diagnostics)
        .def_property_readonly(
            "mesh", [](const TransportResult& result) { return result.mesh; },
            "Copy of the mesh the fields are defined on")
        .def_property_readonly(
            "snapshot_times",
            [](const TransportResult& result) { return to_numpy(result.snapshot_times); },
            "Absolute times requested through SolveOptions.save_times, in order")
        .def_property_readonly(
            "snapshot_fields",
            [](const TransportResult& result) {
                py::list fields;
                for (const auto& field : result.snapshot_fields) {
                    fields.append(to_numpy(field));
                }
                return fields;
            },
            "Owned copies of the nodal field at each snapshot time");

    module.def(
        "solve_transport",
        [](const TransportProblem& problem, const SolveOptions& options) {
            return solve(problem, options);
        },
        py::arg("problem"), py::arg("options"),
        R"doc(Solve every configured scalar-transport term in the C++ core.

The returned time is exactly ``options.final_time``. Unsupported physics and
uncertified automatic reaction stepping raise before integration begins.
``options.save_times`` records the field at each requested absolute time.)doc");
}

}  // namespace biotransport::bindings
