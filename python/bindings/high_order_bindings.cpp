/**
 * @file high_order_bindings.cpp
 * @brief Internal Python adapters for validated high-order C++ kernels.
 */

#include "high_order_bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <biotransport/core/numerics/time_integration/high_order.hpp>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace biotransport {
namespace bindings {
namespace {

using NativeArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using time_integration::State;

State arrayToState(const NativeArray& array, const char* name) {
    const py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw py::value_error(std::string(name) + " must be a one-dimensional array");
    }
    State result(static_cast<std::size_t>(info.size));
    if (info.size > 0) {
        std::memcpy(result.data(), info.ptr, static_cast<std::size_t>(info.size) * sizeof(double));
    }
    return result;
}

py::array_t<double> stateToArray(const State& state) {
    py::array_t<double> result(static_cast<py::ssize_t>(state.size()));
    if (!state.empty()) {
        std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(double));
    }
    return result;
}

State callbackResultToState(const py::object& value) {
    NativeArray array = NativeArray::ensure(value);
    if (!array) {
        throw py::type_error("right-hand side must return a numeric one-dimensional array");
    }
    return arrayToState(array, "right-hand side result");
}

time_integration::RungeKuttaMethod parseMethod(const std::string& method) {
    if (method == "heun") {
        return time_integration::RungeKuttaMethod::HEUN;
    }
    if (method == "rk4") {
        return time_integration::RungeKuttaMethod::CLASSICAL_RK4;
    }
    throw py::value_error("method must be 'heun' or 'rk4'");
}

}  // namespace

void register_high_order_bindings(py::module_& module) {
    module.def(
        "_high_order_laplacian_1d",
        [](const NativeArray& field, double dx, int order) {
            const State input = arrayToState(field, "field");
            State output;
            {
                py::gil_scoped_release release;
                output = time_integration::laplacian1D(input, dx, order);
            }
            return stateToArray(output);
        },
        py::arg("field"), py::arg("dx"), py::arg("order"));

    module.def(
        "_high_order_laplacian_2d",
        [](const NativeArray& field, int nx, int ny, double dx, double dy, int order) {
            const State input = arrayToState(field, "field");
            State output;
            {
                py::gil_scoped_release release;
                output = time_integration::laplacian2D(input, nx, ny, dx, dy, order);
            }
            return stateToArray(output);
        },
        py::arg("field"), py::arg("nx"), py::arg("ny"), py::arg("dx"), py::arg("dy"),
        py::arg("order"));

    module.def(
        "_high_order_gradient_1d",
        [](const NativeArray& field, double dx, int order) {
            const State input = arrayToState(field, "field");
            State output;
            {
                py::gil_scoped_release release;
                output = time_integration::gradient1D(input, dx, order);
            }
            return stateToArray(output);
        },
        py::arg("field"), py::arg("dx"), py::arg("order"));

    module.def("_high_order_stable_dt", &time_integration::stableDiffusionTimeStep,
               py::arg("diffusivity"), py::arg("dx"), py::arg("dy"), py::arg("order"),
               py::arg("safety_factor"), py::arg("is_2d"));

    module.def(
        "_solve_high_order_diffusion",
        [](const NativeArray& initial, int nx, int ny, double dx, double dy, double diffusivity,
           int order, double safety_factor, double end_time, std::optional<double> requested_dt,
           double left, double right, double bottom, double top, py::object callback) {
            time_integration::DiffusionObserver observer;
            if (!callback.is_none()) {
                if (!PyCallable_Check(callback.ptr())) {
                    throw py::type_error("callback must be callable or None");
                }
                observer = [callback = py::reinterpret_borrow<py::function>(callback)](
                               double time, const State& state) {
                    py::gil_scoped_acquire acquire;
                    callback(time, stateToArray(state));
                };
            }

            State initial_state = arrayToState(initial, "initial");
            time_integration::HighOrderDiffusionResult result;
            {
                py::gil_scoped_release release;
                result = time_integration::solveHighOrderDiffusion(
                    initial_state, nx, ny, dx, dy, diffusivity, order, safety_factor, end_time,
                    requested_dt.value_or(0.0), left, right, bottom, top, observer);
            }

            py::dict output;
            output["solution"] = stateToArray(result.solution);
            output["time"] = result.time;
            output["steps"] = result.steps;
            output["dt"] = result.nominal_dt;
            output["last_dt"] = result.last_dt;
            output["order"] = result.interior_order;
            output["boundary_order"] = result.boundary_closure_order;
            return output;
        },
        py::arg("initial"), py::arg("nx"), py::arg("ny"), py::arg("dx"), py::arg("dy"),
        py::arg("diffusivity"), py::arg("order"), py::arg("safety_factor"), py::arg("end_time"),
        py::arg("dt") = py::none(), py::arg("left") = 0.0, py::arg("right") = 0.0,
        py::arg("bottom") = 0.0, py::arg("top") = 0.0, py::arg("callback") = py::none());

    module.def(
        "_integrate_explicit_runge_kutta",
        [](const NativeArray& initial, py::function rhs, double initial_time, double end_time,
           double dt, const std::string& method, bool autonomous, std::size_t maximum_steps) {
            const auto native_method = parseMethod(method);
            time_integration::RHSFunction native_rhs = [rhs = std::move(rhs), autonomous](
                                                           const State& state, double time) {
                py::gil_scoped_acquire acquire;
                py::array_t<double> state_array = stateToArray(state);
                py::object value = autonomous ? rhs(state_array) : rhs(state_array, time);
                return callbackResultToState(value);
            };

            State initial_state = arrayToState(initial, "initial");
            time_integration::RungeKuttaResult result;
            {
                py::gil_scoped_release release;
                result =
                    time_integration::integrateRungeKutta(initial_state, initial_time, end_time, dt,
                                                          native_rhs, native_method, maximum_steps);
            }

            py::dict output;
            output["solution"] = stateToArray(result.state);
            output["initial_time"] = result.initial_time;
            output["time"] = result.time;
            output["steps"] = result.steps;
            output["dt"] = result.nominal_dt;
            output["last_dt"] = result.last_dt;
            output["method"] = method;
            return output;
        },
        py::arg("initial"), py::arg("rhs"), py::arg("initial_time"), py::arg("end_time"),
        py::arg("dt"), py::arg("method") = "rk4", py::arg("autonomous") = false,
        py::arg("maximum_steps") = 10000000,
        R"doc(
Integrate an ODE with validated Heun or classical RK4 stages.

``autonomous=False`` calls ``rhs(state, time)``; ``autonomous=True`` calls
``rhs(state)``.  Python callbacks are evaluated under the GIL at every stage,
so this correctness-oriented adapter does not claim callback acceleration.
)doc");
}

}  // namespace bindings
}  // namespace biotransport
