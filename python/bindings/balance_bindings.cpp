#include "balance_bindings.hpp"

#include <pybind11/stl.h>

#include <biotransport/core/balance.hpp>
#include <optional>

namespace py = pybind11;

namespace biotransport::bindings {

void register_balance_bindings(py::module_& module) {
    py::enum_<BalanceDimension>(module, "BalanceDimension")
        .value("AMOUNT", BalanceDimension::Amount)
        .value("ENERGY", BalanceDimension::Energy)
        .value("VOLUME", BalanceDimension::Volume);

    py::enum_<BalanceUnit>(module, "BalanceUnit")
        .value("MOLE", BalanceUnit::Mole)
        .value("MILLIMOLE", BalanceUnit::Millimole)
        .value("MICROMOLE", BalanceUnit::Micromole)
        .value("JOULE", BalanceUnit::Joule)
        .value("KILOJOULE", BalanceUnit::Kilojoule)
        .value("CUBIC_METER", BalanceUnit::CubicMeter)
        .value("LITER", BalanceUnit::Liter)
        .value("MILLILITER", BalanceUnit::Milliliter);

    py::enum_<BalanceTransferDirection>(module, "BalanceTransferDirection")
        .value("INCOMING", BalanceTransferDirection::Incoming)
        .value("OUTGOING", BalanceTransferDirection::Outgoing);

    py::class_<BalanceTerm>(module, "BalanceTerm")
        .def_readonly("name", &BalanceTerm::name)
        .def_readonly("magnitude", &BalanceTerm::magnitude)
        .def_readonly("unit", &BalanceTerm::unit);

    py::class_<BalanceTransfer>(module, "BalanceTransfer")
        .def_readonly("id", &BalanceTransfer::id)
        .def_readonly("counterparty", &BalanceTransfer::counterparty)
        .def_readonly("magnitude", &BalanceTransfer::magnitude)
        .def_readonly("unit", &BalanceTransfer::unit)
        .def_readonly("direction", &BalanceTransfer::direction);

    py::class_<BalanceAudit>(module, "BalanceAudit")
        .def_readonly("ledger_name", &BalanceAudit::ledger_name)
        .def_readonly("dimension", &BalanceAudit::dimension)
        .def_readonly("unit", &BalanceAudit::unit)
        .def_readonly("initial_inventory", &BalanceAudit::initial_inventory)
        .def_readonly("final_inventory", &BalanceAudit::final_inventory)
        .def_readonly("observed_change", &BalanceAudit::observed_change)
        .def_readonly("boundary_in", &BalanceAudit::boundary_in)
        .def_readonly("boundary_out", &BalanceAudit::boundary_out)
        .def_readonly("generated", &BalanceAudit::generated)
        .def_readonly("consumed", &BalanceAudit::consumed)
        .def_readonly("transfer_in", &BalanceAudit::transfer_in)
        .def_readonly("transfer_out", &BalanceAudit::transfer_out)
        .def_readonly("expected_change", &BalanceAudit::expected_change)
        .def_readonly("closure_residual", &BalanceAudit::closure_residual)
        .def("is_closed", &BalanceAudit::isClosed, py::arg("absolute_tolerance"));

    py::class_<BalanceLedger>(module, "BalanceLedger")
        .def(py::init<std::string, BalanceUnit>(), py::arg("name"), py::arg("unit"))
        .def_property_readonly("name", &BalanceLedger::name)
        .def_property_readonly("unit", &BalanceLedger::unit)
        .def_property_readonly("dimension", &BalanceLedger::dimension)
        .def_property_readonly("has_initial_inventory", &BalanceLedger::hasInitialInventory)
        .def_property_readonly("has_final_inventory", &BalanceLedger::hasFinalInventory)
        .def_property_readonly("initial_inventory", &BalanceLedger::initialInventory)
        .def_property_readonly("final_inventory", &BalanceLedger::finalInventory)
        .def_property_readonly("boundary_in_terms", &BalanceLedger::boundaryInTerms)
        .def_property_readonly("boundary_out_terms", &BalanceLedger::boundaryOutTerms)
        .def_property_readonly("generated_terms", &BalanceLedger::generatedTerms)
        .def_property_readonly("consumed_terms", &BalanceLedger::consumedTerms)
        .def_property_readonly("transfers", &BalanceLedger::transfers)
        .def("set_initial_inventory", &BalanceLedger::setInitialInventory, py::arg("magnitude"),
             py::return_value_policy::reference_internal)
        .def("set_final_inventory", &BalanceLedger::setFinalInventory, py::arg("magnitude"),
             py::return_value_policy::reference_internal)
        .def("add_boundary_in", &BalanceLedger::addBoundaryIn, py::arg("name"),
             py::arg("magnitude"), py::return_value_policy::reference_internal)
        .def("add_boundary_out", &BalanceLedger::addBoundaryOut, py::arg("name"),
             py::arg("magnitude"), py::return_value_policy::reference_internal)
        .def("add_generated", &BalanceLedger::addGenerated, py::arg("name"), py::arg("magnitude"),
             py::return_value_policy::reference_internal)
        .def("add_consumed", &BalanceLedger::addConsumed, py::arg("name"), py::arg("magnitude"),
             py::return_value_policy::reference_internal)
        .def(
            "add_transfer_in",
            [](BalanceLedger& ledger, std::string id, std::string sender, double magnitude,
               std::optional<BalanceUnit> unit) -> BalanceLedger& {
                if (unit) {
                    return ledger.addTransferIn(std::move(id), std::move(sender), magnitude, *unit);
                }
                return ledger.addTransferIn(std::move(id), std::move(sender), magnitude);
            },
            py::arg("id"), py::arg("sender"), py::arg("magnitude"), py::arg("unit") = py::none(),
            py::return_value_policy::reference_internal)
        .def(
            "add_transfer_out",
            [](BalanceLedger& ledger, std::string id, std::string receiver, double magnitude,
               std::optional<BalanceUnit> unit) -> BalanceLedger& {
                if (unit) {
                    return ledger.addTransferOut(std::move(id), std::move(receiver), magnitude,
                                                 *unit);
                }
                return ledger.addTransferOut(std::move(id), std::move(receiver), magnitude);
            },
            py::arg("id"), py::arg("receiver"), py::arg("magnitude"), py::arg("unit") = py::none(),
            py::return_value_policy::reference_internal)
        .def("audit", &BalanceLedger::audit,
             R"doc(Audit this ledger using observed minus expected as the closure residual.)doc");

    py::class_<MatchedBalanceTransfer>(module, "MatchedBalanceTransfer")
        .def_readonly("id", &MatchedBalanceTransfer::id)
        .def_readonly("sender", &MatchedBalanceTransfer::sender)
        .def_readonly("receiver", &MatchedBalanceTransfer::receiver)
        .def_readonly("dimension", &MatchedBalanceTransfer::dimension)
        .def_readonly("base_unit", &MatchedBalanceTransfer::base_unit)
        .def_readonly("magnitude_base", &MatchedBalanceTransfer::magnitude_base);

    py::class_<DimensionBalanceAudit>(module, "DimensionBalanceAudit")
        .def_readonly("dimension", &DimensionBalanceAudit::dimension)
        .def_readonly("base_unit", &DimensionBalanceAudit::base_unit)
        .def_readonly("observed_change", &DimensionBalanceAudit::observed_change)
        .def_readonly("external_expected_change", &DimensionBalanceAudit::external_expected_change)
        .def_readonly("internal_transfer_net", &DimensionBalanceAudit::internal_transfer_net)
        .def_readonly("closure_residual", &DimensionBalanceAudit::closure_residual)
        .def_readonly("representation_adjustment",
                      &DimensionBalanceAudit::representation_adjustment);

    py::class_<BalanceReconciliation>(module, "BalanceReconciliation")
        .def_readonly("ledgers", &BalanceReconciliation::ledgers)
        .def_readonly("matched_transfers", &BalanceReconciliation::matched_transfers)
        .def_readonly("dimensions", &BalanceReconciliation::dimensions)
        .def("is_closed", &BalanceReconciliation::isClosed,
             py::arg("amount_absolute_tolerance") = 0.0, py::arg("energy_absolute_tolerance") = 0.0,
             py::arg("volume_absolute_tolerance") = 0.0);

    module.def("balance_dimension_name", &balanceDimensionName, py::arg("dimension"));
    module.def("balance_unit_symbol", &balanceUnitSymbol, py::arg("unit"));
    module.def("balance_unit_dimension", &balanceDimension, py::arg("unit"));
    module.def("balance_base_unit", &baseBalanceUnit, py::arg("dimension"));
    module.def("convert_balance_value", &convertBalanceValue, py::arg("value"),
               py::arg("from_unit"), py::arg("to_unit"));
    module.def("reconcile_balances", &reconcileBalances, py::arg("ledgers"),
               py::arg("relative_transfer_tolerance") = 1.0e-12,
               py::arg("absolute_transfer_tolerance_base") = 0.0,
               R"doc(Validate named internal transfers and audit coupled model ledgers.

This reconciles accounting records; it does not couple or advance PDE solvers.)doc");
}

}  // namespace biotransport::bindings
