#include "../test_support/science_test.hpp"
#include <biotransport/core/balance.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace biotransport;

namespace {

template <typename Exception, typename Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

void testSignedBudgetClosesWithDocumentedConvention() {
    BalanceLedger ledger("reactor solute", BalanceUnit::Millimole);
    ledger.setInitialInventory(100.0)
        .setFinalInventory(92.0)
        .addBoundaryIn("feed", 10.0)
        .addBoundaryOut("effluent", 5.0)
        .addGenerated("reaction production", 2.0)
        .addConsumed("reaction uptake", 15.0);

    const BalanceAudit audit = ledger.audit();
    SCIENCE_REQUIRE_NEAR(audit.observed_change, -8.0, 0.0, 0.0, "observed inventory change");
    SCIENCE_REQUIRE_NEAR(audit.expected_change, -8.0, 0.0, 0.0, "signed expected inventory change");
    SCIENCE_REQUIRE_NEAR(audit.closure_residual, 0.0, 0.0, 0.0, "closed ledger residual");
    SCIENCE_REQUIRE(audit.isClosed(0.0), "an exactly closed ledger must report closed");

    ledger.setFinalInventory(93.5);
    const BalanceAudit open = ledger.audit();
    SCIENCE_REQUIRE_NEAR(open.closure_residual, 1.5, 0.0, 0.0,
                         "positive unexplained accumulation residual");
}

void testCompatibleTransferUnitsCancelAcrossLedgers() {
    BalanceLedger donor("donor", BalanceUnit::Mole);
    donor.setInitialInventory(10.0).setFinalInventory(8.0).addTransferOut(
        "solute handoff", "receiver", 2.0, BalanceUnit::Mole);

    BalanceLedger receiver("receiver", BalanceUnit::Millimole);
    receiver.setInitialInventory(1000.0).setFinalInventory(3000.0).addTransferIn(
        "solute handoff", "donor", 2000.0, BalanceUnit::Millimole);

    const BalanceReconciliation result = reconcileBalances({donor, receiver});
    SCIENCE_REQUIRE(result.matched_transfers.size() == 1,
                    "one declared transfer pair must be matched once");
    SCIENCE_REQUIRE_NEAR(result.matched_transfers.front().magnitude_base, 2.0, 0.0, 0.0,
                         "matched transfer in moles");
    SCIENCE_REQUIRE(result.dimensions.size() == 1,
                    "two amount ledgers must produce one dimension audit");
    const DimensionBalanceAudit& total = result.dimensions.front();
    SCIENCE_REQUIRE(total.dimension == BalanceDimension::Amount,
                    "aggregate dimension must remain amount");
    SCIENCE_REQUIRE_NEAR(total.observed_change, 0.0, 0.0, 0.0, "coupled amount inventory change");
    SCIENCE_REQUIRE_NEAR(total.external_expected_change, 0.0, 0.0, 0.0,
                         "coupled external expected change");
    SCIENCE_REQUIRE_NEAR(total.internal_transfer_net, 0.0, 0.0, 0.0,
                         "validated internal transfers cancel");
    SCIENCE_REQUIRE_NEAR(total.representation_adjustment, 0.0, 0.0, 0.0,
                         "exact transfer conversion needs no representation adjustment");
    SCIENCE_REQUIRE_NEAR(total.closure_residual, 0.0, 0.0, 0.0, "coupled amount closure residual");
    SCIENCE_REQUIRE(result.isClosed(), "closed coupled ledgers must reconcile exactly");
}

void testDecimalSiPrefixTransfersCancelAcrossLedgers() {
    BalanceLedger donor("decimal donor", BalanceUnit::Mole);
    donor.setInitialInventory(0.1).setFinalInventory(0.0).addTransferOut(
        "decimal handoff", "decimal receiver", 0.1, BalanceUnit::Mole);

    BalanceLedger receiver("decimal receiver", BalanceUnit::Millimole);
    receiver.setInitialInventory(0.0).setFinalInventory(100.0).addTransferIn(
        "decimal handoff", "decimal donor", 100.0, BalanceUnit::Millimole);

    const BalanceReconciliation result = reconcileBalances({donor, receiver});
    SCIENCE_REQUIRE_NEAR(result.matched_transfers.front().magnitude_base, 0.1, 0.0, 0.0,
                         "matched decimal transfer in moles");
    SCIENCE_REQUIRE_NEAR(result.dimensions.front().observed_change, 0.0, 0.0, 0.0,
                         "equivalent decimal inventories cancel in base units");
    SCIENCE_REQUIRE_NEAR(result.dimensions.front().closure_residual, 0.0, 0.0, 0.0,
                         "decimal SI-prefix reconciliation closes exactly");
    SCIENCE_REQUIRE(result.isClosed(),
                    "equivalent decimal SI-prefix ledgers must reconcile exactly");
}

void testExactSiPrefixConversions() {
    struct Conversion {
        double value;
        BalanceUnit from;
        BalanceUnit to;
        double expected;
        const char* quantity;
    };
    const std::vector<Conversion> conversions{
        {2000.0, BalanceUnit::Millimole, BalanceUnit::Mole, 2.0, "millimoles to moles"},
        {2.0, BalanceUnit::Mole, BalanceUnit::Millimole, 2000.0, "moles to millimoles"},
        {2000000.0, BalanceUnit::Micromole, BalanceUnit::Mole, 2.0, "micromoles to moles"},
        {2.0, BalanceUnit::Mole, BalanceUnit::Micromole, 2000000.0, "moles to micromoles"},
        {2.0, BalanceUnit::Kilojoule, BalanceUnit::Joule, 2000.0, "kilojoules to joules"},
        {2000.0, BalanceUnit::Joule, BalanceUnit::Kilojoule, 2.0, "joules to kilojoules"},
        {2000.0, BalanceUnit::Liter, BalanceUnit::CubicMeter, 2.0, "liters to cubic metres"},
        {2.0, BalanceUnit::CubicMeter, BalanceUnit::Liter, 2000.0, "cubic metres to liters"},
        {2000000.0, BalanceUnit::Milliliter, BalanceUnit::CubicMeter, 2.0,
         "milliliters to cubic metres"},
        {2.0, BalanceUnit::CubicMeter, BalanceUnit::Milliliter, 2000000.0,
         "cubic metres to milliliters"},
        {0x1.3102644584d9ep-4, BalanceUnit::Micromole, BalanceUnit::Mole, 0x1.3fd3526af11d5p-24,
         "micromoles use one binary64 rounding"},
    };

    for (const auto& conversion : conversions) {
        SCIENCE_REQUIRE_NEAR(convertBalanceValue(conversion.value, conversion.from, conversion.to),
                             conversion.expected, 0.0, 0.0, conversion.quantity);
    }
}

void testAggregateUsesPortableCompensatedSummation() {
    BalanceLedger large_gain("large gain", BalanceUnit::Mole);
    large_gain.setInitialInventory(0.0).setFinalInventory(1.0e16);

    BalanceLedger unit_gain("unit gain", BalanceUnit::Mole);
    unit_gain.setInitialInventory(0.0).setFinalInventory(1.0);

    BalanceLedger large_loss("large loss", BalanceUnit::Mole);
    large_loss.setInitialInventory(1.0e16).setFinalInventory(0.0);

    const BalanceReconciliation result = reconcileBalances({large_gain, unit_gain, large_loss});
    const DimensionBalanceAudit& total = result.dimensions.front();
    SCIENCE_REQUIRE_NEAR(total.observed_change, 1.0, 0.0, 0.0,
                         "compensated aggregate preserves a unit-scale residual");
    SCIENCE_REQUIRE_NEAR(total.external_expected_change, 0.0, 0.0, 0.0,
                         "empty external budget remains zero");
    SCIENCE_REQUIRE_NEAR(total.closure_residual, 1.0, 0.0, 0.0,
                         "aggregate residual is independent of long-double width");
    SCIENCE_REQUIRE(!result.isClosed(),
                    "a resolved unit-scale aggregate residual must not report closed");
}

void testLedgerUsesPortableCompensatedBudget() {
    BalanceLedger ledger("compensated ledger", BalanceUnit::Mole);
    ledger.setInitialInventory(0.0)
        .setFinalInventory(1.0)
        .addBoundaryIn("large input", 1.0e16)
        .addGenerated("unit generation", 1.0)
        .addConsumed("large consumption", 1.0e16);

    const BalanceAudit audit = ledger.audit();
    SCIENCE_REQUIRE_NEAR(audit.expected_change, 1.0, 0.0, 0.0,
                         "compensated ledger preserves a unit-scale budget");
    SCIENCE_REQUIRE_NEAR(audit.closure_residual, 0.0, 0.0, 0.0,
                         "compensated ledger closes exactly");
    SCIENCE_REQUIRE(audit.isClosed(0.0), "the compensated per-ledger budget must report closed");

    const BalanceReconciliation result = reconcileBalances({ledger});
    SCIENCE_REQUIRE_NEAR(result.dimensions.front().external_expected_change, 1.0, 0.0, 0.0,
                         "aggregate external budget preserves the ledger result");
    SCIENCE_REQUIRE_NEAR(result.dimensions.front().closure_residual, 0.0, 0.0, 0.0,
                         "aggregate compensated budget closes exactly");
    SCIENCE_REQUIRE(result.isClosed(), "the compensated aggregate budget must report closed");
}

void testAggregateConvertsNativeSubtotalOnce() {
    BalanceLedger ledger("native subtotal", BalanceUnit::Millimole);
    ledger.setInitialInventory(0.0)
        .setFinalInventory(1.125)
        .addBoundaryIn("whole input", 1.0)
        .addBoundaryIn("fractional input", 0.125);

    const BalanceAudit audit = ledger.audit();
    SCIENCE_REQUIRE(audit.isClosed(0.0), "the native-unit ledger must close exactly");

    const BalanceReconciliation result = reconcileBalances({ledger});
    const DimensionBalanceAudit& total = result.dimensions.front();
    SCIENCE_REQUIRE_NEAR(total.observed_change, 0.001125, 0.0, 0.0,
                         "observed subtotal converts once to base units");
    SCIENCE_REQUIRE_NEAR(total.external_expected_change, 0.001125, 0.0, 0.0,
                         "external subtotal converts once to base units");
    SCIENCE_REQUIRE_NEAR(total.closure_residual, 0.0, 0.0, 0.0,
                         "native subtotal reconciliation closes exactly");
    SCIENCE_REQUIRE(result.isClosed(), "the native subtotal aggregate must report closed");
}

void testMixedUnitTransferRoundoffIsAccountedInternally() {
    constexpr double transfer_micromoles = 1.125;
    const double donor_change =
        convertBalanceValue(transfer_micromoles, BalanceUnit::Micromole, BalanceUnit::Mole);
    const double receiver_change =
        convertBalanceValue(transfer_micromoles, BalanceUnit::Micromole, BalanceUnit::Millimole);

    BalanceLedger donor("roundoff donor", BalanceUnit::Mole);
    donor.setInitialInventory(donor_change)
        .setFinalInventory(0.0)
        .addTransferOut("roundoff handoff", "roundoff receiver", transfer_micromoles,
                        BalanceUnit::Micromole);

    BalanceLedger receiver("roundoff receiver", BalanceUnit::Millimole);
    receiver.setInitialInventory(0.0)
        .setFinalInventory(receiver_change)
        .addTransferIn("roundoff handoff", "roundoff donor", transfer_micromoles,
                       BalanceUnit::Micromole);

    SCIENCE_REQUIRE(donor.audit().isClosed(0.0) && receiver.audit().isClosed(0.0),
                    "both local-unit transfer ledgers must close exactly");

    const BalanceReconciliation result = reconcileBalances({donor, receiver});
    const DimensionBalanceAudit& total = result.dimensions.front();
    SCIENCE_REQUIRE_NEAR(total.internal_transfer_net, -0x1p-72, 0.0, 0.0,
                         "local-unit transfer roundoff is reported internally");
    SCIENCE_REQUIRE_NEAR(total.observed_change, total.internal_transfer_net, 0.0, 0.0,
                         "observed inventory net matches local transfer representation");
    SCIENCE_REQUIRE_NEAR(total.external_expected_change, 0.0, 0.0, 0.0,
                         "internal transfers remain excluded from the external budget");
    SCIENCE_REQUIRE_NEAR(total.representation_adjustment, 0.0, 0.0, 0.0,
                         "transfer-only roundoff needs no decomposition adjustment");
    SCIENCE_REQUIRE_NEAR(total.closure_residual, 0.0, 0.0, 0.0,
                         "mixed-unit transfer representation reconciles exactly");
    SCIENCE_REQUIRE(result.isClosed(), "the mixed-unit transfer aggregate must report closed");
}

void testExternalAndInternalRoundingPreservesClosure() {
    BalanceLedger donor("combined donor", BalanceUnit::Millimole);
    donor.setInitialInventory(0.2).setFinalInventory(0.0).addTransferOut(
        "combined handoff", "combined receiver", 0.2, BalanceUnit::Millimole);

    BalanceLedger receiver("combined receiver", BalanceUnit::Millimole);
    receiver.setInitialInventory(0.0)
        .setFinalInventory(1.2)
        .addBoundaryIn("external input", 1.0)
        .addTransferIn("combined handoff", "combined donor", 0.2, BalanceUnit::Millimole);

    SCIENCE_REQUIRE(donor.audit().isClosed(0.0) && receiver.audit().isClosed(0.0),
                    "combined external and transfer ledgers must close exactly");

    const BalanceReconciliation result = reconcileBalances({donor, receiver});
    const DimensionBalanceAudit& total = result.dimensions.front();
    SCIENCE_REQUIRE_NEAR(total.internal_transfer_net, 0.0, 0.0, 0.0,
                         "matched same-unit internal transfers cancel exactly");
    SCIENCE_REQUIRE_NEAR(total.representation_adjustment, -0x1p-62, 0.0, 0.0,
                         "decomposition roundoff is reported separately");
    SCIENCE_REQUIRE_NEAR(total.closure_residual, 0.0, 0.0, 0.0,
                         "converted complete expectations preserve aggregate closure");
    SCIENCE_REQUIRE(result.isClosed(),
                    "combined external and transfer aggregate must report closed");
}

void testRepresentationAdjustmentPreservesBookkeepingLabels() {
    BalanceLedger donor("hierarchical donor", BalanceUnit::Mole);
    donor.setInitialInventory(0.0)
        .setFinalInventory(1.0)
        .addBoundaryIn("large input", 1.0e16)
        .addGenerated("unit generation", 1.0)
        .addTransferOut("large handoff", "hierarchical receiver", 1.0e16);

    BalanceLedger receiver("hierarchical receiver", BalanceUnit::Mole);
    receiver.setInitialInventory(0.0).setFinalInventory(1.0e16).addTransferIn(
        "large handoff", "hierarchical donor", 1.0e16);

    BalanceLedger sink("external sink", BalanceUnit::Mole);
    sink.setInitialInventory(1.0e16).setFinalInventory(0.0).addBoundaryOut("large output", 1.0e16);

    SCIENCE_REQUIRE(
        donor.audit().isClosed(0.0) && receiver.audit().isClosed(0.0) && sink.audit().isClosed(0.0),
        "all hierarchical bookkeeping ledgers must close exactly");

    const BalanceReconciliation result = reconcileBalances({donor, receiver, sink});
    const DimensionBalanceAudit& total = result.dimensions.front();
    SCIENCE_REQUIRE_NEAR(total.observed_change, 1.0, 0.0, 0.0,
                         "complete converted observation preserves the unit term");
    SCIENCE_REQUIRE_NEAR(total.external_expected_change, 0.0, 0.0, 0.0,
                         "external rounded subtotals retain their physical label");
    SCIENCE_REQUIRE_NEAR(total.internal_transfer_net, 0.0, 0.0, 0.0,
                         "matched internal transfers retain a zero net");
    SCIENCE_REQUIRE_NEAR(total.representation_adjustment, 1.0, 0.0, 0.0,
                         "hierarchical rounding is exposed as representation adjustment");
    SCIENCE_REQUIRE_NEAR(total.closure_residual, 0.0, 0.0, 0.0,
                         "complete expected bookkeeping remains exactly closed");
    SCIENCE_REQUIRE(result.isClosed(), "hierarchical bookkeeping aggregate must report closed");
}

void testDimensionAuditAggregateInitializationRemainsCompatible() {
    const DimensionBalanceAudit legacy{
        BalanceDimension::Amount, BalanceUnit::Mole, 1.0, 2.0, 3.0, 4.0};
    SCIENCE_REQUIRE_NEAR(legacy.closure_residual, 4.0, 0.0, 0.0,
                         "legacy sixth aggregate field remains closure residual");
    SCIENCE_REQUIRE_NEAR(legacy.representation_adjustment, 0.0, 0.0, 0.0,
                         "new aggregate field defaults when omitted");
}

void testMixedDimensionsRemainSeparate() {
    BalanceLedger amount("amount process", BalanceUnit::Micromole);
    amount.setInitialInventory(5.0).setFinalInventory(7.0).addGenerated("synthesis", 2.0);

    BalanceLedger energy("thermal process", BalanceUnit::Kilojoule);
    energy.setInitialInventory(12.0)
        .setFinalInventory(10.5)
        .addBoundaryIn("heater", 0.5)
        .addBoundaryOut("cooling jacket", 2.0);

    BalanceLedger volume("fluid process", BalanceUnit::Liter);
    volume.setInitialInventory(4.0)
        .setFinalInventory(3.75)
        .addBoundaryIn("pump inlet", 0.25)
        .addBoundaryOut("pump outlet", 0.5);

    const BalanceReconciliation result = reconcileBalances({amount, energy, volume});
    SCIENCE_REQUIRE(result.dimensions.size() == 3,
                    "mixed physical dimensions require separate aggregate audits");
    SCIENCE_REQUIRE_NEAR(result.dimensions[0].closure_residual, 0.0, 1.0e-21, 0.0,
                         "amount aggregate closure in mol");
    SCIENCE_REQUIRE_NEAR(result.dimensions[1].closure_residual, 0.0, 0.0, 0.0,
                         "energy aggregate closure in J");
    SCIENCE_REQUIRE_NEAR(result.dimensions[2].closure_residual, 0.0, 1.0e-18, 0.0,
                         "volume aggregate closure in cubic metres");
}

void testUnmatchedAndUnknownTransfersFailLoudly() {
    BalanceLedger donor("donor", BalanceUnit::Mole);
    donor.setInitialInventory(2.0).setFinalInventory(1.0).addTransferOut("unpaired", "receiver",
                                                                         1.0);
    BalanceLedger receiver("receiver", BalanceUnit::Mole);
    receiver.setInitialInventory(0.0).setFinalInventory(0.0);
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([&] { (void)reconcileBalances({donor, receiver}); }),
        "an outgoing transfer without an incoming pair must be rejected");

    BalanceLedger unknown("known", BalanceUnit::Mole);
    unknown.setInitialInventory(1.0).setFinalInventory(0.0).addTransferOut("orphan", "not supplied",
                                                                           1.0);
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { (void)reconcileBalances({unknown}); }),
                    "a transfer to an unknown ledger must be rejected");
}

void testDoubleCountedAndMismatchedTransfersFailLoudly() {
    BalanceLedger first("first", BalanceUnit::Mole);
    first.setInitialInventory(2.0).setFinalInventory(1.0).addTransferOut("shared ID", "receiver",
                                                                         1.0);
    BalanceLedger second("second", BalanceUnit::Mole);
    second.setInitialInventory(2.0).setFinalInventory(1.0).addTransferOut("shared ID", "receiver",
                                                                          1.0);
    BalanceLedger receiver("receiver", BalanceUnit::Mole);
    receiver.setInitialInventory(0.0).setFinalInventory(1.0).addTransferIn("shared ID", "first",
                                                                           1.0);
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([&] { (void)reconcileBalances({first, second, receiver}); }),
        "a globally reused outgoing transfer ID must be rejected as double-counted");

    BalanceLedger donor("donor", BalanceUnit::Mole);
    donor.setInitialInventory(2.0).setFinalInventory(1.0).addTransferOut("bad magnitude",
                                                                         "acceptor", 1.0);
    BalanceLedger acceptor("acceptor", BalanceUnit::Millimole);
    acceptor.setInitialInventory(0.0).setFinalInventory(900.0).addTransferIn("bad magnitude",
                                                                             "donor", 900.0);
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([&] { (void)reconcileBalances({donor, acceptor}); }),
        "different physical magnitudes must not be reconciled");
}

void testDimensionAndUnitMismatchesAreRejected() {
    BalanceLedger amount("amount", BalanceUnit::Mole);
    amount.setInitialInventory(1.0).setFinalInventory(1.0);
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        amount.addTransferOut("wrong unit", "other", 1.0, BalanceUnit::Joule);
                    }),
                    "an energy transfer unit must not enter an amount ledger");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([] {
                        (void)convertBalanceValue(1.0, BalanceUnit::Liter, BalanceUnit::Mole);
                    }),
                    "conversions across physical dimensions must be rejected");

    BalanceLedger energy("energy", BalanceUnit::Joule);
    energy.setInitialInventory(0.0).setFinalInventory(1.0).addTransferIn("cross dimension",
                                                                         "amount", 1.0);
    amount.addTransferOut("cross dimension", "energy", 1.0);
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([&] { (void)reconcileBalances({amount, energy}); }),
        "paired transfers cannot cross incompatible ledger dimensions");
}

void testInvalidMagnitudesAndIncompleteLedgersAreRejected() {
    BalanceLedger ledger("validated", BalanceUnit::Joule);
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { ledger.setInitialInventory(-1.0); }),
                    "negative inventory magnitude must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        ledger.addBoundaryIn("nan", std::numeric_limits<double>::quiet_NaN());
                    }),
                    "non-finite boundary magnitude must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] {
                        ledger.addGenerated("infinite", std::numeric_limits<double>::infinity());
                    }),
                    "infinite generation magnitude must be rejected");
    SCIENCE_REQUIRE(throws<std::invalid_argument>(
                        [&] { ledger.addTransferOut("negative transfer", "peer", -1.0); }),
                    "negative transfer magnitude must be rejected");
    ledger.setInitialInventory(0.0).addBoundaryIn("heater", 1.0);
    SCIENCE_REQUIRE(throws<std::invalid_argument>([&] { ledger.addBoundaryIn("heater", 1.0); }),
                    "duplicate named terms must be rejected before they can be double-counted");
    SCIENCE_REQUIRE(throws<std::logic_error>([&] { (void)ledger.audit(); }),
                    "both inventories are required for a closure audit");
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>([] { (void)BalanceLedger("   ", BalanceUnit::Liter); }),
        "blank ledger names must be rejected");
    BalanceLedger complete("complete", BalanceUnit::Joule);
    complete.setInitialInventory(0.0).setFinalInventory(0.0);
    SCIENCE_REQUIRE(
        throws<std::invalid_argument>(
            [&] { (void)reconcileBalances({complete}, std::numeric_limits<double>::quiet_NaN()); }),
        "non-finite reconciliation tolerances must be rejected");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "coupled balance accounting",
        {{"signed budget closes with documented convention",
          testSignedBudgetClosesWithDocumentedConvention},
         {"compatible transfer units cancel across ledgers",
          testCompatibleTransferUnitsCancelAcrossLedgers},
         {"decimal SI-prefix transfers cancel across ledgers",
          testDecimalSiPrefixTransfersCancelAcrossLedgers},
         {"SI prefix conversions use exact integer factors", testExactSiPrefixConversions},
         {"aggregate uses portable compensated summation",
          testAggregateUsesPortableCompensatedSummation},
         {"ledger uses portable compensated budget", testLedgerUsesPortableCompensatedBudget},
         {"aggregate converts native subtotal once", testAggregateConvertsNativeSubtotalOnce},
         {"mixed-unit transfer roundoff is accounted internally",
          testMixedUnitTransferRoundoffIsAccountedInternally},
         {"external and internal rounding preserves closure",
          testExternalAndInternalRoundingPreservesClosure},
         {"representation adjustment preserves bookkeeping labels",
          testRepresentationAdjustmentPreservesBookkeepingLabels},
         {"dimension audit aggregate initialization remains compatible",
          testDimensionAuditAggregateInitializationRemainsCompatible},
         {"mixed dimensions remain separate", testMixedDimensionsRemainSeparate},
         {"unmatched and unknown transfers fail loudly",
          testUnmatchedAndUnknownTransfersFailLoudly},
         {"double-counted and mismatched transfers fail loudly",
          testDoubleCountedAndMismatchedTransfersFailLoudly},
         {"dimension and unit mismatches are rejected", testDimensionAndUnitMismatchesAreRejected},
         {"invalid magnitudes and incomplete ledgers are rejected",
          testInvalidMagnitudesAndIncompleteLedgersAreRejected}});
}
