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
    SCIENCE_REQUIRE_NEAR(total.closure_residual, 0.0, 0.0, 0.0, "coupled amount closure residual");
    SCIENCE_REQUIRE(result.isClosed(), "closed coupled ledgers must reconcile exactly");
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
         {"mixed dimensions remain separate", testMixedDimensionsRemainSeparate},
         {"unmatched and unknown transfers fail loudly",
          testUnmatchedAndUnknownTransfersFailLoudly},
         {"double-counted and mismatched transfers fail loudly",
          testDoubleCountedAndMismatchedTransfersFailLoudly},
         {"dimension and unit mismatches are rejected", testDimensionAndUnitMismatchesAreRejected},
         {"invalid magnitudes and incomplete ledgers are rejected",
          testInvalidMagnitudesAndIncompleteLedgersAreRejected}});
}
