"""Tests for native, unit-aware multiphysics balance accounting."""

import math
import unittest

import biotransport as bt


class TestBalanceAccounting(unittest.TestCase):
    def test_signed_single_ledger_budget(self):
        ledger = bt.BalanceLedger("reactor solute", bt.BalanceUnit.MILLIMOLE)
        returned = (
            ledger.set_initial_inventory(100.0)
            .set_final_inventory(92.0)
            .add_boundary_in("feed", 10.0)
            .add_boundary_out("effluent", 5.0)
            .add_generated("reaction production", 2.0)
            .add_consumed("reaction uptake", 15.0)
        )
        self.assertIs(returned, ledger)

        audit = ledger.audit()
        self.assertEqual(audit.observed_change, -8.0)
        self.assertEqual(audit.expected_change, -8.0)
        self.assertEqual(audit.closure_residual, 0.0)
        self.assertTrue(audit.is_closed(0.0))
        self.assertEqual(ledger.boundary_in_terms[0].name, "feed")
        self.assertEqual(ledger.boundary_in_terms[0].unit, bt.BalanceUnit.MILLIMOLE)

    def test_compatible_units_and_named_transfer_cancel(self):
        donor = bt.BalanceLedger("donor", bt.BalanceUnit.MOLE)
        donor.set_initial_inventory(10.0).set_final_inventory(8.0).add_transfer_out(
            "solute handoff", "receiver", 2.0
        )
        receiver = bt.BalanceLedger("receiver", bt.BalanceUnit.MILLIMOLE)
        receiver.set_initial_inventory(1000.0).set_final_inventory(
            3000.0
        ).add_transfer_in("solute handoff", "donor", 2000.0, bt.BalanceUnit.MILLIMOLE)

        result = bt.reconcile_balances([donor, receiver])
        self.assertTrue(result.is_closed())
        self.assertEqual(len(result.matched_transfers), 1)
        self.assertEqual(result.matched_transfers[0].sender, "donor")
        self.assertEqual(result.matched_transfers[0].receiver, "receiver")
        self.assertAlmostEqual(result.matched_transfers[0].magnitude_base, 2.0)
        self.assertEqual(len(result.dimensions), 1)
        self.assertEqual(result.dimensions[0].dimension, bt.BalanceDimension.AMOUNT)
        self.assertEqual(result.dimensions[0].internal_transfer_net, 0.0)
        self.assertAlmostEqual(result.dimensions[0].closure_residual, 0.0)

        # Reconciliation results own their audit records; later ledger edits do not alias them.
        donor.set_final_inventory(9.0)
        self.assertEqual(result.ledgers[0].final_inventory, 8.0)

    def test_mixed_dimensions_are_audited_separately(self):
        amount = bt.BalanceLedger("amount", bt.BalanceUnit.MICROMOLE)
        amount.set_initial_inventory(5.0).set_final_inventory(7.0).add_generated(
            "synthesis", 2.0
        )
        energy = bt.BalanceLedger("energy", bt.BalanceUnit.KILOJOULE)
        energy.set_initial_inventory(12.0).set_final_inventory(10.5).add_boundary_in(
            "heater", 0.5
        ).add_boundary_out("cooler", 2.0)
        volume = bt.BalanceLedger("volume", bt.BalanceUnit.LITER)
        volume.set_initial_inventory(4.0).set_final_inventory(3.75).add_boundary_in(
            "inlet", 0.25
        ).add_boundary_out("outlet", 0.5)

        result = bt.reconcile_balances([amount, energy, volume])
        self.assertEqual(
            [audit.dimension for audit in result.dimensions],
            [
                bt.BalanceDimension.AMOUNT,
                bt.BalanceDimension.ENERGY,
                bt.BalanceDimension.VOLUME,
            ],
        )
        self.assertTrue(result.is_closed(1e-18, 1e-12, 1e-18))

    def test_unmatched_unknown_and_double_counted_transfers_fail(self):
        donor = bt.BalanceLedger("donor", bt.BalanceUnit.MOLE)
        donor.set_initial_inventory(2.0).set_final_inventory(1.0).add_transfer_out(
            "unpaired", "receiver", 1.0
        )
        receiver = bt.BalanceLedger("receiver", bt.BalanceUnit.MOLE)
        receiver.set_initial_inventory(0.0).set_final_inventory(0.0)
        with self.assertRaises(ValueError):
            bt.reconcile_balances([donor, receiver])

        unknown = bt.BalanceLedger("known", bt.BalanceUnit.MOLE)
        unknown.set_initial_inventory(1.0).set_final_inventory(0.0).add_transfer_out(
            "orphan", "not supplied", 1.0
        )
        with self.assertRaises(ValueError):
            bt.reconcile_balances([unknown])

        second = bt.BalanceLedger("second", bt.BalanceUnit.MOLE)
        second.set_initial_inventory(2.0).set_final_inventory(1.0).add_transfer_out(
            "unpaired", "receiver", 1.0
        )
        receiver.add_transfer_in("unpaired", "donor", 1.0).set_final_inventory(1.0)
        with self.assertRaises(ValueError):
            bt.reconcile_balances([donor, second, receiver])

    def test_magnitude_endpoint_and_dimension_mismatches_fail(self):
        donor = bt.BalanceLedger("donor", bt.BalanceUnit.MOLE)
        donor.set_initial_inventory(2.0).set_final_inventory(1.0).add_transfer_out(
            "handoff", "receiver", 1.0
        )
        receiver = bt.BalanceLedger("receiver", bt.BalanceUnit.MILLIMOLE)
        receiver.set_initial_inventory(0.0).set_final_inventory(900.0).add_transfer_in(
            "handoff", "donor", 900.0
        )
        with self.assertRaises(ValueError):
            bt.reconcile_balances([donor, receiver])

        amount = bt.BalanceLedger("amount", bt.BalanceUnit.MOLE)
        with self.assertRaises(ValueError):
            amount.add_transfer_out("wrong unit", "energy", 1.0, bt.BalanceUnit.JOULE)
        with self.assertRaises(ValueError):
            bt.convert_balance_value(1.0, bt.BalanceUnit.LITER, bt.BalanceUnit.MOLE)

        amount.set_initial_inventory(1.0).set_final_inventory(0.0).add_transfer_out(
            "cross dimension", "energy", 1.0
        )
        energy = bt.BalanceLedger("energy", bt.BalanceUnit.JOULE)
        energy.set_initial_inventory(0.0).set_final_inventory(1.0).add_transfer_in(
            "cross dimension", "amount", 1.0
        )
        with self.assertRaises(ValueError):
            bt.reconcile_balances([amount, energy])

    def test_invalid_inputs_and_duplicate_terms_fail_before_audit(self):
        with self.assertRaises(ValueError):
            bt.BalanceLedger("   ", bt.BalanceUnit.LITER)

        ledger = bt.BalanceLedger("validated", bt.BalanceUnit.JOULE)
        with self.assertRaises(ValueError):
            ledger.set_initial_inventory(-1.0)
        with self.assertRaises(ValueError):
            ledger.add_boundary_in("nan", math.nan)
        with self.assertRaises(ValueError):
            ledger.add_generated("infinite", math.inf)
        with self.assertRaises(ValueError):
            ledger.add_transfer_out("negative transfer", "peer", -1.0)

        ledger.set_initial_inventory(0.0).add_boundary_in("heater", 1.0)
        with self.assertRaises(ValueError):
            ledger.add_boundary_in("heater", 1.0)
        with self.assertRaises(RuntimeError):
            ledger.audit()
        with self.assertRaises(ValueError):
            bt.reconcile_balances([])

        complete = bt.BalanceLedger("complete", bt.BalanceUnit.JOULE)
        complete.set_initial_inventory(0.0).set_final_inventory(0.0)
        with self.assertRaises(ValueError):
            bt.reconcile_balances([complete], relative_transfer_tolerance=math.nan)

    def test_positive_residual_means_unexplained_accumulation(self):
        ledger = bt.BalanceLedger("open balance", bt.BalanceUnit.LITER)
        ledger.set_initial_inventory(10.0).set_final_inventory(10.4).add_boundary_in(
            "measured inlet", 0.25
        )
        audit = ledger.audit()
        self.assertAlmostEqual(audit.expected_change, 0.25)
        self.assertAlmostEqual(audit.observed_change, 0.4)
        self.assertAlmostEqual(audit.closure_residual, 0.15)
        self.assertFalse(audit.is_closed(0.1))
        self.assertTrue(audit.is_closed(0.2))


if __name__ == "__main__":
    unittest.main()
