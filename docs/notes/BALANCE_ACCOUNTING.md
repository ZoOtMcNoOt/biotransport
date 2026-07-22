# Balance accounting

BioTransport's balance API audits inventories and exchanges reported by models. It is an
accounting layer: it does **not** couple PDEs, choose coupling algorithms, advance solvers, or infer
fluxes from solution fields.

## Convention

Each `BalanceLedger` represents one model or control volume and exactly one physical dimension:
amount of substance, energy, or volume. The ledger's unit applies to its inventories and named
boundary/source terms.

The signed budget is

```text
observed change = final inventory - initial inventory

expected change = boundary in - boundary out
                + generated - consumed
                + transfers in - transfers out

closure residual = observed change - expected change
```

A positive residual is unexplained accumulation; a negative residual is an unexplained loss.
Inputs such as inventories, flows integrated over the audit interval, generation, consumption, and
transfer amounts are non-negative magnitudes. Direction supplies their sign. All inputs must be
finite.

The supported explicit units are:

| Dimension | Units | Reconciliation base unit |
| --- | --- | --- |
| Amount of substance | mol, mmol, umol | mol |
| Energy | J, kJ | J |
| Volume | m^3, L, mL | m^3 |

Compatible units are converted when transfers are reconciled. Conversions or transfers between
different dimensions are rejected rather than silently reinterpreted.

## One ledger

Python:

```python
import biotransport as bt

solute = bt.BalanceLedger("reactor solute", bt.BalanceUnit.MILLIMOLE)
solute.set_initial_inventory(100.0).set_final_inventory(92.0)
solute.add_boundary_in("feed", 10.0)
solute.add_boundary_out("effluent", 5.0)
solute.add_generated("reaction production", 2.0)
solute.add_consumed("reaction uptake", 15.0)

audit = solute.audit()
assert audit.expected_change == -8.0
assert audit.closure_residual == 0.0
```

C++ uses the same model with camel-case methods:

```cpp
#include <biotransport/core/balance.hpp>

using namespace biotransport;

BalanceLedger solute("reactor solute", BalanceUnit::Millimole);
solute.setInitialInventory(100.0)
    .setFinalInventory(92.0)
    .addBoundaryIn("feed", 10.0)
    .addBoundaryOut("effluent", 5.0)
    .addGenerated("reaction production", 2.0)
    .addConsumed("reaction uptake", 15.0);

const BalanceAudit audit = solute.audit();
```

Initial and final inventories are required before `audit()`. Reusing a term name within the same
category is rejected to catch accidental double entry. An explicit absolute tolerance is required
when deciding whether a single ledger is closed; that tolerance is in the ledger's declared unit.

## Reconcile model-to-model transfers

Record a transfer on both participating ledgers with a globally unique transfer ID:

```python
donor = bt.BalanceLedger("donor", bt.BalanceUnit.MOLE)
donor.set_initial_inventory(10.0).set_final_inventory(8.0)
donor.add_transfer_out("solute handoff", "receiver", 2.0)

receiver = bt.BalanceLedger("receiver", bt.BalanceUnit.MILLIMOLE)
receiver.set_initial_inventory(1000.0).set_final_inventory(3000.0)
receiver.add_transfer_in("solute handoff", "donor", 2000.0)

coupled = bt.reconcile_balances([donor, receiver])
assert coupled.is_closed()
```

`reconcile_balances` requires every transfer ID to appear exactly once outgoing and exactly once
incoming. It also requires matching sender, receiver, dimension, and physical magnitude. It raises
on unknown counterparties, unmatched records, reused IDs, endpoint disagreements, magnitude
disagreements, and cross-dimension pairs.

After validation, each dimension is aggregated separately in its base unit. Internal transfers are
excluded from the aggregate external expected change, so they cancel once and cannot masquerade as
external production or loss. Boundary exchanges, generation, and consumption remain external terms.

The optional transfer tolerances compare the two recorded transfer magnitudes after conversion:

```python
bt.reconcile_balances(
    ledgers,
    relative_transfer_tolerance=1e-12,
    absolute_transfer_tolerance_base=0.0,
)
```

The absolute transfer tolerance is in mol, J, or m^3 according to the transfer dimension. Closure
tolerances passed to `BalanceReconciliation.is_closed` are likewise separate base-unit tolerances
for amount, energy, and volume.

## Scientific scope

The API provides traceable arithmetic and guards against common bookkeeping errors. A caller must
still integrate its solver fields, boundary fluxes, reactions, and sources over compatible spatial
and temporal domains before recording them. A closed ledger supports conservation verification; it
does not by itself establish numerical convergence, constitutive-model validity, or physical
coupling fidelity.
