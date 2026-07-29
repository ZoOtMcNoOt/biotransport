#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace biotransport {

/** Physical dimensions supported by the balance-accounting API. */
enum class BalanceDimension { Amount = 0, Energy = 1, Volume = 2 };

/** Explicit units accepted by balance ledgers and inter-model transfers. */
enum class BalanceUnit {
    Mole,
    Millimole,
    Micromole,
    Joule,
    Kilojoule,
    CubicMeter,
    Liter,
    Milliliter
};

enum class BalanceTransferDirection { Incoming, Outgoing };

inline const char* balanceDimensionName(BalanceDimension dimension) {
    switch (dimension) {
        case BalanceDimension::Amount:
            return "amount";
        case BalanceDimension::Energy:
            return "energy";
        case BalanceDimension::Volume:
            return "volume";
    }
    throw std::invalid_argument("unknown balance dimension");
}

inline BalanceDimension balanceDimension(BalanceUnit unit) {
    switch (unit) {
        case BalanceUnit::Mole:
        case BalanceUnit::Millimole:
        case BalanceUnit::Micromole:
            return BalanceDimension::Amount;
        case BalanceUnit::Joule:
        case BalanceUnit::Kilojoule:
            return BalanceDimension::Energy;
        case BalanceUnit::CubicMeter:
        case BalanceUnit::Liter:
        case BalanceUnit::Milliliter:
            return BalanceDimension::Volume;
    }
    throw std::invalid_argument("unknown balance unit");
}

inline const char* balanceUnitSymbol(BalanceUnit unit) {
    switch (unit) {
        case BalanceUnit::Mole:
            return "mol";
        case BalanceUnit::Millimole:
            return "mmol";
        case BalanceUnit::Micromole:
            return "umol";
        case BalanceUnit::Joule:
            return "J";
        case BalanceUnit::Kilojoule:
            return "kJ";
        case BalanceUnit::CubicMeter:
            return "m^3";
        case BalanceUnit::Liter:
            return "L";
        case BalanceUnit::Milliliter:
            return "mL";
    }
    throw std::invalid_argument("unknown balance unit");
}

inline BalanceUnit baseBalanceUnit(BalanceDimension dimension) {
    switch (dimension) {
        case BalanceDimension::Amount:
            return BalanceUnit::Mole;
        case BalanceDimension::Energy:
            return BalanceUnit::Joule;
        case BalanceDimension::Volume:
            return BalanceUnit::CubicMeter;
    }
    throw std::invalid_argument("unknown balance dimension");
}

inline double balanceUnitScaleToBase(BalanceUnit unit) {
    switch (unit) {
        case BalanceUnit::Mole:
        case BalanceUnit::Joule:
        case BalanceUnit::CubicMeter:
            return 1.0;
        case BalanceUnit::Millimole:
            return 1.0e-3;
        case BalanceUnit::Micromole:
            return 1.0e-6;
        case BalanceUnit::Kilojoule:
            return 1.0e3;
        case BalanceUnit::Liter:
            return 1.0e-3;
        case BalanceUnit::Milliliter:
            return 1.0e-6;
    }
    throw std::invalid_argument("unknown balance unit");
}

namespace balance_detail {

inline void requireName(const std::string& name, const char* quantity) {
    const bool only_whitespace = std::all_of(name.begin(), name.end(), [](unsigned char character) {
        return std::isspace(character) != 0;
    });
    if (name.empty() || only_whitespace) {
        throw std::invalid_argument(std::string(quantity) + " must not be empty");
    }
}

inline void requireMagnitude(double value, const char* quantity) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(quantity) + " must be finite and non-negative");
    }
}

inline void requireTolerance(double value, const char* quantity) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(quantity) + " must be finite and non-negative");
    }
}

inline double finiteResult(double value, const char* quantity) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(std::string(quantity) + " is not representable as a double");
    }
    return value;
}

inline std::size_t dimensionIndex(BalanceDimension dimension) {
    switch (dimension) {
        case BalanceDimension::Amount:
            return 0;
        case BalanceDimension::Energy:
            return 1;
        case BalanceDimension::Volume:
            return 2;
    }
    throw std::invalid_argument("unknown balance dimension");
}

inline int decimalExponentToBase(BalanceUnit unit) {
    switch (unit) {
        case BalanceUnit::Mole:
        case BalanceUnit::Joule:
        case BalanceUnit::CubicMeter:
            return 0;
        case BalanceUnit::Millimole:
        case BalanceUnit::Liter:
            return -3;
        case BalanceUnit::Micromole:
        case BalanceUnit::Milliliter:
            return -6;
        case BalanceUnit::Kilojoule:
            return 3;
    }
    throw std::invalid_argument("unknown balance unit");
}

inline double convertCompatibleValue(double value, BalanceUnit from, BalanceUnit to) {
    if (balanceDimension(from) != balanceDimension(to)) {
        throw std::invalid_argument("cannot convert between incompatible balance dimensions");
    }

    // SI prefixes are exact decimal ratios. Apply their integer numerator or
    // denominator directly at the public binary64 precision. This avoids both
    // promoting a rounded approximation to 10^-3/10^-6 and double-rounding a
    // wider intermediate when returning the public double result.
    switch (decimalExponentToBase(from) - decimalExponentToBase(to)) {
        case -6:
            return value / 1000000.0;
        case -3:
            return value / 1000.0;
        case 0:
            return value;
        case 3:
            return value * 1000.0;
        case 6:
            return value * 1000000.0;
    }
    throw std::logic_error("unsupported balance-unit conversion ratio");
}

class CompensatedSum {
public:
    void add(double value) {
        if (!std::isfinite(value)) {
            throw std::overflow_error("balance accumulation received a non-finite value");
        }
        const double updated = sum_ + value;
        if (!std::isfinite(updated)) {
            throw std::overflow_error("balance accumulation is not representable as a double");
        }
        const double adjustment =
            std::abs(sum_) >= std::abs(value) ? (sum_ - updated) + value : (value - updated) + sum_;
        correction_ += adjustment;
        if (!std::isfinite(correction_)) {
            throw std::overflow_error("balance accumulation correction is not representable");
        }
        sum_ = updated;
    }

    double value(const char* quantity) const {
        const double result = sum_ + correction_;
        if (!std::isfinite(result)) {
            throw std::overflow_error(std::string(quantity) + " is not representable as a double");
        }
        return result;
    }

private:
    double sum_ = 0.0;
    double correction_ = 0.0;
};

}  // namespace balance_detail

/** Convert a finite signed value between compatible balance units. */
inline double convertBalanceValue(double value, BalanceUnit from, BalanceUnit to) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("balance value must be finite");
    }
    const double converted = balance_detail::convertCompatibleValue(value, from, to);
    if (!std::isfinite(converted)) {
        throw std::overflow_error("converted balance value is not representable as a double");
    }
    return converted;
}

struct BalanceTerm {
    std::string name;
    double magnitude = 0.0;
    BalanceUnit unit = BalanceUnit::Mole;
};

struct BalanceTransfer {
    std::string id;
    std::string counterparty;
    double magnitude = 0.0;
    BalanceUnit unit = BalanceUnit::Mole;
    BalanceTransferDirection direction = BalanceTransferDirection::Incoming;
};

/** Audited scalar budget in the ledger's declared unit.
 *
 * The sign convention is
 * expected_change = boundary_in - boundary_out + generated - consumed
 *                 + transfer_in - transfer_out
 * and closure_residual = observed_change - expected_change. A positive residual therefore means
 * unexplained accumulation.
 */
struct BalanceAudit {
    std::string ledger_name;
    BalanceDimension dimension = BalanceDimension::Amount;
    BalanceUnit unit = BalanceUnit::Mole;
    double initial_inventory = 0.0;
    double final_inventory = 0.0;
    double observed_change = 0.0;
    double boundary_in = 0.0;
    double boundary_out = 0.0;
    double generated = 0.0;
    double consumed = 0.0;
    double transfer_in = 0.0;
    double transfer_out = 0.0;
    double expected_change = 0.0;
    double closure_residual = 0.0;

    bool isClosed(double absolute_tolerance) const {
        balance_detail::requireTolerance(absolute_tolerance, "closure tolerance");
        return std::abs(closure_residual) <= absolute_tolerance;
    }
};

/** One model or control-volume ledger for a single conserved dimension. */
class BalanceLedger {
public:
    BalanceLedger(std::string name, BalanceUnit unit) : name_(std::move(name)), unit_(unit) {
        balance_detail::requireName(name_, "ledger name");
        (void)balanceDimension(unit_);
    }

    const std::string& name() const noexcept { return name_; }
    BalanceUnit unit() const noexcept { return unit_; }
    BalanceDimension dimension() const { return balanceDimension(unit_); }

    bool hasInitialInventory() const noexcept { return initial_inventory_.has_value(); }
    bool hasFinalInventory() const noexcept { return final_inventory_.has_value(); }

    double initialInventory() const {
        if (!initial_inventory_) {
            throw std::logic_error("initial inventory has not been set for ledger '" + name_ + "'");
        }
        return *initial_inventory_;
    }

    double finalInventory() const {
        if (!final_inventory_) {
            throw std::logic_error("final inventory has not been set for ledger '" + name_ + "'");
        }
        return *final_inventory_;
    }

    BalanceLedger& setInitialInventory(double magnitude) {
        balance_detail::requireMagnitude(magnitude, "initial inventory");
        initial_inventory_ = magnitude;
        return *this;
    }

    BalanceLedger& setFinalInventory(double magnitude) {
        balance_detail::requireMagnitude(magnitude, "final inventory");
        final_inventory_ = magnitude;
        return *this;
    }

    BalanceLedger& addBoundaryIn(std::string name, double magnitude) {
        addTerm(boundary_in_, std::move(name), magnitude, "boundary-in term");
        return *this;
    }

    BalanceLedger& addBoundaryOut(std::string name, double magnitude) {
        addTerm(boundary_out_, std::move(name), magnitude, "boundary-out term");
        return *this;
    }

    BalanceLedger& addGenerated(std::string name, double magnitude) {
        addTerm(generated_, std::move(name), magnitude, "generation term");
        return *this;
    }

    BalanceLedger& addConsumed(std::string name, double magnitude) {
        addTerm(consumed_, std::move(name), magnitude, "consumption term");
        return *this;
    }

    BalanceLedger& addTransferIn(std::string id, std::string sender, double magnitude) {
        return addTransfer(std::move(id), std::move(sender), magnitude, unit_,
                           BalanceTransferDirection::Incoming);
    }

    BalanceLedger& addTransferIn(std::string id, std::string sender, double magnitude,
                                 BalanceUnit unit) {
        return addTransfer(std::move(id), std::move(sender), magnitude, unit,
                           BalanceTransferDirection::Incoming);
    }

    BalanceLedger& addTransferOut(std::string id, std::string receiver, double magnitude) {
        return addTransfer(std::move(id), std::move(receiver), magnitude, unit_,
                           BalanceTransferDirection::Outgoing);
    }

    BalanceLedger& addTransferOut(std::string id, std::string receiver, double magnitude,
                                  BalanceUnit unit) {
        return addTransfer(std::move(id), std::move(receiver), magnitude, unit,
                           BalanceTransferDirection::Outgoing);
    }

    const std::vector<BalanceTerm>& boundaryInTerms() const noexcept { return boundary_in_; }
    const std::vector<BalanceTerm>& boundaryOutTerms() const noexcept { return boundary_out_; }
    const std::vector<BalanceTerm>& generatedTerms() const noexcept { return generated_; }
    const std::vector<BalanceTerm>& consumedTerms() const noexcept { return consumed_; }
    const std::vector<BalanceTransfer>& transfers() const noexcept { return transfers_; }

    BalanceAudit audit() const {
        const double initial = initialInventory();
        const double final = finalInventory();
        const double boundary_in = sumTerms(boundary_in_, "boundary-in total");
        const double boundary_out = sumTerms(boundary_out_, "boundary-out total");
        const double generated = sumTerms(generated_, "generation total");
        const double consumed = sumTerms(consumed_, "consumption total");

        balance_detail::CompensatedSum transfer_in;
        balance_detail::CompensatedSum transfer_out;
        balance_detail::CompensatedSum expected_change;
        for (const auto& term : boundary_in_) {
            expected_change.add(term.magnitude);
        }
        for (const auto& term : boundary_out_) {
            expected_change.add(-term.magnitude);
        }
        for (const auto& term : generated_) {
            expected_change.add(term.magnitude);
        }
        for (const auto& term : consumed_) {
            expected_change.add(-term.magnitude);
        }
        for (const auto& transfer : transfers_) {
            const double local = convertBalanceValue(transfer.magnitude, transfer.unit, unit_);
            if (transfer.direction == BalanceTransferDirection::Incoming) {
                transfer_in.add(local);
                expected_change.add(local);
            } else {
                transfer_out.add(local);
                expected_change.add(-local);
            }
        }

        BalanceAudit result;
        result.ledger_name = name_;
        result.dimension = dimension();
        result.unit = unit_;
        result.initial_inventory = initial;
        result.final_inventory = final;
        result.observed_change =
            balance_detail::finiteResult(final - initial, "observed inventory change");
        result.boundary_in = boundary_in;
        result.boundary_out = boundary_out;
        result.generated = generated;
        result.consumed = consumed;
        result.transfer_in = transfer_in.value("incoming-transfer total");
        result.transfer_out = transfer_out.value("outgoing-transfer total");
        result.expected_change = expected_change.value("expected inventory change");
        result.closure_residual = balance_detail::finiteResult(
            result.observed_change - result.expected_change, "balance closure residual");
        return result;
    }

private:
    void addTerm(std::vector<BalanceTerm>& terms, std::string name, double magnitude,
                 const char* category) {
        balance_detail::requireName(name, category);
        balance_detail::requireMagnitude(magnitude, category);
        const auto duplicate = std::find_if(terms.begin(), terms.end(),
                                            [&](const auto& term) { return term.name == name; });
        if (duplicate != terms.end()) {
            throw std::invalid_argument(std::string(category) + " '" + name +
                                        "' is already recorded in ledger '" + name_ + "'");
        }
        terms.push_back({std::move(name), magnitude, unit_});
    }

    BalanceLedger& addTransfer(std::string id, std::string counterparty, double magnitude,
                               BalanceUnit unit, BalanceTransferDirection direction) {
        balance_detail::requireName(id, "transfer ID");
        balance_detail::requireName(counterparty, "transfer counterparty");
        balance_detail::requireMagnitude(magnitude, "transfer magnitude");
        if (counterparty == name_) {
            throw std::invalid_argument(
                "a balance transfer cannot name its own ledger as the "
                "counterparty");
        }
        if (balanceDimension(unit) != dimension()) {
            throw std::invalid_argument("transfer unit '" + std::string(balanceUnitSymbol(unit)) +
                                        "' is incompatible with " +
                                        balanceDimensionName(dimension()) + " ledger '" + name_ +
                                        "'");
        }
        const auto duplicate =
            std::find_if(transfers_.begin(), transfers_.end(),
                         [&](const auto& transfer) { return transfer.id == id; });
        if (duplicate != transfers_.end()) {
            throw std::invalid_argument("transfer ID '" + id + "' is already recorded in ledger '" +
                                        name_ + "'");
        }
        transfers_.push_back({std::move(id), std::move(counterparty), magnitude, unit, direction});
        return *this;
    }

    static double sumTerms(const std::vector<BalanceTerm>& terms, const char* quantity) {
        balance_detail::CompensatedSum sum;
        for (const auto& term : terms) {
            sum.add(term.magnitude);
        }
        return sum.value(quantity);
    }

    std::string name_;
    BalanceUnit unit_;
    std::optional<double> initial_inventory_;
    std::optional<double> final_inventory_;
    std::vector<BalanceTerm> boundary_in_;
    std::vector<BalanceTerm> boundary_out_;
    std::vector<BalanceTerm> generated_;
    std::vector<BalanceTerm> consumed_;
    std::vector<BalanceTransfer> transfers_;
};

struct MatchedBalanceTransfer {
    std::string id;
    std::string sender;
    std::string receiver;
    BalanceDimension dimension = BalanceDimension::Amount;
    BalanceUnit base_unit = BalanceUnit::Mole;
    double magnitude_base = 0.0;
};

/** Aggregate budget for one dimension, expressed in its SI base unit. */
struct DimensionBalanceAudit {
    BalanceDimension dimension = BalanceDimension::Amount;
    BalanceUnit base_unit = BalanceUnit::Mole;
    double observed_change = 0.0;
    double external_expected_change = 0.0;
    double internal_transfer_net = 0.0;
    double closure_residual = 0.0;
    double representation_adjustment = 0.0;
};

struct BalanceReconciliation {
    std::vector<BalanceAudit> ledgers;
    std::vector<MatchedBalanceTransfer> matched_transfers;
    std::vector<DimensionBalanceAudit> dimensions;

    bool isClosed(double amount_absolute_tolerance = 0.0, double energy_absolute_tolerance = 0.0,
                  double volume_absolute_tolerance = 0.0) const {
        balance_detail::requireTolerance(amount_absolute_tolerance, "amount closure tolerance");
        balance_detail::requireTolerance(energy_absolute_tolerance, "energy closure tolerance");
        balance_detail::requireTolerance(volume_absolute_tolerance, "volume closure tolerance");
        const std::array<double, 3> tolerances{amount_absolute_tolerance, energy_absolute_tolerance,
                                               volume_absolute_tolerance};
        return std::all_of(dimensions.begin(), dimensions.end(), [&](const auto& audit) {
            return std::abs(audit.closure_residual) <=
                   tolerances[balance_detail::dimensionIndex(audit.dimension)];
        });
    }
};

/**
 * Validate and reconcile multiple model ledgers.
 *
 * Every transfer ID must occur exactly once as outgoing and exactly once as incoming, with the
 * same sender, receiver, dimension, and physical magnitude. Validated internal transfers are
 * omitted from the aggregate external expected change; their net after local-unit binary64
 * conversion is reported separately. Any difference between the converted complete expectation and
 * its external/internal decomposition is reported as a representation adjustment. Absolute transfer
 * tolerance is expressed in the SI base unit for the transfer's dimension.
 */
inline BalanceReconciliation reconcileBalances(const std::vector<BalanceLedger>& ledgers,
                                               double relative_transfer_tolerance = 1.0e-12,
                                               double absolute_transfer_tolerance_base = 0.0) {
    balance_detail::requireTolerance(relative_transfer_tolerance, "relative transfer tolerance");
    balance_detail::requireTolerance(absolute_transfer_tolerance_base,
                                     "absolute transfer tolerance");
    if (ledgers.empty()) {
        throw std::invalid_argument("at least one balance ledger is required");
    }

    std::map<std::string, const BalanceLedger*> by_name;
    for (const auto& ledger : ledgers) {
        if (!by_name.emplace(ledger.name(), &ledger).second) {
            throw std::invalid_argument("duplicate balance ledger name '" + ledger.name() + "'");
        }
    }

    struct TransferEndpoint {
        const BalanceLedger* ledger = nullptr;
        const BalanceTransfer* transfer = nullptr;
        std::string sender;
        std::string receiver;
    };
    struct TransferPair {
        std::optional<TransferEndpoint> outgoing;
        std::optional<TransferEndpoint> incoming;
    };

    std::map<std::string, TransferPair> pairs;
    for (const auto& ledger : ledgers) {
        for (const auto& transfer : ledger.transfers()) {
            if (by_name.count(transfer.counterparty) == 0) {
                throw std::invalid_argument("transfer '" + transfer.id + "' in ledger '" +
                                            ledger.name() + "' names unknown counterparty '" +
                                            transfer.counterparty + "'");
            }
            TransferEndpoint endpoint;
            endpoint.ledger = &ledger;
            endpoint.transfer = &transfer;
            if (transfer.direction == BalanceTransferDirection::Outgoing) {
                endpoint.sender = ledger.name();
                endpoint.receiver = transfer.counterparty;
                auto& slot = pairs[transfer.id].outgoing;
                if (slot) {
                    throw std::invalid_argument("transfer ID '" + transfer.id +
                                                "' has been double-counted as outgoing");
                }
                slot = std::move(endpoint);
            } else {
                endpoint.sender = transfer.counterparty;
                endpoint.receiver = ledger.name();
                auto& slot = pairs[transfer.id].incoming;
                if (slot) {
                    throw std::invalid_argument("transfer ID '" + transfer.id +
                                                "' has been double-counted as incoming");
                }
                slot = std::move(endpoint);
            }
        }
    }

    BalanceReconciliation result;
    result.ledgers.reserve(ledgers.size());
    for (const auto& ledger : ledgers) {
        result.ledgers.push_back(ledger.audit());
    }

    result.matched_transfers.reserve(pairs.size());
    for (const auto& entry : pairs) {
        const std::string& id = entry.first;
        const TransferPair& pair = entry.second;
        if (!pair.outgoing || !pair.incoming) {
            throw std::invalid_argument("transfer ID '" + id +
                                        "' is unmatched; record exactly one outgoing and one "
                                        "incoming entry");
        }
        if (pair.outgoing->sender != pair.incoming->sender ||
            pair.outgoing->receiver != pair.incoming->receiver) {
            throw std::invalid_argument("transfer ID '" + id +
                                        "' has inconsistent sender or receiver ledgers");
        }

        const BalanceDimension sender_dimension = pair.outgoing->ledger->dimension();
        const BalanceDimension receiver_dimension = pair.incoming->ledger->dimension();
        if (sender_dimension != receiver_dimension) {
            throw std::invalid_argument("transfer ID '" + id +
                                        "' crosses incompatible balance dimensions");
        }

        const BalanceTransfer& outgoing = *pair.outgoing->transfer;
        const BalanceTransfer& incoming = *pair.incoming->transfer;
        const BalanceUnit base_unit = baseBalanceUnit(sender_dimension);
        const double outgoing_base =
            convertBalanceValue(outgoing.magnitude, outgoing.unit, base_unit);
        const double incoming_base =
            convertBalanceValue(incoming.magnitude, incoming.unit, base_unit);
        const double error = std::abs(outgoing_base - incoming_base);
        const double scale = std::max(std::abs(outgoing_base), std::abs(incoming_base));
        const double allowed =
            std::max(absolute_transfer_tolerance_base, relative_transfer_tolerance * scale);
        if (error > allowed) {
            throw std::invalid_argument("transfer ID '" + id +
                                        "' has inconsistent incoming and outgoing magnitudes");
        }

        result.matched_transfers.push_back({id, pair.outgoing->sender, pair.outgoing->receiver,
                                            sender_dimension, base_unit,
                                            outgoing_base + (incoming_base - outgoing_base) / 2.0});
    }

    struct DimensionAccumulator {
        bool present = false;
        balance_detail::CompensatedSum observed;
        balance_detail::CompensatedSum expected;
        balance_detail::CompensatedSum external_expected;
        balance_detail::CompensatedSum internal_transfer_net;
    };
    std::array<DimensionAccumulator, 3> accumulators{};
    for (std::size_t index = 0; index < result.ledgers.size(); ++index) {
        const auto& ledger = ledgers[index];
        const auto& audit = result.ledgers[index];
        auto& accumulator = accumulators[balance_detail::dimensionIndex(audit.dimension)];
        accumulator.present = true;
        const BalanceUnit base_unit = baseBalanceUnit(audit.dimension);
        // Every public balance value is binary64. Round each unit conversion
        // through that same boundary before compensated accumulation so
        // equivalent values (for example, 0.1 mol and 100 mmol) cancel on
        // every platform.
        accumulator.observed.add(convertBalanceValue(audit.observed_change, audit.unit, base_unit));
        accumulator.expected.add(convertBalanceValue(audit.expected_change, audit.unit, base_unit));
        balance_detail::CompensatedSum ledger_external_expected;
        for (const auto& term : ledger.boundaryInTerms()) {
            ledger_external_expected.add(term.magnitude);
        }
        for (const auto& term : ledger.boundaryOutTerms()) {
            ledger_external_expected.add(-term.magnitude);
        }
        for (const auto& term : ledger.generatedTerms()) {
            ledger_external_expected.add(term.magnitude);
        }
        for (const auto& term : ledger.consumedTerms()) {
            ledger_external_expected.add(-term.magnitude);
        }
        accumulator.external_expected.add(
            convertBalanceValue(ledger_external_expected.value("ledger external expected change"),
                                audit.unit, base_unit));
        accumulator.internal_transfer_net.add(
            convertBalanceValue(audit.transfer_in, audit.unit, base_unit));
        accumulator.internal_transfer_net.add(
            -convertBalanceValue(audit.transfer_out, audit.unit, base_unit));
    }

    const std::array<BalanceDimension, 3> dimensions{
        BalanceDimension::Amount, BalanceDimension::Energy, BalanceDimension::Volume};
    for (BalanceDimension dimension : dimensions) {
        const auto& accumulator = accumulators[balance_detail::dimensionIndex(dimension)];
        if (!accumulator.present) {
            continue;
        }
        DimensionBalanceAudit audit;
        audit.dimension = dimension;
        audit.base_unit = baseBalanceUnit(dimension);
        audit.observed_change = accumulator.observed.value("aggregate observed change");
        const double expected_change = accumulator.expected.value("aggregate expected change");
        audit.external_expected_change =
            accumulator.external_expected.value("aggregate external expected change");
        audit.internal_transfer_net =
            accumulator.internal_transfer_net.value("aggregate internal transfer net");
        balance_detail::CompensatedSum representation_adjustment;
        representation_adjustment.add(expected_change);
        representation_adjustment.add(-audit.external_expected_change);
        representation_adjustment.add(-audit.internal_transfer_net);
        audit.representation_adjustment =
            representation_adjustment.value("aggregate representation adjustment");
        balance_detail::CompensatedSum closure_residual;
        closure_residual.add(audit.observed_change);
        closure_residual.add(-expected_change);
        audit.closure_residual = closure_residual.value("aggregate balance closure residual");
        result.dimensions.push_back(audit);
    }

    return result;
}

}  // namespace biotransport
