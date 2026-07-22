#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
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

inline double narrowFinite(long double value, const char* quantity) {
    if (!std::isfinite(value) ||
        std::abs(value) > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::overflow_error(std::string(quantity) + " is not representable as a double");
    }
    return static_cast<double>(value);
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

}  // namespace balance_detail

/** Convert a finite signed value between compatible balance units. */
inline double convertBalanceValue(double value, BalanceUnit from, BalanceUnit to) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("balance value must be finite");
    }
    if (balanceDimension(from) != balanceDimension(to)) {
        throw std::invalid_argument("cannot convert between incompatible balance dimensions");
    }
    const long double converted =
        static_cast<long double>(value) * balanceUnitScaleToBase(from) / balanceUnitScaleToBase(to);
    return balance_detail::narrowFinite(converted, "converted balance value");
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

        long double transfer_in = 0.0L;
        long double transfer_out = 0.0L;
        for (const auto& transfer : transfers_) {
            const double local = convertBalanceValue(transfer.magnitude, transfer.unit, unit_);
            if (transfer.direction == BalanceTransferDirection::Incoming) {
                transfer_in += local;
            } else {
                transfer_out += local;
            }
        }

        BalanceAudit result;
        result.ledger_name = name_;
        result.dimension = dimension();
        result.unit = unit_;
        result.initial_inventory = initial;
        result.final_inventory = final;
        result.observed_change = balance_detail::narrowFinite(
            static_cast<long double>(final) - initial, "observed inventory change");
        result.boundary_in = boundary_in;
        result.boundary_out = boundary_out;
        result.generated = generated;
        result.consumed = consumed;
        result.transfer_in = balance_detail::narrowFinite(transfer_in, "incoming-transfer total");
        result.transfer_out = balance_detail::narrowFinite(transfer_out, "outgoing-transfer total");
        result.expected_change =
            balance_detail::narrowFinite(static_cast<long double>(boundary_in) - boundary_out +
                                             generated - consumed + transfer_in - transfer_out,
                                         "expected inventory change");
        result.closure_residual = balance_detail::narrowFinite(
            static_cast<long double>(result.observed_change) - result.expected_change,
            "balance closure residual");
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
        long double sum = 0.0L;
        for (const auto& term : terms) {
            sum += term.magnitude;
        }
        return balance_detail::narrowFinite(sum, quantity);
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
 * same sender, receiver, dimension, and physical magnitude. The aggregate expected change omits
 * validated internal transfers, so a transfer cannot be double-counted as an external source.
 * Absolute transfer tolerance is expressed in the SI base unit for the transfer's dimension.
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
        long double observed = 0.0L;
        long double external_expected = 0.0L;
    };
    std::array<DimensionAccumulator, 3> accumulators{};
    for (const auto& audit : result.ledgers) {
        auto& accumulator = accumulators[balance_detail::dimensionIndex(audit.dimension)];
        accumulator.present = true;
        const long double scale = balanceUnitScaleToBase(audit.unit);
        accumulator.observed += static_cast<long double>(audit.observed_change) * scale;
        accumulator.external_expected += (static_cast<long double>(audit.boundary_in) -
                                          audit.boundary_out + audit.generated - audit.consumed) *
                                         scale;
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
        audit.observed_change =
            balance_detail::narrowFinite(accumulator.observed, "aggregate observed change");
        audit.external_expected_change = balance_detail::narrowFinite(
            accumulator.external_expected, "aggregate external expected change");
        audit.internal_transfer_net = 0.0;
        audit.closure_residual =
            balance_detail::narrowFinite(accumulator.observed - accumulator.external_expected,
                                         "aggregate balance closure residual");
        result.dimensions.push_back(audit);
    }

    return result;
}

}  // namespace biotransport
