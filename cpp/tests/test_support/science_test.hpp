#ifndef BIOTRANSPORT_TESTS_TEST_SUPPORT_SCIENCE_TEST_HPP
#define BIOTRANSPORT_TESTS_TEST_SUPPORT_SCIENCE_TEST_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace science_test {

class Failure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline std::string number(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

[[noreturn]] inline void fail(const char* expression, const char* file, int line,
                              const std::string& message) {
    std::ostringstream stream;
    stream << file << ':' << line << ": requirement `" << expression << "` failed";
    if (!message.empty()) {
        stream << ": " << message;
    }
    throw Failure(stream.str());
}

inline void require(bool condition, const char* expression, const char* file, int line,
                    const std::string& message) {
    if (!condition) {
        fail(expression, file, line, message);
    }
}

inline void requireFinite(double value, const char* expression, const char* file, int line,
                          const std::string& quantity) {
    if (!std::isfinite(value)) {
        fail(expression, file, line, quantity + " must be finite; actual=" + number(value));
    }
}

inline void requireNear(double actual, double expected, double absolute_tolerance,
                        double relative_tolerance, const char* expression, const char* file,
                        int line, const std::string& quantity) {
    requireFinite(actual, expression, file, line, quantity);
    requireFinite(expected, expression, file, line, quantity + " reference");

    const double error = std::abs(actual - expected);
    const double allowed = std::max(absolute_tolerance, relative_tolerance * std::abs(expected));
    if (error > allowed) {
        fail(expression, file, line,
             quantity + " outside tolerance; actual=" + number(actual) +
                 ", expected=" + number(expected) + ", abs_error=" + number(error) +
                 ", allowed=" + number(allowed));
    }
}

inline void report(const std::string& quantity, double value, const std::string& units = {}) {
    std::cout << "    " << quantity << " = " << std::setprecision(10) << value;
    if (!units.empty()) {
        std::cout << ' ' << units;
    }
    std::cout << '\n';
}

struct Case {
    const char* name;
    void (*body)();
};

inline int runSuite(const char* suite_name, std::initializer_list<Case> cases) noexcept {
    std::cout << "[science] " << suite_name << '\n';

    std::size_t failures = 0;
    for (const auto& test_case : cases) {
        try {
            test_case.body();
            std::cout << "  [PASS] " << test_case.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "  [FAIL] " << test_case.name << "\n    " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "  [FAIL] " << test_case.name << "\n    unknown non-standard exception\n";
        }
    }

    if (failures == 0) {
        std::cout << "[science] all " << cases.size() << " checks passed\n";
        return 0;
    }

    std::cerr << "[science] " << failures << " of " << cases.size() << " checks failed\n";
    return 1;
}

}  // namespace science_test

#define SCIENCE_REQUIRE(condition, message) \
    ::science_test::require(static_cast<bool>(condition), #condition, __FILE__, __LINE__, message)

#define SCIENCE_REQUIRE_FINITE(value, quantity) \
    ::science_test::requireFinite((value), #value, __FILE__, __LINE__, quantity)

#define SCIENCE_REQUIRE_NEAR(actual, expected, absolute_tolerance, relative_tolerance, quantity)  \
    ::science_test::requireNear((actual), (expected), (absolute_tolerance), (relative_tolerance), \
                                #actual, __FILE__, __LINE__, quantity)

#endif  // BIOTRANSPORT_TESTS_TEST_SUPPORT_SCIENCE_TEST_HPP
