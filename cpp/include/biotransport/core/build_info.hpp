#ifndef BIOTRANSPORT_CORE_BUILD_INFO_HPP
#define BIOTRANSPORT_CORE_BUILD_INFO_HPP

/**
 * @file build_info.hpp
 * @brief Path-free compile-time provenance for reproducible result manifests.
 *
 * This header intentionally exposes compiler and feature identities without
 * embedding source directories, compiler executable paths, usernames, hostnames,
 * command lines, or environment variables.
 */

#include <string>

#ifdef BIOTRANSPORT_ENABLE_EIGEN
#include <Eigen/Core>
#endif

namespace biotransport {
namespace build {

struct NativeBuildInfo {
    std::string compiler_id;
    std::string compiler_version;
    long cpp_standard = 0;
    std::string cpp_standard_name;
    bool eigen_enabled = false;
    std::string eigen_version;
    bool openmp_compile_definition = false;
    bool openmp_enabled = false;
    int openmp_specification_date = 0;
    bool assertions_enabled = true;
};

namespace detail {

inline std::string compilerId() {
#if defined(__INTEL_LLVM_COMPILER)
    return "IntelLLVM";
#elif defined(__INTEL_COMPILER)
    return "Intel";
#elif defined(__clang__) && defined(_MSC_VER)
    return "Clang-cl";
#elif defined(__apple_build_version__) && defined(__clang__)
    return "AppleClang";
#elif defined(__clang__)
    return "Clang";
#elif defined(_MSC_VER)
    return "MSVC";
#elif defined(__GNUC__)
    return "GNU";
#else
    return "unknown";
#endif
}

inline std::string numericVersion(int major, int minor, int patch) {
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

inline std::string compilerVersion() {
#if defined(__INTEL_LLVM_COMPILER)
    return std::to_string(__INTEL_LLVM_COMPILER);
#elif defined(__INTEL_COMPILER)
#ifdef __INTEL_COMPILER_UPDATE
    return std::to_string(__INTEL_COMPILER) + "." + std::to_string(__INTEL_COMPILER_UPDATE);
#else
    return std::to_string(__INTEL_COMPILER);
#endif
#elif defined(__clang__)
    return numericVersion(__clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(_MSC_FULL_VER)
    return numericVersion(_MSC_VER / 100, _MSC_VER % 100, static_cast<int>(_MSC_FULL_VER % 100000));
#elif defined(_MSC_VER)
    return std::to_string(_MSC_VER);
#elif defined(__GNUC__)
    return numericVersion(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    return "unknown";
#endif
}

constexpr long cppStandard() {
#ifdef _MSVC_LANG
    return _MSVC_LANG;
#else
    return __cplusplus;
#endif
}

inline std::string cppStandardName(long value) {
    if (value >= 202302L)
        return "C++23";
    if (value >= 202002L)
        return "C++20";
    if (value >= 201703L)
        return "C++17";
    if (value >= 201402L)
        return "C++14";
    if (value >= 201103L)
        return "C++11";
    return "pre-C++11 or compiler macro unavailable";
}

}  // namespace detail

/**
 * @brief Return metadata for the native code that compiled the current caller.
 *
 * Because this is header-only, the Python metadata adapter reports the exact
 * compiler and feature definitions used for the loaded extension module.
 */
inline NativeBuildInfo nativeBuildInfo() {
    NativeBuildInfo info;
    info.compiler_id = detail::compilerId();
    info.compiler_version = detail::compilerVersion();
    info.cpp_standard = detail::cppStandard();
    info.cpp_standard_name = detail::cppStandardName(info.cpp_standard);

#ifdef BIOTRANSPORT_ENABLE_EIGEN
    info.eigen_enabled = true;
    info.eigen_version =
        detail::numericVersion(EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif

#ifdef BIOTRANSPORT_ENABLE_OPENMP
    info.openmp_compile_definition = true;
#endif

#if defined(BIOTRANSPORT_ENABLE_OPENMP) && defined(_OPENMP)
    info.openmp_enabled = true;
    info.openmp_specification_date = _OPENMP;
#endif

#ifdef NDEBUG
    info.assertions_enabled = false;
#endif
    return info;
}

}  // namespace build
}  // namespace biotransport

#endif  // BIOTRANSPORT_CORE_BUILD_INFO_HPP
