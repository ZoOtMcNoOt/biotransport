#include "../test_support/science_test.hpp"
#include <array>
#include <biotransport/physics/fluid_dynamics/non_newtonian.hpp>
#include <biotransport/physics/fluid_dynamics/velocity_bc_applicator.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace biotransport;

namespace {

template <typename Exception, typename Callable>
void requireThrows(Callable&& callable, const std::string& context) {
    bool caught = false;
    try {
        callable();
    } catch (const Exception&) {
        caught = true;
    }
    SCIENCE_REQUIRE(caught, context);
}

void constitutiveParametersAreBounded() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();

    requireThrows<std::invalid_argument>([&] { NewtonianModel invalid(nan); },
                                         "NaN viscosity must be rejected");
    requireThrows<std::invalid_argument>([&] { PowerLawModel invalid(1.0, 0.5, 0.0); },
                                         "zero power-law cutoff must be rejected");
    requireThrows<std::invalid_argument>([&] { CarreauModel invalid(1.0, 0.1, 1.0, 1.1); },
                                         "unbounded Carreau index must be rejected");
    requireThrows<std::invalid_argument>(
        [&] { CarreauYasudaModel invalid(1.0, 2.0, 1.0, 2.0, 0.5); },
        "Carreau-Yasuda plateau ordering must be enforced");
    requireThrows<std::invalid_argument>([&] { CrossModel invalid(1.0, 0.001, 1.0, 2.0); },
                                         "non-monotone Cross parameters must be rejected");
    requireThrows<std::invalid_argument>([&] { BinghamModel invalid(0.1, infinity); },
                                         "infinite plastic viscosity must be rejected");
    requireThrows<std::invalid_argument>([&] { HerschelBulkleyModel invalid(0.1, 0.5, 0.8, nan); },
                                         "NaN regularization must be rejected");
    requireThrows<std::invalid_argument>([&] { CassonModel invalid(-0.1, 0.003); },
                                         "negative Casson yield stress must be rejected");

    const CarreauModel model(0.056, 0.00345, 3.313, 0.3568);
    requireThrows<std::invalid_argument>([&] { (void)model.viscosity(nan); },
                                         "non-finite shear rate must be rejected");

    const BinghamModel bingham(0.01, 0.004);
    requireThrows<std::invalid_argument>([&] { (void)bingham.binghamNumber(1.0, 0.0); },
                                         "zero characteristic speed must be rejected");

    const CrossModel monotone_cross(0.056, 0.00345, 1.007, 1.028);
    SCIENCE_REQUIRE_FINITE(monotone_cross.viscosity(100.0),
                           "a monotone literature-style Cross fit with m > 1");
}

void canonicalConstitutiveResponses() {
    const NewtonianModel newtonian(0.004);
    SCIENCE_REQUIRE_NEAR(newtonian.viscosity(123.0), 0.004, 0.0, 0.0,
                         "Newtonian dynamic viscosity");
    SCIENCE_REQUIRE_NEAR(newtonian.shearStress(-10.0), -0.04, 1e-15, 1e-15,
                         "signed Newtonian shear stress");

    const PowerLawModel power_law(0.02, 0.5, 1e-12);
    SCIENCE_REQUIRE_NEAR(power_law.viscosity(4.0), 0.01, 1e-15, 1e-15, "power-law viscosity");
    SCIENCE_REQUIRE_NEAR(power_law.shearStress(4.0), 0.04, 1e-15, 1e-15, "power-law shear stress");

    const CarreauModel carreau(0.056, 0.00345, 3.313, 0.3568);
    SCIENCE_REQUIRE_NEAR(carreau.viscosity(0.0), 0.056, 1e-15, 1e-15, "Carreau zero-shear plateau");
    SCIENCE_REQUIRE(std::abs(carreau.viscosity(1e12) - 0.00345) < 1e-8,
                    "Carreau viscosity must approach its high-shear plateau");

    const CassonModel casson(0.005, 0.003, 1e-6);
    const double positive_stress = casson.shearStress(10.0);
    SCIENCE_REQUIRE_NEAR(casson.shearStress(-10.0), -positive_stress, 1e-15, 1e-15,
                         "Casson response must be odd in signed shear rate");
    SCIENCE_REQUIRE_NEAR(casson.shearStress(0.0), 0.0, 0.0, 0.0,
                         "regularized Casson stress at zero shear");
    SCIENCE_REQUIRE_FINITE(casson.viscosity(0.0), "regularized zero-shear viscosity");
}

void bloodHelpersMatchDocumentedCorrelations() {
    const CassonModel casson = bloodCassonModel(0.45);
    SCIENCE_REQUIRE_NEAR(casson.yieldStress(), 0.9e-7 * 39.0 * 39.0 * 39.0, 1e-14, 1e-14,
                         "Merrill yield-stress parameter at H=45%");
    SCIENCE_REQUIRE_NEAR(casson.plasticViscosity(),
                         0.0012 * (1.0 + 0.025 * 45.0 + 7.35e-4 * 45.0 * 45.0), 1e-14, 1e-14,
                         "Merrill high-shear viscosity at H=45%");

    const CarreauModel carreau = bloodCarreauModel(0.45);
    SCIENCE_REQUIRE_NEAR(carreau.mu0(), 0.056, 1e-14, 1e-14,
                         "reference blood Carreau zero-shear viscosity");
    SCIENCE_REQUIRE_NEAR(carreau.muInf(), 0.00345, 1e-14, 1e-14,
                         "reference blood Carreau high-shear viscosity");

    SCIENCE_REQUIRE_FINITE(bloodCassonModel(0.60).viscosity(100.0), "upper-domain blood viscosity");
    requireThrows<std::invalid_argument>([&] { (void)bloodCassonModel(0.600001); },
                                         "hematocrit beyond 60% must fail loudly");
    requireThrows<std::invalid_argument>([&] { (void)bloodCassonModel(0.67); },
                                         "the former singular hematocrit must be rejected");
    requireThrows<std::invalid_argument>(
        [&] { (void)bloodCarreauModel(std::numeric_limits<double>::quiet_NaN()); },
        "NaN hematocrit must be rejected");
}

void pipeRheometryUsesModelSlope() {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double radius = 0.002;
    const double pressure_gradient = -1000.0;
    const double viscosity = 0.004;
    const double flow_rate =
        pi * std::pow(radius, 4.0) * std::abs(pressure_gradient) / (8.0 * viscosity);

    SCIENCE_REQUIRE_NEAR(pipeWallShearRate(-flow_rate, radius),
                         pipeWallShearRate(flow_rate, radius), 0.0, 0.0,
                         "nominal pipe shear-rate magnitude under reverse flow");
    SCIENCE_REQUIRE_NEAR(
        apparentViscosityPipe(NewtonianModel(viscosity), flow_rate, radius, pressure_gradient),
        viscosity, 1e-10, 1e-8, "Newtonian Poiseuille apparent viscosity");

    const PowerLawModel power_law(0.5, 0.5, 1e-14);
    const double wall_stress = 2.0;
    const double wall_rate = std::pow(wall_stress / power_law.K(), 1.0 / power_law.n());
    const double power_law_flow =
        pi * std::pow(radius, 3.0) * wall_rate / (3.0 + 1.0 / power_law.n());
    const double power_law_gradient = -2.0 * wall_stress / radius;
    const double expected_wall_viscosity = wall_stress / wall_rate;
    SCIENCE_REQUIRE_NEAR(
        apparentViscosityPipe(power_law, power_law_flow, radius, power_law_gradient),
        expected_wall_viscosity, 1e-8, 2e-6,
        "Rabinowitsch-Mooney correction for a power-law fluid");
    SCIENCE_REQUIRE_NEAR(
        apparentViscosityPipe(power_law, -power_law_flow, radius, -power_law_gradient),
        expected_wall_viscosity, 1e-8, 2e-6, "reverse-flow apparent viscosity magnitude");

    requireThrows<std::domain_error>(
        [&] { (void)apparentViscosityPipe(power_law, 0.0, radius, pressure_gradient); },
        "zero flow must not divide silently");
    requireThrows<std::domain_error>(
        [&] { (void)apparentViscosityPipe(power_law, flow_rate, radius, 0.0); },
        "zero pressure gradient must fail loudly");
    requireThrows<std::invalid_argument>([&] { (void)pipeWallShearRate(flow_rate, 0.0); },
                                         "zero pipe radius must be rejected");
}

void velocityBoundarySemanticsAreHonest() {
    const StructuredMesh mesh(3, 3, 0.0, 1.0, 0.0, 1.0);
    const int stride = mesh.nx() + 1;
    std::vector<double> u(static_cast<std::size_t>(mesh.numNodes()));
    std::vector<double> v(static_cast<std::size_t>(mesh.numNodes()));
    for (int j = 0; j <= mesh.ny(); ++j) {
        for (int i = 0; i <= mesh.nx(); ++i) {
            u[static_cast<std::size_t>(j * stride + i)] = 10.0 * j + i;
            v[static_cast<std::size_t>(j * stride + i)] = -10.0 * j - i;
        }
    }

    std::array<VelocityBC, 4> boundaries{VelocityBC::NoSlip(), VelocityBC::Outflow(),
                                         VelocityBC::NoSlip(), VelocityBC::NoSlip()};
    applyVelocityBoundaryConditions(mesh, boundaries, u, v);
    SCIENCE_REQUIRE_NEAR(u[static_cast<std::size_t>(stride + mesh.nx())], 12.0, 0.0, 0.0,
                         "outflow is zero-normal-gradient velocity extrapolation");
    SCIENCE_REQUIRE_NEAR(v[static_cast<std::size_t>(stride + mesh.nx())], -12.0, 0.0, 0.0,
                         "outflow extrapolates both velocity components");

    boundaries[0] = VelocityBC::StressFree();
    const auto u_before_rejection = u;
    const auto v_before_rejection = v;
    requireThrows<std::invalid_argument>(
        [&] { applyVelocityBoundaryConditions(mesh, boundaries, u, v); },
        "unimplemented traction-free semantics must be rejected");
    SCIENCE_REQUIRE(u == u_before_rejection && v == v_before_rejection,
                    "unsupported boundary conditions must be rejected before mutation");

    auto short_field = u;
    short_field.pop_back();
    requireThrows<std::invalid_argument>(
        [&] { applyVelocityBoundaryConditions(mesh, boundaries, short_field, v); },
        "mismatched field storage must be rejected before indexing");

    boundaries = {VelocityBC::Dirichlet(std::numeric_limits<double>::quiet_NaN(), 0.0),
                  VelocityBC::NoSlip(), VelocityBC::NoSlip(), VelocityBC::NoSlip()};
    requireThrows<std::domain_error>(
        [&] { applyVelocityBoundaryConditions(mesh, boundaries, u, v); },
        "non-finite prescribed velocity must be rejected");

    boundaries[0] = VelocityBC::NoSlip();
    boundaries[0].type = static_cast<VelocityBCType>(999);
    requireThrows<std::invalid_argument>(
        [&] { applyVelocityBoundaryConditions(mesh, boundaries, u, v); },
        "unknown velocity boundary enum values must be rejected");
}

}  // namespace

int main() {
    return science_test::runSuite(
        "non-Newtonian rheology and velocity boundary contracts",
        {{"constitutive parameter domains are bounded", constitutiveParametersAreBounded},
         {"canonical constitutive responses are correct", canonicalConstitutiveResponses},
         {"blood helpers match documented correlations", bloodHelpersMatchDocumentedCorrelations},
         {"pipe rheometry uses the selected model slope", pipeRheometryUsesModelSlope},
         {"velocity boundary semantics are fail-loud", velocityBoundarySemanticsAreHonest}});
}
