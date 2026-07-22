#include <biotransport/biotransport.hpp>
#include <cmath>

int main() {
    biotransport::StructuredMesh mesh(4, 0.0, 1.0);
    biotransport::TransportProblem problem(mesh);
    problem.diffusivity(0.1).constantSource(1.0).initialCondition(0.0);

    const auto result = biotransport::solve(problem, 0.01);
    return result.time == 0.01 && std::abs(result.concentration[2] - 0.01) < 1e-14 ? 0 : 1;
}
