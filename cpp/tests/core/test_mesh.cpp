#include "../test_support/science_test.hpp"
#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cmath>

// Test the StructuredMesh class
void testStructuredMesh1D() {
    // Create a 1D mesh
    biotransport::StructuredMesh mesh(10, 0.0, 1.0);

    // Test properties
    SCIENCE_REQUIRE(mesh.nx() == 10, "1D mesh must report the requested number of cells");
    SCIENCE_REQUIRE(mesh.is1D() == true, "mesh built with the 1D constructor must be 1D");
    SCIENCE_REQUIRE(mesh.numNodes() == 11, "1D mesh must have nx+1 nodes");
    SCIENCE_REQUIRE(mesh.numCells() == 10, "1D mesh must have nx cells");
    SCIENCE_REQUIRE_NEAR(mesh.dx(), 0.1, 1e-10, 0.0, "1D grid spacing dx");

    // Test coordinates
    SCIENCE_REQUIRE_NEAR(mesh.x(0), 0.0, 1e-10, 0.0, "x coordinate of node 0");
    SCIENCE_REQUIRE_NEAR(mesh.x(5), 0.5, 1e-10, 0.0, "x coordinate of node 5");
    SCIENCE_REQUIRE_NEAR(mesh.x(10), 1.0, 1e-10, 0.0, "x coordinate of node 10");

    // Test indices
    SCIENCE_REQUIRE(mesh.index(0) == 0, "1D linear index of node 0");
    SCIENCE_REQUIRE(mesh.index(5) == 5, "1D linear index of node 5");
    SCIENCE_REQUIRE(mesh.index(10) == 10, "1D linear index of node 10");
}

void testStructuredMesh2D() {
    // Create a 2D mesh
    biotransport::StructuredMesh mesh(5, 5, 0.0, 1.0, 0.0, 1.0);

    // Test properties
    SCIENCE_REQUIRE(mesh.nx() == 5, "2D mesh must report the requested nx");
    SCIENCE_REQUIRE(mesh.ny() == 5, "2D mesh must report the requested ny");
    SCIENCE_REQUIRE(mesh.is1D() == false, "mesh built with the 2D constructor must not be 1D");
    SCIENCE_REQUIRE(mesh.numNodes() == 36,
                    "2D mesh must have (nx+1)*(ny+1) nodes");                 // (5+1) * (5+1)
    SCIENCE_REQUIRE(mesh.numCells() == 25, "2D mesh must have nx*ny cells");  // 5 * 5
    SCIENCE_REQUIRE_NEAR(mesh.dx(), 0.2, 1e-10, 0.0, "2D grid spacing dx");
    SCIENCE_REQUIRE_NEAR(mesh.dy(), 0.2, 1e-10, 0.0, "2D grid spacing dy");

    // Test coordinates
    SCIENCE_REQUIRE_NEAR(mesh.x(0), 0.0, 1e-10, 0.0, "x coordinate of column 0");
    SCIENCE_REQUIRE_NEAR(mesh.x(5), 1.0, 1e-10, 0.0, "x coordinate of column 5");
    SCIENCE_REQUIRE_NEAR(mesh.y(0, 0), 0.0, 1e-10, 0.0, "y coordinate of row 0");
    SCIENCE_REQUIRE_NEAR(mesh.y(0, 5), 1.0, 1e-10, 0.0, "y coordinate of row 5");

    // Test indices
    SCIENCE_REQUIRE(mesh.index(0, 0) == 0, "2D linear index of node (0,0)");
    SCIENCE_REQUIRE(mesh.index(5, 0) == 5, "2D linear index of node (5,0)");
    SCIENCE_REQUIRE(mesh.index(0, 5) == 30, "2D linear index of node (0,5)");
    SCIENCE_REQUIRE(mesh.index(5, 5) == 35, "2D linear index of node (5,5)");
}

int main() {
    return science_test::runSuite("structured mesh",
                                  {{"1D structured mesh", testStructuredMesh1D},
                                   {"2D structured mesh", testStructuredMesh2D}});
}
