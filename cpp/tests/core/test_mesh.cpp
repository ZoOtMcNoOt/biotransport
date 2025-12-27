#include <biotransport/core/mesh/structured_mesh.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

// Test the StructuredMesh class
void testStructuredMesh1D() {
    std::cout << "Testing 1D structured mesh..." << std::endl;

    // Create a 1D mesh
    biotransport::StructuredMesh mesh(10, 0.0, 1.0);

    // Test properties
    assert(mesh.nx() == 10);
    assert(mesh.is1D() == true);
    assert(mesh.numNodes() == 11);
    assert(mesh.numCells() == 10);
    assert(std::abs(mesh.dx() - 0.1) < 1e-10);

    // Test coordinates
    assert(std::abs(mesh.x(0) - 0.0) < 1e-10);
    assert(std::abs(mesh.x(5) - 0.5) < 1e-10);
    assert(std::abs(mesh.x(10) - 1.0) < 1e-10);

    // Test indices
    assert(mesh.index(0) == 0);
    assert(mesh.index(5) == 5);
    assert(mesh.index(10) == 10);

    std::cout << "1D structured mesh tests passed!" << std::endl;
}

void testStructuredMesh2D() {
    std::cout << "Testing 2D structured mesh..." << std::endl;

    // Create a 2D mesh
    biotransport::StructuredMesh mesh(5, 5, 0.0, 1.0, 0.0, 1.0);

    // Test properties
    assert(mesh.nx() == 5);
    assert(mesh.ny() == 5);
    assert(mesh.is1D() == false);
    assert(mesh.numNodes() == 36);  // (5+1) * (5+1)
    assert(mesh.numCells() == 25);  // 5 * 5
    assert(std::abs(mesh.dx() - 0.2) < 1e-10);
    assert(std::abs(mesh.dy() - 0.2) < 1e-10);

    // Test coordinates
    assert(std::abs(mesh.x(0) - 0.0) < 1e-10);
    assert(std::abs(mesh.x(5) - 1.0) < 1e-10);
    assert(std::abs(mesh.y(0, 0) - 0.0) < 1e-10);
    assert(std::abs(mesh.y(0, 5) - 1.0) < 1e-10);

    // Test indices
    assert(mesh.index(0, 0) == 0);
    assert(mesh.index(5, 0) == 5);
    assert(mesh.index(0, 5) == 30);
    assert(mesh.index(5, 5) == 35);

    std::cout << "2D structured mesh tests passed!" << std::endl;
}

int main() {
    // Run tests
    testStructuredMesh1D();
    testStructuredMesh2D();

    std::cout << "All mesh tests passed!" << std::endl;
    return 0;
}
