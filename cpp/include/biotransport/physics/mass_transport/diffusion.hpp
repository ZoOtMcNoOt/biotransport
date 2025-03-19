#ifndef BIOTRANSPORT_DIFFUSION_HPP
#define BIOTRANSPORT_DIFFUSION_HPP

#include <vector>
#include <array>
#include <biotransport/core/mesh/mesh.hpp>

namespace biotransport {

enum class BoundaryType {
    DIRICHLET,  // Fixed value
    NEUMANN     // Fixed flux
};

/**
 * Solver for diffusion problems.
 */
class DiffusionSolver {
public:
    /**
     * Create a diffusion solver.
     * 
     * @param mesh The mesh to solve on
     * @param diffusivity The diffusion coefficient
     */
    DiffusionSolver(const StructuredMesh& mesh, double diffusivity);
    
    /**
     * Set the initial condition.
     * 
     * @param values The initial values (one per node)
     */
    virtual void setInitialCondition(const std::vector<double>& values);
    
    /**
     * Set a Dirichlet (fixed value) boundary condition.
     * 
     * @param boundary_id Boundary ID (0=left, 1=right, 2=bottom, 3=top)
     * @param value The fixed value
     */
    void setDirichletBoundary(int boundary_id, double value);
    
    /**
     * Set a Neumann (fixed flux) boundary condition.
     * 
     * @param boundary_id Boundary ID (0=left, 1=right, 2=bottom, 3=top)
     * @param flux The fixed flux
     */
    void setNeumannBoundary(int boundary_id, double flux);
    
    /**
     * Solve the diffusion equation for a specified time.
     * 
     * @param dt Time step size
     * @param num_steps Number of time steps
     */
    virtual void solve(double dt, int num_steps);
    
    /**
     * Get the current solution.
     */
    const std::vector<double>& solution() const { return solution_; }
    
protected:
    const StructuredMesh& mesh_;
    double diffusivity_;
    std::vector<double> solution_;
    
    // Boundary conditions
    std::array<BoundaryType, 4> boundary_types_;
    std::array<double, 4> boundary_values_;
    
    // Apply boundary conditions
    void applyBoundaryConditions(std::vector<double>& new_solution);
    
    // Check CFL condition
    bool checkStability(double dt) const;
};

} // namespace biotransport

#endif // BIOTRANSPORT_DIFFUSION_HPP