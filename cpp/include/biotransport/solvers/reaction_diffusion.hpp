#ifndef BIOTRANSPORT_REACTION_DIFFUSION_HPP
#define BIOTRANSPORT_REACTION_DIFFUSION_HPP

#include <biotransport/solvers/diffusion.hpp>
#include <functional>

namespace biotransport {

/**
 * Reaction-Diffusion solver for problems of the form:
 * ∂u/∂t = D∇²u + R(u,x,t)
 */
class ReactionDiffusionSolver : public DiffusionSolver {
public:
    /**
     * Function type for the reaction term.
     * 
     * @param u Current solution value
     * @param x Spatial coordinate
     * @param t Current time
     * @return Reaction rate
     */
    using ReactionFunction = std::function<double(double u, double x, double y, double t)>;
    
    /**
     * Create a reaction-diffusion solver.
     * 
     * @param mesh The mesh to solve on
     * @param diffusivity The diffusion coefficient
     * @param reaction The reaction function
     */
    ReactionDiffusionSolver(const StructuredMesh& mesh, 
                           double diffusivity,
                           ReactionFunction reaction);
    
    /**
     * Solve the reaction-diffusion equation.
     * 
     * @param dt Time step size
     * @param num_steps Number of time steps
     */
    void solve(double dt, int num_steps) override;
    
private:
    ReactionFunction reaction_;
    double time_ = 0.0;  // Current simulation time
};

} // namespace biotransport

#endif // BIOTRANSPORT_REACTION_DIFFUSION_HPP