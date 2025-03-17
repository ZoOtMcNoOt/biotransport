#include <biotransport/solvers/reaction_diffusion.hpp>

namespace biotransport {

ReactionDiffusionSolver::ReactionDiffusionSolver(
    const StructuredMesh& mesh,
    double diffusivity,
    ReactionFunction reaction)
    : DiffusionSolver(mesh, diffusivity), reaction_(reaction)
{
    // All initialization handled by the base class and member initializers
}

void ReactionDiffusionSolver::solve(double dt, int num_steps) {
    if (dt <= 0.0 || num_steps <= 0) {
        throw std::invalid_argument("Time step and number of steps must be positive");
    }
    
    // Check stability
    if (!checkStability(dt)) {
        std::cerr << "Warning: Time step may be too large for stability" << std::endl;
    }
    
    std::vector<double> new_solution(solution_.size());
    
    // Time stepping loop
    for (int step = 0; step < num_steps; ++step) {
        // 1D case
        if (mesh_.is1D()) {
            // Interior nodes
            for (int i = 1; i < mesh_.nx(); ++i) {
                int idx = mesh_.index(i);
                double x = mesh_.x(i);
                
                // Diffusion term (same as in base class)
                double diffusion_term = diffusivity_ * dt / (mesh_.dx() * mesh_.dx()) * 
                    (solution_[idx+1] - 2*solution_[idx] + solution_[idx-1]);
                
                // Reaction term
                double reaction_term = dt * reaction_(solution_[idx], x, 0.0, time_);
                
                // Update
                new_solution[idx] = solution_[idx] + diffusion_term + reaction_term;
            }
        }
        // 2D case
        else {
            // Interior nodes
            for (int j = 1; j < mesh_.ny(); ++j) {
                for (int i = 1; i < mesh_.nx(); ++i) {
                    int idx = mesh_.index(i, j);
                    double x = mesh_.x(i);
                    double y = mesh_.y(i, j);
                    
                    int idx_left = mesh_.index(i-1, j);
                    int idx_right = mesh_.index(i+1, j);
                    int idx_bottom = mesh_.index(i, j-1);
                    int idx_top = mesh_.index(i, j+1);
                    
                    // Diffusion term
                    double d2u_dx2 = (solution_[idx_right] - 2*solution_[idx] + solution_[idx_left]) / 
                                    (mesh_.dx() * mesh_.dx());
                    double d2u_dy2 = (solution_[idx_top] - 2*solution_[idx] + solution_[idx_bottom]) / 
                                    (mesh_.dy() * mesh_.dy());
                    double diffusion_term = diffusivity_ * dt * (d2u_dx2 + d2u_dy2);
                    
                    // Reaction term
                    double reaction_term = dt * reaction_(solution_[idx], x, y, time_);
                    
                    // Update
                    new_solution[idx] = solution_[idx] + diffusion_term + reaction_term;
                }
            }
        }
        
        // Apply boundary conditions
        applyBoundaryConditions(new_solution);
        
        // Update solution and time
        solution_ = new_solution;
        time_ += dt;
    }
}

} // namespace biotransport