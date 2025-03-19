#include <biotransport/solvers/diffusion.hpp>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace biotransport {

DiffusionSolver::DiffusionSolver(const StructuredMesh& mesh, double diffusivity)
    : mesh_(mesh), diffusivity_(diffusivity)
{
    if (diffusivity <= 0.0) {
        throw std::invalid_argument("Diffusivity must be positive");
    }
    
    // Initialize solution vector
    solution_.resize(mesh.numNodes(), 0.0);
    
    // Default boundary conditions (Dirichlet, value = 0)
    for (int i = 0; i < 4; ++i) {
        boundary_types_[i] = BoundaryType::DIRICHLET;
        boundary_values_[i] = 0.0;
    }
}

void DiffusionSolver::setInitialCondition(const std::vector<double>& values) {
    if (values.size() != solution_.size()) {
        throw std::invalid_argument("Initial condition size doesn't match mesh");
    }
    
    solution_ = values;
}

void DiffusionSolver::setDirichletBoundary(int boundary_id, double value) {
    if (boundary_id < 0 || boundary_id > 3) {
        throw std::invalid_argument("Invalid boundary ID");
    }
    
    boundary_types_[boundary_id] = BoundaryType::DIRICHLET;
    boundary_values_[boundary_id] = value;
}

void DiffusionSolver::setNeumannBoundary(int boundary_id, double flux) {
    if (boundary_id < 0 || boundary_id > 3) {
        throw std::invalid_argument("Invalid boundary ID");
    }
    
    boundary_types_[boundary_id] = BoundaryType::NEUMANN;
    boundary_values_[boundary_id] = flux;
}

void DiffusionSolver::solve(double dt, int num_steps) {
    if (dt <= 0.0 || num_steps <= 0) {
        throw std::invalid_argument("Time step and number of steps must be positive");
    }
    
    // Check stability (CFL condition)
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
                new_solution[idx] = solution_[idx] + 
                    diffusivity_ * dt / (mesh_.dx() * mesh_.dx()) * 
                    (solution_[idx+1] - 2*solution_[idx] + solution_[idx-1]);
            }
        }
        // 2D case
        else {
            // Interior nodes
            for (int j = 1; j < mesh_.ny(); ++j) {
                for (int i = 1; i < mesh_.nx(); ++i) {
                    int idx = mesh_.index(i, j);
                    int idx_left = mesh_.index(i-1, j);
                    int idx_right = mesh_.index(i+1, j);
                    int idx_bottom = mesh_.index(i, j-1);
                    int idx_top = mesh_.index(i, j+1);
                    
                    // Finite difference approximation
                    double d2u_dx2 = (solution_[idx_right] - 2*solution_[idx] + solution_[idx_left]) / 
                                     (mesh_.dx() * mesh_.dx());
                    double d2u_dy2 = (solution_[idx_top] - 2*solution_[idx] + solution_[idx_bottom]) / 
                                     (mesh_.dy() * mesh_.dy());
                    
                    new_solution[idx] = solution_[idx] + diffusivity_ * dt * (d2u_dx2 + d2u_dy2);
                }
            }
        }
        
        // Apply boundary conditions
        applyBoundaryConditions(new_solution);
        
        // Update solution
        solution_ = new_solution;
    }
}

void DiffusionSolver::applyBoundaryConditions(std::vector<double>& new_solution) {
    // 1D case
    if (mesh_.is1D()) {
        // Left boundary (ID = 0)
        if (boundary_types_[0] == BoundaryType::DIRICHLET) {
            new_solution[mesh_.index(0)] = boundary_values_[0];
        } else { // Neumann
            new_solution[mesh_.index(0)] = new_solution[mesh_.index(1)] - 
                boundary_values_[0] * mesh_.dx();
        }
        
        // Right boundary (ID = 1)
        if (boundary_types_[1] == BoundaryType::DIRICHLET) {
            new_solution[mesh_.index(mesh_.nx())] = boundary_values_[1];
        } else { // Neumann
            new_solution[mesh_.index(mesh_.nx())] = new_solution[mesh_.index(mesh_.nx()-1)] + 
                boundary_values_[1] * mesh_.dx();
        }
    }
    // 2D case
    else {
        // Left boundary (ID = 0)
        if (boundary_types_[0] == BoundaryType::DIRICHLET) {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                new_solution[mesh_.index(0, j)] = boundary_values_[0];
            }
        } else { // Neumann
            for (int j = 0; j <= mesh_.ny(); ++j) {
                new_solution[mesh_.index(0, j)] = new_solution[mesh_.index(1, j)] - 
                    boundary_values_[0] * mesh_.dx();
            }
        }
        
        // Right boundary (ID = 1)
        if (boundary_types_[1] == BoundaryType::DIRICHLET) {
            for (int j = 0; j <= mesh_.ny(); ++j) {
                new_solution[mesh_.index(mesh_.nx(), j)] = boundary_values_[1];
            }
        } else { // Neumann
            for (int j = 0; j <= mesh_.ny(); ++j) {
                new_solution[mesh_.index(mesh_.nx(), j)] = new_solution[mesh_.index(mesh_.nx()-1, j)] + 
                    boundary_values_[1] * mesh_.dx();
            }
        }
        
        // Bottom boundary (ID = 2)
        if (boundary_types_[2] == BoundaryType::DIRICHLET) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                new_solution[mesh_.index(i, 0)] = boundary_values_[2];
            }
        } else { // Neumann
            for (int i = 0; i <= mesh_.nx(); ++i) {
                new_solution[mesh_.index(i, 0)] = new_solution[mesh_.index(i, 1)] - 
                    boundary_values_[2] * mesh_.dy();
            }
        }
        
        // Top boundary (ID = 3)
        if (boundary_types_[3] == BoundaryType::DIRICHLET) {
            for (int i = 0; i <= mesh_.nx(); ++i) {
                new_solution[mesh_.index(i, mesh_.ny())] = boundary_values_[3];
            }
        } else { // Neumann
            for (int i = 0; i <= mesh_.nx(); ++i) {
                new_solution[mesh_.index(i, mesh_.ny())] = new_solution[mesh_.index(i, mesh_.ny()-1)] + 
                    boundary_values_[3] * mesh_.dy();
            }
        }
    }
}

bool DiffusionSolver::checkStability(double dt) const {
    // CFL condition for explicit scheme: dt <= dx^2/(2*D) in 1D
    //                                   dt <= dx^2/(4*D) in 2D
    double dx2 = mesh_.dx() * mesh_.dx();
    
    if (mesh_.is1D()) {
        return dt <= dx2 / (2.0 * diffusivity_);
    } else {
        double dy2 = mesh_.dy() * mesh_.dy();
        double min_h2 = std::min(dx2, dy2);
        return dt <= min_h2 / (4.0 * diffusivity_);
    }
}

} // namespace biotransport