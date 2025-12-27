# Biotransport Library Architecture

## Overview

The biotransport library is a C++ scientific computing library for mass transport simulations including diffusion, reaction-diffusion, advection-diffusion, and fluid dynamics (Darcy flow, Stokes flow).

## Directory Structure

```
cpp/
├── include/biotransport/
│   ├── biotransport.hpp           # Convenience header (includes everything)
│   ├── core/
│   │   ├── mesh/
│   │   │   ├── structured_mesh.hpp
│   │   │   └── mesh_iterators.hpp  # Unified 1D/2D iteration
│   │   ├── boundary.hpp
│   │   ├── utils.hpp
│   │   └── numerics/
│   │       └── solvers/
│   │           └── iterative.hpp   # Jacobi, Gauss-Seidel, SOR
│   ├── physics/
│   │   ├── reactions.hpp           # Reaction term functors library
│   │   ├── mass_transport/
│   │   │   ├── gray_scott.hpp      # Specialized: Turing patterns
│   │   │   ├── tumor_drug_delivery.hpp
│   │   │   └── membrane_diffusion.hpp
│   │   ├── heat_transfer/
│   │   │   └── bioheat_cryotherapy.hpp
│   │   └── fluid_dynamics/
│   │       ├── darcy_flow.hpp      # Declaration only
│   │       ├── stokes.hpp          # Declaration only
│   │       ├── navier_stokes.hpp
│   │       └── non_newtonian.hpp
│   └── solvers/
│       ├── solver_base.hpp         # CRTP base template
│       ├── diffusion_solvers.hpp   # All diffusion/reaction-diffusion
│       ├── advection_diffusion_solver.hpp
│       └── explicit_fd.hpp
└── src/
    ├── core/
    │   └── mesh/structured_mesh.cpp
    └── physics/
        ├── mass_transport/
        │   ├── gray_scott.cpp
        │   └── tumor_drug_delivery.cpp
        ├── heat_transfer/
        │   └── bioheat_cryotherapy.cpp
        └── fluid_dynamics/
            ├── darcy_flow.cpp      # Implementation
            └── stokes.cpp          # Implementation
```

## Key Design Patterns

### 1. CRTP Solver Base (Curiously Recurring Template Pattern)

**File:** `solvers/solver_base.hpp`

Eliminates duplicated time-stepping code across all explicit solvers:

```cpp
template<typename Derived>
class ExplicitSolverBase {
public:
    void solve(double dt, int num_steps) {
        // Common time-stepping loop
        for (int step = 0; step < num_steps; ++step) {
            // Derived class implements computeNodeUpdate()
            static_cast<Derived*>(this)->computeNodeUpdate(idx, i, j, ops, dt);
        }
    }
};

class DiffusionSolver : public ExplicitSolverBase<DiffusionSolver> {
    double computeNodeUpdate(int idx, int i, int j, 
                             const StencilOps& ops, double dt) {
        return D_ * ops.laplacian(solution_, idx) * dt;
    }
};
```

### 2. Mesh Iterators

**File:** `core/mesh/mesh_iterators.hpp`

Unifies 1D/2D iteration and eliminates duplicated branching:

```cpp
MeshIterator iter(mesh);

// Iterate over interior nodes (works for 1D or 2D)
iter.forEachInterior([&](int idx, int i, int j) {
    // idx = linear index, (i,j) = grid indices
    solution[idx] += D * ops.laplacian(solution, idx) * dt;
});

// Stencil operations
StencilOps ops(mesh);
double lap = ops.laplacian(field, idx);
double grad_x = ops.gradX(field, idx);
double upwind = ops.upwindGradX(field, idx, velocity);
```

### 3. Reactions Library

**File:** `physics/reactions.hpp`

Library of reaction term functors - no need for separate solver classes:

```cpp
using namespace biotransport::reactions;

// Pre-built reactions
auto decay = linearDecay(0.1);           // -k*c
auto mm = michaelisMenten(1.0, 0.5);     // Vmax*c/(Km+c)
auto logistic = logisticGrowth(0.5, 1.0); // r*c*(1-c/K)

// Combine reactions
auto combined = combine(decay, constantSource(0.2));

// Use with ReactionDiffusionSolver
ReactionDiffusionSolver solver(mesh, D, combined);
```

### 4. Declaration/Implementation Separation

Large solver classes (DarcyFlowSolver, StokesSolver) now have:
- **Header (*.hpp):** Class declaration, inline trivial methods
- **Source (*.cpp):** Full implementation

Benefits:
- Faster compilation (no template instantiation in every TU)
- Smaller binary size
- Cleaner headers for users

## Migration Guide

### Using Consolidated Solvers

**Before (old structure):**
```cpp
#include <biotransport/physics/mass_transport/diffusion.hpp>
#include <biotransport/physics/mass_transport/linear_reaction_diffusion.hpp>
#include <biotransport/physics/mass_transport/michaelis_menten_reaction_diffusion.hpp>
```

**After (new structure):**
```cpp
#include <biotransport/solvers/diffusion_solvers.hpp>
// All diffusion variants in one header
```

### Using Reactions Library

**Before:** Create separate solver class for each reaction type

**After:**
```cpp
#include <biotransport/physics/reactions.hpp>

// Michaelis-Menten with decay
auto reaction = reactions::combine(
    reactions::michaelisMenten(Vmax, Km),
    reactions::linearDecay(k)
);
ReactionDiffusionSolver solver(mesh, D, reaction);
```

## Build Configuration

Add source files to CMakeLists.txt:
```cmake
set(SOURCES
    # Core
    src/core/mesh/structured_mesh.cpp
    src/core/utils.cpp
    
    # Specialized mass transport
    src/physics/mass_transport/gray_scott.cpp
    src/physics/mass_transport/tumor_drug_delivery.cpp
    
    # Heat transfer
    src/physics/heat_transfer/bioheat_cryotherapy.cpp
    
    # Fluid dynamics
    src/physics/fluid_dynamics/darcy_flow.cpp
    src/physics/fluid_dynamics/stokes.cpp
)
```

## Notes

- Base diffusion/reaction-diffusion solvers are now header-only (template-based)
- Fluid dynamics solvers have declarations in headers, implementations in .cpp files
- Specialized solvers (GrayScott, TumorDrugDelivery) retain their own files
