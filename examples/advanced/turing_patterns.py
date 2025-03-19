"""
Example of Turing pattern formation using reaction-diffusion.

This example simulates the Gray-Scott model, a reaction-diffusion system
that can produce a variety of patterns through diffusion-driven instability.
This model has been used to study pattern formation in biological systems.

The Gray-Scott model describes the reaction:
    U + 2V -> 3V
    V -> P

Where U and V are two chemicals, and P is an inert product.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from biotransport import StructuredMesh, DiffusionSolver
from biotransport.utils import get_result_path

# Create results subdirectory for this example
EXAMPLE_NAME = "turing_patterns"

# Since our library doesn't directly support systems of PDEs,
# we'll implement a simple solver for the Gray-Scott model ourselves,
# leveraging the StructuredMesh class from our library.

class GrayScottModel:
    """
    Gray-Scott reaction-diffusion model implementation.
    """
    def __init__(self, mesh, Du, Dv, f, k):
        """
        Initialize the Gray-Scott model.

        Args:
            mesh: StructuredMesh object
            Du: Diffusion coefficient for U
            Dv: Diffusion coefficient for V
            f: Feed rate
            k: Kill rate
        """
        self.mesh = mesh
        self.Du = Du  # Diffusion rate of U
        self.Dv = Dv  # Diffusion rate of V
        self.f = f    # Feed rate
        self.k = k    # Kill rate

        # Initialize concentration fields
        self.nx = mesh.nx()
        self.ny = mesh.ny()
        self.u = np.ones((self.ny + 1, self.nx + 1))  # Chemical U (substrate)
        self.v = np.zeros((self.ny + 1, self.nx + 1))  # Chemical V (activator)

        # Create a central square of V
        center_x, center_y = self.nx // 2, self.ny // 2
        r = min(self.nx, self.ny) // 10
        for j in range(center_y - r, center_y + r):
            for i in range(center_x - r, center_x + r):
                self.u[j, i] = 0.5
                self.v[j, i] = 0.25

        # Add some random perturbations to break symmetry
        self.v += np.random.random((self.ny + 1, self.nx + 1)) * 0.01

        # Store all states for animation
        self.u_history = [self.u.copy()]
        self.v_history = [self.v.copy()]

    def laplacian(self, field):
        """
        Compute the Laplacian of a field using finite differences.
        """
        laplacian = np.zeros_like(field)

        # Interior points
        laplacian[1:-1, 1:-1] = (
                                        field[1:-1, 0:-2] +  # left
                                        field[1:-1, 2:] +    # right
                                        field[0:-2, 1:-1] +  # bottom
                                        field[2:, 1:-1] -    # top
                                        4 * field[1:-1, 1:-1]  # center
                                ) / (self.mesh.dx() * self.mesh.dx())

        # Apply periodic boundary conditions
        # Left and right boundaries
        laplacian[1:-1, 0] = (
                                     field[1:-1, -2] +    # left (wraps around)
                                     field[1:-1, 1] +     # right
                                     field[0:-2, 0] +     # bottom
                                     field[2:, 0] -       # top
                                     4 * field[1:-1, 0]     # center
                             ) / (self.mesh.dx() * self.mesh.dx())

        laplacian[1:-1, -1] = (
                                      field[1:-1, -2] +    # left
                                      field[1:-1, 1] +     # right (wraps around)
                                      field[0:-2, -1] +    # bottom
                                      field[2:, -1] -      # top
                                      4 * field[1:-1, -1]    # center
                              ) / (self.mesh.dx() * self.mesh.dx())

        # Top and bottom boundaries
        laplacian[0, 1:-1] = (
                                     field[0, 0:-2] +     # left
                                     field[0, 2:] +       # right
                                     field[-2, 1:-1] +    # bottom (wraps around)
                                     field[1, 1:-1] -     # top
                                     4 * field[0, 1:-1]     # center
                             ) / (self.mesh.dx() * self.mesh.dx())

        laplacian[-1, 1:-1] = (
                                      field[-1, 0:-2] +    # left
                                      field[-1, 2:] +      # right
                                      field[-2, 1:-1] +    # bottom
                                      field[1, 1:-1] -     # top (wraps around)
                                      4 * field[-1, 1:-1]    # center
                              ) / (self.mesh.dx() * self.mesh.dx())

        # Corner points
        laplacian[0, 0] = (
                                  field[0, -2] +       # left (wraps around)
                                  field[0, 1] +        # right
                                  field[-2, 0] +       # bottom (wraps around)
                                  field[1, 0] -        # top
                                  4 * field[0, 0]        # center
                          ) / (self.mesh.dx() * self.mesh.dx())

        laplacian[0, -1] = (
                                   field[0, -2] +       # left
                                   field[0, 1] +        # right (wraps around)
                                   field[-2, -1] +      # bottom (wraps around)
                                   field[1, -1] -       # top
                                   4 * field[0, -1]       # center
                           ) / (self.mesh.dx() * self.mesh.dx())

        laplacian[-1, 0] = (
                                   field[-1, -2] +      # left (wraps around)
                                   field[-1, 1] +       # right
                                   field[-2, 0] +       # bottom
                                   field[1, 0] -        # top (wraps around)
                                   4 * field[-1, 0]       # center
                           ) / (self.mesh.dx() * self.mesh.dx())

        laplacian[-1, -1] = (
                                    field[-1, -2] +      # left
                                    field[-1, 1] +       # right (wraps around)
                                    field[-2, -1] +      # bottom
                                    field[1, -1] -       # top (wraps around)
                                    4 * field[-1, -1]      # center
                            ) / (self.mesh.dx() * self.mesh.dx())

        return laplacian

    def step(self, dt, num_steps=1, store_history=False):
        """
        Advance the simulation by num_steps with step size dt.

        Args:
            dt: Time step size
            num_steps: Number of time steps
            store_history: Whether to store the history for animation
        """
        for _ in range(num_steps):
            # Compute Laplacians
            laplacian_u = self.laplacian(self.u)
            laplacian_v = self.laplacian(self.v)

            # Reaction terms
            uvv = self.u * self.v * self.v
            reaction_u = -uvv + self.f * (1.0 - self.u)
            reaction_v = uvv - (self.f + self.k) * self.v

            # Update concentrations
            self.u += dt * (self.Du * laplacian_u + reaction_u)
            self.v += dt * (self.Dv * laplacian_v + reaction_v)

            # Ensure concentrations stay in valid range [0, 1]
            self.u = np.clip(self.u, 0.0, 1.0)
            self.v = np.clip(self.v, 0.0, 1.0)

            if store_history:
                self.u_history.append(self.u.copy())
                self.v_history.append(self.v.copy())

# Create a 2D mesh
nx, ny = 200, 200
Lx, Ly = 2.0, 2.0
mesh = StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

# Different pattern types (adjust f and k parameters)
pattern_types = {
    "spots": {"f": 0.025, "k": 0.06},
    "stripes": {"f": 0.022, "k": 0.051},
    "maze": {"f": 0.029, "k": 0.057},
    "holes": {"f": 0.039, "k": 0.065}
}

# Select pattern type
pattern_type = "maze"
f = pattern_types[pattern_type]["f"]
k = pattern_types[pattern_type]["k"]

# Create the Gray-Scott model
Du, Dv = 0.16, 0.08  # Diffusion coefficients (Dv < Du is required for patterns)
model = GrayScottModel(mesh, Du, Dv, f, k)

# Time integration parameters
dt = 1.0
total_steps = 10000
display_interval = 100

# Solve the system
for step in range(0, total_steps + 1, display_interval):
    if step > 0:
        print(f"Computing steps {step-display_interval+1} to {step}...")
        model.step(dt, display_interval)

    # Plot the current state
    plt.figure(figsize=(10, 8))

    # Use the V concentration for visualization
    plt.imshow(model.v, origin='lower', cmap='viridis',
               extent=[0, Lx, 0, Ly])

    plt.colorbar(label='V Concentration')
    plt.title(f'Gray-Scott Model: {pattern_type.capitalize()} Pattern (Step {step})')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save figure
    plt.savefig(get_result_path(f'pattern_{pattern_type}_{step:05d}.png', EXAMPLE_NAME))
    plt.close()

# Create an animation of the pattern formation
print("Creating animation...")

# We'll store a subset of steps for the animation to keep file size reasonable
animation_steps = 50
step_size = total_steps // animation_steps

# Run the simulation again, storing history this time
model = GrayScottModel(mesh, Du, Dv, f, k)
for i in range(animation_steps + 1):
    if i > 0:
        model.step(dt, step_size, store_history=True)
    print(f"Animation frame {i}/{animation_steps}")

# Create the animation
fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(model.v_history[0], origin='lower', cmap='viridis',
                extent=[0, Lx, 0, Ly], vmin=0, vmax=0.4)
plt.colorbar(img, label='V Concentration')
ax.set_title(f'Gray-Scott Model: {pattern_type.capitalize()} Pattern')
ax.set_xlabel('X')
ax.set_ylabel('Y')

def update_frame(i):
    img.set_array(model.v_history[i])
    ax.set_title(f'Gray-Scott Model: {pattern_type.capitalize()} Pattern (Step {i*step_size})')
    return [img]

ani = animation.FuncAnimation(fig, update_frame, frames=len(model.v_history), interval=100, blit=True)

# Save the animation
animation_file = get_result_path(f'pattern_{pattern_type}_animation.mp4', EXAMPLE_NAME)
ani.save(animation_file, writer='ffmpeg', fps=10, dpi=100)

print(f"Animation saved to {animation_file}")

# Final plots showing both chemicals
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(model.u, origin='lower', cmap='Blues',
           extent=[0, Lx, 0, Ly])
plt.colorbar(label='U Concentration')
plt.title('Chemical U (Substrate)')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.imshow(model.v, origin='lower', cmap='viridis',
           extent=[0, Lx, 0, Ly])
plt.colorbar(label='V Concentration')
plt.title('Chemical V (Activator)')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.savefig(get_result_path(f'pattern_{pattern_type}_final.png', EXAMPLE_NAME))

results_dir = get_result_path('', EXAMPLE_NAME)
print(f"Simulation complete. Results saved to '{results_dir}'.")