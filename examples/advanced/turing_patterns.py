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
import os
import subprocess
from biotransport import StructuredMesh
from biotransport.utils import get_result_path

# Create results subdirectory for this example
EXAMPLE_NAME = "turing_patterns"

# Check if ffmpeg is available
def ffmpeg_available():
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

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
        self.steps = [0]

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

    def step(self, dt, num_steps=1, store_history=False, store_interval=100):
        """
        Advance the simulation by num_steps with step size dt.

        Args:
            dt: Time step size
            num_steps: Number of time steps
            store_history: Whether to store the history for animation
            store_interval: How often to store history steps
        """
        total_steps = 0
        for step in range(num_steps):
            # Compute Laplacians
            laplacian_u = self.laplacian(self.u)
            laplacian_v = self.laplacian(self.v)

            # Reaction terms
            uvv = self.u * self.v * self.v
            reaction_u = -uvv + self.f * (1.0 - self.u)
            reaction_v = uvv - (self.f + self.k) * self.v

            # Update concentrations (copy to avoid simultaneous update issues)
            u_new = self.u + dt * (self.Du * laplacian_u + reaction_u)
            v_new = self.v + dt * (self.Dv * laplacian_v + reaction_v)

            # Ensure concentrations stay in valid range [0, 1]
            self.u = np.clip(u_new, 0.0, 1.0)
            self.v = np.clip(v_new, 0.0, 1.0)

            total_steps += 1

            if store_history and total_steps % store_interval == 0:
                self.u_history.append(self.u.copy())
                self.v_history.append(self.v.copy())
                self.steps.append(total_steps)

        return total_steps


# Dictionary of pattern types with their parameters
pattern_types = {
    "spots": {"f": 0.025, "k": 0.060, "description": "Isolated spots form"},
    "stripes": {"f": 0.022, "k": 0.051, "description": "Parallel stripes develop"},
    "maze": {"f": 0.029, "k": 0.057, "description": "Meandering labyrinth structure"},
    "holes": {"f": 0.039, "k": 0.065, "description": "Isolated holes in a connected medium"}
}

# Function to run simulation and visualize results
def run_simulation(pattern_name, save_animation=True):
    # Create a 2D mesh
    nx, ny = 200, 200
    Lx, Ly = 2.0, 2.0
    mesh = StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

    # Get pattern parameters
    params = pattern_types[pattern_name]
    f = params["f"]
    k = params["k"]
    description = params["description"]
    print(f"\nSimulating {pattern_name.capitalize()} pattern (f={f}, k={k})")
    print(f"Description: {description}")

    # Create the Gray-Scott model
    Du, Dv = 0.16, 0.08  # Diffusion coefficients (Dv < Du is required for patterns)
    model = GrayScottModel(mesh, Du, Dv, f, k)

    # Time integration parameters
    dt = 1.0
    total_steps = 25000  # Increased for better pattern development
    display_steps = [0, 500, 1000, 2500, 5000, 10000, 25000]

    # Save initial state
    plt.figure(figsize=(10, 8))
    plt.imshow(model.v, origin='lower', cmap='viridis',
               extent=[0, Lx, 0, Ly], vmin=0, vmax=0.4)
    plt.colorbar(label='V Concentration')
    plt.title(f'Gray-Scott Model: {pattern_name.capitalize()} Pattern (Initial)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(get_result_path(f'pattern_{pattern_name}_initial.png', EXAMPLE_NAME))
    plt.close()

    current_step = 0
    # Run simulation, saving at key time points
    for target_step in display_steps[1:]:
        steps_to_run = target_step - current_step
        if steps_to_run <= 0:
            continue

        print(f"Computing steps {current_step+1} to {target_step}...")
        current_step += model.step(dt, steps_to_run)

        # Plot and save the current state
        plt.figure(figsize=(10, 8))
        plt.imshow(model.v, origin='lower', cmap='viridis',
                   extent=[0, Lx, 0, Ly], vmin=0, vmax=0.4)
        plt.colorbar(label='V Concentration')
        plt.title(f'Gray-Scott Model: {pattern_name.capitalize()} Pattern (Step {current_step})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(get_result_path(f'pattern_{pattern_name}_step_{current_step:05d}.png', EXAMPLE_NAME))
        plt.close()

    # Create animation data
    if save_animation:
        print("Preparing animation frames...")
        # Reset model and store history for animation
        model = GrayScottModel(mesh, Du, Dv, f, k)
        animation_frames = 50
        steps_per_frame = total_steps // animation_frames

        # Run model again, storing history at regular intervals
        current_step = 0
        while current_step < total_steps:
            steps_to_run = min(steps_per_frame, total_steps - current_step)
            current_step += model.step(dt, steps_to_run, store_history=True, store_interval=steps_per_frame)
            print(f"Animation progress: {current_step}/{total_steps} steps")

        # Save individual frames (always do this as backup)
        print("Saving animation frames...")
        frames_dir = get_result_path(f'frames_{pattern_name}', EXAMPLE_NAME)
        os.makedirs(frames_dir, exist_ok=True)

        for i, (step, v) in enumerate(zip(model.steps, model.v_history)):
            plt.figure(figsize=(8, 8))
            plt.imshow(v, origin='lower', cmap='viridis',
                       extent=[0, Lx, 0, Ly], vmin=0, vmax=0.4)
            plt.colorbar(label='V Concentration')
            plt.title(f'Gray-Scott Model: {pattern_name.capitalize()} Pattern (Step {step})')
            plt.xlabel('X')
            plt.ylabel('Y')
            frame_file = os.path.join(frames_dir, f'frame_{i:04d}.png')
            plt.savefig(frame_file)
            plt.close()

        # Save GIF using imageio (more reliable than matplotlib's animation)
        try:
            import imageio

            frames = []
            for i in range(len(model.v_history)):
                frame_file = os.path.join(frames_dir, f'frame_{i:04d}.png')
                frames.append(imageio.imread(frame_file))

            gif_file = get_result_path(f'pattern_{pattern_name}_animation.gif', EXAMPLE_NAME)
            imageio.mimsave(gif_file, frames, duration=0.1)
            print(f"GIF animation saved to: {gif_file}")
        except ImportError:
            print("imageio not found - skipping GIF creation")
            print("To create GIFs, install imageio: pip install imageio")

        # Try to create an MP4 using ffmpeg if available
        if ffmpeg_available():
            try:
                mp4_file = get_result_path(f'pattern_{pattern_name}_animation.mp4', EXAMPLE_NAME)
                print(f"Creating MP4 animation using ffmpeg: {mp4_file}")

                # Create MP4 from frames using ffmpeg
                cmd = [
                    'ffmpeg', '-y', '-framerate', '10',
                    '-i', os.path.join(frames_dir, 'frame_%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure dimensions are even
                    mp4_file
                ]
                subprocess.run(cmd, check=True)
                print(f"MP4 animation saved to: {mp4_file}")
            except Exception as e:
                print(f"Error creating MP4: {e}")
        else:
            print("ffmpeg not found - skipping MP4 creation")

    # Final comparison plot showing U and V
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
    plt.savefig(get_result_path(f'pattern_{pattern_name}_final_comparison.png', EXAMPLE_NAME))
    plt.close()

    return model

# Run simulations for different pattern types
print("Demonstrating Turing pattern formation with the Gray-Scott model")

# Create a comparison of all pattern types
def compare_patterns():
    plt.figure(figsize=(15, 12))

    for i, (pattern_name, params) in enumerate(pattern_types.items()):
        # Create a small version for the comparison
        nx, ny = 100, 100
        Lx, Ly = 2.0, 2.0
        mesh = StructuredMesh(nx, ny, 0.0, Lx, 0.0, Ly)

        # Create the model
        Du, Dv = 0.16, 0.08
        model = GrayScottModel(mesh, Du, Dv, params["f"], params["k"])

        # Run for a fixed number of steps
        dt = 1.0
        print(f"Computing pattern: {pattern_name}...")
        model.step(dt, 10000)  # Run for 10000 steps

        # Plot
        plt.subplot(2, 2, i+1)
        plt.imshow(model.v, origin='lower', cmap='viridis',
                   extent=[0, Lx, 0, Ly], vmin=0, vmax=0.4)
        plt.colorbar(label='V Concentration')
        plt.title(f'{pattern_name.capitalize()} (f={params["f"]}, k={params["k"]})')
        plt.xlabel('X')
        plt.ylabel('Y')

    plt.tight_layout()
    plt.savefig(get_result_path('pattern_comparison.png', EXAMPLE_NAME))
    plt.close()

# First create a pattern comparison
compare_patterns()

# Run a full simulation for the selected pattern type
selected_pattern = "maze"  # Change this to explore different patterns
model = run_simulation(selected_pattern, save_animation=True)

results_dir = get_result_path('', EXAMPLE_NAME)
print(f"\nSimulation complete. Results saved to '{results_dir}'.")
print(f"Check out the different pattern types in 'pattern_comparison.png'.")
print(f"For a detailed view of {selected_pattern} pattern evolution, see the step images and animation.")