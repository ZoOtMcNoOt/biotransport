"""Gray–Scott reaction-diffusion (Turing patterns), simulated in C++.

This is a classic *dimensionless* 2-species reaction–diffusion system that
produces spots/stripes/mazes depending on (f, k). The Python script is mainly:
- parameter + initial-condition setup
- calling the C++ solver
- saving plots

For headless runs (e.g., `run_examples.py`), we run a smaller/faster config.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import time
from datetime import datetime

import biotransport as bt

EXAMPLE_NAME = "turing_patterns"
_HEADLESS = os.environ.get("MPLBACKEND", "").lower() == "agg"
nx, ny = (128, 128) if _HEADLESS else (256, 256)
Du = 0.16
Dv = 0.08
dt = 1.0
total_steps = 6000 if _HEADLESS else 20000

# Some well-known Gray–Scott parameter sets (f, k) for distinct patterns
# (Adapted from John Pearson’s classification or widely known references)
pattern_sets = {
    "spots": (0.035, 0.065, "Isolated spots"),
    "stripes": (0.022, 0.051, "Stripe patterns"),
    "labyrinth": (0.03, 0.055, "Maze/labyrinth patterns"),
    "targets": (0.026, 0.051, "Target (ring) patterns"),
}

# Custom colormaps for U and V fields
v_cmap = LinearSegmentedColormap.from_list(
    "v_colormap",
    [
        (0, "#000033"),
        (0.3, "#0000FF"),
        (0.6, "#FF00FF"),
        (0.8, "#FFFF00"),
        (1.0, "#FFFFFF"),
    ],
)
u_cmap = LinearSegmentedColormap.from_list(
    "u_colormap", [(0, "#FFFF00"), (0.3, "#FF0000"), (0.6, "#800080"), (1.0, "#000000")]
)


def initialize_fields(init_type="random"):
    u = np.ones((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    yy, xx = np.ogrid[:ny, :nx]

    if init_type == "random":
        u = 1.0 - 0.03 * np.random.random((ny, nx)).astype(np.float32)
        v = 0.03 * np.random.random((ny, nx)).astype(np.float32)
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    if init_type == "center_square":
        cx, cy = nx // 2, ny // 2
        r = max(4, min(nx, ny) // 20)
        u[cy - r : cy + r, cx - r : cx + r] = 0.5
        v[cy - r : cy + r, cx - r : cx + r] = 0.25
    elif init_type == "spots":
        m = np.random.random((ny, nx)) > 0.995
        u[m] = 0.5
        v[m] = 0.25
    elif init_type == "circles":
        for _ in range(6):
            cx = np.random.randint(nx // 4, 3 * nx // 4)
            cy = np.random.randint(ny // 4, 3 * ny // 4)
            r = np.random.randint(5, 15)
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
            u[m] = 0.5
            v[m] = 0.25
    elif init_type == "target_rings":
        cx, cy = nx // 2, ny // 2
        r0 = min(nx, ny) // 8
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r0**2
        u[m] = 0.5
        v[m] = 0.25
        for _ in range(2):
            cx = np.random.randint(nx // 4, 3 * nx // 4)
            cy = np.random.randint(ny // 4, 3 * ny // 4)
            r = np.random.randint(10, 20)
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
            u[m] = 0.5
            v[m] = 0.25

    noise = (0.01 if _HEADLESS else 0.002) * np.random.random((ny, nx)).astype(
        np.float32
    )
    u += noise
    v += noise
    return np.clip(u, 0, 1), np.clip(v, 0, 1)


def simulate_gray_scott(f, k, init_type="random", steps_between_frames=1000):
    print(f"Starting Gray-Scott with f={f}, k={k}, init={init_type}")
    start_time = time.time()

    # Initialize fields
    u0, v0 = initialize_fields(init_type)

    mesh = bt.mesh_2d(nx - 1, ny - 1, x_max=1.0, y_max=1.0)
    solver = bt.GrayScottSolver(mesh, Du, Dv, f, k)

    check_interval = 500 if _HEADLESS else 1000
    result = solver.simulate(
        u0.ravel(order="C").tolist(),
        v0.ravel(order="C").tolist(),
        total_steps,
        dt,
        steps_between_frames=steps_between_frames,
        check_interval=check_interval,
        stable_tol=1e-4,
        min_frames_before_early_stop=6,
    )

    u_stack, v_stack = result.u_frames(), result.v_frames()
    u_frames = [u_stack[i].copy() for i in range(u_stack.shape[0])]
    v_frames = [v_stack[i].copy() for i in range(v_stack.shape[0])]
    frame_steps = list(result.frame_steps)

    elapsed = time.time() - start_time
    print(f"Simulation finished in {elapsed:.1f} s, total steps = {result.steps_run}")
    return u_frames, v_frames, frame_steps


def visualize_frames(u_frames, v_frames, frame_steps, f, k, pattern_name, description):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_frames = len(v_frames)
    if n_frames < 1:
        print("No frames to visualize.")
        return

    # Pick up to 6 frames, spread across time, always including the final frame.
    if n_frames <= 6:
        show_indices = list(range(n_frames))
    else:
        show_indices = np.linspace(0, n_frames - 1, 6).astype(int).tolist()
        show_indices[-1] = n_frames - 1

    u_vmin, u_vmax = 0.0, 1.0
    start_idx = max(1, n_frames // 4)
    v_stack = np.stack(v_frames[start_idx:], axis=0)
    v_p1, v_p99 = np.percentile(v_stack, [1.0, 99.0])
    if not np.isfinite(v_p1) or not np.isfinite(v_p99) or v_p99 <= v_p1:
        v_p1, v_p99 = float(v_stack.min()), float(v_stack.max())
        if v_p99 <= v_p1:
            v_p1, v_p99 = 0.0, 1.0

    plt.figure(figsize=(18, 8))
    for i, idx in enumerate(show_indices):
        actual_step = frame_steps[idx]
        plt.subplot(2, len(show_indices), i + 1)
        u_contrast = 1.0 - u_frames[idx]
        plt.imshow(
            u_contrast, cmap=u_cmap, interpolation="bilinear", vmin=0.0, vmax=1.0
        )
        plt.title(f"1-U: step {actual_step}")
        plt.axis("off")

        plt.subplot(2, len(show_indices), i + 1 + len(show_indices))
        plt.imshow(
            v_frames[idx], cmap=v_cmap, interpolation="bilinear", vmin=v_p1, vmax=v_p99
        )
        plt.title(f"V: step {actual_step}")
        plt.axis("off")

    plt.suptitle(f"Gray-Scott (f={f}, k={k}) - {description}", fontsize=16)
    plt.tight_layout()
    evol_path = bt.get_result_path(
        f"evolution_{pattern_name}_{timestamp}.png", EXAMPLE_NAME
    )
    plt.savefig(evol_path, dpi=150)

    plt.figure(figsize=(10, 8))
    v_final = v_frames[-1]
    plt.imshow(v_final, cmap=v_cmap, interpolation="bilinear", vmin=v_p1, vmax=v_p99)
    plt.title(f"Final V: f={f}, k={k} - {description}")
    plt.colorbar(label="V concentration")
    plt.axis("off")
    plt.tight_layout()
    final_v_path = bt.get_result_path(
        f"final_v_{pattern_name}_{timestamp}.png", EXAMPLE_NAME
    )
    plt.savefig(final_v_path, dpi=300)

    plt.figure(figsize=(10, 8))
    u_final = u_frames[-1]
    plt.imshow(u_final, cmap=u_cmap, interpolation="bilinear", vmin=u_vmin, vmax=u_vmax)
    plt.title(f"Final U: f={f}, k={k} - {description}")
    plt.colorbar(label="U concentration")
    plt.axis("off")
    plt.tight_layout()
    final_u_path = bt.get_result_path(
        f"final_u_{pattern_name}_{timestamp}.png", EXAMPLE_NAME
    )
    plt.savefig(final_u_path, dpi=300)

    plt.close("all")
    print(f"Saved evolution snapshots and final images for pattern '{pattern_name}'.")


if __name__ == "__main__":
    np.random.seed(0)
    selected_pattern = (
        "stripes" if _HEADLESS else "labyrinth"
    )  # "spots", "stripes", "labyrinth", or "targets"
    init_type = (
        "center_square" if _HEADLESS else "center_square"
    )  # "random", "circles", "spots", "target_rings", or "center_square"
    steps_between_frames = 500 if _HEADLESS else 2000

    f, k, desc = pattern_sets[selected_pattern]

    print(f"Running Gray-Scott for pattern: {selected_pattern} with init={init_type}")
    u_frames, v_frames, frame_steps = simulate_gray_scott(
        f, k, init_type, steps_between_frames
    )
    visualize_frames(u_frames, v_frames, frame_steps, f, k, selected_pattern, desc)

    print("Done. Check the generated images for final patterns.")
