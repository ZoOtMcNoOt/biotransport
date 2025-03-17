"""
Visualization tools for biotransport simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_1d_solution(mesh, solution, title=None, xlabel='Position', ylabel='Value'):
    """
    Plot a 1D solution on a mesh.
    
    Args:
        mesh: The 1D mesh
        solution: The solution values
        title: Plot title
        xlabel: x-axis label
        ylabel: y-axis label
    """
    if not mesh.is_1d():
        raise ValueError("Mesh must be 1D for 1D plotting")
    
    x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, solution, 'b-')
    
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    return plt.gcf()

def plot_2d_solution(mesh, solution, title=None, colorbar_label='Value'):
    """
    Plot a 2D solution on a mesh as a contour plot.
    
    Args:
        mesh: The 2D mesh
        solution: The solution values
        title: Plot title
        colorbar_label: Label for the colorbar
    """
    if mesh.is_1d():
        raise ValueError("Mesh must be 2D for 2D plotting")
    
    # Create grid for plotting
    x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
    y = np.array([mesh.y(0, j) for j in range(mesh.ny() + 1)])
    X, Y = np.meshgrid(x, y)
    
    # Reshape solution to 2D array
    Z = np.zeros((mesh.ny() + 1, mesh.nx() + 1))
    for j in range(mesh.ny() + 1):
        for i in range(mesh.nx() + 1):
            idx = mesh.index(i, j)
            Z[j, i] = solution[idx]
    
    # Plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, label=colorbar_label)
    
    if title:
        plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    return plt.gcf()

def plot_2d_surface(mesh, solution, title=None, zlabel='Value'):
    """
    Plot a 2D solution as a 3D surface.
    
    Args:
        mesh: The 2D mesh
        solution: The solution values
        title: Plot title
        zlabel: z-axis label
    """
    if mesh.is_1d():
        raise ValueError("Mesh must be 2D for 3D surface plotting")
    
    # Create grid for plotting
    x = np.array([mesh.x(i) for i in range(mesh.nx() + 1)])
    y = np.array([mesh.y(0, j) for j in range(mesh.ny() + 1)])
    X, Y = np.meshgrid(x, y)
    
    # Reshape solution to 2D array
    Z = np.zeros((mesh.ny() + 1, mesh.nx() + 1))
    for j in range(mesh.ny() + 1):
        for i in range(mesh.nx() + 1):
            idx = mesh.index(i, j)
            Z[j, i] = solution[idx]
    
    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(zlabel)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig