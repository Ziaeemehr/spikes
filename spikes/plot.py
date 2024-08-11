import numpy as np
import matplotlib.pyplot as plt


def plot_direction_field(
    A: np.ndarray,
    B: np.ndarray,
    x1_range: tuple = (-2, 2),
    x2_range: tuple = (-2, 2),
    num_points: int = 20,
    ax=None,
    xlabel: str = "x1",
    ylabel: str = "x2",
    title: str = "Direction Field for the System $dX/dt = AX + B$",
    **kwargs
):
    """
    Plots the direction field for the system dx/dt = A * X + B.

    Parameters
    ----------
    A : numpy.ndarray
        A 2x2 numpy array representing the coefficient matrix for the linear system.
    B : numpy.ndarray
        A 1x2 numpy array representing the constant vector.
    x1_range : tuple, optional
        The range for x1 values. Default is (-10, 10).
    x2_range : tuple, optional
        The range for x2 values. Default is (-10, 10).
    num_points : int, optional
        The number of points per axis in the grid. Default is 20.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object with the direction field plotted.
    """

    # Create a grid of points (x1, x2)
    x1 = np.linspace(x1_range[0], x1_range[1], num_points)
    x2 = np.linspace(x2_range[0], x2_range[1], num_points)
    X, Y = np.meshgrid(x1, x2)

    # Define the system of differential equations
    def dX_dt(X, A, B):
        return A @ X + B

    # Calculate U and V (the x and y components of the arrows)
    U, V = np.zeros(X.shape), np.zeros(Y.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_vec = np.array([X[i,j], Y[i,j]])
            derivs = dX_dt(x_vec, A, B)
            U[i,j] = derivs[0]
            V[i,j] = derivs[1]

    # Normalize the arrows
    N = np.sqrt(U**2 + V**2)
    U, V = U/N, V/N

    # Plot the direction field using quiver
    if ax is None:
        ax = plt.gca()
    ax.quiver(X, Y, U, V, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim([x1_range[0], x1_range[1]])
    ax.set_ylim([x2_range[0], x2_range[1]])
    return ax
