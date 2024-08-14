import sympy as sp
import numpy as np

from typing import List
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
    **kwargs,
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
            x_vec = np.array([X[i, j], Y[i, j]])
            derivs = dX_dt(x_vec, A, B)
            U[i, j] = derivs[0]
            V[i, j] = derivs[1]

    # Normalize the arrows
    N = np.sqrt(U**2 + V**2)
    U, V = U / N, V / N

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


def plot_nullclines(
    f,
    g,
    symbols: List[str],
    x_range=(-2, 2),
    y_range=(-2, 2),
    num_points=400,
    figsize: tuple = (4, 4),
    title: str = "Nullclines of the system",
    grid=False,
    ax=None,
    **kwargs: dict,
):
    """
    Plots the nullclines of a two-dimensional system of differential equations.

    Parameters
    ----------
    f : sympy expression
        The right-hand side of the first differential equation dx/dt = f(x, y).
    g : sympy expression
        The right-hand side of the second differential equation dy/dt = g(x, y).
    x_range : tuple, optional
        The range of x values for plotting (default is (-2, 2)).
    y_range : tuple, optional
        The range of y values for plotting (default is (-2, 2)).
    num_points : int, optional
        The number of points to plot (default is 400).
    **kwargs : dict
        Additional keyword arguments to be passed to `plt.plot`.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object with the nullclines plotted.

    """

    # Define symbols
    x, y = sp.Symbol(symbols[0]), sp.Symbol(symbols[1])

    # if ls not in kwargs set it to "--"
    if "ls" not in kwargs:
        kwargs["ls"] = "-"
    if "lw" not in kwargs:
        kwargs["lw"] = 1.5

    # Solve for nullclines
    nullcline_x = sp.solve(f, y)  # Nullcline for dx/dt = 0
    nullcline_y = sp.solve(g, y)  # Nullcline for dy/dt = 0

    # Convert to lambdified functions for plotting
    nullcline_x_func = [sp.lambdify(x, expr) for expr in nullcline_x]
    nullcline_y_func = [sp.lambdify(x, expr) for expr in nullcline_y]

    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], num_points)

    if "figsize" in kwargs:
        plt.figure(figsize=figsize)

    # Plotting the nullclines
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    # Plot nullcline for dx/dt = 0 (f(x, y) = 0)
    for func in nullcline_x_func:
        ax.plot(x_vals, func(x_vals), label=f"$d{symbols[0]}/dt = 0$", c="r", **kwargs)

    # Plot nullcline for dy/dt = 0 (g(x, y) = 0)
    for func in nullcline_y_func:
        plt.plot(x_vals, func(x_vals), label=f"$d{symbols[1]}/dt = 0$", c="b", **kwargs)

    # Customize the plot
    # ax.axhline(0, color='black', linewidth=0.5)
    # ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(f"{symbols[0]}")
    ax.set_ylabel(f"{symbols[1]}")
    ax.set_title(title)
    ax.legend(fontsize=10)
    if grid:
        ax.grid(True, ls="--")
    return ax
