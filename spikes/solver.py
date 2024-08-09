import numpy as np
import sympy as sp
from scipy.linalg import eig, inv
from scipy.integrate import odeint
from IPython.display import display, Math
from sympy.utilities.lambdify import lambdify


def solve_system_of_equations(
    A_np: np.ndarray,
    B_np: np.ndarray,
    X0_np: np.ndarray,
    t_range: np.ndarray,
    t0: float = 0,
    verbose: bool = True,
):
    """
    Solve a system of first-order linear differential equations of any size.

    Parameters:
        A_np (ndarray): Coefficient matrix A.
        B_np (ndarray): Constant vector B.
        X0_np (ndarray): Initial condition vector X0.
        t_range (array-like): Range of time points over which to evaluate the solution.
        t0 (float): Initial time, default is 0.

    Returns:
        final_solution (list): List of symbolic solutions.
        x_values (list): List of evaluated solution values over t_range.
    """
    t = sp.Symbol('t')

    # Number of equations (and variables)
    n = A_np.shape[0]

    # Define the system of equations symbolically
    X = sp.Matrix([sp.Function(f'x{i+1}')(t) for i in range(n)])
    A = sp.Matrix(A_np)
    B = sp.Matrix(B_np)
    X0 = sp.Matrix(X0_np)

    dx_dt = A * X + B
    system = [sp.Eq(sp.diff(X[i], t), dx_dt[i]) for i in range(n)]

    # Solve the homogeneous system
    sol_hom = sp.dsolve(system)

    # Find the constants using initial conditions
    constants = sp.solve(
        [sol.rhs.subs(t, t0) - x0 for sol, x0 in zip(sol_hom, X0)]
    )

    # Substitute the constants into the solutions
    sol_hom_constants = [sol.rhs.subs(constants) for sol in sol_hom]

    # Display symbolic solutions
    final_solution = [sol_hom for sol_hom in sol_hom_constants]
    
    # Evaluate the solutions over the given time range
    x_functions = [lambdify((t,), sol, "numpy") for sol in final_solution]
    x_values = [x_func(t_range) for x_func in x_functions]

    return final_solution, x_values


def solve_linear_system_numerical(A, B, X0, t):
    """
    Solves the differential equation dX/dt = AX + B.

    Parameters:
    A (numpy.ndarray): Coefficient matrix.
    B (numpy.ndarray): Constant vector.
    X0 (numpy.ndarray): Initial condition vector.
    t (numpy.ndarray): Array of time points at which to solve.

    Returns:
    numpy.ndarray: Array of solution vectors at each time point.
    """
    # Equilibrium state
    X_eq = -inv(A) @ B

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)

    # Solve for the constants a1, a2, b1, b2
    def system(X, t):
        return A @ X + B

    sol = odeint(system, X0, t)

    return {"x": sol, "xeq": X_eq, "ev": eigenvalues, "evec": eigenvectors}


def solve_linear_system_analytically(A, B, X0, t):
    '''Depricated'''
    # Equilibrium state
    X_eq = -inv(A) @ B
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)
    lambda1, lambda2 = np.real(eigenvalues)
    v1, v2 = eigenvectors.T
    
    # Ensure lambda1 != lambda2 for the provided solution
    if np.isclose(lambda1, lambda2):
        raise ValueError("The eigenvalues must be distinct.")
    
    # Calculate the coefficients c1 and c2 using initial conditions
    def initial_condition_coefficients(X0):
        V = np.column_stack((v1, v2))
        c = np.linalg.solve(V, X0 - X_eq)
        return c
    
    c1, c2 = initial_condition_coefficients(X0)
    
    # Construct the solution
    X_t = np.array([c1 * v1 * np.exp(lambda1 * t_) + c2 * v2 * np.exp(lambda2 * t_) + X_eq for t_ in t])
    
    return {
        "x": X_t,
        "xeq": X_eq,
        "ev": eigenvalues,
        "evec": eigenvectors
    }

def solve_linear_system_sympy(A, B, X0, verbose=True):
    '''Depricated'''
    t = sp.symbols('t')
    A = sp.Matrix(A)
    B = sp.Matrix(B)
    X0 = sp.Matrix(X0)
    
    # Equilibrium state
    X_eq = -A.inv() * B
    
    # Eigenvalues and eigenvectors
    eigenvals = A.eigenvals()
    eigenvects = A.eigenvects()
    
    print("Eigenvalues:", list(eigenvals.keys()))
    
    if len(eigenvals) == 2 or list(eigenvals.values()).count(1) == 2:
        if verbose:
            print("The matrix A have two distinct eigenvalues.")

        lambda1, lambda2 = eigenvals.keys()
        v1 = eigenvects[0][2][0].normalized()
        v2 = eigenvects[1][2][0].normalized()
    
        # Construct the general solution
        c1, c2 = sp.symbols('c1 c2')
        X_hom = c1 * v1 * sp.exp(lambda1 * t) + c2 * v2 * sp.exp(lambda2 * t)
        X_t = X_hom + X_eq
        
        # Calculate the coefficients c1 and c2 using initial conditions
        C = sp.Matrix([c1, c2])
        V = sp.Matrix.hstack(v1, v2)
        C_vals = sp.linsolve((V, X0 - X_eq))
    
        c1_val, c2_val = list(C_vals)[0]
        X_t = X_t.subs({c1: c1_val, c2: c2_val})
        
        # LaTeX representation
        latex_solution = sp.latex(sp.simplify(X_t))
    
    if len(eigenvals) == 1:
        if verbose:
            print("The matrix A must have one repeated eigenvalue.")
        return None
        
        
    return X_t#, latex_solution
