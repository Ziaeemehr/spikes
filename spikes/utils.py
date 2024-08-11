import spikes 
import sympy

def characteristic_polynomial(matrix):
    """
    Computes the characteristic polynomial of a given matrix.

    Parameters
    ----------
    matrix: (list of lists) 
        A 2D list representing the matrix.

    Returns
    -------
    sympy.Poly: 
        The characteristic polynomial of the matrix.
    """
    # Define the symbolic variable
    x = sympy.Symbol('x')
    
    # Convert the input matrix to a sympy.Matrix
    A = sympy.Matrix(matrix)
    
    # Compute the characteristic polynomial
    char_poly = A.charpoly(x)
    
    # Get the polynomial coefficients
    coefficients = char_poly.all_coeffs()
    
    # Create the polynomial using sympy.Poly
    p = sympy.Poly(coefficients, x)
    
    return p


def routh(p):
    """ Construct the Routh-Hurwitz array given a polynomial in s

    Parameters
    ----------
    p: sympy.Poly
        The characteristic polynomial of coefficient matrix
    
    Returns
    -------
    value: sympy.Matrix
        The Routh-Hurwitz array
    
    References https://github.com/alchemyst/Dynamics-and-Control/blob/master/tbcontrol/symbolic.py
    """
    coefficients = p.all_coeffs()
    N = len(coefficients)
    M = sympy.zeros(N, (N+1)//2 + 1)

    r1 = coefficients[0::2]
    r2 = coefficients[1::2]
    M[0, :len(r1)] = [r1]
    M[1, :len(r2)] = [r2]
    for i in range(2, N):
        for j in range(N//2):
            S = M[[i-2, i-1], [0, j+1]]
            M[i, j] = sympy.simplify(-S.det()/M[i-1, 0])
        # If a row of the routh table becomes zero,Take the derivative of the previous row and substitute it instead
        # Ref: Norman S. Nise, Control Systems Engineering, 8th Edition, Chapter 6, Section 3
        if M[i, :] == sympy.Matrix([[0]*(M.shape[1])]):
            # Find the coefficients on taking the derivative of the Auxiliary polynomial
            diff_arr = list(range(N-i, -1, -2))
            diff_arr.extend([0]*(M.shape[1] - len(diff_arr)))
            diff_arr = sympy.Matrix([diff_arr])
            # Multiply the coefficients with the value in previous row
            M[i, :] = sympy.matrix_multiply_elementwise(diff_arr, M[i-1, :])
    return M[:, :-1]