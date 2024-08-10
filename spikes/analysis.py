import numpy as np

import numpy as np

def classify_eigenvalues_dynamics(eigenvalues, tolerance=1e-10):
    eigenvalues = np.array(eigenvalues)
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    # Check if all eigenvalues have magnitude less than 1
    all_inside_unit_circle = np.all(np.abs(eigenvalues) < 1 - tolerance)
    any_on_unit_circle = np.any(np.abs(np.abs(eigenvalues) - 1) < tolerance)
    
    if np.allclose(eigenvalues, 0, atol=tolerance):
        return "Superattracting fixed point"
    elif all_inside_unit_circle:
        if np.allclose(imag_parts, 0, atol=tolerance):
            return "Attracting node"
        else:
            return "Attracting spiral"
    elif any_on_unit_circle and np.all(np.abs(eigenvalues) <= 1 + tolerance):
        if np.allclose(real_parts, 1, atol=tolerance) and np.allclose(imag_parts, 0, atol=tolerance):
            return "Non-hyperbolic fixed point"
        elif np.allclose(real_parts, -1, atol=tolerance) and np.allclose(imag_parts, 0, atol=tolerance):
            return "Period-2 cycle"
        else:
            return "Periodic or quasiperiodic behavior"
    else:
        if np.allclose(imag_parts, 0, atol=tolerance):
            if np.all(real_parts > 1 + tolerance):
                return "Repelling node"
            elif np.all(real_parts < -1 - tolerance):
                return "Repelling node with period-2 behavior"
            else:
                return "Saddle point"
        else:
            return "Repelling spiral"