import numpy as np
from typing import Tuple

def is_positive_semi_definite(matrix: np.ndarray) -> bool:

    if not np.allclose(matrix, matrix.T):
        return False  
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= -1e-8) 

def calculate_max_sq_sharpe_ratio(mu: np.ndarray, sigma: np.ndarray) -> float:
    
    sigma_inv = np.linalg.pinv(sigma)
    sr_squared = mu.T @ sigma_inv @ mu
    return float(sr_squared)

