import numpy as np
import pandas as pd
from typing import Dict, Optional
from utils import is_positive_semi_definite

def create_characteristics_matrix(n_assets: int, n_factors: int, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=0.8, scale=0.4, size=(n_assets, n_factors))

def create_spanning_mu_vector(
    sigma_t: np.ndarray,
    w_t: np.ndarray,
    factor_risk_premia: np.ndarray
) -> np.ndarray:
    spanning_basis = sigma_t @ w_t
    mu_t = spanning_basis @ factor_risk_premia
    return mu_t

def simulate_asset_returns(
    phi_t: np.ndarray,
    sigma_f_t: np.ndarray,
    sigma_epsilon_t: np.ndarray,
    n_periods: int,
    mu_t: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    n_assets, n_factors = phi_t.shape
    assert sigma_f_t.shape == (n_factors, n_factors), "Factor covariance matrix has incorrect shape."
    assert sigma_epsilon_t.shape == (n_assets, n_assets), "Residual covariance matrix has incorrect shape."
    assert is_positive_semi_definite(sigma_f_t), "Factor covariance matrix is not positive semi-definite."
    assert is_positive_semi_definite(sigma_epsilon_t), "Residual covariance matrix is not positive semi-definite."

    if mu_t is None:
        mu_t = np.zeros(n_assets)
    else:
        mu_t = mu_t.flatten()
        assert mu_t.shape == (n_assets,), f"mu_t must have shape ({n_assets},), but has {mu_t.shape}"
        
    mean_f = np.zeros(n_factors)
    mean_epsilon = np.zeros(n_assets)

    rng = np.random.default_rng(random_state)
    generative_factor_shocks = rng.multivariate_normal(mean_f, sigma_f_t, n_periods)
    generative_residual_shocks = rng.multivariate_normal(mean_epsilon, sigma_epsilon_t, n_periods)

    asset_returns_array = mu_t + (generative_factor_shocks @ phi_t.T + generative_residual_shocks)
    index = pd.RangeIndex(start=0, stop=n_periods, step=1)
    asset_cols = [f'Asset_{i+1}' for i in range(n_assets)]
    factor_cols = [f'Factor_{i+1}' for i in range(n_factors)]

    df_asset_returns = pd.DataFrame(asset_returns_array, index=index, columns=asset_cols)
    df_generative_factor_shocks = pd.DataFrame(generative_factor_shocks, index=index, columns=factor_cols)
    df_generative_residual_shocks = pd.DataFrame(generative_residual_shocks, index=index, columns=asset_cols)
    
    return {
        "asset_returns": df_asset_returns,
        "generative_factor_shocks": df_generative_factor_shocks,
        "generative_residual_shocks": df_generative_residual_shocks
    }
