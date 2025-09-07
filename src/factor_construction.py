import numpy as np
import pandas as pd

def construct_ols_factors(
    df_asset_returns: pd.DataFrame,
    phi_t: np.ndarray
) -> pd.DataFrame:
    w_t_transpose = np.linalg.pinv(phi_t)
    factor_returns_array = df_asset_returns.values @ w_t_transpose.T
    factor_cols = [f'Factor_{i+1}' for i in range(phi_t.shape[1])]
    return pd.DataFrame(factor_returns_array, index=df_asset_returns.index, columns=factor_cols)

def construct_gls_factors(
    df_asset_returns: pd.DataFrame,
    phi_t: np.ndarray,
    d_t: np.ndarray
) -> pd.DataFrame:
    d_t_inv = np.linalg.pinv(d_t)
    stabilizer = np.eye(d_t_inv.shape[0]) * 1e-9
    s_t = np.linalg.cholesky(d_t_inv + stabilizer)
    s_t_phi_t = s_t @ phi_t
    w_t_transpose = np.linalg.pinv(s_t_phi_t) @ s_t
    factor_returns_array = df_asset_returns.values @ w_t_transpose.T
    factor_cols = [f'Factor_{i+1}' for i in range(phi_t.shape[1])]
    return pd.DataFrame(factor_returns_array, index=df_asset_returns.index, columns=factor_cols)
