import numpy as np
import pandas as pd
from typing import Dict, Any
from utils import calculate_max_sq_sharpe_ratio

def verify_covariance_decomposition(
    df_asset_returns: pd.DataFrame,
    phi_t: np.ndarray,
    sigma_f_t: np.ndarray,
    sigma_epsilon_t: np.ndarray
) -> Dict[str, Any]:
    factor_component_cov = phi_t @ sigma_f_t @ phi_t.T
    theoretical_sigma_t = factor_component_cov + sigma_epsilon_t
    empirical_sigma_t = df_asset_returns.cov().to_numpy()
    is_close = np.allclose(theoretical_sigma_t, empirical_sigma_t, atol=1e-3)
    return {
        "test": "Lemma 3.1 (Covariance Decomposition)",
        "result": is_close,
        "details": f"Checked if empirical Σ_t matches theoretical Σ_t."
    }

def verify_residual_covariance_singularity(
    df_asset_returns: pd.DataFrame,
    df_tradable_factors: pd.DataFrame,
    phi_t: np.ndarray
) -> Dict[str, Any]:
    implied_residuals = df_asset_returns.values - (df_tradable_factors.values @ phi_t.T)
    residual_cov = np.cov(implied_residuals, rowvar=False)
    n_assets = df_asset_returns.shape[1]
    rank = np.linalg.matrix_rank(residual_cov)
    is_singular = rank < n_assets
    return {
        "test": "Corollary 3.4 (Residual Covariance Singularity)",
        "result": is_singular,
        "details": {
            "Number of Assets": n_assets,
            "Rank of Residual Covariance": rank,
            "Is Singular": is_singular
        }
    }

def verify_sharpe_ratio_spanning(
    df_asset_returns: pd.DataFrame,
    df_tradable_factors: pd.DataFrame,
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    mu_assets = df_asset_returns.mean().values
    sigma_assets = df_asset_returns.cov().values
    mu_factors = df_tradable_factors.mean().values
    sigma_factors = df_tradable_factors.cov().values
    sr2_assets = calculate_max_sq_sharpe_ratio(mu_assets, sigma_assets)
    sr2_factors = calculate_max_sq_sharpe_ratio(mu_factors, sigma_factors)
    diff = abs(sr2_assets - sr2_factors)
    is_equal = diff < tolerance
    return {
        "test": "Proposition 5.1 (Sharpe Ratio Spanning)",
        "result": is_equal,
        "details": {
            "Max Sq Sharpe Ratio (Full Universe)": sr2_assets,
            "Max Sq Sharpe Ratio (Factors)": sr2_factors,
            "Absolute Difference": diff,
            f"Condition (SR2_assets ≈ SR2_factors)": is_equal
        }
    }

def verify_gls_spanning_and_correlation(
    df_asset_returns: pd.DataFrame,
    df_gls_factors: pd.DataFrame,
    phi_t: np.ndarray
) -> Dict[str, Any]:
    spanning_test = verify_sharpe_ratio_spanning(df_asset_returns, df_gls_factors)
    implied_residuals = df_asset_returns.values - (df_gls_factors.values @ phi_t.T)
    df_implied_residuals = pd.DataFrame(implied_residuals, columns=[f"Res_{i+1}" for i in range(df_asset_returns.shape[1])])
    factor_residual_cov = np.cov(df_gls_factors.values, df_implied_residuals.values, rowvar=False)[
        :df_gls_factors.shape[1], df_gls_factors.shape[1]:
    ]
    is_uncorrelated = np.allclose(factor_residual_cov, 0, atol=1e-3)
    return {
        "test": "Proposition 6.3 & Lemma 6.2 (GLS Properties)",
        "result": spanning_test['result'] and is_uncorrelated,
        "details": {
            "Spanning Result": spanning_test,
            "Factor-Residual Correlation Test": {
                "Max Absolute Covariance": np.max(np.abs(factor_residual_cov)),
                "Is Uncorrelated": is_uncorrelated
            }
        }
    }
