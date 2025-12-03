# Linear Factor Models Implementation

## Overview

This project provides a Python implementation of linear factor models, designed to simulate asset returns and verify key theoretical propositions and lemmas in financial economics. It focuses on the relationship between asset returns, factor models, and covariance structures, providing a practical tool for understanding these concepts.

The core functionality includes:
- **Data Simulation**: Generating synthetic asset returns based on factor structures.
- **Factor Construction**: Implementing Ordinary Least Squares (OLS) and Generalized Least Squares (GLS) methods for factor extraction.
- **Theoretical Verification**: rigorous testing of theoretical claims such as Covariance Decomposition (Lemma 3.1), Residual Covariance Singularity (Corollary 3.4), and Sharpe Ratio Spanning (Proposition 5.1).

## Project Structure

The project is organized as follows:

```
linear-factor-models/
├── notebooks/          # Jupyter notebooks for demonstration and results
│   └── tt.ipynb   # Main notebook showcasing simulations and verifications
├── src/                # Source code for the project
│   ├── data_simulation.py    # Modules for simulating asset returns and shocks
│   ├── factor_construction.py # OLS and GLS factor construction algorithms
│   ├── verification_tests.py  # Functions to verify theoretical propositions
│   └── utils.py              # Utility functions (e.g., positive semi-definite checks)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd linear-factor-models
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Demonstration Notebook

The primary way to interact with this project is through the `tt.ipynb` notebook. This notebook walks through the entire process of simulation, factor construction, and verification.

To run the notebook:
```bash
jupyter lab notebooks/tt.ipynb
```

### Using the Modules Directly

You can also use the Python modules in your own scripts. Here is a basic example:

```python
import numpy as np
from src.data_simulation import create_characteristics_matrix, simulate_asset_returns
from src.factor_construction import construct_ols_factors

# 1. Setup parameters
n_assets = 100
n_factors = 3
n_periods = 200

# 2. Create characteristics
phi_t = create_characteristics_matrix(n_assets, n_factors)

# 3. Define covariance matrices (simplified for example)
sigma_f_t = np.eye(n_factors)
sigma_epsilon_t = np.eye(n_assets)

# 4. Simulate returns
simulation_data = simulate_asset_returns(
    phi_t=phi_t,
    sigma_f_t=sigma_f_t,
    sigma_epsilon_t=sigma_epsilon_t,
    n_periods=n_periods
)
df_returns = simulation_data["asset_returns"]

# 5. Construct Factors
df_factors = construct_ols_factors(df_returns, phi_t)

print(df_factors.head())
```

## Key Theoretical Verifications

The `src/verification_tests.py` module contains functions to verify the following:

*   **Lemma 3.1 (Covariance Decomposition)**: Checks if the empirical covariance matrix matches the theoretical decomposition into factor and residual components.
*   **Corollary 3.4 (Residual Covariance Singularity)**: Verifies that the residual covariance matrix is singular when the number of assets exceeds the number of factors.
*   **Proposition 5.1 (Sharpe Ratio Spanning)**: Tests if the maximum squared Sharpe ratio of the full asset universe is spanned by the factors.
*   **Proposition 6.3 & Lemma 6.2 (GLS Properties)**: Verifies properties related to Generalized Least Squares factors, including spanning and orthogonality with residuals.
