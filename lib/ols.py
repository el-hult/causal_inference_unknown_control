import numpy as np


def myOLS(X, y):
    """
    make a very standard regression, reporting heteroscadastic consistent errors

    Notes:
        Scott Long & Laurie H. Ervin (2000) Using Heteroscedasticity Consistent
        Standard Errors in the Linear Regression Model, The American Statistician,
        54:3, 217-224, DOI: 10.1080/00031305.2000.10474549
            for simple statement of equations for HCx
        J. M. Wooldridge, Econometric analysis of cross section and panel data, 2nd ed.
        Cambridge, Mass: MIT Press, 2010.
            a general derivation for HC0, that works also in nonlinear least squares is
            in equation 12.52
    """
    regressors = X
    dependant_variable = y
    beta_hat, _, _, _ = np.linalg.lstsq(regressors, dependant_variable, rcond=None)
    Q = regressors.T @ regressors
    Q_inv = np.linalg.pinv(Q)
    predictions = beta_hat @ regressors.T
    residuals = dependant_variable - predictions
    n = dependant_variable.size
    p = beta_hat.size
    s2 = residuals @ residuals / (n - p)  # OLS estimate of noise
    OLS_se = np.sqrt(s2 * np.diag(Q_inv))
    tmp = regressors @ Q_inv
    tmp2 = (residuals[:, np.newaxis] ** 2) * tmp
    # trick:  np.diag(a)@B = a[:,np.newaxis] * B, but the latter is MUCH faster. :)
    HC0_se = np.sqrt(np.einsum("ji,ji->i", tmp, tmp2))
    # trick: np.diag(A.T@ B) = np.einsum('ji,ji->i',A,B)
    HC1_se = HC0_se * n / (n - p)  # small sample correction
    return dict(params=beta_hat, bse=OLS_se, HC0_se=HC0_se, HC1_se=HC1_se)
