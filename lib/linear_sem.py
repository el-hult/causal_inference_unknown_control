import numpy as np
import scipy.linalg


def intsqrt(d2):
    """Takes the square root, and assert it is an int"""
    d = np.sqrt(d2)
    if d != int(d):
        raise ValueError("Supplied value is not a perfect square")
    return int(d)


def ace(theta, L):
    """Compute the Average Causal Effect between node 0 and node 1 in the SEM
    specified by the parameter vector theta and parametrization matrix L
    
    N.B. does not depend on the noise covariance. See lemma 1 in the article.
    """
    d2 = L.shape[0]
    d = intsqrt(d2)
    Z = np.eye(d)
    Z[0, 0] = 0
    vecW = L @ theta
    W = vecW.reshape(d, d).T
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    return M[1, 0]


def ace_grad(theta, L):
    """Compute the gradient of the causal effect
    under linearity assumptions and vec(W)=L@theta"""
    d2 = L.shape[0]
    d = intsqrt(d2)
    Z = np.eye(d)
    Z[0, 0] = 0
    vecW = L @ theta
    W = vecW.reshape(d, d).T
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    MZ = M @ Z
    prod = scipy.linalg.kron(MZ, M.T)
    myMat = prod @ L
    return myMat[d, :]


selected_graphs = {
    "2forward": np.array([[0, 0.4], [0, 0]]),
    "2backwards": np.array([[0, 0], [0.4, 0]]),
    "3fork": np.array([[0, 0.4, 0], [0, 0, 0], [0.7, 0.2, 0]]),
    "3mediator": np.array([[0, 0.4, 0.7], [0, 0, 0], [0, 0.2, 0]]),
    "3collider": np.array([[0, 0, 0.7], [0, 0, 0.2], [0, 0, 0]]),
    "4collider": np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]]),
    "calibration": np.array(
        [
            [0.0, -1.0, 1.6, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.2, 0.0, -0.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ),
    "random10.1": np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.77],
            [0.0, 0.0, 1.79, 0.0, 0.0, 0.0, 0.69, 1.45, 0.0, 0.0],
            [-0.97, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.52, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.76, 0.0, -0.6],
            [1.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.58, 0.0, -1.8, -1.75, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.98, 0.0, 0.0],
        ]
    ),
}
