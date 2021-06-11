import itertools
import warnings

import numpy as np
import scipy.optimize
import scipy.linalg

from lib.linear_sem import intsqrt, ace
from lib.linear_algebra import make_L_no_diag


def h(W):
    # tr exp(W ◦ W) − d
    d_nodes = W.shape[0]
    return np.trace(scipy.linalg.expm(np.multiply(W, W))) - d_nodes


def grad_h(W):
    # 2 W ◦ exp(W ◦ W).T
    return np.multiply(2 * W, scipy.linalg.expm(np.multiply(W, W)).transpose())


def W_from_theta(theta, L_parametrization):
    Wvec = L_parametrization @ theta
    d2 = L_parametrization.shape[0]
    d = intsqrt(d2)
    W = Wvec.reshape(d, d).T
    return W


def make_h_paramterized(M):
    def h_from_v(v):
        W = W_from_theta(v, M)
        return h(W)

    def grad_h_from_v(v):
        W = W_from_theta(v, M)
        grad_w = grad_h(W)  # gradient with respect to the matrix W
        grad_v = grad_w.T.flatten() @ M  # vectorize and apply chain rule
        return grad_v

    return h_from_v, grad_h_from_v


def make_notears_loss(sigma_hat: np.ndarray, L: np.ndarray):
    """Paramterized versions. From Remark 3.7 i mest.pdf note but with Wref=0

    a = L@v - vec(I)

    loss = 0.5 ( a.T @ (I kron SigmaHat) @ a)
    grad = L.T @ @ (I kron SigmaHat) @ a

    sigma_hat is the covariance of the data

    """
    id = np.eye(sigma_hat.shape[0])
    i = id.T.flatten()
    Q = scipy.linalg.kron(id, sigma_hat)

    def notears_loss(v: np.ndarray):
        # logging.debug(f"{L},{v},{vref},{i}")
        a = L @ v - i
        return a @ Q @ a / 2.0

    def notears_gradient(v: np.ndarray):
        a = L @ v - i
        return L.T @ Q @ a

    return notears_loss, notears_gradient


def relaxed_notears(
    data_cov: np.ndarray, L, W_initial, dag_tolerance: float, optim_opts: dict = None,
) -> dict:
    """Get Notears solution with a guarantee of zero on diagonal, to accepted tolerance
    """
    opts = optim_opts.copy() if optim_opts else {}

    f_fun, f_grad = make_notears_loss(data_cov, L)
    h_fun, h_grad = make_h_paramterized(L)
    theta_initial = np.linalg.pinv(L) @ W_initial.T.flatten()

    alpha_initial = 0.0  # lagrangian multiplier
    nitermax = opts.pop("nitermax", 100)
    s_start = opts.pop(
        "slack_start", 10.0
    )  # slack in inequality constraint. gives wierd results if the start is feasible.
    log_every = opts.pop("log_every", 10)
    rho_start = opts.pop("penalty_start", 1.0)  # quadratic penalty multiplier
    rho_max = opts.pop("penalty_max", 1e20)
    mu = opts.pop("penalty_growth_rate", 2.0)  # the rho increase factor
    constraint_violation_decrease_factor = opts.pop(
        "minimum_progress_rate", 0.25
    )  # the accepted least decrease in infeasibility
    tolerated_constraint_violation = opts.pop("tolerated_constraint_violation", 1e-12)
    verbose = opts.pop("verbose", False)
    ftol = opts.pop(
        "lbfgs_ftol", 2.220446049250313e-09
    )  # defaults from scipy documentation
    gtol = opts.pop("lbfgs_gtol", 1e-5)  # defaults from scipy documentation
    if len(opts.keys()) != 0:
        warnings.warn(f"Unknown options keys: {opts.keys()}")

    def lagrangian(theta_and_s, rho, alpha):
        theta = theta_and_s[:-1]
        s = theta_and_s[-1]
        h_fun_theta = h_fun(theta)
        c = h_fun_theta + s ** 2 - dag_tolerance
        t1 = (rho / 2) * c ** 2
        t2 = alpha * c + t1
        return f_fun(theta) + t2

    def lagrangian_grad(theta_and_s, rho, alpha):
        theta = theta_and_s[:-1]
        s = theta_and_s[-1]
        h_fun_theta = h_fun(theta)
        h_grad_theta = h_grad(theta)
        f_grad_t = f_grad(theta)
        c = h_fun_theta + s ** 2 - dag_tolerance
        t1 = alpha + rho * c
        t2 = h_grad_theta * t1
        grad_wrt_theta = f_grad_t + t2
        grad_wrt_s = 2.0 * s * t1
        return np.hstack([grad_wrt_theta, grad_wrt_s])

    def solve_inner(rho, theta_and_s, alpha):
        res = scipy.optimize.minimize(
            fun=lagrangian,
            jac=lagrangian_grad,
            x0=theta_and_s,
            args=(rho, alpha),
            method="L-BFGS-B",
            options={
                "disp": None,  # None means that the iprint argument is used
                # 'iprint':0 # 0 = one output, at last iteration
                "ftol": ftol,
                "gtol": gtol,
            },
        )
        return res["x"], res["message"]

    theta = theta_initial
    alpha = alpha_initial
    s = s_start
    rho = rho_start
    nit = 0
    solved_complete = False
    running = True
    theta_inner = None
    message = ""
    while running:

        theta_and_s_inner, _ = solve_inner(rho, np.hstack([theta, s]), alpha)
        theta_inner = theta_and_s_inner[:-1]
        s_inner = theta_and_s_inner[-1]
        h_fun_theta_inner = h_fun(theta_inner)
        h_fun_theta = h_fun(theta)
        current_inner_constraint_violation = (
            h_fun_theta_inner + s_inner ** 2 - dag_tolerance
        )
        current_constraint_violation = h_fun_theta + s ** 2 - dag_tolerance

        if current_constraint_violation != 0.0:
            while current_inner_constraint_violation >= max(
                constraint_violation_decrease_factor * current_constraint_violation,
                tolerated_constraint_violation,
            ):
                rho = mu * rho
                theta_and_s_inner, message_inner = solve_inner(
                    rho, np.hstack([theta, s]), alpha
                )
                theta_inner = theta_and_s_inner[:-1]
                s_inner = theta_and_s_inner[-1]
                h_fun_theta_inner = h_fun(theta_inner)
                current_inner_constraint_violation = (
                    h_fun_theta_inner + s_inner ** 2 - dag_tolerance
                )
                nit = nit + 1
                if rho > rho_max:
                    break
                elif nit == nitermax:
                    break
                if nit % log_every == 0 and verbose:
                    print(
                        f"nit={nit}\t|theta_and_s|={np.linalg.norm(theta_and_s_inner)}"
                        f"\trho={rho:.3g}\t|c|={current_constraint_violation}"
                        f"\tmessage_inner={message_inner}"
                    )

        if current_inner_constraint_violation < tolerated_constraint_violation:
            if verbose:
                message = "Found a feasible solution!"
                print(message)
            solved_complete = True
            running = False
        elif rho > rho_max:
            message = "Rho > rho_max. Stopping"
            warnings.warn(message)
            running = False
        elif nit >= nitermax:
            running = False
            message = "Maximum number of iterations reached. Stopping."
            warnings.warn(message)

        # Outer loop of augmented lagrangian
        alpha = alpha + rho * h_fun_theta_inner
        theta = theta_inner
        s = s_inner
        # log.info("Stepping outer")

    theta_final = theta_inner
    f_final = f_fun(theta_final)
    W_final = W_from_theta(theta_final, L)
    return dict(
        f_final=f_final,
        theta=theta_final,
        s=s,
        w=W_final,
        success=solved_complete,
        rho=rho,
        alpha=alpha,
        nit=nit,
        message=message,
    )


def mest_covarance(
    W_hat,
    data_covariance,
    L_parametrization_matrix,
    normal_data: bool = True,
    cmoment4=None,
    noise_cov: np.ndarray = None,
) -> np.ndarray:
    """Compute the Least-squares covariance matrix assuming:
    - Equality-constrained by a epsilon-relaxed h-function
    - the W matrix is parametrized by vec(W) = L*theta
    - The noise (and consequently the data, v) is centered
    
    Args:
      normal_data: use Isserlis theorem to compute 4th cross moment, assuming data is
        multivariate normal (true if data generator is Linear with normal Noise)
      cmoment4: 4dimensional tensor of cross moments (4th cross moments) E[v⊗v⊗v⊗v]
      noise_cov: latent noise matrix

    Returns:
        the covariance matrix for sqrt(n)*(theta_n - theta_circ)
    """

    if not normal_data:
        assert (
            cmoment4 is not None
        ), "Under non-normal noise, the 4th moment must be supplied"

    # aliases and simple variabels
    d = W_hat.shape[0]
    d2 = d ** 2
    id = np.eye(d)
    if noise_cov is None:
        noise_cov = id
    noise_prec = np.linalg.inv(noise_cov)
    data_cov = data_covariance
    W_I = W_hat - id
    L = L_parametrization_matrix

    #
    # Make the computations
    #

    K_expected_loss_hessian = L.T @ np.kron(np.linalg.inv(noise_cov), data_cov) @ L

    if normal_data:
        # The permutation matrix that works like P@vecop(A) = vecop(A.T)
        P = np.zeros((d2, d2))
        for i, j in itertools.product(range(d), repeat=2):
            P[d * i + j, d * j + i] = 1

        # simpler formula for J, valid when using Isserlis' theorem and data is normal
        Jtilde = (
            np.kron(W_I.T @ data_cov @ W_I, data_cov)
            + np.kron(W_I.T @ data_cov, data_cov @ W_I) @ P
        )
    else:
        varvar = np.tensordot(data_cov, data_cov, 0)
        Jtilde = np.einsum(
            "iqok,jr,pl,qr,op->ijkl",
            cmoment4 - varvar,
            noise_prec,
            noise_prec,
            W_I,
            W_I,
        ).reshape(d2, d2)
        # b[d*j+i, d*l+k] = a[i, j, k, l] is the same as b=a.reshape(d**2,d**2)
    J_score_covariance = L.T @ Jtilde @ L
    assert np.allclose(J_score_covariance, J_score_covariance.T), "Not symmetric!"

    grad_h_theta = grad_h(W_hat).T.flatten() @ L  # assume vec(W) = L@theta
    plane_normal_vec = grad_h_theta / np.linalg.norm(grad_h_theta, ord=2)
    Pi_projector = np.eye(plane_normal_vec.size) - np.outer(
        plane_normal_vec, plane_normal_vec
    )

    Pi = Pi_projector
    K = K_expected_loss_hessian
    Kinv = np.linalg.inv(K)
    J = J_score_covariance
    estimator_covariance_mat = Kinv @ Pi @ J @ Pi @ Kinv

    return estimator_covariance_mat


def ace_circ(W, noise_cov, dag_tolerance, optim_opts=None):
    """Solve the relaxed NOTEARS problem in the infinite-data limit

    assumes a linear SEM with possibly non-gaussian noise.
    assumes centered data (mean=0)
    
    returns W_circ and ace_circ"""
    d = W.shape[0]
    id = np.eye(d)
    M = np.linalg.pinv(id - W.T)
    data_cov = M @ noise_cov @ M.T
    L = make_L_no_diag(d)
    res = relaxed_notears(
        data_cov,
        L=L,
        W_initial=np.zeros((d, d)),
        dag_tolerance=dag_tolerance,
        optim_opts=optim_opts,
    )
    assert res["success"]
    W_circ = res["w"]
    theta = res["theta"]
    ace_circ = ace(theta, L=L)
    return W_circ, ace_circ
