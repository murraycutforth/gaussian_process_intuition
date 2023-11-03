import numpy as np

rng = np.random.default_rng()


def gp_prior(x, kernel_func, num_samples=10, **kernel_kwargs):
    """Draw num_samples different GP priors
    """
    sigma = kernel_func(x, x, **kernel_kwargs)
    mu = np.zeros_like(x)
    y_samples = [rng.multivariate_normal(mu, sigma) for _ in range(num_samples)]

    return y_samples, sigma


def gp_regression(x1, x2, y1, kernel_func, x1_noise=None, num_samples=10, **kernel_kwargs):
    """Evaluate the posterior P(y_2 | x_2, x_1, y_1), which is also a multivariate normal
    """
    sigma_11 = kernel_func(x1, x1, **kernel_kwargs)
    sigma_21 = kernel_func(x2, x1, **kernel_kwargs)
    sigma_22 = kernel_func(x2, x2, **kernel_kwargs)

    if x1_noise is not None:
        sigma_11 += np.diag(x1_noise)

    mu_posterior = sigma_21 @ np.linalg.solve(sigma_11, y1)
    sigma_posterior = sigma_22 - sigma_21 @ np.linalg.solve(sigma_11, sigma_21.T)

    y_2_samples = [rng.multivariate_normal(mu_posterior, sigma_posterior) for _ in range(num_samples)]

    return y_2_samples, mu_posterior, sigma_posterior
