"""Exploration of Gaussian Process parameters!

What do we want to examine in this code?

Let's try and define the search space here.

First, there is the kernel. We have a bunch of standard kernels:
 - RBF / squared exponential
 - Matern
 - Rational quadratic
 - Exponential squared
 - Constant
 - White noise
 - Linear (this is non-stationary, i.e. not just a function of |x-x'|)

 It would be interesting to plot the GP priors obtained from these kernel functions, to see what the functions
 look like before being conditioned on observations, for different parameter values.

 Then, we can combine kernels using multiplication (AND) or addition (OR).

 It would also be interesting to then look at the behaviour of these kernels when applied to interpolation and
 extrapolation of data generated from various underlying functions, with or without noise:

 """
from itertools import product
import logging
from pathlib import Path

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from gpr import gp_prior, gp_regression
from kernels import squared_exponential, constant, white, exp_sine_squared, matern, rational_quadratic

rng = np.random.default_rng()
logger = logging.getLogger(__name__)
outdir = Path("output")


def SE_prior():
    """Plot GP priors with SE kernel with various parameters
    """
    l_space = np.logspace(-3, 2, 6)
    A_space = [0.5, 1.0, 2.0]
    x = np.linspace(-0.5, 1.5, 200)

    fig, all_axs = plt.subplots(len(l_space), len(A_space), sharex=True, sharey=True, figsize=(15, 15))

    for i, (l, A) in enumerate(product(l_space, A_space)):
        logger.info(f"Sampling SE prior with l={l}, A={A}")
        y_samples, sigma = gp_prior(x, kernel_func=squared_exponential, l=l, A=A)

        ax = all_axs.ravel()[i]

        for y in y_samples:
            ax.plot(x, y, color="blue")

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, l in enumerate(l_space):
        all_axs[i, 0].set_ylabel(f"l={l}")

    for i, A in enumerate(A_space):
        all_axs[0, i].set_title(f"A={A}")

    fig.savefig(outdir / "squared_expo_prior.png")


def white_prior():
    """Plot GP priors with white noise kernel with various parameters
    """
    A_space = [0.5, 1.0, 2.0]
    x = np.linspace(-0.5, 1.5, 100)

    fig, all_axs = plt.subplots(1, len(A_space), sharex=True, sharey=True, figsize=(15, 3))

    for i, A in enumerate(A_space):
        logger.info(f"Sampling SE prior with A={A}")
        y_samples, sigma = gp_prior(x, kernel_func=white, sigma=A)

        ax = all_axs[i]

        for y in y_samples:
            ax.plot(x, y, color="blue")

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, A in enumerate(A_space):
        all_axs[i].set_title(f"A={A}")

    fig.savefig(outdir / "white_prior.png")


def constant_prior():
    """Plot GP priors with white noise kernel with various parameters
    """
    v_space = [0.5, 1.0, 2.0, 4.0, 8.0]
    x = np.linspace(-0.5, 1.5, 100)

    fig, all_axs = plt.subplots(1, len(v_space), sharex=True, sharey=True, figsize=(15, 3))

    for i, v in enumerate(v_space):
        logger.info(f"Sampling constant prior with v={v}")
        y_samples, sigma = gp_prior(x, kernel_func=constant, num_samples=20, val=v)

        ax = all_axs[i]

        for y in y_samples:
            ax.plot(x, y, color="blue")

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, v in enumerate(v_space):
        all_axs[i].set_title(f"v={v}")

    fig.savefig(outdir / "constant_prior.png")


def exp_sine_squared_prior():
    """Plot GP priors with exp sine^2 kernel with various parameters
    """
    period_space = np.logspace(-3, 1, 4, base=2)
    l_space = np.logspace(-1, 2, 4)
    x = np.linspace(-0.5, 1.5, 200)

    fig, all_axs = plt.subplots(len(l_space), len(period_space), sharex=True, sharey=True, figsize=(15, 15))

    for i, (l, period) in enumerate(product(l_space, period_space)):
        logger.info(f"Sampling SE prior with l={l}, period={period}")
        y_samples, sigma = gp_prior(x, kernel_func=exp_sine_squared, l=l, A=1.0, period=period, num_samples=5)

        ax = all_axs.ravel()[i]

        for y in y_samples:
            ax.plot(x, y)

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, l in enumerate(l_space):
        all_axs[i, 0].set_ylabel(f"l={l}")

    for i, period in enumerate(period_space):
        all_axs[0, i].set_title(f"period={period:.1f}")

    fig.savefig(outdir / "exp_sine_squared_prior.png")


def matern_prior():
    """Plot GP priors with matern kernel with various parameters
    """
    v_space = [0.5, 1.5, 2.5, 10]
    l_space = np.logspace(-1, 2, 4)
    x = np.linspace(-0.5, 1.5, 200)

    fig, all_axs = plt.subplots(len(l_space), len(v_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l, v) in enumerate(product(l_space, v_space)):
        logger.info(f"Sampling SE prior with l={l}, v={v}")
        y_samples, sigma = gp_prior(x, kernel_func=matern, l=l, A=1.0, v=v)

        ax = all_axs.ravel()[i]

        for y in y_samples:
            ax.plot(x, y, color="blue")

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, l in enumerate(l_space):
        all_axs[i, 0].set_ylabel(f"l={l}")

    for i, v in enumerate(v_space):
        all_axs[0, i].set_title(f"v={v:.1f}")

    fig.savefig(outdir / "matern_prior.png")




def rational_quadratic_prior():
    """Plot GP priors with rational quadratic kernel with various parameters
    """
    alpha_space = np.logspace(-2, 1, 4)
    l_space = [0.05, 0.2, 0.8, 3.2]
    x = np.linspace(-0.5, 1.5, 200)

    fig, all_axs = plt.subplots(len(l_space), len(alpha_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l, alpha) in enumerate(product(l_space, alpha_space)):
        logger.info(f"Sampling SE prior with l={l}, alpha={alpha}")
        y_samples, sigma = gp_prior(x, kernel_func=rational_quadratic, l=l, A=1.0, alpha=alpha)

        ax = all_axs.ravel()[i]

        for y in y_samples:
            ax.plot(x, y, color="blue")

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, l in enumerate(l_space):
        all_axs[i, 0].set_ylabel(f"l={l}")

    for i, alpha in enumerate(alpha_space):
        all_axs[0, i].set_title(f"alpha={alpha:.2g}")

    fig.savefig(outdir / "rational_quadratic_prior.png")


def sine_squared_plus_rational_quadratic_prior():
    """Plot GP priors with rational quadratic kernel added to sine squared with various parameters
    """
    l1_space = [0.05, 0.2, 0.8, 3.2]
    l2_space = [0.05, 0.2, 0.8, 3.2]
    x = np.linspace(-0.5, 1.5, 200)

    fig, all_axs = plt.subplots(len(l1_space), len(l2_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l1, l2) in enumerate(product(l1_space, l2_space)):
        logger.info(f"Sampling SE prior with l1={l1}, l2={l2}")
        kernel_func = lambda x1, x2: rational_quadratic(x1, x2, A=1.0, l=l1, alpha=1.0) + exp_sine_squared(x1, x2, A=1.0, l=l2, period=0.5)
        y_samples, sigma = gp_prior(x, kernel_func=kernel_func, num_samples=5)

        ax = all_axs.ravel()[i]

        for y in y_samples:
            ax.plot(x, y)

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, l1 in enumerate(l1_space):
        all_axs[i, 0].set_ylabel(f"l1={l1}")

    for i, l2 in enumerate(l2_space):
        all_axs[0, i].set_title(f"l2={l2:.2g}")

    fig.savefig(outdir / "exp_sine_squared_plus_rational_quadratic_prior.png")



def sine_squared_times_rational_quadratic_prior():
    """Plot GP priors with rational quadratic kernel multiplied by sine squared with various parameters
    """
    l1_space = [0.05, 0.2, 0.8, 3.2]
    l2_space = [0.05, 0.2, 0.8, 3.2]
    x = np.linspace(-0.5, 1.5, 200)

    fig, all_axs = plt.subplots(len(l1_space), len(l2_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l1, l2) in enumerate(product(l1_space, l2_space)):
        logger.info(f"Sampling SE prior with l1={l1}, l2={l2}")
        kernel_func = lambda x1, x2: rational_quadratic(x1, x2, A=1.0, l=l1, alpha=1.0) * exp_sine_squared(x1, x2, A=1.0, l=l2, period=0.5)
        y_samples, sigma = gp_prior(x, kernel_func=kernel_func, num_samples=5)

        ax = all_axs.ravel()[i]

        for y in y_samples:
            ax.plot(x, y)

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, l1 in enumerate(l1_space):
        all_axs[i, 0].set_ylabel(f"l1={l1}")

    for i, l2 in enumerate(l2_space):
        all_axs[0, i].set_title(f"l2={l2:.2g}")

    fig.savefig(outdir / "exp_sine_squared_times_rational_quadratic_prior.png")


def sine_squared_plus_rbf_prior():
    """Plot GP priors with rational quadratic kernel added to sine squared with various parameters
    """
    l1_space = [0.05, 0.2, 0.8, 3.2]
    l2_space = [0.05, 0.2, 0.8, 3.2]
    x = np.linspace(-0.5, 1.5, 200)

    fig, all_axs = plt.subplots(len(l1_space), len(l2_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l1, l2) in enumerate(product(l1_space, l2_space)):
        logger.info(f"Sampling SE prior with l1={l1}, l2={l2}")
        kernel_func = lambda x1, x2: squared_exponential(x1, x2, A=1.0, l=l1) + exp_sine_squared(x1, x2, A=1.0, l=l2, period=0.5)
        y_samples, sigma = gp_prior(x, kernel_func=kernel_func, num_samples=5)

        ax = all_axs.ravel()[i]

        for y in y_samples:
            ax.plot(x, y)

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, l1 in enumerate(l1_space):
        all_axs[i, 0].set_ylabel(f"l1={l1}")

    for i, l2 in enumerate(l2_space):
        all_axs[0, i].set_title(f"l2={l2:.2g}")

    fig.savefig(outdir / "exp_sine_squared_plus_rbf_prior.png")



def sine_squared_times_rbf_prior():
    """Plot GP priors with rbf kernel multiplied by sine squared with various parameters
    """
    l1_space = [0.05, 0.2, 0.8, 3.2]
    l2_space = [0.05, 0.2, 0.8, 3.2]
    x = np.linspace(-0.5, 1.5, 200)

    fig, all_axs = plt.subplots(len(l1_space), len(l2_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l1, l2) in enumerate(product(l1_space, l2_space)):
        logger.info(f"Sampling SE prior with l1={l1}, l2={l2}")
        kernel_func = lambda x1, x2: squared_exponential(x1, x2, A=1.0, l=l1) * exp_sine_squared(x1, x2, A=1.0, l=l2, period=0.5)
        y_samples, sigma = gp_prior(x, kernel_func=kernel_func, num_samples=5)

        ax = all_axs.ravel()[i]

        for y in y_samples:
            ax.plot(x, y)

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="50%", pad=0.1)
        cax.imshow(sigma, cmap='viridis')
        cax.axis('off')

        sigma_min = sigma.min()
        sigma_max = sigma.max()
        cax.text(0, 0, f"Sigma: \nMax={sigma_max:.2g}\nMin={sigma_min:.2g}")

    for i, l1 in enumerate(l1_space):
        all_axs[i, 0].set_ylabel(f"l1={l1}")

    for i, l2 in enumerate(l2_space):
        all_axs[0, i].set_title(f"l2={l2:.2g}")

    fig.savefig(outdir / "exp_sine_squared_times_rbf_prior.png")


def posterior_rbf_sine_low_noise():
    l1_space = [0.01, 0.05, 0.2, 0.8]
    P_space = [5, 10, 20]
    kernel = squared_exponential

    fig, all_axs = plt.subplots(len(l1_space), len(P_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l1, P) in enumerate(product(l1_space, P_space)):
        x1 = rng.uniform(0, 1, 2 * P)
        x2 = np.linspace(-0.5, 1.5, 200)
        y1 = np.sin(P * x1)
        logger.info(f"Sampling posterior with l1={l1}")

        y2, mu, sigma = gp_regression(x1, x2, y1, kernel, A=1.0, l=l1, num_samples=5, x1_noise=0.05 * np.ones_like(x1))

        ax = all_axs.ravel()[i]

        ax.plot(np.linspace(0, 1, 100), np.sin(P * np.linspace(0, 1, 100)), ls="--", color="black")
        ax.scatter(x1, y1, marker="o", color="black")

        ax.fill_between(x2, mu - np.sqrt(sigma.diagonal()), mu + np.sqrt(sigma.diagonal()), alpha=0.6)

    for i, l1 in enumerate(l1_space):
        all_axs[i, 0].set_ylabel(f"l={l1}")

    fig.savefig(outdir / "posterior_rbf_sine.png")


def posterior_rbf_sine_high_noise():
    l1_space = [0.01, 0.05, 0.2, 0.8]
    P_space = [5, 10, 20]
    kernel = squared_exponential

    fig, all_axs = plt.subplots(len(l1_space), len(P_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l1, P) in enumerate(product(l1_space, P_space)):
        x1 = rng.uniform(0, 1, 2 * P)
        x2 = np.linspace(-0.5, 1.5, 200)
        y1 = np.sin(P * x1)
        logger.info(f"Sampling posterior with l1={l1}")

        y2, mu, sigma = gp_regression(x1, x2, y1, kernel, A=1.0, l=l1, num_samples=5, x1_noise=0.5 * np.ones_like(x1))

        ax = all_axs.ravel()[i]

        ax.plot(np.linspace(0, 1, 100), np.sin(P * np.linspace(0, 1, 100)), ls="--", color="black")
        ax.scatter(x1, y1, marker="o", color="black")

        ax.fill_between(x2, mu - np.sqrt(sigma.diagonal()), mu + np.sqrt(sigma.diagonal()), alpha=0.6)

    for i, l1 in enumerate(l1_space):
        all_axs[i, 0].set_ylabel(f"l={l1}")

    fig.savefig(outdir / "posterior_rbf_sine_high_noise.png")


def posterior_expsinesquared_sine():
    l1_space = [0.05, 0.2, 0.8, 3.2]
    P_space = [5, 10, 20]
    kernel = exp_sine_squared

    fig, all_axs = plt.subplots(len(l1_space), len(P_space), sharex=True, sharey=True, figsize=(15, 10))

    for i, (l1, P) in enumerate(product(l1_space, P_space)):
        x1 = rng.uniform(0, 1, 2 * P)
        x2 = np.linspace(-1.5, 2.5, 400)
        y1 = np.sin(P * x1)
        logger.info(f"Sampling posterior with l1={l1}")

        y2, mu, sigma = gp_regression(x1, x2, y1, kernel, A=1.0, l=l1, period=1, x1_noise=0.05 * np.ones_like(x1))

        ax = all_axs.ravel()[i]

        ax.plot(np.linspace(0, 1, 100), np.sin(P * np.linspace(0, 1, 100)), ls="--", color="black")
        ax.scatter(x1, y1, marker="o", color="black")

        ax.fill_between(x2, mu - np.sqrt(sigma.diagonal()), mu + np.sqrt(sigma.diagonal()), alpha=0.6)

    for i, l1 in enumerate(l1_space):
        all_axs[i, 0].set_ylabel(f"l={l1}")

    fig.savefig(outdir / "posterior_expsinesquared_sine.png")



def posterior(kernel, kernel_args, x1, x2, y1):
    pass

def main():
    if not outdir.exists():
        outdir.mkdir()

    SE_prior()
    white_prior()
    constant_prior()
    exp_sine_squared_prior()
    matern_prior()
    rational_quadratic_prior()
    sine_squared_plus_rational_quadratic_prior()
    sine_squared_times_rational_quadratic_prior()
    sine_squared_plus_rbf_prior()
    sine_squared_times_rbf_prior()
    posterior_rbf_sine_low_noise()
    posterior_rbf_sine_high_noise()
    posterior_expsinesquared_sine()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()