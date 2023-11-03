# gaussian_process_intuition
Simple visualisation of some GP priors and posteriors to help build intuition. Some examples shown in this readme.


## Priors

A gaussian process is a distribution over functions, with the key property that when you *marginalise* over a finite subset of points in the function domain, you have a multivariate Gaussian distribution.

The types of functions which can be obtained from a GP depend on the *covariance function*, or kernel. This tells us the covariance between a pair of positions.


### RBF

The most common kernel is the RBF, or squared exponential:

$$
k(x_i, x_j) = A \exp(- \frac{(x_i - x_j)^2}{2 l^2})
$$

The image below shows a bunch of sample functions from the multivariate normal obtained using a covariance matrix based on sampling $x$ in $[-0.5, 1.5]$, as well as the covariance matrix in each case.

![squared_expo_prior](https://github.com/murraycutforth/gaussian_process_intuition/assets/11088372/ddde345d-0ae7-4732-900b-4e827f8c1e50)


### Exp-sine-squared

$$
k(x_i, x_j) = \exp(- \frac{2}{l^2} \sin^2(\frac{\pi |x_i - x_j|_2}{P}))
$$

This is commonly used to model periodic functions, with period $P$.

![exp_sine_squared_prior](https://github.com/murraycutforth/gaussian_process_intuition/assets/11088372/3019713b-1562-416f-ab67-108b2de61df0)


### Combining kernel functions: addition

$$
k(x_i, x_j) = \exp(- \frac{2}{l^2} \sin^2(\frac{\pi |x_i - x_j|_2}{P})) + \exp(- \frac{(x_i - x_j)^2}{2 l^2})
$$

Similar to an "OR" operation:

![exp_sine_squared_plus_rbf_prior](https://github.com/murraycutforth/gaussian_process_intuition/assets/11088372/cbb02cdc-3833-4bd3-8155-050c04d23874)


### Combining kernel functions: multiplication

$$
k(x_i, x_j) = \exp(- \frac{2}{l^2} \sin^2(\frac{\pi |x_i - x_j|_2}{P})) \exp(- \frac{(x_i - x_j)^2}{2 l^2})
$$

Similar to "AND" operation:

![exp_sine_squared_times_rbf_prior](https://github.com/murraycutforth/gaussian_process_intuition/assets/11088372/9b5195cd-29af-425a-8ae9-4bf83efb7d05)


## Posteriors

Given some function values $f_1$ at positions $x_1$, we can compute the posterior distribution $p(f_2 | x_1, x_2, f_1)$, which is also a multivariate normal (since $f_1$ and the unknown $f_2$ are jointly Gaussian).
This is super important so I'll reproduce it here.

With $y_i = f(X_i)$

$$
\left(\begin{matrix} f_1 \\
f_2 \end{matrix} \right) \sim \mathcal{N}\left(
\left(\begin{matrix} \mu_1 \\
\mu_2 \end{matrix} \right),
\left(\begin{matrix} \Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22} \end{matrix} \right) \right)
$$

Then

$$
p(y_2 | y_1) = 
\mathcal{N}\left(\Sigma_{21} \Sigma_{11}^{-1} y_1, \Sigma_{22} - \Sigma_{21} \Sigma_{11}^{-1} \Sigma_{12} \right)
$$

Using the assumption that both mean functions are zero, which to my understanding can be satisfied by preprocessing the data.
The standard deviation of each individual random variable is just the square root of the corresponding element on the diagonal of the conditional covariance matrix.

![posterior_rbf_sine](https://github.com/murraycutforth/gaussian_process_intuition/assets/11088372/4fd71fa2-baae-4087-8f10-4a97b460dea7)


We can account for uncertainty in our observations by adding a diagonal matrix $\sigma^2 I$ to the covariance matrix:

![posterior_rbf_sine_high_noise](https://github.com/murraycutforth/gaussian_process_intuition/assets/11088372/4ac535aa-3eb9-4e33-bca2-660b103a382c)


And we can use a exp-sin-squared kernel, in this case with a period of 1.

![posterior_expsinesquared_sine](https://github.com/murraycutforth/gaussian_process_intuition/assets/11088372/9a81545a-ac0d-4665-a275-41548d3d503a)
