Adventures in Covariance
================

-   Consider fitting a model to estimate the wait time at a variety of
    cafes. We might estimate that the average wait time varies by cafe,
    and can represent with a hierarchical model:

$$
\begin{gather}
\mu_i = \alpha_{\text{CAFE}[i]}
\end{gather}
$$

-   The time of day may matter as well — if all cafes experience a
    similar slow down in the afternoon ($A_i$), we can model using our
    previous methods:

$$
\begin{gather}
\mu_i = \alpha_{\text{CAFE}[i]} + \beta A_i
\end{gather}
$$

-   If, however, different cafes had different changes in the afternoon,
    the slope parameter can also be modeled hierarchically:

$$
\begin{gather}
\mu_i = \alpha_{\text{CAFE}[i]} + \beta_{\text{CAFE}[i]} A_i
\end{gather}
$$

-   This is also known as a *varying effects* model.
-   Here’s a fact that will allow us to squeeze more info out of the
    data: there is *covariance* in the intercept/slopes!
-   For example, at a really popular cafe, the wait will be really long
    in the morning, but less so in the afternoon. At a bone-dry cafe,
    the difference between morning and afternoon will be less
    noticeable, because you can expect a short wait time even in the
    morning.
-   We can use this covariance to pool information across parameter
    types — intercepts and slopes.
-   This chapter will explore the varying effects strategy & extend to
    more subtle model types, such as *Gaussian Processes*.
-   Ordinary varying effects only work with discrete, unordered
    categories like country or individual, but we can also use pooling
    with continuous categories like age or location.
-   We’ll also toe-dip back into causal inference and introduce
    *instrumental variables*, which allow ways of inferring cause
    without closing backdoor paths in a DAG.
-   The material in this chapter is difficult!

## 14.1 Varying slopes by construction

-   A joint multivariate Gaussian distribution is generally used to
    model the covariance of the intercepts and slopes.
-   There’s not inherent reason why the multivariate distribution of
    slopes/intercepts must be Gaussian, but there are practical and
    epidemiological reasons:
    -   **practical**: there aren’t many multivariate distributions that
        are easy to work with — the only common ones are the MV Gaussian
        & MV Student-t.
    -   **epistemological**: if all we want to say about the
        intercepts/slopes is their means, variances, and covariances,
        then the Gaussian is the maximum entropy distribution.
-   Let’s simulate some data from coffee shops to get a better idea.

### 14.1.1 Simulate the population

``` r
a <- 3.5        # average morning wait time
b <- -1         # average difference in afternoon wait time
sigma_a <- 1    # std dev in intercepts
sigma_b <- 0.5  # std dev in slopes
rho <- -0.7     # correlation between intercepts and slopes
```

-   To use these values to simulate a sample of cafes, we need a 2-d MV
    Gaussian which requires:
    1.  a vector of 2 means
    2.  a 2-by-2 matrix of variances and covariances
-   The means are easy:

``` r
Mu <- c(a, b)
```

-   The matrix of variances is arranged like this:

$$
\begin{gather}
\begin{pmatrix}
\text{variance of intercepts} & \text{covariance of intercepts and slopes} \\
\text{covariance of intercepts and slopes} & \text{variance of slopes}
\end{pmatrix}
\end{gather}
$$

$$
\begin{gather}
\begin{pmatrix}
\sigma_{\alpha}^2 & \sigma_{\alpha} \sigma_{\beta} \rho \\
\sigma_{\alpha} \sigma_{\beta} \rho & \sigma_{\beta}^2
\end{pmatrix}
\end{gather}
$$

-   We can build this covariance matrix in a few ways. The first is to
    use `matrix()`, which takes a vector and starts filling in a matrix
    by column:

``` r
# here's how the matrix is filled
matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
```

    ##      [,1] [,2]
    ## [1,]    1    3
    ## [2,]    2    4

``` r
# construct the matrix directly
cov_ab <- sigma_a * sigma_b * rho
Sigma <- matrix(c(sigma_a^2, cov_ab, cov_ab, sigma_b^2), ncol = 2)
Sigma
```

    ##       [,1]  [,2]
    ## [1,]  1.00 -0.35
    ## [2,] -0.35  0.25

-   Alternatively, we can treat the standard deviations and covariances
    separately then use matrix multiplication to combine:

``` r
sigmas <- c(sigma_a, sigma_b) # std devs
Rho <- matrix(c(1, rho, rho, 1), nrow = 2) # correlation matrix
Sigma <- diag(sigmas) %*% Rho %*% diag(sigmas)
Sigma
```

    ##       [,1]  [,2]
    ## [1,]  1.00 -0.35
    ## [2,] -0.35  0.25

-   To simulate the properties of each cafe, we just sample randomly
    from the MV Gaussian:

``` r
library(rethinking)
```

    ## Loading required package: rstan

    ## Warning: package 'rstan' was built under R version 4.2.1

    ## Loading required package: StanHeaders

    ## Warning: package 'StanHeaders' was built under R version 4.2.1

    ## 
    ## rstan version 2.26.13 (Stan version 2.26.1)

    ## For execution on a local, multicore CPU with excess RAM we recommend calling
    ## options(mc.cores = parallel::detectCores()).
    ## To avoid recompilation of unchanged Stan programs, we recommend calling
    ## rstan_options(auto_write = TRUE)
    ## For within-chain threading using `reduce_sum()` or `map_rect()` Stan functions,
    ## change `threads_per_chain` option:
    ## rstan_options(threads_per_chain = 1)

    ## Do not specify '-march=native' in 'LOCAL_CPPFLAGS' or a Makevars file

    ## Loading required package: cmdstanr

    ## This is cmdstanr version 0.5.3

    ## - CmdStanR documentation and vignettes: mc-stan.org/cmdstanr

    ## - CmdStan path: C:/Users/E1735399/Documents/.cmdstan/cmdstan-2.30.1

    ## - CmdStan version: 2.30.1

    ## 
    ## A newer version of CmdStan is available. See ?install_cmdstan() to install it.
    ## To disable this check set option or environment variable CMDSTANR_NO_VER_CHECK=TRUE.

    ## Loading required package: parallel

    ## rethinking (Version 2.23)

    ## 
    ## Attaching package: 'rethinking'

    ## The following object is masked from 'package:rstan':
    ## 
    ##     stan

    ## The following object is masked from 'package:stats':
    ## 
    ##     rstudent

``` r
# simulate intercepts/slopes
N_cafes <- 20
set.seed(5)
vary_effects <- MASS::mvrnorm(N_cafes, Mu, Sigma)
vary_effects
```

    ##           [,1]       [,2]
    ##  [1,] 4.223962 -1.6093565
    ##  [2,] 2.010498 -0.7517704
    ##  [3,] 4.565811 -1.9482646
    ##  [4,] 3.343635 -1.1926539
    ##  [5,] 1.700971 -0.5855618
    ##  [6,] 4.134373 -1.1444539
    ##  [7,] 3.794469 -1.6264661
    ##  [8,] 3.946598 -1.7152794
    ##  [9,] 3.864267 -0.9071677
    ## [10,] 3.467614 -0.6804054
    ## [11,] 2.242875 -0.6181516
    ## [12,] 4.159506 -1.6592120
    ## [13,] 4.300283 -2.1125474
    ## [14,] 3.506948 -1.4406430
    ## [15,] 4.382086 -1.8798983
    ## [16,] 3.521133 -1.3506986
    ## [17,] 4.216713 -0.9192799
    ## [18,] 5.913003 -1.2313624
    ## [19,] 3.477306 -0.3570341
    ## [20,] 3.774899 -1.0570457

``` r
# assign to vectors
a_cafe <- vary_effects[,1]
b_cafe <- vary_effects[,2]

# plot!
plot(a_cafe, b_cafe,
     col = rangi2,
     xlab = "intercepts (`a_cafe`)",
     ylab = "slopes (`b_cafe`)")

# overlay population distribution
for (i in c(0.1, 0.3, 0.5, 0.8, 0.99))
  lines(ellipse::ellipse(Sigma, centre = Mu, level = i), 
        col = col.alpha("black", 0.2))
```

![](chapter_14_notes_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->
