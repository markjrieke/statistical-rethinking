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
    similar slow down in the afternoon we can model using our previous
    methods:

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
set_ulam_cmdstan(FALSE)

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

### 14.1.2 Simulate observations

-   The above gives the properties of 20 individual cafes.
-   Now let’s simulate 10 visits at each — 5 in the morning and 5 in the
    afternoon.

``` r
set.seed(22)
N_visits <- 10
afternoon <- rep(0:1, N_visits * N_cafes/2)
cafe_id <- rep(1:N_cafes, each = N_visits)
mu <- a_cafe[cafe_id] + b_cafe[cafe_id]*afternoon
sigma <- 0.5 # std dev within cafes
wait <- rnorm(N_visits * N_cafes, mu, sigma)
d <- data.frame(cafe = cafe_id, afternoon = afternoon, wait = wait)

str(d)
```

    ## 'data.frame':    200 obs. of  3 variables:
    ##  $ cafe     : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ afternoon: int  0 1 0 1 0 1 0 1 0 1 ...
    ##  $ wait     : num  3.97 3.86 4.73 2.76 4.12 ...

-   Some rethinking — here we’re simulating data then analyzing with a
    model that reflects the exact correct structure of the data
    generating process. In the real world, we’re never so lucky.
-   We’re always forced to analyze data with a model that is
    *misspecified* — that is, the model is different than the true data
    generating process.

### 14.1.3 The varying slopes model

-   Now let’s do the reverse!

$$
\begin{gather}
W_i \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i = \alpha_{\text{CAFE}[i]} + \beta_{\text{CAFE}[i]} A_i \\
\begin{bmatrix}
\alpha_{\text{CAFE}} \\
\beta_{\text{CAFE}}
\end{bmatrix}
\sim \text{MVNormal}
\left(\begin{bmatrix} \alpha \\ \beta \end{bmatrix}, \text{S}\right) \\
\text{S} = \begin{pmatrix} \sigma_\alpha & 0 \\ 0 & \sigma_\beta \end{pmatrix} \text{R} \begin{pmatrix} \sigma_\alpha & 0 \\ 0 & \sigma_\beta \end{pmatrix} \\
\alpha \sim \text{Normal}(5, 2) \\
\beta \sim \text{Normal}(-1, 0.5) \\
\sigma \sim \text{Exponential}(1) \\
\sigma_\alpha \sim \text{Exponential}(1) \\
\sigma_\beta \sim \text{Exponential}(1) \\
\text{R} \sim \text{LKJcorr}(2)
\end{gather}
$$

-   Okay, lots to unpack here — the wait time $W_i$ is defined by a
    linear model.
-   The terms of that linear model stat that each cafe has an intercept
    $\alpha_{\text{CAFE}}$ and slope $\beta_{\text{CAFE}}$ whose prior
    distribution is defined by the two-dimensional Gaussian with means
    $\alpha$ and $\beta$ and covariance matrix $\text{S}$.
-   This prior will adaptively regularize the individual intercepts,
    slopes, and the correlation among them.
-   $\text{S}$ can be factored into separate standard deviations
    $\sigma_\alpha$ and $\sigma_\beta$ and a correlation matrix
    $\text{R}$.
-   Then come all our hyper-priors that do the adaptive regularization.
-   The prior for $\text{R}$ is a “distribution of matrices”. In our
    case, $\text{R}$ looks like this:

$$
\begin{gather}
R = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}
\end{gather}
$$

-   $\rho$ is the correlation between intercepts and slopes (in larger
    matrices with additional varying slopes, it gets more complicated).
-   A $\text{LKJcorr}$ distribution takes a single parameter $\eta$ to
    control how skeptical the prior is of extremely high correlations.
-   $\eta = 1$ is a flat prior and $\eta = 2$ means that the prior is
    weakly skeptical of extreme correlations near -1 or +1.
-   Here’s a few visualizations just for kicks:

``` r
R <- rlkjcorr(1e4, K = 2, eta = 2)
dens(R[,1,2], xlab = "correlation")
```

![](chapter_14_notes_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

-   Let’s model! (note the use of `c()`)

``` r
set.seed(867530)
m14.1 <-
  ulam(
    alist(
      # model
      wait ~ normal(mu, sigma),
      mu <- a_cafe[cafe] + b_cafe[cafe] * afternoon,
      
      # priors
      c(a_cafe, b_cafe)[cafe] ~ multi_normal(c(a, b), Rho, sigma_cafe),
      
      # hyper-priors
      a ~ normal(5, 2),
      b ~ normal(-1, 0.5),
      sigma_cafe ~ exponential(1),
      sigma ~ exponential(1),
      Rho ~ lkj_corr(2)
    ),
    
    data = d,
    chains = 4,
    cores = 4
  )
```

-   The `multi_normal` distribution takes a vector of means, `c(a, b)`,
    a correlation matrix, `Rho`, and a vector of standard deviations,
    `sigma_cafe`, then constructs the matrix internally.
-   The `stancode()` shows how.

``` r
stancode(m14.1)
```

    ## data{
    ##      vector[200] wait;
    ##     array[200] int afternoon;
    ##     array[200] int cafe;
    ## }
    ## parameters{
    ##      vector[20] b_cafe;
    ##      vector[20] a_cafe;
    ##      real a;
    ##      real b;
    ##      vector<lower=0>[2] sigma_cafe;
    ##      real<lower=0> sigma;
    ##      corr_matrix[2] Rho;
    ## }
    ## model{
    ##      vector[200] mu;
    ##     Rho ~ lkj_corr( 2 );
    ##     sigma ~ exponential( 1 );
    ##     sigma_cafe ~ exponential( 1 );
    ##     b ~ normal( -1 , 0.5 );
    ##     a ~ normal( 5 , 2 );
    ##     {
    ##     vector[2] YY[20];
    ##     vector[2] MU;
    ##     MU = [ a , b ]';
    ##     for ( j in 1:20 ) YY[j] = [ a_cafe[j] , b_cafe[j] ]';
    ##     YY ~ multi_normal( MU , quad_form_diag(Rho , sigma_cafe) );
    ##     }
    ##     for ( i in 1:200 ) {
    ##         mu[i] = a_cafe[cafe[i]] + b_cafe[cafe[i]] * afternoon[i];
    ##     }
    ##     wait ~ normal( mu , sigma );
    ## }

-   Let’s jump into inspecting the posterior distribution.

``` r
plot_posterior_correlation <- function(model, eta) {
  
  post <- extract.samples(model)
  dens(post$Rho[,1,2], xlim = c(-1, 1), col = rangi2, lwd = 1) # posterior
  R <- rlkjcorr(1e4, K = 2, eta) # prior
  dens(R[,1,2], add = TRUE, lty = 2)
  
}

plot_posterior_correlation(m14.1, 2)
```

![](chapter_14_notes_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

-   The posterior distribution of correlation between intercepts and
    slopes is mostly negative, whereas the prior is spread evenly.
-   Per McElreath’s suggestion, let’s fit models with different priors
    for the correlation.

``` r
# flat prior model
m14.1_flat <-
  ulam(
    alist(
      # model
      wait ~ normal(mu, sigma),
      mu <- a_cafe[cafe] + b_cafe[cafe]*afternoon,
      
      # priors
      c(a_cafe, b_cafe)[cafe] ~ multi_normal(c(a, b), Rho, sigma_cafe),
      
      # hyper-priors
      a ~ normal(5, 2),
      b ~ normal(-1, 0.5),
      sigma_cafe ~ exponential(1),
      sigma ~ exponential(1),
      Rho ~ lkj_corr(1)
    ),
    
    data = d,
    chains = 4,
    cores = 4
  )

plot_posterior_correlation(m14.1_flat, 1)
```

![](chapter_14_notes_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
# strongly regularized
m14.1_strong <-
  ulam(
    alist(
      # model
      wait ~ normal(mu, sigma),
      mu <- a_cafe[cafe] + b_cafe[cafe]*afternoon,
      
      # priors
      c(a_cafe, b_cafe)[cafe] ~ multi_normal(c(a, b), Rho, sigma_cafe),
      a ~ normal(5, 2),
      b ~ normal(-1, 0.5),
      sigma_cafe ~ exponential(1),
      sigma ~ exponential(1),
      Rho ~ lkj_corr(4)
    ),
    
    data = d,
    chains = 4, 
    cores = 4
  )

plot_posterior_correlation(m14.1_strong, 4)
```

![](chapter_14_notes_files/figure-gfm/unnamed-chunk-11-2.png)<!-- -->

-   Next, let’s consider shrinkage. Information is pooled across
    intercepts and slopes of each cafe through an inferred correlation.
-   Let’s plot the posterior mean varying effects compared to the raw
    unpooled estimates to see the consequence of shrinkage.

``` r
# compute the unpooled estimates directly from the data
a1 <- sapply(1:N_cafes, function(i) mean(wait[cafe_id == i & afternoon == 0]))
b1 <- sapply(1:N_cafes, function(i) mean(wait[cafe_id == i & afternoon == 1])) - a1

# extract posterior means of partially pooled estimates
post <- extract.samples(m14.1)
a2 <- apply(post$a_cafe, 2, mean)
b2 <- apply(post$b_cafe, 2, mean)

# plot both & connect with lines
plot(
  a1, 
  b1,
  xlab = "intercept",
  ylab = "slope",
  pch = 16, 
  col = rangi2,
  ylim = c(min(b1) - 0.1, max(b1) + 0.1),
  xlim = c(min(a1) - 0.1, max(a1) + 0.1)
)
points(a2, b2, pch = 1)
for (i in 1:N_cafes) lines(c(a1[i], a2[i]), c(b1[i], b2[i]))

# compute the posterior mean bivariate gaussian
Mu_est <- c(mean(post$a), mean(post$b))
rho_est <- mean(post$Rho[,1,2])
sa_est <- mean(post$sigma_cafe[,1])
sb_est <- mean(post$sigma_cafe[,2])
cov_ab <- sa_est*sb_est*rho_est
Sigma_est <- matrix(c(sa_est^2, cov_ab, cov_ab, sb_est^2), ncol = 2)

# draw contours
for (i in c(0.1, 0.3, 0.5, 0.7, 0.9))
  lines(ellipse::ellipse(Sigma_est, centre = Mu_est, level = i),
        col = col.alpha("black", 0.2))
```

![](chapter_14_notes_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

-   The open points are the posterior means & the blue points are
    unpooled estimates.
-   Note that shrinkage is not always in a direct line towards the
    center.
-   We can also look at the same information on the outcome scale

``` r
# convert varying effects to waiting times
wait_morning_1 <- a1
wait_afternoon_1 <- a1 + b1
wait_morning_2 <- a2
wait_afternoon_2 <- a2 + b2

# plot both & connect with lines
plot(
  wait_morning_1,
  wait_afternoon_1,
  xlab = "morning wait",
  ylab = "afternoon wait",
  pch = 16,
  col = rangi2,
  xlim = c(min(wait_morning_1) - 0.1, max(wait_morning_1) + 0.1),
  ylim = c(min(wait_afternoon_1) - 0.1, max(wait_afternoon_1) + 0.1)
)
points(wait_morning_2, wait_afternoon_2, pch = 1)
for (i in 1:N_cafes)
  lines(c(wait_morning_1[i], wait_morning_2[i]),
        c(wait_afternoon_1[i], wait_afternoon_2[i]))

# add line for y = x
abline(a = 0, b = 1, lty = 2)

# add shrinkage distribution by simulation
v <- MASS::mvrnorm(1e4, Mu_est, Sigma_est)
v[,2] <- v[,1] + v[,2] # calculate afternoon wait
Sigma_est2 <- cov(v)
Mu_est2 <- Mu_est
Mu_est2[2] <- Mu_est[1] + Mu_est[2]

# draw contours
for (i in c(0.1, 0.3, 0.5, 0.7, 0.9))
  lines(ellipse::ellipse(Sigma_est2, centre = Mu_est2, level = i),
        col = col.alpha("black", 0.5))
```

![](chapter_14_notes_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

-   Appreciate the fact that shrinkage on the parameter scale
    *naturally* produces shrinkage on the scale we actually care about:
    the outcome scale.
-   Also, there is an implied population of wait times shown by the gray
    contours.
-   That population is potively correlated — cafes with long morning
    waits tend to also have long afternoon waits. But the wait times in
    the morning across the board are generally longer than the afternoon
    (under the dashed line).
