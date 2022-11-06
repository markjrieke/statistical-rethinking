Geocentric Models
================

-   Ptolomey’s geocentric model of the solar system using epicycles
    could provide useful information about the future location of the
    planets, but was wrong.
-   *Linear regression* is our geocentric golem — it’s often not
    necessarily correct, but can be useful.

## 4.1 Why normal distributions are normal

### 4.1.1 Normal by addition

-   Let’s simulate 16 random steps forward/backward for 1000 people.
-   Even though these are effectively coin flips, we’ll end up with a
    normal-ish distribution of of final position at step 16 for these
    1000 people.

``` r
pos <- replicate(1000, sum(runif(16, -1, 1)))

hist(pos)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
plot(density(pos))
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-1-2.png)<!-- -->

-   Any process that adds together random values from the same
    distribution converges to a normal distribution.

### 4.1.2 Normal by multiplication

-   Let’s the growth rate of an organism is determined by a a dozen loci
    that interact, such that each increase growth by a percentage. This
    means the effects multiply. Here’s an example of 1:

``` r
prod(1 + runif(12, 0, 0.1))
```

    ## [1] 1.855755

-   Now let’s repeat this for 10,000 organisms:

``` r
growth <- replicate(10000, prod(1 + runif(12, 0, 0.1)))
rethinking::dens(growth, norm.comp = TRUE)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

-   This is because small multiplications are approximately additive
    (e.g., 1.1 \* 1.1 = 1.21)

``` r
big <- replicate(10000, prod(1 + runif(12, 0, 0.5)))
small <- replicate(10000, prod(1 + runif(12, 0, 0.01)))

rethinking::dens(big)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
rethinking::dens(small)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

### 4.1.3 Normal by log-multiplication

-   Large deviates that are multiplied together don’t produce gaussian
    distributions, but they do generate log-normal distributions

``` r
log.big <- replicate(10000, log(prod(1 + runif(12, 0, 0.5))))
rethinking::dens(log.big)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### 4.1.4 Using Gaussian distributions

-   There are two justifications for using Gaussian distributions:
    -   **Ontological**: The world is full of Gaussian distributions.
        Other members of the *Exponential Family* of distributions
        (Gamma, Poisson) also arise in nature.
    -   **Epistemological**: When all we know or are willing to say
        about a distribution is their mean and variance, then ghe
        Gaussian is the most natural way to express this. This is also
        justified with information theory via measures of *maximum
        entropy*.

## 4.2 A language for describing models

1.  We have a set of variables to work with. Some of these are
    observable — we call these *data*. Others are unobservable — we call
    these *parameters*.
2.  We define each variable either in terms of other variables or in
    terms of a probability distribution.
3.  The combination of variables and their probability distributions
    defines a *joint generative model*.

-   The biggest difficulty in working with this framework is the subject
    matter — which variables matter and how does theory tell us to
    connect them?

### 4.2.1 Re-describing the globe tossing model

$$
\\begin{gather}
W \\sim Binomial(N, p) \\\\
p \\sim Uniform(0, 1)\\\\
\\end{gather}
$$

-   Both lines in the above model are *stochastic* — that is no single
    variable on the left is known with certainty, but rather is
    described probabilistically.

## 4.3 Gaussian model of height

### 4.3.1 The data

``` r
library(rethinking)
data("Howell1")
d <- Howell1

str(d)
```

    ## 'data.frame':    544 obs. of  4 variables:
    ##  $ height: num  152 140 137 157 145 ...
    ##  $ weight: num  47.8 36.5 31.9 53 41.3 ...
    ##  $ age   : num  63 63 65 41 51 35 32 27 19 54 ...
    ##  $ male  : int  1 0 0 1 0 1 0 1 0 1 ...

``` r
precis(d)
```

    ##               mean         sd      5.5%     94.5%     histogram
    ## height 138.2635963 27.6024476 81.108550 165.73500 ▁▁▁▁▁▁▁▂▁▇▇▅▁
    ## weight  35.6106176 14.7191782  9.360721  54.50289 ▁▂▃▂▂▂▂▅▇▇▃▂▁
    ## age     29.3443934 20.7468882  1.000000  66.13500     ▇▅▅▃▅▂▂▁▁
    ## male     0.4724265  0.4996986  0.000000   1.00000    ▇▁▁▁▁▁▁▁▁▇

``` r
d2 <- d[d$age >= 18,]
```

### 4.3.2 The model

``` r
hist(d2$height)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

-   Can model height using a normal distribution
-   Notation note: *iid* = *independent and identically distributed*.
    This assumption is often wrong (for example, heights amongst family
    members are not independent), but also often useful.

$$
\\begin{gather}
h\_{i} \\sim Normal(\\mu, \\sigma) \\\\
\\mu \\sim Normal(178, 20) \\\\
\\sigma \\sim Uniform(0, 50)
\\end{gather}
$$

``` r
# ***average*** height somewhere between 140 & 220 cm
curve(dnorm(x, 178, 20), 
      from = 100,
      to = 250)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
curve(dunif(x, 0, 50),
      from = -10,
      to = 60)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
# prior predictive simulation
sample_mu <- rnorm(1e4, 178, 20)
sample_sigma <- runif(1e4, 0, 50)
prior_h <- rnorm(1e4, sample_mu, sample_sigma)
dens(prior_h)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

-   Prior prediction is useful for setting reasonable priors.
-   For example, if the prior for the mean, *μ*, was
    *μ* ∼ *N**o**r**m**a**l*(178,100), we’d end up with implausible
    (negative!) prior heights:

``` r
# this set of priors is unreasonable!
sample_mu <- rnorm(1e4, 178, 100)
prior_h <- rnorm(1e4, sample_mu, sample_sigma)
dens(prior_h)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

-   In this particular case, we have enough data and a simple enough
    model that a sill prior is harmless. But that won’t always be the
    case.

### 4.3.3 Grid approximation of the posterior distribution

-   For posterity, let’s generate the analytical solution to the
    posterior (won’t be useful in the long term, but approximations will
    be far quicker/less computationally expensive/just as good for
    practical purposes):

``` r
# generate grid of mu/sigma
mu.list <- seq(from = 150, to = 160, length.out = 100)
sigma.list <- seq(from = 7, to = 9, length.out = 100)

# generate every combination of mu/sigma (10000 total)
post <- expand.grid(mu = mu.list, sigma = sigma.list)

# generate log-likelihood of data based on each combination of mu/sigma
post$LL <- 
  sapply(1:nrow(post),
         function(i) sum(
           dnorm(d2$height, post$mu[i], post$sigma[i], log = TRUE)
         ))

# compute the log product of the likelihood & prior
post$prod <- post$LL + dnorm(post$mu, 178, 20, TRUE) + dunif(post$sigma, 0, 50, TRUE)

# find the posterior, convert from log to response scale
post$prob <- exp(post$prod - max(post$prod))
```

``` r
contour_xyz(post$mu, post$sigma, post$prob)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
image_xyz(post$mu, post$sigma, post$prob)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

### 4.3.4 Sampling from the posterior

-   Let’s sample parameter values from the posterior distribution

``` r
# sample.rows = row indices --- rows with greater plausibility are more likely 
# to be sampled
sample.rows <- 
  sample(1:nrow(post), size = 1e4, replace = TRUE, prob = post$prob)

sample.mu <- post$mu[sample.rows]
sample.sigma <- post$sigma[sample.rows]

plot(sample.mu,
     sample.sigma,
     cex = 1,
     pch = 16,
     col = col.alpha(rangi2, 0.1))
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
hist(sample.mu)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
hist(sample.sigma)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-18-2.png)<!-- -->

``` r
PI(sample.mu)
```

    ##       5%      94% 
    ## 153.9394 155.2525

``` r
PI(sample.sigma)
```

    ##       5%      94% 
    ## 7.323232 8.252525

-   One interesting thing to note is that *σ* is *right-skewed*.
-   This *basically* has to do with the fact that it must be positive —
    if variance is estimated to be near zero, then it can’t be much
    smaller, but could be a lot bigger.

``` r
# using a sample of 20 points will display this issue further
d3 <- sample(d2$height, size = 20)

# repeat analysis from above, this time looking at only 20 points
# priors:
mu.list <- seq(from = 150, to = 170, length.out = 200)
sigma.list <- seq(from = 4, to = 20, length.out = 200)

# combine all possible combinations
post2 <- expand.grid(mu = mu.list, sigma = sigma.list)

# log likelihood
post2$LL <- 
  sapply(
    1:nrow(post2),
    function(i) 
      sum(dnorm(d3, mean = post2$mu[i], sd = post2$sigma[i], log = TRUE))
  )

# log product of likelihood & prior
post2$prod <- post2$LL + dnorm(post2$mu, 178, 20, TRUE) + dunif(post2$sigma, 0, 50, TRUE)

# log posterior converted to response scale
post2$prob <- exp(post2$prod - max(post2$prod))

# sample from the posterior
sample2.rows <- sample(1:nrow(post2), size = 1e4, replace = TRUE, prob = post2$prob)
sample2.mu <- post2$mu[sample2.rows]
sample2.sigma <- post2$sigma[sample2.rows]

plot(sample2.mu, 
     sample2.sigma, 
     cex = 1, 
     col = col.alpha(rangi2, 0.1),
     xlab = "mu",
     ylab = "sigma",
     pch = 16)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
dens(sample2.sigma, norm.comp = TRUE)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

### 4.3.5 Finding the posterior distribution with `quap`

-   `quap()` will allow us to make a *quadratic approximation* of the
    posterior.
-   `quap()` will find the peak at the *maximum a posteriori* estimate
    (MAP), then approximate the posterior using the curvature at the
    MAP.
-   This is similar to what many non-Bayesian procedures do, just
    without any priors.
-   `quap()` allows us to define the formulas very similarly to the
    mathematical syntax:

$$
\\begin{gather}
h\_{i} \\sim Normal(\\mu, \\sigma) \\\\
\\mu \\sim Normal(178, 20) \\\\
\\sigma \\sim Uniform(0, 50)
\\end{gather}
$$

``` r
flist <-
  alist(
    height ~ dnorm(mu, sigma),
    mu ~ dnorm(178, 20),
    sigma ~ dunif(0, 50)
  )
```

-   Very simple to fit the model to the data:

``` r
m4.1 <- quap(flist, data = d2)

precis(m4.1)
```

    ##             mean        sd       5.5%     94.5%
    ## mu    154.607057 0.4120042 153.948594 155.26552
    ## sigma   7.731512 0.2914029   7.265794   8.19723

-   The table from `precis()` provides the Gaussian approximations for
    each parameter’s *marginal* distribution.
-   This means the plausibility of each value of *μ* after averaging
    over the plausibilities of each value of *σ* is given by a Gaussian
    distribution with mean 154.6 & std. dev of 0.4.
-   `quap()` will start at random parameter values by default, but you
    can also give it an explicit starting value:

``` r
start <- 
  list(
    mu = mean(d2$height),
    sigma = sd(d2$height)
  )

# use start values specified above
m4.1 <- quap(flist, data = d2, start = start)

# quap still finds pretty much the same estimates:
precis(m4.1)
```

    ##             mean        sd       5.5%      94.5%
    ## mu    154.607024 0.4119947 153.948576 155.265471
    ## sigma   7.731333 0.2913860   7.265642   8.197024

-   Previous priors were very weak — we can also specify very stron
    priors:

``` r
m4.2 <-
  quap(
    alist(
      height ~ dnorm(mu, sigma),
      mu ~ dnorm(178, 0.1), # very strong prior!!!
      sigma ~ dunif(0, 50)
    ),
    data = d2
  )

precis(m4.2)
```

    ##            mean        sd      5.5%     94.5%
    ## mu    177.86375 0.1002354 177.70356 178.02395
    ## sigma  24.51757 0.9289236  23.03297  26.00216

-   Strong priors regularize *a lot* so the estimate for *μ* has hardly
    moved.
-   Even though *μ* hasn’t moved, *σ* has *had* to move quite a bit to
    compensate.

### 4.3.6 Sampling from a `quap`

-   Quadratic approximation is just a multidimensional gaussian (*μ* and
    *σ* both contribute a dimension) distribution.
-   Just like a mean and standard deviation are enough to describe a
    one-dimensional Gaussian distribution, a list of means and a matrix
    of variances and covariances are enough to describe a
    multidimensional Gaussian distribution.

``` r
vcov(m4.1)
```

    ##                 mu        sigma
    ## mu    0.1697396109 0.0002180307
    ## sigma 0.0002180307 0.0849058224

-   `vcov()` returns a *variance-covariance* matrix which tells us how
    each parameter relates to every other parameter in the posterior
    distribution.
-   A variance-covariance matrix can be factored into two elements:
    1.  A vector of variances for the parameters.
    2.  A correlation matrix that tells us how changes in any parameter
        lead to correlated changes in the others.

``` r
diag(vcov(m4.1))
```

    ##         mu      sigma 
    ## 0.16973961 0.08490582

``` r
cov2cor(vcov(m4.1))
```

    ##                mu       sigma
    ## mu    1.000000000 0.001816174
    ## sigma 0.001816174 1.000000000

-   The two element vector returned by `diag()` is a list of variances
    (if you take the square root, you get the standard deviations that
    are shown by `precis()`).
-   Instead of sampling single values from a Gaussian, we sample vectors
    of values from a multi-dimensional gaussian.

``` r
post <- extract.samples(m4.1, n = 1e4)

head(post)
```

    ##         mu    sigma
    ## 1 154.3330 7.374420
    ## 2 154.8001 7.498716
    ## 3 154.4903 7.793386
    ## 4 154.3520 8.292934
    ## 5 154.5383 7.718690
    ## 6 154.7274 7.824662

``` r
precis(post)
```

    ##             mean       sd       5.5%      94.5%    histogram
    ## mu    154.615312 0.414371 153.950134 155.279164      ▁▁▅▇▂▁▁
    ## sigma   7.735643 0.292622   7.270406   8.198819 ▁▁▁▂▅▇▇▃▁▁▁▁

``` r
plot(post)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

## 4.4 Linear prediction

-   Thus far, we’ve only modeled height. Oftentimes, we want measure how
    an outcome is related to some other variable (e.g., predictors).
-   For example, height and weight are positvely associated:

``` r
plot(d2$height ~ d2$weight)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

### 4.4.1 The linear model strategy

-   The strategy is to make the parameter for the mean int oa linear
    function of the predictor(s).
-   For our height example:

$$
\\begin{gather}
h_i \\sim Normal(\\mu_i, \\sigma) \\\\
\\mu_i = \\alpha + \\beta(x_i - \\overline{x}) \\\\
\\alpha \\sim Normal(178, 20) \\\\
\\beta \\sim Normal(0, 10) \\\\
\\sigma \\sim Uniform(0, 50)
\\end{gather}
$$

#### 4.4.1.1 Probability of the data

-   Read *h*<sub>*i*</sub> and *μ*<sub>*i*</sub> as “each *h*” and “each
    *μ*.
-   The mean of each now depends on unique values on each row *i*.

#### 4.4.1.2 Linear model

-   *μ*<sub>*i*</sub> is now *deterministic*, because it described by
    other parameters.
-   We’re asking two things in this regression:

1.  What is the expected height when $x_i = \\overline{x}$? The
    parameter *α* answers this question — in this case *α* is a centered
    intercept.
2.  What is the change in expected height when *x*<sub>*i*</sub> changes
    by 1 unit? The parameter *β* answers this question — in this case
    the slope.

#### 4.4.1.3 Priors

-   Let’s check if the priors are reasonable:

``` r
set.seed(2971)
N <- 100
a <- rnorm(N, 178, 20)
b <- rnorm(N, 0, 10)

plot(NULL,
     xlim = range(d2$weight),
     ylim = c(-100, 400),
     xlab = "weight", ylab = "height")

abline(h = 0, lty = 2)
abline(h = 272, lty = 1, lwd = 0.5)
mtext("b ~ dnorm(0, 10)")
xbar <- mean(d2$weight)
for (i in 1:N) curve(a[i] + b[i]*(x - xbar),
                     from = min(d2$weight),
                     to = max(d2$weight),
                     add = TRUE,
                     col = col.alpha("black", 0.2))
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

-   This is an unreasonable prior!
-   No one is shorter than 0 cm and there are not so many people who are
    taller than the world’s tallest person (272 cm).
-   A log-normal prior may be more reasonable.

``` r
b <- rlnorm(1e4, 0, 1)
dens(b, xlim = c(0, 5), adj = 0.1)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->

``` r
set.seed(2971)
N <- 100
a <- rnorm(N, 178, 20)
b <- rlnorm(N, 0, 1)

plot(NULL,
     xlim = range(d2$weight),
     ylim = c(-100, 400),
     xlab = "weight", ylab = "height")

abline(h = 0, lty = 2)
abline(h = 272, lty = 1, lwd = 0.5)
mtext("b ~ dnorm(0, 10)")
xbar <- mean(d2$weight)
for (i in 1:N) curve(a[i] + b[i]*(x - xbar),
                     from = min(d2$weight),
                     to = max(d2$weight),
                     add = TRUE,
                     col = col.alpha("black", 0.2))
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-32-1.png)<!-- -->

-   This is a much more reasonable prior.
-   One thing to note — adjusting a prior in light of the observed
    sample just to get some desired result is the Bayesian equivalent of
    p-hacking.
-   In this example, we didn’t set a prior based on comparing to the
    data, but rather used our *prior knowledge* about what reasonable
    height/weight relationships might look like.

### 4.4.2 Finding the posterior distribution.

-   An update to the model:

$$
\\begin{gather}
h_i \\sim Normal(\\mu_i, \\sigma)\\\\
\\mu_i = \\alpha + \\beta(x_i - \\overline{x}) \\\\
\\alpha \\sim Normal(178, 20) \\\\
\\beta \\sim LogNormal(0, 1) \\\\
\\sigma \\sim Uniform(0, 50)\\
\\end{gather}
$$

``` r
m4.3 <-
  quap(
    alist(
      height ~ dnorm(mu, sigma),
      mu <- a + b*(weight - xbar),
      a ~ dnorm(178, 20),
      b ~ dlnorm(0, 1),
      sigma ~ dunif(0, 50)
    ),
    data = d2
  )
```

### 4.4.3 Interpreting the posterior distribution

-   There are two broad categories of processing the posterior:
    1.  Reading tables
    2.  Plotting simulations
-   Most models are very hard to understand from tables of numbers
    alone.
-   Plotting allows you to inquire about things that are hard to learn
    from tables:
    1.  Whether or not the model fitting process worked correctly
    2.  The *absolute* magnitude (rather than the *relative* magnitude)
        of a relationship between an outcome and predictor.
    3.  The uncertainty surrounding an average relationship.
    4.  The uncertainty surrounding the implied predictions of the
        model.

#### 4.4.3.1 Tables of marginal distributions

``` r
precis(m4.3)
```

    ##              mean         sd        5.5%       94.5%
    ## a     154.6013671 0.27030766 154.1693633 155.0333710
    ## b       0.9032807 0.04192363   0.8362787   0.9702828
    ## sigma   5.0718809 0.19115478   4.7663786   5.3773831

-   Look at `b` — this can be read as *a person 1 kg heavier is expected
    to be 0.90 cm taller*.
-   89% of the posterior probability lies between 0.84 and 0.97 — a
    relationship close to 0 or above 1 is highly implausible given these
    data and this model.
-   In this case, there is very little correlation amongst the
    parameters:

``` r
round(vcov(m4.3), 3)
```

    ##           a     b sigma
    ## a     0.073 0.000 0.000
    ## b     0.000 0.002 0.000
    ## sigma 0.000 0.000 0.037

``` r
pairs(m4.3)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-35-1.png)<!-- -->

#### 4.4.3.2 Plotting posterior inference against the data

-   Plotting against the data helps interpret the posterior and provides
    an informal check on model assumptions.

``` r
plot(height ~ weight, data = d2, col = rangi2)

# get maximum a posteriri estimates for a/b
post <- extract.samples(m4.3)
a_map <- mean(post$a)
b_map <- mean(post$b)

# add map line ot plot
curve(a_map + b_map*(x - xbar), add = TRUE)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-36-1.png)<!-- -->

#### 4.4.3.3 Adding uncertainty around the mean

-   The above plot is a single line defined by *α* & *β*. Let’s add more

``` r
post[1:5, ]
```

    ##          a         b    sigma
    ## 1 154.2172 0.8589918 4.506438
    ## 2 154.4955 0.8845459 5.030822
    ## 3 154.5208 0.8393668 4.995964
    ## 4 154.8405 0.8407996 5.016383
    ## 5 154.5813 0.9485057 5.143586

``` r
plot_n_samples <- function(N) {
  
  # filter to just the first N samples in d2
  dN <- d2[1:N, ]
  
  # create a new model
  mN <- 
    quap(
      alist(height ~ dnorm(mu, sigma),
            mu <- a + b*(weight - mean(weight)),
            a ~ dnorm(178, 20),
            b ~ dlnorm(0, 1),
            sigma ~ dunif(0, 50)),
      data = dN
    )
  
  # extract 20 samples from the posterior
  post <- extract.samples(mN, n = 20)
  
  # display raw data & sample size
  plot(dN$weight,
       dN$height,
       xlim = range(d2$weight),
       ylim = range(d2$height),
       col = rangi2,
       xlab = "weight",
       ylab = "height")
  
  mtext(concat("N = ", N))
  
  # plot the lines with transparency
  for (i in 1:20)
    curve(post$a[i] + post$b[i] * (x - mean(dN$weight)),
          col = col.alpha("black", 0.3),
          add = TRUE)
  
}

plot_n_samples(10)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-38-1.png)<!-- -->

``` r
plot_n_samples(50)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-38-2.png)<!-- -->

``` r
plot_n_samples(150)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-38-3.png)<!-- -->

``` r
plot_n_samples(352)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-38-4.png)<!-- -->

#### 4.4.3.4 Plotting regression intervals and contours

-   We can also plot uncertainty around the mean estimate for individual
    points (here’s a weight of 50kg as an example):

``` r
# get posterior estimates of mean height based on a weight of 50kg
mu_at_50 <- post$a + post$b * (50 - xbar)

# plot posterior
dens(mu_at_50, 
     col = rangi2,
     lwd = 2,
     xlab = "mu | weight = 50")
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-39-1.png)<!-- -->

``` r
PI(mu_at_50)
```

    ##       5%      94% 
    ## 158.5755 159.6733

-   For all points, we can use `link()`:

``` r
mu <- link(m4.3)
str(mu)
```

    ##  num [1:1000, 1:352] 157 157 157 158 157 ...

``` r
# define a sequence of weights to compute predictions for
# these values will be on the horizontal axis
weight.seq <- seq(from = 25, to = 70, by = 1)

# use link to compute mu
# for each sample from the posterior
# and for each weight in weight.seq
mu <- 
  link(
    m4.3,
    data = data.frame(weight = weight.seq)
  )

str(mu)
```

    ##  num [1:1000, 1:46] 136 138 137 137 135 ...

``` r
plot(height ~ weight, d2, type = "n")

# loop over samples and plot each mu value
for (i in 1:100) points(weight.seq, mu[i,], pch = 16, col = col.alpha(rangi2, 0.1))
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-42-1.png)<!-- -->

-   Instead of plotting the raw mu values at each weight, we can
    summarise with an uncertainty interval.

``` r
# summarise the distribution of mu
# "compute the mean of each column (dimenstion 2)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob = 0.89)

# plot the summary on top of data
plot(height ~ weight, data = d2, col = col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

-   To summarise, here’s the `rethinking` approach to generating
    predictions and intervals from the posterior fit of a model:
    1.  Use `link()` to generate posterior values for *μ* (`link()` uses
        the original data by default or you can pass new data).
    2.  Use summary functions (like `mean()` or `PI()`) to find
        averages, lower, and upper bounds for each value.
    3.  Use plotting functions like `lines()` and `shade()` to draw the
        lines and intervals.

#### 4.4.3.5 Prediction intervals

-   What we’ve done so far is generate the uncertainty around
    *μ*<sub>*i*</sub>, the mean.
-   Actual predictions of height also depends on the distribution (e.g.,
    *σ*).
-   `rethinking::sim()` handles this for us.

``` r
# simulate plausible heights for weights in weight.seq under this model 
sim.height <- sim(m4.3, data = list(weight = weight.seq))
str(sim.height)
```

    ##  num [1:1000, 1:46] 141 136 130 136 135 ...

``` r
# summarise the distribution of plausible heights
height.PI <- apply(sim.height, 2, PI, prob = 0.89)

# plot raw data
plot(height ~ weight, d2, col = col.alpha(rangi2, 0.5))

# draw MAP line
lines(weight.seq, mu.mean)

# draw PI around mu
shade(mu.PI, weight.seq)

# draw PI around height
shade(height.PI, weight.seq)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-44-1.png)<!-- -->

-   Increasing the number of samples will smooth out some of the
    roughness in the shaded areas

## 4.5 Curves from lines

-   Two commonplace methods that use linear regression to build curves
    are *polynomial regression* and *B-splines*.
-   Both work by transforming a single predictor into several synthetic
    variables, but splines have some clear advantages.

### 4.5.1 Polynomial regression

-   Use powers of variables (eg, squares, cubes) as extra predictors.

``` r
# the full export of the Howell1 dataset shows a clear curved relationship when
# lower weights (from non-adults) are included
plot(height ~ weight, d)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-45-1.png)<!-- -->

-   A basic linear model doesn’t do great here!

``` r
# extend weight seq for plotting
weight.seq <- seq(from = 1, to = 70, by = 1)

# update linear model
mu <- link(m4.3, data = data.frame(weight = weight.seq))
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob = 0.89)

# model height
sim.height <- sim(m4.3, data = data.frame(weight = weight.seq))
height.PI <- apply(sim.height, 2, PI, prob = 0.89)

# plot
plot(height ~ weight, d, col = col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(height.PI, weight.seq)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-46-1.png)<!-- -->

-   We can model this with a *second order* polynomial by changing the
    equation for *μ*<sub>*i*</sub> to:
    *μ*<sub>*i*</sub> = *α* + *β*<sub>1</sub>*x*<sub>*i*</sub> + *β*<sub>2</sub>*x*<sub>*i*</sub><sup>2</sup>
-   Fitting this model is easy — interpreting it is hard.
-   First thing to do is standardize the predictor

``` r
# standardize the weight & then square it
d$weight_s <- (d$weight - mean(d$weight))/sd(d$weight)
d$weight_s2 <- d$weight_s^2

# model using the standardized weight
m4.5 <-
  quap(
    alist(height ~ dnorm(mu, sigma),
          mu <- a + b1*weight_s + b2*weight_s2,
          a ~ dnorm(178, 20),
          b1 ~ dlnorm(0, 1),
          b2 ~ dnorm(0, 1),
          sigma ~ dunif(0, 50)),
    data = d
  )

precis(m4.5)
```

    ##             mean        sd       5.5%      94.5%
    ## a     146.058088 0.3690226 145.468319 146.647857
    ## b1     21.732531 0.2889276  21.270769  22.194293
    ## b2     -7.803404 0.2742195  -8.241659  -7.365148
    ## sigma   5.775300 0.1765283   5.493174   6.057426

-   a is still the intercept, so it tells us the expected value of
    height when weight is at it’s mean value (0 when on the standardized
    scale).
-   B1/B2 are the linear/square components of the curve — graphing makes
    this a bit more transparent:

``` r
# define a new weight sequence on the standardized scale
weight.seq <- seq(from = -2.2, to = 2, length.out = 30)

# new data to pass to model
pred_dat <- 
  list(weight_s = weight.seq,
       weight_s2 = weight.seq^2)

# model mu
mu <- link(m4.5, data = pred_dat)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob = 0.89)

# model height
sim.height <- sim(m4.5, data = pred_dat)
height.PI <- apply(sim.height, 2, PI, prob = 0.89)

# plot
plot(height ~ weight_s, d, col = col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(height.PI, weight.seq)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-48-1.png)<!-- -->

-   This is better, but a cubic is *even better*:

``` r
# model add a cubic term & model
d$weight_s3 <- d$weight_s^3
m4.6 <-
  quap(
    alist(height ~ dnorm(mu, sigma),
          mu <- a + b1*weight_s + b2*weight_s2 + b3*weight_s3,
          a ~ dnorm(178, 20),
          b1 ~ dlnorm(0, 1),
          b2 ~ dnorm(0, 10),
          b3 ~ dnorm(0, 10),
          sigma ~ dunif(0, 50)),
    data = d
  )

# new data to pass to model
pred_dat <- 
  list(weight_s = weight.seq,
       weight_s2 = weight.seq^2,
       weight_s3 = weight.seq^3)

# model mu
mu <- link(m4.6, data = pred_dat)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob = 0.89)

# model height
sim.height <- sim(m4.6, data = pred_dat)
height.PI <- apply(sim.height, 2, PI, prob = 0.89)

# plot
plot(height ~ weight_s, d, col = col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(height.PI, weight.seq)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-49-1.png)<!-- -->

### 4.5.2 Splines

-   A spline is a smooth function built out of smaller, component
    functions.
-   There are many types, but let’s look at the *B-spline*, which is
    fairly commonplace.

``` r
# load the cherry blossoms dataset to explore b-splines
data("cherry_blossoms")
d <- cherry_blossoms

# simple exploration of the dataset 
# doy = day of year of the first cherry blossom bloom in japan
precis(d)
```

    ##                   mean          sd      5.5%      94.5%       histogram
    ## year       1408.000000 350.8845964 867.77000 1948.23000   ▇▇▇▇▇▇▇▇▇▇▇▇▁
    ## doy         104.540508   6.4070362  94.43000  115.00000        ▁▂▅▇▇▃▁▁
    ## temp          6.141886   0.6636479   5.15000    7.29470        ▁▃▅▇▃▂▁▁
    ## temp_upper    7.185151   0.9929206   5.89765    8.90235 ▁▂▅▇▇▅▂▂▁▁▁▁▁▁▁
    ## temp_lower    5.098941   0.8503496   3.78765    6.37000 ▁▁▁▁▁▁▁▃▅▇▃▂▁▁▁

``` r
plot(doy ~ year, data = d)
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-50-1.png)<!-- -->

-   B-splines do not directly transform the predictor, but instead work
    with a synthetic set of *basis functions*:

*μ*<sub>*i*</sub> = *α* + *w*<sub>*i*</sub>*β*<sub>*i*, 1</sub> + *w*<sub>2</sub>*β*<sub>*i*, 2</sub> + …

-   *β*<sub>*i*, *n*</sub> is the *n*-th basis function’s value on row
    *i*. *w* parameters are the weights for each.
-   (this is just linear regression)
-   Let’s explore placing 15 *knots* for a cubic b-spline on the cherry
    blossom data (for more info, see figure 4.1.2 on page 116).

``` r
# d2 is a frame with no NAs for doy
d2 <- d[complete.cases(d$doy),] 

# split up d2 into 15 quantiles
num_knots <- 15
knot_list <- quantile(d2$year, probs = seq(0, 1, length.out = num_knots))

# create cubic b-spline (degree = 3) out of the knots
library(splines)
B <- 
  bs(
    d2$year,
    knots = knot_list[-c(1, num_knots)],
    degree = 3,
    intercept = TRUE
  )

# plot cubic basis functions
plot(
  NULL, 
  xlim = range(d2$year), 
  ylim = c(0, 1),
  xlab = "year",
  ylab = "basis"
)

for (i in 1:ncol(B)) lines(d2$year, B[,i])
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-51-1.png)<!-- -->

-   Now we can plop these basis functions into a linear model to get the
    weight of each

$$
\\begin{gather}
D_i \\sim Normal(\\mu_i, \\sigma) \\\\
\\mu_i = \\alpha + \\sum\_{k = 1} ^{K} w_k B\_{k, i} \\\\
\\alpha \\sim Normal(100, 10) \\\\
w_k \\sim Normal(0, 10) \\\\
\\sigma \\sim Exponential(1)
\\end{gather}
$$

-   Some quick notes on the mathematical model:
    -   *w*<sub>*k*</sub> indicates that each basis function has the
        same prior
    -   The *Exponential* prior for *σ* forces it to be greater than 0
        (this is the least informative/max entropy case of the *gamma
        distribution*).

``` r
# model w/quap
m4.7 <-
  quap(
    alist(D ~ dnorm(mu, sigma),
          mu <- a + B %*% w,
          a ~ dnorm(100, 10),
          w ~ dnorm(0, 10),
          sigma ~ dexp(1)),
    data = list(D = d2$doy,
                B = B),
    start = list(w = rep(0, ncol(B)))
  )

precis(m4.7, depth = 2)
```

    ##              mean        sd         5.5%       94.5%
    ## w[1]   -3.0257576 3.8610453  -9.19645377   3.1449386
    ## w[2]   -0.8654717 3.8700408  -7.05054429   5.3196009
    ## w[3]   -1.0274023 3.5848362  -6.75666296   4.7018583
    ## w[4]    4.8375713 2.8770768   0.23944691   9.4356957
    ## w[5]   -0.8367738 2.8742885  -5.43044198   3.7568944
    ## w[6]    4.3348685 2.9147966  -0.32353933   8.9932764
    ## w[7]   -5.3288914 2.8001759  -9.80411328  -0.8536696
    ## w[8]    7.8568441 2.8020289   3.37866074  12.3350274
    ## w[9]   -1.0141020 2.8809954  -5.61848908   3.5902852
    ## w[10]   3.0589236 2.9100511  -1.59190009   7.7097473
    ## w[11]   4.6409908 2.8916730   0.01953893   9.2624427
    ## w[12]  -0.1321205 2.8693627  -4.71791628   4.4536752
    ## w[13]   5.5477451 2.8873841   0.93314763  10.1623426
    ## w[14]   0.7424729 2.9992560  -4.05091745   5.5358632
    ## w[15]  -0.8336141 3.2934266  -6.09714589   4.4299178
    ## w[16]  -6.9422214 3.3756783 -12.33720733  -1.5472355
    ## w[17]  -7.6602777 3.2226660 -12.81072041  -2.5098350
    ## a     103.3470640 2.3697442  99.55975507 107.1343729
    ## sigma   5.8761926 0.1437276   5.64648813   6.1058971

``` r
# plot the weight * basis function
post <- 
  extract.samples(m4.7)

# return mean for each w
w <-
  apply(post$w, 2, mean)

# plot
plot(
  NULL,
  xlim = range(d2$year),
  ylim = c(-6, 6),
  xlab = "year",
  ylab = "basis * weight"
)

for (i in 1:ncol(B)) lines(d2$year, w[i] * B[,i])
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-53-1.png)<!-- -->

``` r
# get posterior draws of mu
mu <- link(m4.7)

# get 97% percentile interval & plot
mu_PI <- apply(mu, 2, PI, 0.97)
plot(d2$year, 
     d2$doy,
     col = col.alpha(rangi2, 0.3),
     pch = 16)

shade(mu_PI, 
      d2$year,
      col = col.alpha("black", 0.5))
```

![](chapter_4_notes_files/figure-gfm/unnamed-chunk-54-1.png)<!-- -->

-   More knots = more wiggliness. How many knots is “correct?” (We’ll
    address in a few chapters).
-   This is just te beginning of an entire class of models, *Generalized
    Additive Models* (GAMS).

## 4.6 Summary

-   Gaussian distribution is a good geocentric tool.
-   Use `quap()` to use quadratic approximation of the posterior.
