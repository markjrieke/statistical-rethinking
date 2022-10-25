The Many Variables & The Spurious Waffles
================

-   States with the highest number of Waffle Houses also have the
    highest divorce rate.
-   This is a spurious relationship!
-   *Multiple regression* helps deflect spurious relationships (more on
    that later…):
    -   “Control” for confounds
    -   Multiple and complex causation
    -   Interactions

## 5.1 Spurious association

-   The rate that adults marry is a great predictor of divorce rate, but
    does marriage *cause* divorce?

``` r
library(rethinking)
data("WaffleDivorce")
d <- WaffleDivorce

# standardize variables
d$D <- standardize(d$Divorce)
d$M <- standardize(d$Marriage)
d$A <- standardize(d$MedianAgeMarriage)
```

![
D_i \\sim Normal(\\mu_i, \\sigma) \\\\
\\mu_i = \\alpha + \\beta_A A_i \\\\
\\alpha \\sim Normal(0, 0.2) \\\\
\\beta_A \\sim Normal(0, 0.5) \\\\
\\sigma \\sim Exponential(1)
](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%0AD_i%20%5Csim%20Normal%28%5Cmu_i%2C%20%5Csigma%29%20%5C%5C%0A%5Cmu_i%20%3D%20%5Calpha%20%2B%20%5Cbeta_A%20A_i%20%5C%5C%0A%5Calpha%20%5Csim%20Normal%280%2C%200.2%29%20%5C%5C%0A%5Cbeta_A%20%5Csim%20Normal%280%2C%200.5%29%20%5C%5C%0A%5Csigma%20%5Csim%20Exponential%281%29%0A "
D_i \sim Normal(\mu_i, \sigma) \\
\mu_i = \alpha + \beta_A A_i \\
\alpha \sim Normal(0, 0.2) \\
\beta_A \sim Normal(0, 0.5) \\
\sigma \sim Exponential(1)
")

``` r
# model
m5.1 <-
  quap(
    alist(D ~ dnorm(mu, sigma),
          mu <- a + bA * A,
          a ~ dnorm(0, 0.2),
          bA ~ dnorm(0, 0.5),
          sigma ~ dexp(1)),
    data = d
  )

# simulate from priors:
set.seed(10)
prior <- extract.prior(m5.1)
mu <- link(m5.1, post = prior, data = list(A = c(-2, 2)))

# plot!
plot(NULL, 
     xlim = c(-2,2), 
     ylim = c(-2,2),
     xlab = "Median age marriage (std)",
     ylab = "Divorce rate (std)",
     main = "Prior simulation")

for (i in 1:50) lines(c(-2,2), mu[i,], col = col.alpha("black", 0.4))
```

![](chapter_5_notes_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
# plotting function for posterior:


# compute percentile interval of mean
A_seq <- seq(from = -3, to = 3.2, length.out = 30)
mu <- link(m5.1, data = list(A = A_seq))
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI)

# plot it all!
plot(D ~ A,
     data = d,
     col = rangi2,
     xlab = "Median age marriage (std)",
     ylab = "Divorce rate (std)",
     main = "Posterior")

lines(A_seq,
      mu.mean,
      lwd = 2)

shade(mu.PI, A_seq)
```

![](chapter_5_notes_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

``` r
# model divorce rate as a fn of marriage rate
m5.2 <-
  quap(
    alist(D ~ dnorm(mu, sigma),
          mu <- a + bM * M,
          a ~ dnorm(0, 0.2),
          bM ~ dnorm(0, 0.5),
          sigma ~ dexp(1)),
    data = d
  )

# plot posterior for m5.2
M_seq <- seq(from = -2, to = 3, length.out = 30)
mu <- link(m5.2, data = list(M = M_seq))
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI)

plot(D ~ M,
     data = d,
     col = rangi2,
     xlab = "Marriage rate (std)",
     ylab = "Divorce rate (std)",
     main = "Posterior")

lines(M_seq,
      mu.mean,
      lwd = 2)

shade(mu.PI, M_seq)
```

![](chapter_5_notes_files/figure-gfm/unnamed-chunk-2-3.png)<!-- -->

-   There is a weakly positive relationship between marriage and divorce
    rates, and a negative relationship between median age at marriage
    and divorce rate.
-   Comparing these single-variable models separately isn’t great — they
    could both provide value, be redundant, or one can eliminate the
    value of the other.
-   Goal: think *causally*, then fit a bigger regression.

### 5.1.1 Think before you regress

-   Three variables at play: D, M, and A.
-   A *Directed Acyclic Graph* (DAG) can help us think about the
    relationship between the variables.
-   A possible DAG for the divorce rate example could be:

``` r
library(dagitty)
dag5.1 <- dagitty("dag{A -> D; A -> M; M -> D}")
coordinates(dag5.1) <- 
  list(x = c(A = 0, D = 1, M = 2),
       y = c(A = 0, D = 1, M = 0))

drawdag(dag5.1)
```

![](chapter_5_notes_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

-   In this diagram, age influences divorce rates in two ways: directly
    (A -\> D) and indirectly through marriage rates (A -\> M -\> D)
-   To infer the strength of these different arrows, we’d need more than
    one model. `m5.1` only shows the *total* influence of age at
    marriage on the divorce rate (direct and indirect). It could
    possible that there is no direct effect and it is associated with D
    entirely through the indirect path. This is known as *mediation*.
-   Another alternative is that there is no relationship between M & D
    (this is still consistent with `m5.2`, because M in this DAG picks
    up information from A):

``` r
dag5.2 <- dagitty("dag{A -> M; A -> D}")
coordinates(dag5.2) <- coordinates(dag5.1)
drawdag(dag5.2)
```

![](chapter_5_notes_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

-   So, which is it?
