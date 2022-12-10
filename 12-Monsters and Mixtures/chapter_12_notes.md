Monsters and Mixtures
================

-   This chapter is all about constructing hybrid statistical models by
    piecing together smipler components of previous chapters.
-   There are three common and useful examples:
    1.  **Over-dispersion**: extensions of binomial and Poisson models
        to cope with unmeasured sources of variation.
    2.  **Zero-inlated/zero-augmented models**: mixes a binary event
        with an ordinary GLM.
    3.  **Ordered categorical models**: like the name suggests, models
        where the outcome has an implied order to the categories.
-   This is not an exhaustive list! There are many more.

## 12.1 Over-dispersed counts

-   Earlier, “outliers” in normal models were able to be dealt with the
    student-t likelihood, which is a mixture of normal likelihoods. We
    can do something similar in count models!
-   When counts are more variable than a pure process, they exhibit
    *over-dispersion*.
-   For example, the expected value of a binomial process is $Np$ and
    the variance/dispersion is $Np(1 - p)$.
-   When the observed variance exceeds this (after conditioning on
    variables) it implies that some omitted variable is producing
    additional dispersion!
-   The best solution would be to be able to discover the source of
    dispersion and include it in the model, but that often isn’t
    possible. Even without, however, we can mitigate the effects of
    over-dispersion.
-   We’ll consider *continuous mixture* models where the linear model
    isn’t attached to the observations themselves but instead to a
    distribution of observations.

### 12.1.1 Beta-binomial

-   A *beta-binomial* model is a mixture of binomial distributions that
    assumes each binomial count has its own probability of success.
-   The beta distribution can be used to describe a distribution of
    probabilities — this also makes the math easier, since it’s the
    conjugate pair for the binomial.

``` r
library(rethinking)

beta_plot <- function(pbar, theta) {
  
  curve(dbeta2(x, pbar, theta),
        from = 0,
        to = 1,
        xlab = "probability",
        ylab = "density")
  
}

beta_plot(0.5, 5)
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
beta_plot(0.8, 10)
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-1-2.png)<!-- -->

-   Let’s build out a model for `UCBadmit`, where the admission rate is
    modeled purely on gender.

$$
\begin{gather}
A_i \sim BetaBinomial(N_i, \overline{p_i}, \theta) \\
logit(\overline{p_i}) = \alpha_{GID[i]} \\
\alpha_j \sim Normal(0, 1.5) \\
\theta = \phi + 2 \\ 
\phi \sim Exponential(1)
\end{gather}
$$

``` r
# load data
data("UCBadmit")
d <- UCBadmit

# prep for stan
d$gid <- ifelse(d$applicant.gender == "male", 1L, 2L)
dat <- list(A = d$admit, N = d$applications, gid = d$gid)

# model!
m12.1 <- 
  ulam(
    alist(A ~ dbetabinom(N, pbar, theta),
          logit(pbar) <- a[gid],
          a[gid] ~ dnorm(0, 1.5),
          transpars> theta <<- phi + 2.0,
          phi ~ dexp(1)),
    data = dat,
    chains = 4
  )
```

    ## Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    ## 
    ## Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 1 finished in 0.3 seconds.
    ## Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 2 finished in 0.2 seconds.
    ## Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 3 finished in 0.2 seconds.
    ## Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 4 finished in 0.2 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 0.2 seconds.
    ## Total execution time: 1.9 seconds.

-   Here, McElreath added `transpars>` (transformed parameters) so that
    Stan will return `theta` in the samples.

``` r
post <- extract.samples(m12.1) 
post$da <- post$a[,1] - post$a[,2]
precis(post, depth = 2)
```

    ##             mean        sd       5.5%     94.5%       histogram
    ## a[1]  -0.4290155 0.3884849 -1.0377471 0.2068656   ▁▁▁▂▃▇▇▇▅▂▂▁▁
    ## a[2]  -0.3247134 0.4153179 -0.9826741 0.3410041 ▁▁▁▁▂▅▇▇▇▃▂▁▁▁▁
    ## phi    1.0347497 0.8160561  0.1071035 2.6004500   ▇▇▃▂▂▁▁▁▁▁▁▁▁
    ## theta  3.0347497 0.8160561  2.1071053 4.6004500   ▇▇▃▂▂▁▁▁▁▁▁▁▁
    ## da    -0.1043021 0.5625613 -0.9780437 0.7916987        ▁▁▃▇▅▂▁▁

-   Here, `a[1]` and `a[2]` are the log-odds of admission for
    male/female applicants. The difference between the two, `da` is
    highly uncertain and basically centered around 0.
-   In the previous chapter, a binomial model for these data that
    omitted department ended up being misleading, since there’s an
    indirect path from gender to admission through department. Here, the
    model is not confounded, despite not containing the department
    variable! How?
-   The beta-binomial model allows each row to have its own unobserved
    intercept:

``` r
ucb_beta <- function(gid) {
  
  # draw posterior mean beta distribution
  curve(dbeta2(x, mean(logistic(post$a[,gid])), mean(post$theta)),
        from = 0, 
        to = 1,
        ylab = "Density",
        xlab = "Probability of Admission",
        ylim = c(0, 3),
        lwd = 2)
  
  # draw 50 samples from the posterior
  for (i in 1:50) {
    
    p <- logistic(post$a[i, gid])
    theta <- post$theta[i]
    curve(dbeta2(x, p, theta),
          add = TRUE,
          col = col.alpha("black", 0.2))
    
  }
  
  if (gid == 1) gender <- "Male" else gender <- "Female"
  
  mtext(paste("Distribution of", gender, "Admission Rates"))
  
}

ucb_beta(1)
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
ucb_beta(2)
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
postcheck(m12.1)
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->

### 12.1.2 Negative-binomial or Gamma-Poisson

-   A *negative-binomial* model is more usefully described as a
    *gamma-Poisson* model and assumes that each Poisson count
    observation has its own rate.
-   This is similar to the beta-binomial model, where the beta
    distribution describes a distribution of probabilities and the gamma
    distribution describes a distribution of rates (+ the conjugate
    pairing makes the math easier).
-   A gamma-Poisson distribution has two terms — one for the mean rate,
    $\lambda$ and another for the dispersion/scale of the rates, $\phi$.

$$
\begin{gather}
y_i \sim \text{Gamma-Poisson}(\lambda_i, \phi)
\end{gather}
$$

-   Here, the variance is described by
    $\lambda + \frac{\lambda^2}{\phi}$, so larger values for $\phi$ mean
    the distribution is more similar to a pure Poisson process.
-   Let’s look back at the island civilization tool model:

``` r
# load and prep data for stan
data("Kline")
d <- Kline
d$P <- standardize(log(d$population))
d$contact_id <- ifelse(d$contact == "high", 2L, 1L)

dat2 <-
  list(
    tools = d$total_tools,
    P = d$population,
    cid = d$contact_id
  )

# model!
m12.2 <- ulam(
  alist(tools ~ dgampois(lambda, phi),
        lambda <- exp(a[cid]) * P^b[cid]/g,
        a[cid] ~ dnorm(1, 1),
        b[cid] ~ dexp(1),
        g ~ dexp(1),
        phi ~ dexp(1)),
  data = dat2,
  chains = 4, 
  log_lik = TRUE
)
```

    ## Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    ## 
    ## Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 1 finished in 0.6 seconds.
    ## Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 2 finished in 0.5 seconds.
    ## Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 3 finished in 0.7 seconds.
    ## Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 4 finished in 0.5 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 0.6 seconds.
    ## Total execution time: 3.1 seconds.

``` r
precis(m12.2, depth = 2)
```

    ##           mean         sd        5.5%     94.5%     n_eff    Rhat4
    ## a[1] 0.9382833 0.83343811 -0.43210201 2.2565351 1040.7623 1.001052
    ## a[2] 1.0804288 0.94593187 -0.42954208 2.4950511 1353.8774 0.999898
    ## b[1] 0.2425519 0.09927092  0.08816549 0.4010911  648.0695 1.001876
    ## b[2] 0.2559066 0.13302830  0.05942911 0.4766992  892.9532 1.000956
    ## g    1.0549679 0.87590001  0.19618674 2.6322510  906.8250 1.002323
    ## phi  3.7226259 1.68760007  1.53487140 6.7478833 1209.6542 1.001541

-   See figure 12.2 on page 375 for a comparison of the outputs from
    `m11.11` and `m12.2`.

### 12.1.3 Over-dispersion, entropy, and information criteria

-   Both the beta-binomial and gamma-Poisson are similarly the maximum
    entropy distributions as their binomial/Poisson counterparts.
-   In terms of model comparison, you can treat a beta-binomial like a
    binomial and a gamma-Poisson like a Poisson.
-   You shouldn’t, however, use WAIC or PSIS for these models, since
    beta-binomial/gamma-Poisson models cannot be
    aggregated/disaggregated across rows without changing causal
    assumptions.
-   Multilevel models help reduce this obstacle, since they can handle
    heterogeneity in probabilities/rates at any level of aggregation.

## 12.2 Zero-inflated outcomes

-   Often, things we measure are not pure processes, but are mixtures of
    multiple processes. Whenever there are multiple causes for the same
    observation, then a *mixture model* may be useful.
-   Count variables are especially prone to mixture models — very often
    a count of 0 can arise in more than one way (i.e., the rate of
    events is low or that the process to generate events hasn’t even
    started).

### 12.2.1 Example: Zero-inflated Poisson

-   Recall the monk manuscript writing example from chapter 11. Let’s
    say that some days, instead of working on manuscripts, the monks
    take the day of to go drinking.
-   On those days — no manuscripts will be produced. On the days they
    are working, there will still be some days that no manuscripts are
    produced, but that is due to the rate of the poisson model!
-   A mixture model can help solve this problem (see figure 12.3 on page
    377 for the process diagram).

$$
\begin{gather}
\text{Pr}(0|p, \lambda) = \text{Pr}(\text{drink}|p) + \text{Pr}(\text{work}|p) \times \text{Pr}(0|\lambda) \\
\text{Pr}(0|p, \lambda) = p + (1 - p)\ \text{exp} (-\lambda) \\
\text{and ...} \\
\text{Pr}(y|y > 0, p, \lambda) = \text{Pr}(\text{drink}|p)(0) + \text{Pr}(\text{work}|p) \ \text{Pr}(y|\lambda) \\
\text{Pr}(y|y > 0, p, \lambda) = (1 - p) \frac{\lambda^y - \text{exp}(-\lambda)}{y!}
\end{gather}
$$

-   This is just a lot of math to say the following:

> The probability of observing a zero is the probability that the monks
> didn’t drink OR $(+)$ the probability that the monks worked AND
> $(\times)$ failed to finish anything.

-   We can define ZIPoisson as the above distribution with parameters
    $p$ (the probability of 0) and $\lambda$ (the mean of the Poisson).

$$
\begin{gather}
y_i \sim \text{ZIPoisson}(p_i, \lambda_i) \\
\text{logit}(p_i) = \alpha_p + \beta_p x_i \\
\text{log}(\lambda_i) = \alpha_{\lambda} + \beta_{\lambda} x_i
\end{gather}
$$

-   Notice two things:
    1.  There are two linear models and two link functions.
    2.  The parameters of the linear models differ — the association
        between a variable may be different for $p$ and $\lambda$. Also,
        you don’t need to use the same predictors in both models!

``` r
# let's simulate some data!
prob_drink <- 0.2 # 20% of days are taken off for drinking
rate_work <- 1 # average 1 manuscript per day when actually working

# sample over one year
N <- 365

# simulate the days monks drink
set.seed(365)
drink <- rbinom(N, 1, prob_drink)

# simulate the manuscripts completed
y <- (1 - drink) * rpois(N, rate_work)

# plot!
simplehist(y, xlab = "manuscripts completed", lwd = 4)
zeros_drink <- sum(drink)
zeros_work <- sum(y == 0 & drink == 0)
zeros_total <- sum(y == 0)
lines(c(0, 0), c(zeros_work, zeros_total), col = rangi2, lwd = 4)
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
# model
m12.3 <-
  ulam(
    alist(y ~ dzipois(p, lambda),
          logit(p) <- ap,
          log(lambda) <- al,
          ap ~ dnorm(-1.5, 1), # we think monks are less likely to drink 50% of the time
          al ~ dnorm(1, 0.5)),
    data = list(y = y),
    chains = 4
  )
```

    ## Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    ## 
    ## Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 1 finished in 0.7 seconds.
    ## Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 2 finished in 0.6 seconds.
    ## Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 3 finished in 0.7 seconds.
    ## Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 4 finished in 0.7 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 0.7 seconds.
    ## Total execution time: 3.5 seconds.

``` r
# posterior summary
precis(m12.3)
```

    ##           mean         sd       5.5%      94.5%    n_eff    Rhat4
    ## ap -1.28609016 0.36729314 -1.9030041 -0.7951623 541.8312 1.001693
    ## al  0.01077641 0.08767695 -0.1312203  0.1506697 475.7454 1.005102

``` r
# convert to the natural scale
post <- extract.samples(m12.3)
mean(inv_logit(post$ap)) # probability of drinking
```

    ## [1] 0.2227404

``` r
mean(exp(post$al)) # rate finishing manuscripts when not drinking
```

    ## [1] 1.014725

## 12.3 Ordered categorical outcomes

-   It’s very common to have a discrete outcome, like a count, but in
    which values indicate ordered levels (*ahem*, **NPS**).
-   Unlike counts, the differences in values are not necessarily equal.
    It may be much more difficult to move someone’s response from 9 -\>
    10 than from 3 -\> 4.
-   We want to any associated predictor variable to move predictions
    progressively through the multinomial categories (e.g., if the
    preference for black/white movies is positively associated with age,
    the model should sequentially move predictions upwards as age
    increases).
-   The solution to this challenge is the *cumulative link* function, in
    which the probability of a value is the probability of that value
    *or any smaller value*.

### 12.3.1 Example: Moral intuition

-   Let’s consider the trolley problem — someone can pull a lever that
    saves five people but kills one or do nothing. How morally
    permissible is it to pull the lever?
-   Let’s consider three principles of unconscious reasoning that may
    explain variations in responses to this question:
    1.  **The action principle**: harm caused by action is morally worse
        than harm caused by omission.
    2.  **The intention principle**: harm intended as the outcome is
        morally worse than harm foreseen as a side effect.
    3.  **The contact principle**: Physical contact to cause harm is
        morally worse than harm without physical contact.

``` r
data("Trolley")
d <- Trolley
```

### 12.3.2 Describing an ordered distribution with intercepts

``` r
simplehist(d$response, xlim = c(1, 7), xlab = "response")
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
# discrete proportion of each response value
pr_k <- table(d$response)/nrow(d)

# convert to cumulative proportions
cum_pr_k <- cumsum(pr_k)

# plot
plot(
  1:7, 
  cum_pr_k, 
  type = "b", 
  xlab = "response", 
  ylab = "cumulative proportion",
  ylim = c(0, 1)
)
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->

``` r
# logit plot
plot(
  1:7,
  riekelib::logit(cum_pr_k) |> round(2),
  xlab = "response",
  ylab = "log-cumulative odds",
  ylim = c(-2, 2)
)
```

![](chapter_12_notes_files/figure-gfm/unnamed-chunk-8-3.png)<!-- -->
