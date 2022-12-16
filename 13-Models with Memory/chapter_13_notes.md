Models with Memory
================

-   Thus far, all models have used dummy or indicator variables,
    implictly making the assumption that there’s nothing to be learned
    from one category to another.
-   We want, instead, to be able to learn how categories are different
    while also learning how they may be similar!
-   *Multilevel models* help in this regard. Here are some benefits:
    1.  **Improved estimates for repeat sampling**: When there are more
        than one observation from the same individual, location, or
        time, traditional, single-level models either maximally underfit
        or overit the data.
    2.  **Improved estimates for imbalance in sampling**: When some
        individuals, locations, or times are sampled more than others,
        multilvel models automatically cope with differing uncertainty
        (i.e., over-sampled clusters don’t dominate inference unfairly).
    3.  **Estimates of variation**: Multilevel models model variation
        within and between groups explicitly.
    4.  **Avoid averaging, retain variation**: Summarising at a roll-up
        level with an average is dangerous, since it removes variation!

## 13.1 Example: Multilevel tadpoles

``` r
library(rethinking)

# frogs!
data(reedfrogs)
d <- reedfrogs
str(d)
```

    ## 'data.frame':    48 obs. of  5 variables:
    ##  $ density : int  10 10 10 10 10 10 10 10 10 10 ...
    ##  $ pred    : Factor w/ 2 levels "no","pred": 1 1 1 1 1 1 1 1 2 2 ...
    ##  $ size    : Factor w/ 2 levels "big","small": 1 1 1 1 2 2 2 2 1 1 ...
    ##  $ surv    : int  9 10 7 10 9 9 10 9 4 9 ...
    ##  $ propsurv: num  0.9 1 0.7 1 0.9 0.9 1 0.9 0.4 0.9 ...

-   Let’s model the number surviving, `surv`, out of an initial count,
    `density`.
-   Each row is a tank containing tadpoles, so let’s create a *varying
    intercept* model based on each tank.
-   As a comparison point, let’s start with a categorical model.

$$
\begin{gather}
S_i \sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) = \alpha_{TANK[i]} \\
\alpha_j \sim \text{Normal}(0, 1.5)
\end{gather}
$$

``` r
# make the tank cluster variable
d$tank <- 1:nrow(d)

# prep for stan
dat <-
  list(
    S = d$surv,
    N = d$density,
    tank = d$tank
  )

# approximate posterior
m13.1 <-
  ulam(
    alist(S ~ dbinom(N, p),
          logit(p) <- a[tank],
          a[tank] ~ dnorm(0, 1.5)),
    data = dat,
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
    ## Chain 3 finished in 0.6 seconds.
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
    ## Mean chain execution time: 0.6 seconds.
    ## Total execution time: 3.3 seconds.

``` r
precis(m13.1, depth = 2)
```

    ##                mean        sd       5.5%       94.5%    n_eff     Rhat4
    ## a[1]   1.7279799934 0.7810475  0.5605236  3.05497640 3693.341 0.9993318
    ## a[2]   2.3832417860 0.9022468  1.0402167  3.90533400 3250.262 1.0018555
    ## a[3]   0.7547018581 0.6216138 -0.1879645  1.80023285 4848.618 0.9985676
    ## a[4]   2.3964952923 0.9161991  1.0449529  3.98121205 3957.242 0.9986028
    ## a[5]   1.7119935056 0.8181822  0.4971244  3.07789200 3384.650 0.9988680
    ## a[6]   1.7140738180 0.7942460  0.5334207  3.08196495 4061.866 0.9986920
    ## a[7]   2.4020488064 0.8684964  1.1436366  3.88761895 3635.918 0.9983213
    ## a[8]   1.7213726193 0.8020458  0.5285099  3.09697945 3787.792 0.9990263
    ## a[9]  -0.3724378374 0.6128099 -1.3706727  0.56943127 4256.682 0.9988706
    ## a[10]  1.7264658968 0.7676904  0.5825576  2.99434365 3919.965 0.9983915
    ## a[11]  0.7536809975 0.6160551 -0.1893858  1.76348660 4196.472 0.9988911
    ## a[12]  0.3525988154 0.6051254 -0.6265275  1.33663980 4431.735 0.9998605
    ## a[13]  0.7554747662 0.5958390 -0.1834629  1.73133530 5230.435 0.9986761
    ## a[14] -0.0004852317 0.6011134 -0.9706976  0.97154762 3572.723 0.9992424
    ## a[15]  1.7147216864 0.7689089  0.5738608  3.01312805 4270.452 0.9983975
    ## a[16]  1.7331479846 0.8078171  0.5723094  3.11545265 3065.900 0.9999038
    ## a[17]  2.5621169015 0.7051972  1.5043946  3.73657020 3427.909 0.9990502
    ## a[18]  2.1302207170 0.5713726  1.2517548  3.10079805 4345.540 0.9986218
    ## a[19]  1.8131734495 0.5447926  0.9875150  2.71737535 3783.873 0.9994562
    ## a[20]  3.1140262325 0.8389380  1.8925522  4.53336565 4724.712 0.9986445
    ## a[21]  2.1280591780 0.5946800  1.2424648  3.12348770 3222.861 1.0010721
    ## a[22]  2.1476121235 0.6084081  1.2450931  3.17619110 4128.523 0.9989480
    ## a[23]  2.1523031040 0.6244776  1.2554762  3.21351840 4450.284 0.9982617
    ## a[24]  1.5412270762 0.5311365  0.7493188  2.40802365 4298.448 0.9989028
    ## a[25] -1.0913674872 0.4526007 -1.8330902 -0.41783666 5795.532 0.9986803
    ## a[26]  0.0828251393 0.4207167 -0.5766541  0.78383538 5184.286 0.9984651
    ## a[27] -1.5481644363 0.5069168 -2.3678292 -0.78864763 3707.753 0.9987749
    ## a[28] -0.5594846549 0.4063960 -1.2371083  0.08528935 5247.310 0.9985573
    ## a[29]  0.0819387721 0.3931559 -0.5504635  0.72072218 3598.113 0.9986688
    ## a[30]  1.3001285127 0.4915172  0.5581699  2.08998685 4183.327 0.9990632
    ## a[31] -0.7353144256 0.4259769 -1.4231215 -0.06997488 5112.571 0.9983720
    ## a[32] -0.3965028446 0.4149600 -1.0825283  0.27211201 5595.294 0.9985292
    ## a[33]  2.8385407000 0.6433170  1.8937052  3.91035315 3713.249 0.9986763
    ## a[34]  2.4601862500 0.5828518  1.5971674  3.43471575 3707.076 0.9995890
    ## a[35]  2.4443603085 0.5590743  1.5960432  3.33701805 4861.948 0.9987940
    ## a[36]  1.8985861530 0.4835463  1.1900889  2.67263170 4321.817 0.9990898
    ## a[37]  1.9012781335 0.4586234  1.1926212  2.65064090 4398.416 0.9987889
    ## a[38]  3.3724965000 0.8144583  2.2348216  4.76131865 3385.189 0.9991075
    ## a[39]  2.4799810750 0.5879595  1.5978745  3.46626730 3629.688 0.9986496
    ## a[40]  2.1641598505 0.5249783  1.3619873  3.03908420 5430.592 0.9990306
    ## a[41] -1.9086448275 0.4630860 -2.6656976 -1.19400050 3151.532 0.9996887
    ## a[42] -0.6310970514 0.3750291 -1.2561581 -0.01951582 5161.147 0.9988974
    ## a[43] -0.5179215607 0.3391219 -1.0538815  0.03315740 4594.057 1.0000589
    ## a[44] -0.3924622681 0.3335194 -0.9404390  0.12970523 4609.219 0.9984366
    ## a[45]  0.5052236840 0.3382548 -0.0228843  1.03220615 4485.880 0.9982700
    ## a[46] -0.6323293276 0.3529009 -1.2033868 -0.07162851 3874.119 0.9998587
    ## a[47]  1.9065389660 0.4692256  1.1869420  2.68281530 3316.233 0.9988318
    ## a[48] -0.0452419296 0.3285011 -0.5601403  0.46563463 4521.705 0.9991652

-   Here is nothing new — we have 48 different estimates of alpha, one
    for each tank.
-   Let’s do the multilevel version:

$$
\begin{gather}
S_i \sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) = \alpha_{TANK[i]} \\
\alpha_j \sim \text{Normal}(\overline{\alpha}, \sigma) \\
\overline{\alpha} \sim \text{Normal}(0, 1.5) \\
\sigma \sim \text{Exponential}(1)
\end{gather}
$$

-   Now, each tank intercept, $\alpha_j$, is a function of two
    parameters: $\overline{\alpha}$ and $\sigma$. There are two levels
    for $\alpha$ (hence, the name “multilevel”).
-   The two parameters, $\overline{\alpha}$ and $\sigma$ are often
    referred to as *hyperparameters*, and their priors, *hyperpriors*.
-   In principle, there is no limit to the number of “hypers” or levels
    we can model, but in practice there are computational limits and
    limits in our ability to understand the model.
-   We can fit this model with `ulam()`, but not with `quap()`! Since
    `quap()` just approximates posteriors by “climbing a hill,” it can’t
    infer the posterior across multiple levels (there is a more robust
    explanation later).

``` r
# multilevel tadpoles!
m13.2 <-
  ulam(
    alist(S ~ dbinom(N, p),
          logit(p) <- a[tank],
          a[tank] ~ dnorm(a_bar, sigma),
          a_bar ~ dnorm(0, 1.5),
          sigma ~ dexp(1)),
    data = dat,
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
    ## Chain 4 finished in 0.5 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 0.6 seconds.
    ## Total execution time: 3.3 seconds.

``` r
compare(m13.1, m13.2)
```

    ##           WAIC       SE    dWAIC     dSE    pWAIC       weight
    ## m13.2 201.3179 7.263878  0.00000      NA 21.52188 0.9990843497
    ## m13.1 215.3078 4.610643 13.98992 4.00759 25.85083 0.0009156503

-   `m13.2` has only \~21 effective parameters! The prior shrinks all
    the intercept estimates towards the mean $\overline{\alpha}$.
-   This is despite the model having more actual parameters (50) than
    `m13.1` (48).

``` r
precis(m13.2)
```

    ##           mean        sd     5.5%    94.5%    n_eff     Rhat4
    ## a_bar 1.345938 0.2634200 0.945932 1.781546 3479.085 0.9984265
    ## sigma 1.623342 0.2087629 1.311933 1.977055 1600.963 1.0007110

-   `sigma` is a regularizing prior, like from earlier chapters, but now
    the amount of regularization has been learned from the model itself!

``` r
# extract samples
post <- extract.samples(m13.2)

# find mean for each tank
# and transform to log-probability
d$propsurv.est <- logistic(apply(post$a, 2, mean))

# display raw proportions surviving in each tank
plot(
  d$propsurv,
  ylim = c(0, 1),
  pch = 16,
  xaxt = "n",
  xlab = "tank",
  ylab = "proportion survival",
  col = rangi2
)

axis(1, at=c(1, 16, 32, 48), labels = c(1, 16, 32, 48))

# overlay posterior means
points(d$propsurv.est)

# mark posterior mean probability across tanks
abline(h = mean(inv_logit(post$a_bar)), lty = 2)

# draw vertical dividers between tank densities
abline(v = 16.5, lwd = 0.5)
abline(v = 32.5, lwd = 0.5)
text(8, 0, "small tanks")
text(16 + 8, 0, "medium tanks")
text(32 + 8, 0, "large tanks")
```

![](chapter_13_notes_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

-   In every case, the multilevel estimate (open) is closer to the prior
    mean (dashed line) than the raw estimates (blue). This is called
    *shrinkage*.
-   The smaller tanks (with fewer tadpoles) also shrink back towards the
    group mean more than the tanks with many tadpoles.
-   Shrinkage is also proportional to how far away from the group mean
    the estimate is — the further away, the greater the shrinkage.
-   All of these arise because of *pooling* — sharing information across
    groups.

``` r
# show first 100 populations in the posterior
plot(
  NULL,
  xlim = c(-3, 4),
  ylim = c(0, 0.35), 
  xlab = "log-odds survive",
  ylab = "Density"
)

for (i in 1:100) {
  
  curve(dnorm(x, post$a_bar[i], post$sigma[i]), add = TRUE, col = col.alpha("black", 0.2))
  
}
```

![](chapter_13_notes_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
# sample 8000 imaginary tanks from the posterior distribution
sim_tanks <- rnorm(8000, post$a_bar, post$sigma)

# transform to probability and visualize
dens(inv_logit(sim_tanks), lwd = 2, adj = 0.1)
```

![](chapter_13_notes_files/figure-gfm/unnamed-chunk-7-2.png)<!-- -->

-   Thus far, exponential priors on $\sigma$ terms have worked well, and
    they often continue to work well in multilevel models. There are,
    however, sometimes times when there are too few clusters to estimate
    variance from the max-entropy exponential distribution, so a
    half-normal may be more appropriate, i.e.:

$$
\begin{gather}
S_i \sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) = \alpha_{TANK[i]} \\
\alpha_j \sim \text{Normal}(\overline{\alpha}, \sigma) \\
\overline{\alpha} \sim \text{Normal}(0, 1.5) \\
\sigma \sim \text{Half-Normal}(0, 1)
\end{gather}
$$

-   This can be done in `ulam()` with `dhalfnorm()` with a parameter for
    the lower bound of 0: `lower = 0`.

## 13.2 Varying effects and the underfitting/overfitting trade-off

-   Varying intercepts are just regularized estimates, but adaptively
    regularized by estimating how diverse the clusters are while also
    estimating the features of each cluster.
-   Varying effects provide more accurate estimates of the cluster
    intercepts. This is because they do a better job of trading off
    between overfitting/underfitting.
-   Let’s look at predicting the survival of frogs from several ponds
    using a few different methods:
    1.  Complete pooling — assume the population does not vary at all
        from pond to pond.
    2.  No pooling — assume that each pond tells us nothing about any
        other pond.
    3.  Partial pooling — using an adaptive regularizing prior (like the
        last section).
-   Complete pooling will underfit the data, since the estimate for
    $\alpha$ across all ponds is unlikely to fit any particular pond
    well.
-   No pooling will overfit the data, since there is little data about
    each pond in particular.
-   Partial pooling strikes a balance!

### 13.2.1 The model

-   We’ll be simulating data from this model, then use each strategy to
    see how well it recovers the parameters:

$$
\begin{gather}
S_i \sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) = \alpha_{POND[i]} \\
\alpha_j \sim \text{Normal}(\overline{\alpha}, \sigma) \\
\overline{\alpha} \sim \text{Normal}(0, 1.5) \\
\sigma \sim \text{Exponential}(1)
\end{gather}
$$

-   We’ll need to assign values to:
    -   $\overline{\alpha}$, the average log-odds of survival in the
        entire population of ponds.
    -   $\sigma$, the standard deviation of the distribution of log-odds
        of survival among ponds.
    -   $\alpha$, a vector of individual pond intercepts, one for each
        pond.
    -   $N_i$, a sample size for each pond.

### 13.2.2 Assign values to the parameters

``` r
# specify parameters
a_bar <- 1.5
sigma <- 1.5
nponds <- 60
Ni <- as.integer(rep(c(5, 10, 25, 35), each = 15))

# setup df
set.seed(5005)
a_pond <- rnorm(nponds, mean = a_bar, sd = sigma)
d_sim <- data.frame(pond = 1:nponds, Ni = Ni, true_a = a_pond)
```

-   We’ve used `as.integer()` when creating `Ni`. R doesn’t care too
    much about this being a `"numeric"` rather than an `"integer"`, but
    Stan does!

### 13.2.3 Simulate survivors

``` r
d_sim$Si <- rbinom(nponds, prob = logistic(d_sim$true_a), size = d_sim$Ni)
```

### 13.2.4 Compute the no-pooling estimates

``` r
d_sim$p_nopool <- d_sim$Si / d_sim$Ni
```

-   The no pooling estimate we’ve added here is the same thing we’d get
    if we’d fit a model with flat priors that induce no regularization.

### 13.2.5 Compute the partial-pooling estimate

``` r
# prep for stan
dat <- 
  list(
    Si = d_sim$Si, 
    Ni = d_sim$Ni, 
    pond = d_sim$pond
  )

# model!
m13.3 <-
  ulam(
    alist(Si ~ dbinom(Ni, p),
          logit(p) <- a_pond[pond],
          a_pond[pond] ~ dnorm(a_bar, sigma),
          a_bar ~ dnorm(0, 1.5),
          sigma ~ dexp(1)),
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
    ## Chain 1 finished in 0.5 seconds.
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
    ## Chain 3 finished in 0.4 seconds.
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
    ## Chain 4 finished in 0.4 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 0.5 seconds.
    ## Total execution time: 2.7 seconds.

``` r
# display output
precis(m13.3, depth = 2)
```

    ##                   mean        sd         5.5%       94.5%     n_eff     Rhat4
    ## a_pond[1]   1.69057602 0.9696394  0.171820335  3.32659750 3081.1528 1.0004124
    ## a_pond[2]   2.88938764 1.3041403  1.003804025  5.09379965 2470.3985 0.9991508
    ## a_pond[3]  -0.62765160 0.8574147 -2.007948700  0.71526967 2886.3775 0.9982876
    ## a_pond[4]   2.86243656 1.2468991  1.083340350  4.89757560 2425.9140 0.9990962
    ## a_pond[5]   2.85135661 1.2539477  1.014984650  4.99649700 2399.6930 0.9995221
    ## a_pond[6]   2.86770870 1.2788727  1.046039900  5.06942445 2758.8731 0.9996212
    ## a_pond[7]   0.04653569 0.8254234 -1.250867700  1.31744495 3186.5238 0.9997329
    ## a_pond[8]   2.90099655 1.2453263  1.108947450  5.02948865 2537.0973 0.9991087
    ## a_pond[9]   1.66341677 1.0454380  0.112602980  3.47742795 2789.6798 0.9984895
    ## a_pond[10]  1.61034881 0.9614488  0.148074210  3.23400720 3232.4911 1.0008598
    ## a_pond[11]  2.86746880 1.2959425  0.992894865  5.06583530 2152.4240 1.0000455
    ## a_pond[12]  0.06856778 0.8293690 -1.267179950  1.36777185 3303.3015 0.9987685
    ## a_pond[13]  2.87434180 1.2750254  0.993951000  5.09111315 2296.4182 0.9984062
    ## a_pond[14]  2.88623830 1.2651400  1.088326250  5.19428535 2601.8423 0.9991245
    ## a_pond[15]  2.89218655 1.3128219  1.044428450  5.06421280 2299.2256 0.9999248
    ## a_pond[16]  1.57873717 0.7668694  0.409919160  2.95305335 3063.8785 0.9999458
    ## a_pond[17] -1.44502238 0.7706410 -2.725465400 -0.33010133 2401.0632 1.0000048
    ## a_pond[18]  1.05500626 0.6523897  0.059370174  2.16571015 2821.8951 0.9988549
    ## a_pond[19] -0.94245662 0.6546203 -2.033426900  0.05991308 3416.9791 0.9986682
    ## a_pond[20]  1.57356632 0.8032256  0.392057895  2.95944875 2647.8603 1.0003094
    ## a_pond[21] -0.14033520 0.6315116 -1.154259900  0.91912237 3736.4677 0.9988373
    ## a_pond[22]  2.22746104 0.9014484  0.913420025  3.71530290 2747.1989 1.0000703
    ## a_pond[23]  3.30581350 1.2261376  1.622769250  5.41868355 1924.1304 0.9989970
    ## a_pond[24]  0.62911042 0.6641018 -0.411591595  1.73463740 3369.1745 1.0003407
    ## a_pond[25]  3.25463335 1.1773144  1.555464800  5.27866330 2032.5185 0.9990547
    ## a_pond[26]  2.24278845 0.9269187  0.903882715  3.80853075 2653.5248 0.9996193
    ## a_pond[27]  1.05642030 0.6679804  0.029429238  2.20041425 3669.2174 1.0003224
    ## a_pond[28]  2.25707952 0.8967401  0.993532685  3.82391280 1706.6192 1.0017491
    ## a_pond[29]  1.55618807 0.7575435  0.433353455  2.77984475 2790.2503 0.9991458
    ## a_pond[30]  1.05251453 0.6510024  0.050723562  2.14160740 3755.6005 0.9995276
    ## a_pond[31]  2.46170124 0.6630393  1.470582500  3.59403095 2630.4797 0.9992298
    ## a_pond[32]  2.06173119 0.6079320  1.165568900  3.06621775 2740.3258 0.9994743
    ## a_pond[33]  1.72639768 0.5513346  0.899826855  2.67596975 3381.3278 0.9983114
    ## a_pond[34]  1.23947555 0.4442594  0.550527125  1.94816600 2904.8941 0.9992231
    ## a_pond[35]  0.66388206 0.4227995  0.004558146  1.34671690 3283.6148 0.9988225
    ## a_pond[36]  3.85266264 1.0918492  2.330695150  5.76721135 2025.4460 0.9983414
    ## a_pond[37] -0.99319212 0.4536412 -1.715902000 -0.28220333 3328.0283 0.9985758
    ## a_pond[38] -1.18110060 0.4149972 -1.860462100 -0.55234138 2713.6378 0.9994123
    ## a_pond[39]  0.67757601 0.4061146  0.037834357  1.34993980 4116.6726 0.9983234
    ## a_pond[40]  3.88888771 1.0822476  2.348142650  5.75616780 1479.5725 1.0014950
    ## a_pond[41]  3.86466389 1.0738140  2.374247750  5.67138500 1598.5961 1.0004404
    ## a_pond[42]  2.43788371 0.6831780  1.443301850  3.58260570 2948.4069 0.9991983
    ## a_pond[43] -0.12745810 0.3788010 -0.746890740  0.46099720 3904.9809 0.9983916
    ## a_pond[44]  0.66998161 0.4066384  0.056888011  1.31391365 2700.1808 0.9987685
    ## a_pond[45] -1.20588633 0.4683606 -1.975658800 -0.50794980 2353.5577 0.9995668
    ## a_pond[46]  0.01597718 0.3377263 -0.533772630  0.57435622 3078.5637 0.9990045
    ## a_pond[47]  4.03470949 0.9838553  2.621939400  5.78106875 2200.2469 0.9984817
    ## a_pond[48]  2.11231761 0.5348655  1.312427550  3.01754040 2521.3282 0.9993398
    ## a_pond[49]  1.85228236 0.4544399  1.167586700  2.59701530 3962.7588 0.9991617
    ## a_pond[50]  2.76363222 0.6301237  1.828476450  3.81084300 2572.4514 0.9999756
    ## a_pond[51]  2.38926489 0.5244173  1.618218350  3.25917390 3377.4711 0.9989168
    ## a_pond[52]  0.35775929 0.3311128 -0.175081760  0.89059836 3946.0451 0.9984909
    ## a_pond[53]  2.09439501 0.5195456  1.306026000  2.95945330 3107.8709 0.9990561
    ## a_pond[54]  4.09839495 1.0714085  2.597225200  5.98168520 1997.3636 0.9987116
    ## a_pond[55]  1.13183532 0.3932403  0.507778245  1.75752695 3312.5678 0.9999554
    ## a_pond[56]  2.77178749 0.6731752  1.781638900  3.90344540 2945.1970 0.9995765
    ## a_pond[57]  0.72361909 0.3429821  0.202428340  1.25044340 2999.7333 1.0008419
    ## a_pond[58]  4.04396550 1.0021676  2.612761050  5.77530560 1585.7584 1.0007847
    ## a_pond[59]  1.64766666 0.4435001  0.971773495  2.36436735 3051.5512 1.0003471
    ## a_pond[60]  2.39052234 0.5771943  1.543070300  3.34291450 3199.2037 0.9996222
    ## a_bar       1.66959916 0.2610380  1.274270100  2.08440825 1713.0058 1.0003674
    ## sigma       1.67392563 0.2354407  1.333771800  2.07892895  772.0146 1.0018294

``` r
# extract means for comparison
post <- extract.samples(m13.3)
d_sim$p_partpool <- apply(inv_logit(post$a_pond), 2, mean)

# true per-pond survival probabilities
d_sim$p_true <- inv_logit(d_sim$true_a)

# compute error between estimates and true effects
nopool_error <- abs(d_sim$p_nopool - d_sim$p_true)
partpool_error <- abs(d_sim$p_partpool - d_sim$p_true)

# plot!
plot(
  1:60, 
  nopool_error,
  xlab = "pond",
  ylab = "absolute error",
  col = rangi2,
  pch = 16
)

points(1:60, partpool_error)
```

![](chapter_13_notes_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

-   The blue points are the no-pooling estimates, the black circles show
    the varying effect estimates.
-   At the high end, the partially pooled/no pooled estimates are
    similar, but at the low end, the partially pooled estimates are way
    better!

``` r
nopool_avg <- aggregate(nopool_error, list(d_sim$Ni), mean)
partpool_avg <- aggregate(partpool_error, list(d_sim$Ni), mean)

nopool_avg
```

    ##   Group.1          x
    ## 1       5 0.12122935
    ## 2      10 0.10245961
    ## 3      25 0.04057566
    ## 4      35 0.03583510

``` r
partpool_avg
```

    ##   Group.1          x
    ## 1       5 0.08683174
    ## 2      10 0.07617887
    ## 3      25 0.04808259
    ## 4      35 0.03623471

-   Smaller tanks (with less information) are shrunk more towards the
    mean than large tanks (with lots more info).

## 13.3 More than one type of cluster

-   We can (and often should) include more than one type of cluster in a
    model.
-   Let’s look back at the chimpanzee lever-pulling data — each chimp is
    its own cluster, but there are also clusters of experimental groups.
-   The data structure in `data(chimpanzees)` is *cross-classified*,
    since actors are not nested within unique blocks. The model
    specification will still be the same for MCMC.

### 13.3.1 Multilevel chimpanzees

-   Let’s add varying intercepts to our chimpanzee model (recall that we
    had 4 treatment cases):

$$
\begin{gather}
L_i \sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) = \alpha_{ACTOR[i]} + \gamma_{BLOCK[i]} + \beta_{TREATMENT[i]} \\
\beta_j \sim \text{Normal}(0, 0.5) \\ 
\alpha_j \sim \text{Normal}(\overline{\alpha}, \sigma_{\alpha}) \\
\gamma_j \sim \text{Normal}(0, \sigma_{\gamma}) \\
\overline{\alpha} \sim \text{Normal}(0, 1.5) \\
\sigma_{\alpha} \sim \text{Exponential}(1) \\
\sigma_{\gamma} \sim \text{Exponential}(1)
\end{gather}
$$

-   Note that there is only one global mean variable,
    $\overline{\alpha}$. We can’t identify a separate mean for each
    varying intercept type, since both are added to the same linear
    prediction.
-   Doing so won’t be the end of the world, but would be like the
    right/left leg example from Chapter 6.

``` r
# load data
data("chimpanzees")
d <- chimpanzees
d$treatment <- 1 + d$prosoc_left + 2*d$condition

# prep for stan
dat_list <-
  list(
    pulled_left = d$pulled_left,
    actor = d$actor,
    block_id = d$block,
    treatment = as.integer(d$treatment)
  )

# model!
set.seed(13)
m13.4 <-
  ulam(
    alist(
      # model
      pulled_left ~ dbinom(1, p),
      logit(p) <- a[actor] + g[block_id] + b[treatment],
      
      # prior
      b[treatment] ~ dnorm(0, 0.5),
      
      # adaptive priors
      a[actor] ~ dnorm(a_bar, sigma_a),
      g[block_id] ~ dnorm(0, sigma_g),
      
      # hyper-priors
      a_bar ~ dnorm(0, 1.5),
      sigma_a ~ dexp(1),
      sigma_g ~ dexp(1)
    ),
    
    data = dat_list,
    chains = 4,
    cores = 4,
    log_lik = TRUE
  )
```

    ## Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    ## 
    ## Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 1 finished in 6.2 seconds.
    ## Chain 3 finished in 6.1 seconds.
    ## Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 2 finished in 7.6 seconds.
    ## Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 4 finished in 7.7 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 6.9 seconds.
    ## Total execution time: 8.0 seconds.

-   In the book, McElreath expected 22 divergent transitions. McElreath
    notes that this is fine, but the sampler had some trouble
    efficiently exploring the posterior. We’ll fix this later.
-   Let’s explore the model — this is the most complicated model we’ve
    put together so far:

``` r
precis(m13.4, depth = 2)
```

    ##                 mean        sd        5.5%       94.5%     n_eff     Rhat4
    ## b[1]    -0.151271322 0.2917121 -0.62397160  0.31827337  550.6561 1.0025574
    ## b[2]     0.380818175 0.2959266 -0.10239053  0.83639762  532.8338 1.0055399
    ## b[3]    -0.495519979 0.3035780 -0.99081850 -0.01914779  608.2773 1.0032906
    ## b[4]     0.261278965 0.3013455 -0.23274653  0.73978374  491.4415 1.0063042
    ## a[1]    -0.335064116 0.3628073 -0.90140597  0.24966478  531.4998 1.0020375
    ## a[2]     4.769753480 1.3642878  3.02849600  7.05246695  715.7969 1.0013466
    ## a[3]    -0.639852578 0.3643614 -1.22644070 -0.07451138  528.3749 1.0026814
    ## a[4]    -0.638243052 0.3745068 -1.22127630 -0.02540016  559.8474 1.0045913
    ## a[5]    -0.334052981 0.3612879 -0.90435982  0.25533117  592.7247 1.0041918
    ## a[6]     0.610392630 0.3648185  0.03796449  1.19263460  530.8618 1.0018512
    ## a[7]     2.139333271 0.4617689  1.39976525  2.87895035  783.0858 1.0008911
    ## g[1]    -0.182393504 0.2261186 -0.59217019  0.07424323  582.9506 1.0005803
    ## g[2]     0.040514582 0.1791954 -0.22398016  0.34374918 1302.3215 0.9988363
    ## g[3]     0.052076740 0.1775660 -0.21035093  0.35100696 1130.2783 1.0024707
    ## g[4]     0.005591961 0.1783925 -0.26522308  0.28284979 1141.4036 1.0005948
    ## g[5]    -0.035386524 0.1778859 -0.33209991  0.22422660 1248.1184 1.0019953
    ## g[6]     0.109275209 0.1882717 -0.14917160  0.42925906  846.6328 1.0026579
    ## a_bar    0.634942868 0.7077410 -0.52519050  1.74305830 1227.5966 1.0009295
    ## sigma_a  2.052673332 0.6722321  1.22494630  3.23038155  921.5895 0.9985904
    ## sigma_g  0.222820117 0.1716902  0.04519144  0.54579843  429.4329 1.0045743

``` r
precis_plot(precis(m13.4, depth = 2))
```

![](chapter_13_notes_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

-   A couple things worth noting:
    1.  `n_eff` varies quite a lot across parameters. This is common in
        complex models and may be a result of the sampler spending a lot
        of time near a boundary for one parameter (here, that’s
        `sigma_g` spending too much time near 0).
    2.  Compare `sigma_a` to `sigma_g` — the estimated variation among
        actors is a lot larger than the estimated variation among
        blocks. The chimps vary, but the blocks are all the same.

``` r
# fit a model that ignores block
set.seed(14)
m13.5 <-
  ulam(
    alist(pulled_left ~ dbinom(1, p),
          logit(p) <- a[actor] + b[treatment],
          b[treatment] ~ dnorm(0, 0.5),
          a[actor] ~ dnorm(a_bar, sigma_a),
          a_bar ~ dnorm(0, 1.5),
          sigma_a ~ dexp(1)),
    data = dat_list,
    chains = 4,
    cores = 4,
    log_lik = TRUE
  )
```

    ## Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    ## 
    ## Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 3 finished in 4.3 seconds.
    ## Chain 4 finished in 4.2 seconds.
    ## Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 1 finished in 4.4 seconds.
    ## Chain 2 finished in 4.4 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 4.3 seconds.
    ## Total execution time: 4.6 seconds.

``` r
# compare to the model with both clusters
compare(m13.4, m13.5)
```

    ##           WAIC       SE     dWAIC      dSE     pWAIC    weight
    ## m13.5 531.5856 19.19042 0.0000000       NA  8.695202 0.5419143
    ## m13.4 531.9217 19.40196 0.3361032 1.753878 10.594594 0.4580857

-   Here, the `pWAIC` column reports the effective number of parameters.
    `m13.4` has 7 more parameters than `m13.5`, but only 2 more
    effective parameters, because the posterior distribution for
    `sigma_g` ended up close to 0.
-   This means that each of the 6 `g` parameters are pretty inflexible,
    while the `a` parameters resulted in less shrinkage.
-   The difference in `WAIC` between the models is small — the block
    parameters contribute little additional information to the model.
-   There is nothing to gain by selecting either model, but the
    comparison tells a rich story: whether we include block or not
    hardly matters, and the `g` and `sigma_g` parameters tell us why.

### 13.3.2 Even more clusters

-   Let’s fit again with a varying effect for treatment (this will piss
    off certain folks with a specific background in statistical
    semantics).

``` r
set.seed(15)
m13.6 <-
  ulam(
    alist(
      # model
      pulled_left ~ dbinom(1, p),
      logit(p) <- a[actor] + g[block_id] + b[treatment],
      
      # adaptive priors
      b[treatment] ~ dnorm(0, sigma_b),
      a[actor] ~ dnorm(a_bar, sigma_a),
      g[block_id] ~ dnorm(0, sigma_g),
      
      # hyper-priors
      a_bar ~ dnorm(0, 1.5),
      sigma_a ~ dexp(1),
      sigma_g ~ dexp(1),
      sigma_b ~ dexp(1)
    ),
    
    data = dat_list,
    chains = 4, 
    cores = 4,
    log_lik = TRUE
  )
```

    ## Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    ## 
    ## Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    ## Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    ## Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    ## Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    ## Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    ## Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    ## Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    ## Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    ## Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    ## Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    ## Chain 3 finished in 7.5 seconds.
    ## Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    ## Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 4 finished in 8.6 seconds.
    ## Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 1 finished in 8.7 seconds.
    ## Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    ## Chain 2 finished in 8.8 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 8.4 seconds.
    ## Total execution time: 9.1 seconds.

``` r
precis(m13.6, depth = 2)
```

    ##                 mean        sd        5.5%       94.5%     n_eff     Rhat4
    ## b[1]    -0.086007710 0.3298502 -0.57533950  0.42611053  426.1025 1.0050301
    ## b[2]     0.404528269 0.3630766 -0.07218956  0.99097660  408.4146 1.0015473
    ## b[3]    -0.410957900 0.3494859 -0.96192083  0.08883822  412.9184 1.0081533
    ## b[4]     0.308975519 0.3513093 -0.17756353  0.88595720  420.6486 1.0019752
    ## a[1]    -0.390663772 0.3991074 -1.01380300  0.20281940  444.1036 1.0042778
    ## a[2]     4.553820710 1.2158126  2.99920480  6.64409445  544.2434 1.0001459
    ## a[3]    -0.698438640 0.4045194 -1.33840275 -0.09983740  417.1458 1.0049152
    ## a[4]    -0.702559595 0.4075026 -1.35630855 -0.10064754  425.2828 1.0042723
    ## a[5]    -0.397464957 0.3936444 -1.04653685  0.18835521  467.8641 1.0054846
    ## a[6]     0.546664577 0.4112035 -0.09262478  1.15409575  461.9781 1.0046324
    ## a[7]     2.065370554 0.4718346  1.35449420  2.81698820  540.7290 1.0025484
    ## g[1]    -0.154652528 0.2146860 -0.56507879  0.06957587  304.2532 1.0207689
    ## g[2]     0.042103709 0.1695586 -0.19098848  0.35223915  915.6168 1.0038102
    ## g[3]     0.045158582 0.1686725 -0.18478215  0.34252595  775.3787 1.0019710
    ## g[4]     0.002901591 0.1609603 -0.24674455  0.25612233 1205.2585 1.0002284
    ## g[5]    -0.026017469 0.1594274 -0.29042810  0.20971909 1297.0757 1.0040366
    ## g[6]     0.100505832 0.1837296 -0.11052618  0.44782528  441.3487 1.0027480
    ## a_bar    0.573867790 0.6903944 -0.51641714  1.65037200  687.0416 0.9993232
    ## sigma_a  1.956178982 0.6260791  1.17253630  3.03771695  729.6221 1.0009958
    ## sigma_g  0.193380150 0.1607222  0.02559283  0.49464170  166.1626 1.0200381
    ## sigma_b  0.567064700 0.3536679  0.19321751  1.23737490  478.3898 1.0118923

``` r
precis_plot(precis(m13.6, depth = 2))
```

![](chapter_13_notes_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
compare(m13.4, m13.6)
```

    ##           WAIC       SE    dWAIC       dSE    pWAIC    weight
    ## m13.4 531.9217 19.40196 0.000000        NA 10.59459 0.5922075
    ## m13.6 532.6679 19.21735 0.746197 0.5218316 10.54338 0.4077925

-   Here, we don’t get a whole lot of additonal info by setting the
    intercept to vary by treatment — we do get divergent transitions
    though. Let’s deal with those.
