Missing Data and Other Opportunities
================

-   Consider three pancakes. One is burnt on both sides (BB). One is
    burnt on one side (BU). And one is wholly unburnt (UU). If you are
    served a pancake and the side face up is burnt, what is the
    probability the other side is burnt?
-   The intuitive answer is one half — but intuition misleads us!

$$
\begin{align*}
\text{Pr(burnt down|burnt up)} & = \frac{\text{Pr(burnt up & burnt down)}}{\text{Pr(burnt up)}} \\
& = \frac{1/3}{1/2} \\
& = \frac{2}{3}
\end{align*}
$$

-   We can also confirm this via simulation:

``` r
# simulate pancake & return rancomly ordered sides
sim_pancake <- function() {
  
  pancake <- sample(1:3, 1)
  sides <- matrix(c(1, 1, 1, 0, 0, 0), 2, 3)[,pancake]
  sample(sides)
  
}

# sim 10,000 pancakes
pancakes <- replicate(1e4, sim_pancake())
up <- pancakes[1,]
down <- pancakes[2,]

# compute proportion 1/1 (BB) out of all 1/1 and 1/0
num_11_10 <- sum(up == 1)
num_11 <- sum(up == 1 & down == 1)
num_11/num_11_10
```

    ## [1] 0.6739044

-   This is just a matter of counting *sides* rather than *pancakes*.
-   Probability theory is not difficult mathematically — it’s just
    counting! The difficulty is in the interpretation/application.
-   Two commonplace applications that aren’t difficult mathematically
    but need some work are including *measurement error* in models, and
    using *Bayesian imputation* to account for *missing data*.

## 15.1 Measurement error

-   Let’s look back at the divorce/marriage data from chapter 5.

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

data(WaffleDivorce)
d <- WaffleDivorce

# points
plot(d$Divorce ~ d$MedianAgeMarriage,
     ylim = c(4, 15),
     xlab = "Median age marriage",
     ylab = "Divorce rate")

# standard errors
for (i in 1:nrow(d)) {
  
  ci <- d$Divorce[i] + c(-1, 1)*d$Divorce.SE[i]
  x <- d$MedianAgeMarriage[i]
  lines(c(x, x), ci)
  
}
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

-   There’s lots of variation in the measured divorce rate! Since the
    standard errors for some states are smaller, we could reasonably
    expect that the model’s estimate for that state’s rate would also be
    smaller.

### 15.1.1 Error on the outcome

-   Let’s reintroduce the causal model and include an observation error
    on the outcome:

``` r
library(dagitty)
```

    ## Warning: package 'dagitty' was built under R version 4.2.1

``` r
marriage <-
  dagitty(
    "dag{
      D [unobserved]
      e_D [unobserved]
      A -> M
      A -> D
      M -> D
      D -> D_obs
      e_D -> D_obs
    }"
  )

coordinates(marriage) <-
  list(x = c(A = 1, M = 2, D = 2, D_obs = 3, e_D = 4),
       y = c(A = 2, M = 1, D = 3, D_obs = 3, e_D = 3))

drawdag(marriage)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

-   In this version of the DAG, age at marriage & marriage rate causally
    influence the divorce rate. We don’t observe the true divorce rate,
    but we do observe $D_{obs}$, which is a function of both D and some
    error, $\text{e}_D$.
-   The key benefit of Bayes is that we can simply put parameters in the
    gaps in our knowledge (i.e., the true divorce rate):

$$
\begin{align*}
D_{\text{OBS}[i]} & \sim \text{Normal}(D_{\text{TRUE}[i]}, D_{\text{SE}[i]}) \\
D_{\text{TRUE}[i]} & \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i & = \alpha + \beta_A A_A + \beta_M M_M \\
\alpha & \sim \text{Normal}(0,\ 0.2) \\
\beta_A & \sim \text{Normal}(0, \ 0.5) \\
\beta_M & \sim \text{Normal}(0, \ 0.5) \\
\sigma & \sim \text{Exponential}(1)
\end{align*}
$$

-   Here, the true divorce rates just become a parameter in the
    distribution for the observed divorce rate.
-   This also allows information to flow in both directions — the
    uncertainty in measurement influences the regression parameters and
    the regression parameters influence the uncertainty in the
    measurements.

``` r
# prep for stan
dlist <- 
  list(
    D_obs = standardize(d$Divorce),
    D_sd = d$Divorce.SE / sd(d$Divorce),
    M = standardize(d$Marriage),
    A = standardize(d$MedianAgeMarriage),
    N = nrow(d)
  )

# model!
m15.1 <-
  ulam(
    alist(
      # model
      D_obs ~ dnorm(D_true, D_sd),
      vector[N]:D_true ~ dnorm(mu, sigma),
      mu <- a + bA*A + bM*M,
      
      # priors
      a ~ dnorm(0, 0.25),
      bA ~ dnorm(0, 0.5),
      bM ~ dnorm(0, 0.5),
      sigma ~ dexp(1)
    ),
    
    data = dlist,
    chains = 4,
    cores = 4
  )

precis(m15.1, depth = 2)
```

    ##                   mean         sd        5.5%       94.5%    n_eff     Rhat4
    ## D_true[1]   1.16780438 0.37372487  0.58331328  1.76628684 2042.193 1.0008342
    ## D_true[2]   0.67395212 0.54882855 -0.19557769  1.56542765 1922.640 1.0005664
    ## D_true[3]   0.44194592 0.33807176 -0.09126412  0.99072016 2764.111 1.0004091
    ## D_true[4]   1.42247712 0.47413339  0.67811597  2.18133429 2064.769 1.0006598
    ## D_true[5]  -0.90303910 0.12948335 -1.10482218 -0.69041250 3023.070 1.0000806
    ## D_true[6]   0.65452157 0.38669977  0.04353191  1.28258047 2187.813 1.0001565
    ## D_true[7]  -1.36629151 0.35905343 -1.94053504 -0.79752163 2113.484 1.0011387
    ## D_true[8]  -0.34702576 0.48186686 -1.13414175  0.45203107 2423.973 0.9984205
    ## D_true[9]  -1.87390622 0.59154327 -2.82962900 -0.93734142 1495.152 1.0021692
    ## D_true[10] -0.61991811 0.16800941 -0.89368773 -0.35543016 2090.529 1.0003955
    ## D_true[11]  0.76564364 0.28250123  0.32133608  1.22664706 2352.716 0.9994334
    ## D_true[12] -0.54374851 0.47758006 -1.32043806  0.22161455 1789.933 1.0047574
    ## D_true[13]  0.15876928 0.47574522 -0.61566611  0.92455702 1412.599 1.0005673
    ## D_true[14] -0.86673971 0.22506094 -1.23055286 -0.49189371 3066.025 0.9985196
    ## D_true[15]  0.54568745 0.29501779  0.08409063  1.01890483 2563.853 1.0002854
    ## D_true[16]  0.27765696 0.38017543 -0.32039393  0.87944975 2362.003 0.9987135
    ## D_true[17]  0.49058109 0.41534685 -0.16239286  1.14570363 3006.364 0.9987191
    ## D_true[18]  1.26911126 0.34966798  0.71817953  1.84938680 2326.400 0.9991357
    ## D_true[19]  0.43869834 0.36861681 -0.16407074  1.02137614 2598.446 1.0006655
    ## D_true[20]  0.41738855 0.53836099 -0.38736591  1.31193630 1638.942 1.0011805
    ## D_true[21] -0.55975647 0.33337528 -1.09221110 -0.03219480 2373.872 1.0018593
    ## D_true[22] -1.09532530 0.26059141 -1.51565363 -0.68799153 2894.001 0.9989591
    ## D_true[23] -0.26807788 0.25515666 -0.67810289  0.14676896 2793.497 0.9993155
    ## D_true[24] -1.00762907 0.29715457 -1.48288147 -0.53883985 2749.956 0.9995861
    ## D_true[25]  0.43018602 0.40359316 -0.19475538  1.08087128 2426.957 0.9995258
    ## D_true[26] -0.04193324 0.32100581 -0.56258690  0.47091842 2727.232 1.0000368
    ## D_true[27] -0.03518350 0.51500041 -0.85088029  0.80629464 2285.713 1.0000669
    ## D_true[28] -0.16263047 0.39126830 -0.79892313  0.45199223 2122.180 0.9990768
    ## D_true[29] -0.25356973 0.49529449 -1.03511085  0.52862611 2721.115 0.9994426
    ## D_true[30] -1.80230364 0.23332456 -2.16629895 -1.43773128 3001.416 0.9995745
    ## D_true[31]  0.16826086 0.43223149 -0.52342097  0.83302581 2740.475 0.9995587
    ## D_true[32] -1.66371648 0.16747256 -1.93430254 -1.39436797 2972.891 0.9984121
    ## D_true[33]  0.12040358 0.24784474 -0.27021999  0.50876945 2357.881 0.9993920
    ## D_true[34] -0.06719546 0.51518392 -0.88376694  0.71596869 2266.399 0.9992449
    ## D_true[35] -0.11828614 0.21533464 -0.45289335  0.23527633 2750.309 0.9986053
    ## D_true[36]  1.28682000 0.39391710  0.67902362  1.90469791 2184.255 1.0013514
    ## D_true[37]  0.22789688 0.37285737 -0.36985379  0.83725958 1929.575 0.9984769
    ## D_true[38] -1.02744945 0.21457293 -1.38102420 -0.68698530 2988.014 0.9988806
    ## D_true[39] -0.92246327 0.55358815 -1.77676312 -0.04950456 2325.787 1.0001543
    ## D_true[40] -0.67842678 0.32665137 -1.23097273 -0.18636465 2582.436 0.9990403
    ## D_true[41]  0.23827313 0.55154873 -0.65545393  1.13116626 3036.597 0.9990261
    ## D_true[42]  0.73507002 0.34479804  0.19979239  1.29790477 2426.756 0.9996237
    ## D_true[43]  0.18647000 0.18331247 -0.10512835  0.48527562 2841.822 0.9991114
    ## D_true[44]  0.78444024 0.43284484  0.07940729  1.46478142 1965.843 0.9998461
    ## D_true[45] -0.42628773 0.52958676 -1.24553060  0.42414722 2807.098 0.9986435
    ## D_true[46] -0.39589042 0.25223079 -0.80393808 -0.00039385 3100.307 0.9987689
    ## D_true[47]  0.12388369 0.31106067 -0.38723224  0.62805758 3093.467 0.9998303
    ## D_true[48]  0.55305714 0.48368993 -0.19850931  1.34174008 2647.319 0.9997201
    ## D_true[49] -0.64122223 0.27590123 -1.06807401 -0.19002943 2543.547 1.0005587
    ## D_true[50]  0.84581424 0.58747234 -0.09947007  1.73556756 1501.757 1.0012255
    ## a          -0.05873146 0.09998842 -0.21715858  0.10453779 1393.205 1.0026254
    ## bA         -0.60702577 0.15658255 -0.84369264 -0.34578761 1131.080 1.0053751
    ## bM          0.05627484 0.16558513 -0.20508038  0.31696056 1014.571 1.0053933
    ## sigma       0.59283993 0.10637335  0.44245106  0.77205530  778.514 1.0044422

> Note here the use of `vector[N]`. I’m not 100% sure why, but this the
> whole thing sample faster & with fewer diagnostic problems than a
> non-vectorized version.

-   The old model from chapter 5 ound that `bA` was about -1 — now it’s
    almost half that, but still reliably negative. Errors in measuremetn
    can sometimes exaggerate effects or diminish them, depending on the
    context!
-   Look at figure 15.2 on page 495 to see *shrinkage* in action — less
    certain estimates are improved by pooling information with more
    certain estimates.

### 15.1.2 Error on both outcome and predictor

-   What about when there’s measurement error on a predictor as well?
-   We can do the same as before!

``` r
marriage <-
  dagitty(
    "dag{
      M [unobserved]
      D [unobserved]
      e_M [unobserved]
      e_D [unobserved]
      A -> M
      A -> D
      M -> D
      D -> D_obs
      M -> M_obs
      e_D -> D_obs
      e_M -> M_obs
    }"
  )

coordinates(marriage) <-
  list(x = c(A = 1, M = 2, D = 2, M_obs = 3, D_obs = 3, e_M = 4, e_D = 4),
       y = c(A = 2, M = 1, D = 3, M_obs = 1, D_obs = 3, e_M = 1, e_D = 3))

drawdag(marriage)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

-   Fitting the model is much like before:

``` r
# prep data for stan
dlist <-
  list(
    D_obs = standardize(d$Divorce),
    D_sd = d$Divorce.SE/sd(d$Divorce),
    M_obs = standardize(d$Marriage),
    M_sd = d$Marriage.SE/sd(d$Marriage),
    A = standardize(d$MedianAgeMarriage),
    N = nrow(d)
  )

# model!
m15.2 <-
  ulam(
    alist(
      # model
      D_obs ~ dnorm(D_true, D_sd),
      vector[N]:D_true ~ dnorm(mu, sigma),
      mu <- a + bA*A + bM*M_true[i],
      M_obs ~ dnorm(M_true, M_sd),
      vector[N]:M_true ~ dnorm(0, 1),
      
      # priors
      a ~ dnorm(0, 0.2),
      bA ~ dnorm(0, 0.5),
      bM ~ dnorm(0, 0.5),
      sigma ~ dexp(1)
    ),
    
    data = dlist,
    chains = 4,
    cores = 4
  )

precis(m15.2, depth = 2)
```

    ##                    mean         sd        5.5%       94.5%     n_eff     Rhat4
    ## D_true[1]   1.140541943 0.36882245  0.55575732  1.73981057 1903.2371 0.9994531
    ## D_true[2]   0.757603521 0.54506621 -0.06460923  1.69119392 1729.0246 0.9998596
    ## D_true[3]   0.425629304 0.33726502 -0.10781154  0.97477301 2332.3545 1.0007658
    ## D_true[4]   1.469788970 0.45953250  0.76191153  2.20696785 1780.8038 0.9991437
    ## D_true[5]  -0.897159633 0.12805007 -1.09932630 -0.69112644 2856.4974 1.0007999
    ## D_true[6]   0.709152185 0.41586410  0.06334501  1.38582384 2167.1339 0.9986496
    ## D_true[7]  -1.350937964 0.35851195 -1.94813216 -0.79797116 2247.9455 0.9986614
    ## D_true[8]  -0.276037172 0.48516501 -1.04611909  0.50793490 1563.4750 1.0000089
    ## D_true[9]  -1.724767756 0.60605236 -2.67986008 -0.75570579 1512.5236 1.0059297
    ## D_true[10] -0.621442802 0.16596254 -0.88210967 -0.35140062 2559.4159 0.9988664
    ## D_true[11]  0.784261838 0.28707106  0.32390166  1.23573694 2594.9932 0.9984537
    ## D_true[12] -0.452837761 0.46914926 -1.19988976  0.28141412 1677.3668 0.9999887
    ## D_true[13]  0.189144328 0.49632715 -0.62683593  0.96734805 1303.7743 0.9994926
    ## D_true[14] -0.871953231 0.22781929 -1.22959325 -0.51051534 2165.0035 0.9994384
    ## D_true[15]  0.534719913 0.29955283  0.05031746  1.01108545 2031.2534 0.9999267
    ## D_true[16]  0.304659535 0.36885901 -0.28596562  0.88569531 2620.3151 0.9990931
    ## D_true[17]  0.504596371 0.39829279 -0.11986273  1.13408570 2466.3545 0.9989951
    ## D_true[18]  1.252935848 0.35105342  0.69810917  1.81462214 1896.0987 1.0003119
    ## D_true[19]  0.426038683 0.36585275 -0.16350371  1.02824692 2567.9069 0.9990679
    ## D_true[20]  0.248256403 0.56356180 -0.61653506  1.18255472 1137.1235 0.9991503
    ## D_true[21] -0.537607793 0.32067945 -1.04385343 -0.03314072 2359.1278 0.9997448
    ## D_true[22] -1.105942398 0.24629011 -1.49295924 -0.70515347 2063.8017 1.0020740
    ## D_true[23] -0.299481793 0.25926255 -0.71678462  0.11927123 2440.1230 0.9990001
    ## D_true[24] -1.043396061 0.29388260 -1.52583954 -0.58032521 2364.9069 0.9995325
    ## D_true[25]  0.410300073 0.41615115 -0.24405454  1.07748692 1957.6448 1.0002865
    ## D_true[26] -0.063000049 0.31740158 -0.57410081  0.45156935 2799.3946 1.0008652
    ## D_true[27] -0.053919346 0.51405163 -0.89994619  0.79025737 2223.3612 0.9989108
    ## D_true[28] -0.159968734 0.39432758 -0.78522067  0.44643793 2198.4437 0.9986513
    ## D_true[29] -0.287755594 0.50287601 -1.07703414  0.52015537 2141.4793 0.9995017
    ## D_true[30] -1.806040690 0.23444270 -2.18774835 -1.43128915 2401.0108 0.9991926
    ## D_true[31]  0.181818150 0.42238446 -0.49486124  0.86612094 1889.2183 0.9986837
    ## D_true[32] -1.654054925 0.16336989 -1.92258421 -1.39741854 2448.6999 0.9991190
    ## D_true[33]  0.126218291 0.23750048 -0.26166808  0.50671749 2560.3042 1.0006560
    ## D_true[34] -0.005606933 0.50488768 -0.81542368  0.79172069 1430.0799 0.9993832
    ## D_true[35] -0.145504192 0.22704709 -0.49469489  0.20953997 2157.6065 1.0001347
    ## D_true[36]  1.305050252 0.39040959  0.71192715  1.94804846 2056.2788 1.0009923
    ## D_true[37]  0.209116240 0.35259941 -0.34846461  0.77023629 2413.3716 0.9987548
    ## D_true[38] -1.033932921 0.22697020 -1.39541124 -0.67951537 3043.9593 0.9994284
    ## D_true[39] -0.910148410 0.52053826 -1.73910225 -0.05021686 1960.8265 0.9986940
    ## D_true[40] -0.679097674 0.32332516 -1.20137119 -0.17260748 2229.6805 0.9987139
    ## D_true[41]  0.225109169 0.56231158 -0.66611871  1.11204404 1818.8162 1.0001707
    ## D_true[42]  0.711421738 0.33089092  0.18650229  1.23482365 2066.8087 0.9993529
    ## D_true[43]  0.200393238 0.18758781 -0.11093609  0.49162307 2046.5549 0.9991545
    ## D_true[44]  0.865875020 0.43487839  0.17565041  1.55553918 1165.0276 0.9993665
    ## D_true[45] -0.413152496 0.52403384 -1.22858416  0.42447239 2406.8053 0.9994343
    ## D_true[46] -0.356168459 0.25027804 -0.76168812  0.04475670 2391.4470 0.9990157
    ## D_true[47]  0.159539849 0.30972517 -0.35448585  0.66114273 2556.0427 1.0006016
    ## D_true[48]  0.550964428 0.43932266 -0.16156773  1.24610220 2258.2708 0.9986652
    ## D_true[49] -0.656879286 0.27044696 -1.10307345 -0.21410465 2332.4082 0.9999468
    ## D_true[50]  0.858167214 0.54794181 -0.01977136  1.71545118 1635.9749 0.9999698
    ## M_true[1]   0.084519515 0.32484185 -0.43438598  0.60604462 1639.3477 1.0005421
    ## M_true[2]   1.022427160 0.59995794  0.03393143  1.94132474 2087.8328 1.0051291
    ## M_true[3]   0.058960519 0.25619931 -0.34344469  0.47942696 2178.2482 0.9994569
    ## M_true[4]   1.421842645 0.40792535  0.77001208  2.07426519 2522.4161 0.9987033
    ## M_true[5]  -0.267388663 0.09888959 -0.43240858 -0.11449238 2386.7320 1.0000708
    ## M_true[6]   0.827161383 0.31685869  0.32228840  1.33701923 2648.5605 0.9986130
    ## M_true[7]  -0.764474936 0.25764998 -1.18538079 -0.35988452 2177.0626 0.9993855
    ## M_true[8]   0.444968355 0.58841986 -0.51702414  1.41271210 2829.9455 0.9988677
    ## M_true[9]  -0.467052673 0.53777185 -1.30900109  0.41287862 2546.5099 0.9990709
    ## M_true[10] -0.802202669 0.15357430 -1.04830713 -0.55662272 2438.2842 0.9994288
    ## M_true[11]  0.516402170 0.20463814  0.18664363  0.84521941 2200.6461 0.9997946
    ## M_true[12]  0.805552843 0.56135089 -0.07200450  1.69972811 2272.4926 0.9989129
    ## M_true[13]  1.047611030 0.46940365  0.31095117  1.79497616 1180.1712 0.9998634
    ## M_true[14] -0.572972479 0.14770559 -0.80790990 -0.34297373 3206.4902 0.9987018
    ## M_true[15] -0.063457562 0.20424829 -0.39149661  0.27373742 2215.6974 0.9981300
    ## M_true[16]  0.315663249 0.34731247 -0.21775087  0.87395061 2266.4400 0.9986163
    ## M_true[17]  0.441540861 0.37466225 -0.15898961  1.03671451 2339.9653 0.9988274
    ## M_true[18]  0.550015650 0.28316743  0.09151492  1.00000317 2368.6632 0.9997490
    ## M_true[19]  0.139618894 0.28613624 -0.31102181  0.58733085 2086.3502 0.9992164
    ## M_true[20] -1.459234986 0.36365768 -2.03611126 -0.84847390 2008.6199 1.0003644
    ## M_true[21] -0.427934772 0.25728585 -0.82545998 -0.01205946 2353.7674 0.9992656
    ## M_true[22] -1.086588464 0.18272007 -1.38637537 -0.78663520 3017.0611 0.9997681
    ## M_true[23] -0.915870882 0.18336866 -1.21311034 -0.62335577 2917.9147 0.9996829
    ## M_true[24] -1.230613295 0.20002700 -1.54473067 -0.90558246 2425.1222 1.0009683
    ## M_true[25] -0.140727919 0.37875465 -0.74382368  0.44660281 2524.5161 1.0000552
    ## M_true[26] -0.379577624 0.20154554 -0.71498094 -0.05634525 2213.6433 1.0003611
    ## M_true[27] -0.326164710 0.51211149 -1.15265112  0.50259223 2141.4719 0.9989317
    ## M_true[28] -0.146450522 0.34169139 -0.68465422  0.39706992 2668.6443 0.9982490
    ## M_true[29] -0.739494998 0.41217109 -1.39688672 -0.05261391 2305.8793 0.9989674
    ## M_true[30] -1.383725311 0.14962681 -1.62499861 -1.14679423 2074.8977 0.9991333
    ## M_true[31]  0.073894592 0.43259606 -0.61180929  0.75475126 2645.5704 0.9987405
    ## M_true[32] -0.863562410 0.11865160 -1.04788288 -0.67490579 2611.2820 0.9987901
    ## M_true[33]  0.070048724 0.24769734 -0.31442149  0.47181958 2710.8079 0.9984857
    ## M_true[34]  0.984848015 0.59275620  0.08120560  1.93845614 2174.7260 1.0002915
    ## M_true[35] -0.820289907 0.16291782 -1.08030155 -0.55570657 3399.0940 0.9990572
    ## M_true[36]  0.905630737 0.31951599  0.39030534  1.39959014 1924.3210 0.9984665
    ## M_true[37] -0.288465907 0.28439159 -0.75582586  0.16393148 2795.5312 1.0002818
    ## M_true[38] -1.198480183 0.12255373 -1.39055967 -1.00170234 3040.0771 0.9991138
    ## M_true[39] -0.998602245 0.46453985 -1.73036049 -0.25000937 2230.1921 1.0008465
    ## M_true[40] -0.507291281 0.29644013 -0.96843410 -0.02337120 2657.0367 0.9998123
    ## M_true[41]  0.013342760 0.55170271 -0.89769662  0.88200349 2378.8665 0.9987860
    ## M_true[42] -0.169361815 0.21737246 -0.51043026  0.18599036 2502.0553 1.0008824
    ## M_true[43]  0.352689514 0.16491270  0.08714638  0.61341672 3104.2908 0.9991379
    ## M_true[44]  1.944384354 0.42687970  1.26544044  2.63747891 1905.5290 0.9999503
    ## M_true[45] -0.668332805 0.51448237 -1.50798115  0.16260177 2554.5978 0.9986308
    ## M_true[46]  0.087937174 0.20916399 -0.25017393  0.41612001 2387.1950 0.9996909
    ## M_true[47]  0.317577000 0.26021484 -0.10522892  0.72829578 3311.7467 0.9998480
    ## M_true[48]  0.461042849 0.39869255 -0.18135115  1.11393836 3079.0765 0.9987028
    ## M_true[49] -0.745679782 0.20784721 -1.07724881 -0.41044521 2411.7488 0.9989939
    ## M_true[50]  1.297963820 0.71857831  0.15487452  2.48919303 2231.2398 0.9993735
    ## a          -0.036234590 0.09767503 -0.19607284  0.12055158 1261.4291 1.0006119
    ## bA         -0.529716222 0.15845642 -0.78176409 -0.28342393 1000.2304 1.0035078
    ## bM          0.213268411 0.20935831 -0.12228075  0.55131579  628.7724 1.0031420
    ## sigma       0.565275644 0.11258755  0.39643679  0.75701003  522.2215 1.0021766

-   The shrinkage here didn’t change the inference on the divorce rate,
    but it idd update the estimates of marriage rate:

``` r
post <- extract.samples(m15.2)
D_true <- apply(post$D_true, 2, mean)
M_true <- apply(post$M_true, 2, mean)
plot(dlist$M_obs,
     dlist$D_obs,
     pch = 16, 
     col = rangi2,
     xlab = "marriage rate (std)",
     ylab = "divorce rate (std)")

points(M_true, D_true)
for (i in 1:nrow(d)) 
  lines(c(dlist$M_obs[i], M_true[i]),
        c(dlist$D_obs[i], D_true[i]))
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

-   The big take-home point is that when you have a distribution of
    values, don’t reduce it down to a single value to use in a
    regression! Incorporate the entire uncertainty.

### 15.1.3 Measurement terrors

-   In the previous example, there are no new confounds introduced by
    the errors, but this isn’t always the case!
-   consider the following DAG — the errors on $D$ and $M$ are
    correlated through an influence by $P$:

``` r
marriage <- 
  dagitty(
    "dag{
      M [unobserved]
      D [unobserved]
      e_M [unobserved]
      e_D [unobserved]
      A -> M
      A -> D
      M -> D
      M -> M_obs
      D -> D_obs
      e_M -> M_obs
      e_D -> D_obs
      P -> e_M
      P -> e_D
    }"
  )

coordinates(marriage) <-
  list(x = c(A = 1, M = 2, D = 2, M_obs = 3, D_obs = 3, e_M = 4, e_D = 4, P = 5),
       y = c(A = 1, M = 0, D = 2, M_obs = 0, D_obs = 2, e_M = 0, e_D = 2, P = 1))

drawdag(marriage)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

-   Naively regressing $D_{obs}$ on $M_{obs}$ opens a non causal path
    through $P$.
-   If we have information on the measurement process and can model the
    true variables $D$ and $M$, there’s hope, but we need to consider
    the covariance between the errors.
-   Another potential case:

``` r
marriage <-
  dagitty(
    "dag{
      M [unobserved]
      D [unobserved]
      e_M [unobserved]
      e_D [unobserved]
      A -> M
      A -> D
      M -> D
      M -> e_D
      M -> M_obs
      D -> D_obs
      e_M -> M_obs
      e_D -> D_obs
    }"
  )

coordinates(marriage) <-
  list(x = c(A = 1, M = 2, M_obs = 3, e_M = 4, D = 2, D_obs = 3, e_D = 4),
       y = c(A = 1, M = 0, M_obs = 0, e_M = 0, D = 2, D_obs = 2, e_D = 2))

drawdag(marriage)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

-   This might occur if, say, marriages are rare. Then there aren’t as
    many couples that could possibly get divorced, so the smaller sample
    size induces a larger error on the measurement of the divorce rate.
-   If we can average over the uncertainty in the true $M$ and $D$ using
    information about the measurement process, we might do alright.
-   Another worry is when a causal variable is measured less precisely
    than a non-causal one. \*Let’s say for example that we know $D$ and
    $M$ very precisely but now $A$ is measured with error. Let’s also
    assume $M$ has zero causal impact on $D$:

``` r
marriage <-
  dagitty(
    "dag{
      e_A [unobserved]
      A [unobserved]
      e_A -> A_obs
      A -> A_obs
      A -> M
      A -> D
    }"
  )

coordinates(marriage) <-
  list(x = c(e_A = 1, A_obs = 2, A = 3, M = 4, D = 4),
       y = c(e_A = 1, A_obs = 1, A = 1, M = 0, D = 2))

drawdag(marriage)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

-   If we plop a regression of $D$ on $A_{obs}$ and $M$ it’ll suggest
    that $M$ *strongly* influences $D$. This is because $M$ contains
    proxy information about the true $A$, but measured much more
    precisely than $A_{obs}$.
-   Here’s a simulation to show the example:

``` r
N <- 500
A <- rnorm(N)
M <- rnorm(N, -A)
D <- rnorm(N, A)
A_obs <- rnorm(N, A)

plot(M, D)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

## 15.2 Missing Data

-   The insight from measurement error is to realize that any uncertain
    piece of data can be replaced with a distribution that reflects that
    uncertainty, but what about when data is simply missing?
-   So far, we’ve just been doing *complete case analysis*.
-   Another common response to missing data is to replace with an
    assumed value — like the mean, median, or some assumed value like 0.
-   Neither is truly safe — complete case throws away data & fixed
    imputation means the model will think it knows an unknown value with
    absolute certainty.
-   If we think causally about missingness, we may be able to use the
    model to *impute* missing values.
-   Some rethinking: missing data are still meaningful data. The fact
    that a variable has an unobserved value is still an observation.
    Thinking about the process that caused missingness can help solve
    big problems.

### 15.2.1 DAG ate my homework

-   Let’s consider a sample of students who own dogs. The students
    homework $H$ is influenced by how much each student studies $S$.

``` r
N <- 100
S <- rnorm(N)

# grade students on a 10-point scale, influenced by S
H <- rbinom(N, size = 10, inv_logit(S))
```

-   Then, oh-no! some dogs eat some homework. We’ll encode the
    missingness as a 0/1 indicator $D$.
-   When homework has been eaten, we cannot observe the true
    distribution of homework, but we do get to observe the incomplete
    case $H^*$.
-   In DAG form, $H \rightarrow H^* \leftarrow D$.
-   If we want to learn the influence of $S$ on $H$ we have to rely on
    $H^*$ — we’re relying on $S \rightarrow H^*$ being a good
    approximation for $S \rightarrow H$.
-   How good this assumption is depends on the cause of the missing
    values — let’s consider 4 scenarios considered as DAGs.

``` r
# Case 1: missing completely at random (simplest case)
case_1 <-
  dagitty(
    "dag{
      H [unobserved]
      S -> H
      D -> Hm
      H -> Hm
    }"
  )

coordinates(case_1) <-
  list(x = c(S = 1, D = 1, H = 2, Hm = 2),
       y = c(S = 1, D = 2, H = 1, Hm = 2))

drawdag(case_1)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
# simulate case 1
D <- rbern(N)
Hm <- H
Hm[D == 1] <- NA

Hm
```

    ##   [1] NA  1 NA NA NA  5  6  6  9  4  9  7 NA NA  6 NA NA NA  4 NA NA  4  3  5 NA
    ##  [26]  6  3  6  6  4  2 NA NA  4 NA NA  5  2  3 NA NA  8 NA  1  6 NA  6 NA  2  6
    ##  [51]  1  9 NA NA NA  0  6  7 NA  7  9  7  7 NA  1  6 NA  5 NA  8  4  6 NA NA NA
    ##  [76] NA NA  7 NA  2 NA NA NA NA  3 NA  5 NA NA NA  4 NA NA NA  3 NA NA NA NA NA

-   We now have `NA` scattered about the dataset — is this a problem? We
    can decide by considering whether the outcome $H$ is independent of
    $D$. In this case, $H$ is independent of $D$ because $H^*$ is a
    collider.
-   Another way of thinking about it — random missingness doesn’t change
    the overall distribution of homework scores & therefore doesn’t bias
    our estimate on the causal effect of studying.

``` r
# Case 2: Studying influences whether or not dog eats homework
# (maybe studying a lot means less playtime with the dog, who gets restless)
case_2 <-
  dagitty(
    "dag{
      H [unobserved]
      S -> H
      S -> D
      D -> Hm
      H -> Hm
    }"
  )

coordinates(case_2) <-
  coordinates(case_1)

drawdag(case_2)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
D <- ifelse(S > 0, 1, 0)
Hm <- H
Hm[D == 1] <- NA

Hm
```

    ##   [1]  0  1 NA  2  4 NA NA NA NA NA NA NA NA  2 NA NA  0 NA  4 NA NA  4  3 NA NA
    ##  [26] NA  3 NA  6  4  2 NA NA  4 NA NA NA  2  3  2 NA NA NA  1 NA NA NA NA  2 NA
    ##  [51]  1 NA  0 NA  5  0 NA NA NA NA NA NA NA NA  1 NA  3 NA NA NA NA NA NA NA NA
    ##  [76]  4 NA  7  1  2 NA  1 NA NA  3 NA NA NA  3  3 NA NA NA  5  3 NA  2  6  5 NA

-   In this second case the missingness is *not* random — every student
    who studies more than average is missing homework!
-   There is now also a non-causal backdoor path from
    $H \rightarrow H^* \leftarrow D \leftarrow S$. We close this path by
    conditioning on $S$ (which we wanted to do anyway).

``` r
# Case 3: influence on both H and D
# say X is noise in the student's house 
# noisier houses produce worse homework and dogs more likely to misbehave
case_3 <-
  dagitty(
    "dag{
      H [unobserved]
      X [unobserved]
      S -> H
      X -> H
      X -> D
      D -> Hm
      H -> Hm
    }"
  )

coordinates(case_3) <-
  list(x = c(S = 1, D = 1, X = 2, H = 3, Hm = 3),
       y = c(S = 1, D = 3, X = 2, H = 1, Hm = 3))

drawdag(case_3)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
set.seed(501)
N <- 1000
X <- rnorm(N)
S <- rnorm(N)
H <- rbinom(N, size = 10, inv_logit(2 + S - 2*X))
D <- ifelse(X > 1, 1, 0)
Hm <- H
Hm[D == 1] <- NA
Hm
```

    ##    [1] 10  7  9  8 10 10 10  8  7 NA  7  5  9  5  8  8  9 NA 10  5 NA 10  8  4
    ##   [25] 10  9 NA  9 10  6  9  9  2  7 NA  9  7  6  9  8 10  8 NA  7  8 NA NA  8
    ##   [49] 10 NA NA 10 10 NA 10 10  5 10 NA 10 10 NA  5  6 NA 10  5  5 NA NA  7  8
    ##   [73] 10 10  7 NA 10 10 10  8  9  5  7  6  9 NA  8 NA  7  7  7  8  5 10 10  9
    ##   [97]  5 10  9  7 10  9  8  7 10 NA 10  1 10  8 10 10 10  8 NA NA  9  8 10  8
    ##  [121]  7 10 10 NA NA 10  7  9 10  9  9  5 NA 10  6  8  9 10  9  7  9  4  9 10
    ##  [145]  5 NA 10 10  5 10 10  5 NA NA 10  4  3  7  3  4  2 NA NA 10 10  6 NA  7
    ##  [169]  5 10  9 10 10 NA NA NA 10 10 NA  5 10  9  8  9 NA NA  9  9 10  9  9 10
    ##  [193]  9 10  9 NA 10  9  8  3 10 10  7 10  6 NA 10  8  8 10 NA NA NA  8 NA  9
    ##  [217] 10  8  5  6 10 NA NA  7  2 10 10 10 10  4 10  4 10  8  5 10 10  9  8  8
    ##  [241]  9  9  9 10  6 NA  9 10  9 10 10  9  9  9 10  9  8  5  7 NA  6  8  8  9
    ##  [265]  9 10  9 10  7  7  8  6  4 NA 10 10 NA NA  9 10 NA  7 10 NA 10 NA  8  9
    ##  [289] 10  9 NA  4 10  4  9  7  6  5 10  5  7 NA  5 10  9  9 NA 10 10  4  9 10
    ##  [313] 10 10 10 10 NA  6 NA  4 10 10  7  5 NA 10 10  9 NA 10  8 10  5 10 10 10
    ##  [337] 10 NA  6 10 NA 10 10  8  0 10 10  9  9  6  8 10 10  9 NA  8 10  7 10  7
    ##  [361] 10 NA 10  7 10 NA  7 10 NA  9  7 NA 10  4  6  8 10 NA 10  3  9 10  9 10
    ##  [385] 10  8 10  9  8 10 10 NA 10 10  9  9 10 10  8 NA 10  9 NA  7  7 NA NA  3
    ##  [409] 10 10  7 10  5 NA  5 NA NA NA 10 10 10  9  4 NA 10 10 10 NA  9  9 10 NA
    ##  [433] NA NA  9 10  9  6  7  9  2  8 10  7 NA 10 10 10  9 NA NA 10 10  8 10  8
    ##  [457] 10  6  8  8  2  9  6 10 10 10 10  8 10 10 NA NA NA  8  8 NA  4  8 NA  8
    ##  [481]  9 NA NA  6  9  8  9  9  7 NA NA 10  7  4  7 10  3  4  9 10 NA 10  9  9
    ##  [505] 10 10 10 NA  9  7  9  8 NA  8 10 10  9 NA 10  9 NA  7  9  9 NA  9  9  9
    ##  [529] 10 10 NA  6  8  4  8 NA  9 10 10  5 10 NA 10  9  7  6  4 10 10  9  6  9
    ##  [553] NA  5 10  7  7  9 10 NA  9  8 NA 10  9 10  9  8 NA  5  8  9  9 10 10 10
    ##  [577]  9 10 10  7 NA 10  8 10  6  9 NA 10 NA  9 10 10 NA  9  2 10 NA NA 10  7
    ##  [601]  6  5  4 NA  7  8 10  0  9 NA  4  7  9  9  4 10 10 10 10  9 NA 10 10 NA
    ##  [625] NA 10  8 10  6 10  6 NA  9  8  9  8 NA  8  3 10 10  8 NA 10 10  9 10 NA
    ##  [649]  8 10  9  9 10  4 10  8  9 NA  8  8 10  5  9  9  8  9 10 10  8  5 10 NA
    ##  [673] 10  6  9 10 10 NA NA 10  7 NA 10  9 10 10 NA  9 10 NA NA 10  7 10  8  8
    ##  [697] 10  1  9  9 NA 10 10  9  8  8  7 10 10  5 10  7  8 NA 10 NA 10  4  9 NA
    ##  [721] NA  8 10  7  9  9 10  7  7 NA  9  9  9  9 10  8  5 10 NA  8 10 10 NA 10
    ##  [745] NA NA 10 NA 10 10 10 NA  9 10  9  9 10  8  3  9 NA 10  6 10 10  9  7  7
    ##  [769] 10 10  6  7 10  8  9 10 10 NA NA 10 NA NA  9  9  5  9 10 NA 10  9 10  9
    ##  [793]  9 10 10 10  9 NA  7 10  4 NA 10 NA  2  5 10 10  6 10  9 10 10 10 10  8
    ##  [817]  9  9 10 NA 10  9  9  6  9 10 10  9  7  3 NA NA  9  8  9 10 NA 10  9  8
    ##  [841] 10  9 NA NA 10 10 10  9 10 10 NA  7  9 10  9  8  9  6 10  8  4  9  7  9
    ##  [865] NA NA  9 10 10  6  9 10 NA  8 10  8  9  8 10  5 10 10  9  9  6  6  0  9
    ##  [889] 10  2  4 10  8  5 10  5 NA 10 NA  9  9 10 NA 10 10  9 10 NA  8  4 10 NA
    ##  [913]  5 10 10  9  7  9  9 10 10 10 10 10  8 NA  9  2 NA  8 10  8 10  9  7 10
    ##  [937]  9  9 10  8  3 NA  8 10  9 10 NA NA  6  8 NA 10  4  8 NA  6 10 10 10 NA
    ##  [961]  9 10  3 10 10  8  7 10 10  9 10  0  9 10 NA  8  9  7 NA  9 10 10 10  7
    ##  [985] 10 10  8 NA NA  9  6  0  7  8 NA  1  7  6  8 NA

-   Here, regressing $H^*$ on $S$ introduces a new non-causal path:
    $H^* \leftarrow D \leftarrow X \rightarrow H$.
-   Let’s first see what we gt if we fully observe $H$. We haven’t
    observed $X$ so we can’t put it into the model.

``` r
dat_list <-
  list(
    H = H,
    S = S
  )

m15.3 <-
  ulam(
    alist(
      H ~ binomial(10 ,p),
      logit(p) <- a + bS*S,
      a ~ normal(0, 1),
      bS ~ normal(0, 0.5)
    ),
    
    data = dat_list,
    chains = 4
  )
```

    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 0.000199 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 1.99 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.372 seconds (Warm-up)
    ## Chain 1:                0.375 seconds (Sampling)
    ## Chain 1:                0.747 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 0.000129 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 1.29 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 0.402 seconds (Warm-up)
    ## Chain 2:                0.356 seconds (Sampling)
    ## Chain 2:                0.758 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 0.000126 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 1.26 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 0.4 seconds (Warm-up)
    ## Chain 3:                0.357 seconds (Sampling)
    ## Chain 3:                0.757 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 0.00012 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 1.2 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 0.407 seconds (Warm-up)
    ## Chain 4:                0.379 seconds (Sampling)
    ## Chain 4:                0.786 seconds (Total)
    ## Chain 4:

``` r
precis(m15.3)
```

    ##         mean         sd      5.5%     94.5%     n_eff    Rhat4
    ## a  1.1138049 0.02579002 1.0732201 1.1554344  940.5712 1.001608
    ## bS 0.6895852 0.02603040 0.6484871 0.7301994 1035.3165 1.003775

-   The true coefficient on $S$ should be 1 — this estimate is way off!
    This used the complete data $H$ so it can’t be the missingness —
    this is a case of *omitted variable bias*.
-   What impact does the missing data have?

``` r
dat_list0 <-
  list(
    H = H[D == 0],
    S = S[D == 0]
  )

m15.4 <-
  ulam(
    alist(
      H ~ binomial(10, p),
      logit(p) <- a + bS*S,
      a ~ normal(0, 1),
      bS ~ normal(0, 0.5)
    ),
    
    data = dat_list0,
    chains = 4
  )
```

    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 0.000219 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 2.19 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.377 seconds (Warm-up)
    ## Chain 1:                0.359 seconds (Sampling)
    ## Chain 1:                0.736 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 0.00013 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 1.3 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 0.409 seconds (Warm-up)
    ## Chain 2:                0.345 seconds (Sampling)
    ## Chain 2:                0.754 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 0.000117 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 1.17 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 0.366 seconds (Warm-up)
    ## Chain 3:                0.336 seconds (Sampling)
    ## Chain 3:                0.702 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 0.000118 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 1.18 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 0.339 seconds (Warm-up)
    ## Chain 4:                0.358 seconds (Sampling)
    ## Chain 4:                0.697 seconds (Total)
    ## Chain 4:

``` r
precis(m15.4)
```

    ##         mean         sd      5.5%     94.5%     n_eff    Rhat4
    ## a  1.7959583 0.03507498 1.7393783 1.8524947 1034.9354 1.000515
    ## bS 0.8286333 0.03541430 0.7738497 0.8862532  993.2018 1.001287

-   The estimate here actually gets better (still not at 1)! But how?
-   Since the missingness is caused in part by noise, the dataset with
    removed houses actually removes some of the omitted variable bias.
-   This is not a general property of missing data in a DAG of this type
    — if the function for missingness is the following, the estimate
    gets worse:

``` r
D <- ifelse(abs(X) < 1, 1, 0)

dat_list0mod <- 
  list(
    H = H[D == 0],
    S = S[D == 0]
  )

m15.4mod <- 
  ulam(
    alist(
      H ~ binomial(10, p),
      logit(p) <- a + bS*S,
      a ~ normal(0, 1),
      bS ~ normal(0, 1)
    ),
    
    data = dat_list0mod,
    chains = 4
  )
```

    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 8.9e-05 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.89 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.162 seconds (Warm-up)
    ## Chain 1:                0.143 seconds (Sampling)
    ## Chain 1:                0.305 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 6.7e-05 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.67 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 0.172 seconds (Warm-up)
    ## Chain 2:                0.152 seconds (Sampling)
    ## Chain 2:                0.324 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 4.8e-05 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.48 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 0.144 seconds (Warm-up)
    ## Chain 3:                0.154 seconds (Sampling)
    ## Chain 3:                0.298 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 4.4e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.44 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:   1 / 1000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 100 / 1000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 200 / 1000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 300 / 1000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 400 / 1000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 500 / 1000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 501 / 1000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 600 / 1000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 700 / 1000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 800 / 1000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 900 / 1000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 1000 / 1000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 0.161 seconds (Warm-up)
    ## Chain 4:                0.119 seconds (Sampling)
    ## Chain 4:                0.28 seconds (Total)
    ## Chain 4:

``` r
precis(m15.4mod)
```

    ##         mean         sd      5.5%     94.5%    n_eff    Rhat4
    ## a  0.3424152 0.03718837 0.2827800 0.4028754 1270.450 1.001241
    ## bS 0.4936650 0.04059251 0.4292724 0.5580524 1349.162 1.001434

``` r
# Case 4: Much ado about nothing
# here, let's say the quality of homework influences whether or not a dog eats it
# (just roll with it)
case_4 <-
  dagitty(
    "dag{
      H [unobserved]
      S -> H
      H -> D
      D -> Hm
      H -> Hm
    }"
  )

coordinates(case_4) <-
  coordinates(case_1)

drawdag(case_4)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
N <- 100
S <- rnorm(N)
H <- rbinom(N, size = 10, inv_logit(S))
D <- ifelse(H < 5, 1, 0)
Hm <- H
Hm[D == 1] <- NA
Hm
```

    ##   [1] NA NA  5 NA  7  6 10  7  9 NA  9  8  6  8  5  9  7 NA NA NA  8 NA  8  9 NA
    ##  [26] NA NA NA NA  6  7  9  6 NA NA NA NA  7  5 NA NA NA  5  5 NA  5 NA  7  5  6
    ##  [51]  7  6  7  7 NA  7 NA  6  8  6 10  9 NA NA  7  6 NA NA  5  6  6 NA  5  5  5
    ##  [76] NA NA  8  6  6 NA  6  7  6  9  6 NA  5 NA  7  5 NA  5 NA  6 NA NA  7 NA  6

-   Here, there’s not much at all we can do! There’s nothing we can
    condition on to block the non-causal path
    $S \rightarrow H \rightarrow D \rightarrow H^*$.
-   The point here is to illustrate the diverse consequences of missing
    data and the importance of exploring our own scenarios.
-   Even when we cannot completely eliminate the impact of missing data,
    we may be able to show through simulation that the impact is small.

### 15.2.2 Imputing primates

-   *Imputation* allows us to (hopefully) avoid biased information and
    also use all the observed data.
-   In any generative model, information about variables is explained by
    the model regardless of whether or not the data is observed.
-   Let’s return to the primate milk example from chapter 5, where we
    used a *complete case* analysis.
-   Let’s say $M$ is body mass, $B$ is neocortex percent, $K$ is milk
    energy, and $U$ is an unobserved variable:

``` r
milk_basic <-
  dagitty(
    "dag{
      U [unobserved]
      U -> M
      U -> B
      M -> K
      B -> K
    }"
  )

coordinates(milk_basic) <-
  list(x = c(M = 1, U = 2, K = 2, B = 3),
       y = c(M = 1, U = 1, K = 2, B = 1))

drawdag(milk_basic)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

-   Instead of having all these values directly, we have the observed
    $B^*$ that includes some missing values.
-   We don’t know what process causes the missingness, so let’s consider
    some different DAGs that model the process of $B^*$ based on $R_B$ —
    a variable that indicates the species has missing values.
-   First — let’s consider the hypothesis that $R_B$ is missing at
    random — there are no variables that influence it.

``` r
milk_1 <- 
  dagitty(
    "dag{
      U [unobserved]
      B [unobserved]
      U -> M
      U -> B
      M -> K
      B -> K
      B -> Bm
      R_B -> Bm
    }"
  )

coordinates(milk_1) <-
  list(x = c(M = 1, R_B = 2, U = 2, K = 2, Bm = 3, B = 3),
       y = c(M = 1, R_B = 0, U = 1, K = 2, Bm = 0, B = 1))

drawdag(milk_1)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

-   Next, the body mass influences which species have missing values.
    This could happen if smaller primates are less often studied than
    larger ones.
-   This introduces a non-causal path
    $B^* \leftarrow R_B \leftarrow M \rightarrow K$, but conditioning on
    $M$ blocks this path.

``` r
milk_2 <- 
  dagitty(
    "dag{
      U [unobserved]
      B [unobserved]
      U -> M
      U -> B
      M -> R_B
      M -> K
      B -> K
      B -> Bm
      R_B -> Bm
    }"
  )

coordinates(milk_2) <-
  coordinates(milk_1)

drawdag(milk_2)
```

![](chapter_15_notes_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

-   Finally, let’s consider that the brain size itself influences $R_B$.
    This could be because anthropologists are more interested in
    large-brained species.
-   If this is true, we won’t be able to test — the influence of
    $B \rightarrow K$ will be biased via a non-causal path through
    $R_B$.
-   The statistical trick with Bayesian imputation is to model the
    variable with missing values. Each missing value receives a unique
    parameter — the observed data gives us a prior.
-   Here, for example, we might have:

$$
\begin{gather}
B = [0.55, \ B_2, \ B_3, \ B_4, \ 0.65, \ 0.65, \ \dots, \ 0.76, \ 0.75]
\end{gather}
$$

-   The simplest model for missing $B$ values will just draw from its
    own normal distribution:

$$
\begin{align*}
K_i & \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i & = \alpha + \beta_B B_i + \beta_M \ \text{log} \ M_i \\
\color{orange}{B_i} & \color{orange}{\sim} \color{orange}{\text{Normal}(\nu, \sigma_B)} \\
\alpha & \sim \text{Normal}(0, 0.5) \\
\beta_B & \sim \text{Normal}(0, 0.5) \\
\beta_M & \sim \text{Normal}(0, 0.5) \\
\sigma & \sim \text{Exponential}(1) \\
\nu & \sim \text{Normal}(0.5, 1) \\
\sigma_B & \sim \text{Exponential}(1)
\end{align*}
$$

-   This model ignores that $B$ and $M$ are associated through $U$. But
    let’s start with this just to keep things simple.
-   The interpretation of $B_i \sim \text{Normal}(\nu, \sigma_B)$ is a
    bit weird.
-   When $B_i$ is observed, this is a likelihood. When it’s missing,
    it’s a prior.
-   In this case, $B$ is bound by 0 and 1, so a normal distribution
    might not be the best choice, but let’s just roll with it.
-   All implementations of imputation are a bit awkward — since the
    locations of missing values have to be respected, it comes down to a
    lot of index management. `ulam()` handles a lot of this for us.

``` r
# load & prep data for stan
data(milk)
d <- milk
d$neocortex.prop <- d$neocortex.perc/100
d$logmass <- log(d$mass)

dat_list <-
  list(
    K = standardize(d$kcal.per.g),
    B = standardize(d$neocortex.prop),
    M = standardize(d$logmass)
  )

# model!
m15.5 <-
  ulam(
    alist(
      K ~ dnorm(mu, sigma),
      mu <- a + bB*B + bM*M,
      B ~ dnorm(nu, sigma_B),
      c(a, nu) ~ dnorm(0, 0.5),
      c(bB, bM) ~ dnorm(0, 0.5),
      sigma_B ~ dexp(1),
      sigma ~ dexp(1)
    ),
    
    data = dat_list,
    chains = 4,
    cores = 4
  )
```

    ## Found 12 NA values in B and attempting imputation.

``` r
precis(m15.5, depth = 2)
```

    ##                     mean        sd        5.5%      94.5%     n_eff     Rhat4
    ## nu           -0.03375601 0.2123511 -0.36414742  0.3102316 1836.5557 1.0003222
    ## a             0.02881734 0.1643644 -0.23457882  0.2870986 2724.1883 0.9999591
    ## bM           -0.53101002 0.2034316 -0.84807862 -0.2009409 1106.7066 1.0027951
    ## bB            0.48092165 0.2435120  0.08301144  0.8436267  863.0378 1.0026589
    ## sigma_B       1.00985221 0.1683427  0.77322971  1.2878488 1355.4752 1.0002386
    ## sigma         0.85299643 0.1429275  0.64632300  1.0973853 1289.2257 1.0010985
    ## B_impute[1]  -0.52232690 0.9519529 -1.94463981  1.0292449 1757.7138 1.0002946
    ## B_impute[2]  -0.66078752 0.9757835 -2.20175625  0.8989894 1435.5809 0.9992429
    ## B_impute[3]  -0.67420799 0.9949769 -2.29356907  0.8700286 1985.0711 1.0015302
    ## B_impute[4]  -0.22946653 0.9238075 -1.69184938  1.2135133 2613.8307 0.9988613
    ## B_impute[5]   0.45625419 0.8985267 -0.98630192  1.8528731 2578.7824 1.0001380
    ## B_impute[6]  -0.13826425 0.9498108 -1.58157366  1.3860490 2350.3319 1.0007201
    ## B_impute[7]   0.20228855 0.9166428 -1.25091582  1.6700400 2901.8077 0.9988039
    ## B_impute[8]   0.25860181 0.9270789 -1.23833620  1.7053715 2405.0797 1.0002929
    ## B_impute[9]   0.51523224 0.9284591 -0.95702804  1.9504728 2322.2177 1.0006388
    ## B_impute[10] -0.38607625 0.8941252 -1.76647810  0.9854197 1673.9399 0.9991061
    ## B_impute[11] -0.27799321 0.9022535 -1.68435290  1.1306616 2739.3411 0.9988508
    ## B_impute[12]  0.12849691 0.9260181 -1.41589641  1.5847263 2712.2609 0.9998876
