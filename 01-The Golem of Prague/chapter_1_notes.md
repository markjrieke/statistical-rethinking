The Golem of Prague
================

## 1.1 Statistical Golems

-   *Golem*: Hebrew myth of a creation with incredible strength that
    follows orders exactly. Poorly formed orders can lead to disaster!
-   Statistical tests can be thought of as “statistical golems” — when
    we use them without thinking/outside of the narrow context that they
    may be suitable for, we can end up with poor results.

## 1.2 Statistical Rethinking

-   Need to understand the nuts/bolts of the problem/mechanics of the
    statistical test or method to be able to accurately understand it’s
    output.

### 1.2.1 Hypotheses are not models

-   All models are wrong, but some are useful (classic)
-   Statistical models do not express causality, but instead express
    *associations among variables*

### 1.2.2 Measurement Matters

-   Example: The English thought that all swans were white (H0) until
    visiting Australia & seeing non-white swans
-   Before traveling to Australia, no number of observations would have
    disproven H0!

#### 1.2.2.1 Observation Error

-   Not every swan is black or white — what about intermediate shades?
-   The swan example is a bit too optimistic — finding disconfirmations
    is not always easy/viable!
-   Another example is faster-than-light neutrinos — researchers “found”
    faster-than-light neutrinos, but later found the technical error in
    measurement.

#### 1.2.2.2 Continuous Hypotheses

-   Most interesting hypotheses are continuous — e.g., H0: 80% of swans
    are white

### 1.2.3 Falsification is Consensual

-   Coming to a consensus about evidence is difficult and messy
-   Think of the change from geocentric to solar-centric models of the
    solar system.

## 1.3 Tools for Golem Engineering

-   Bayesian data analysis
-   Model comparison
-   Multilevel models
-   Graphical causal models

### 1.3.1 Bayesian Data Analysis

-   Basically, just counting the number of ways the data *could* happen,
    according to our assumptions
-   Even when Bayesian/frequentist procedures give the same answer,
    Bayesian procedures don’t justify inferences with “imagined repeat
    sampling.” Randomness is a property of information, not the world.
    For a coin flip, if we had perfect information, the flip wouldn’t
    appear random, but because we have limited information, it appears
    “random”
-   Many folks interpret non-Baysian results in Bayesian terms!
    (p-values!!!)

### 1.3.2 Model Comparison and Prediction

-   Can use *cross-validation* and *information criteria* to compare
    models by predictive accuracy
-   Want to avoid overfitting!

### 1.3.3 Multilevel Models

-   Partial pooling can actually help with overfitting!
-   Contexts for partial pooling:
    1.  Adjust estimates for repeat sampling
    2.  Adjust estimates for imbalance in sampling
    3.  Study variation
    4.  Avoid averaging!

### 1.3.4 Graphical Causal Models

-   Associations are not inherently causal and statistical models do not
    infer causality
-   Simplest graphical causal model is a Directed Acyclic Graph (DAG)

## 1.4 Summary

-   Don’t choose black-box tools, build up from axioms
-   Chapters 2-3: Foundational
-   Chapters 4-8: Bayesian flavor of linear regression
-   Chapters 9-12: MCMC & GLMs
-   Chapters 13-16: Multilevel models, missing data, measurement error,
    etc.
-   Chapter 17: Return to some issues raised in chapter 1
