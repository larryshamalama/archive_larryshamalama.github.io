---
layout: page
title: Work Product Submission Guidelines
date: 2021-08-23 12:00:00-0400
description: The End of GSoC 2021
---

This post effectively marks the end of my GSoC experience, but just the beginning of my project. As per the Work Submission Guidelines, detailed here are my contributions. The goal of my project is to build a submodule on Dirichlet Processes, a Bayesian nonparametric method for the estimation of probability distributions. I have worked on many aspects that would benefit the submodule, but no pull requests will be merged into the main branch by the end of the summer.

### 1 - Testing

We spent a good portion of the summer discussing about testing for two reasons: (1) tests can serve as checks for my understanding of statistical theory and what I want my code to accomplish and (2) visual tests will ultimately turn into unit tests, which will be integrated in the CD/CI pipeline.

##### 1.A - Generating the right data

Notebook: https://github.com/larryshamalama/pymc3-playground/blob/master/notebooks/step-by-step/test-multiple-dp-samples.ipynb.

Dirichlet processes are specified by two things: a concentration parameter M and a base distribution G0. The first step was to design a visual test that can retrieve the concentration parameter. For instance, if we defined G0 to be $\mathcal{N}(5, 3^2)$, we can attempt to retrieve the mean of mu = 5 via the following test:

```python
mu = 5

rng = np.random.RandomState(seed=34)
x = rng.normal(loc=2., scale=2., size=[1000,])

with pm.Model() as model:
    µ = pm.Normal("µ", mu=0., sigma=5.)
    G0 = pm.Normal("G0", mu=µ, sigma=2., observed=x)

    trace = pm.sample(draws=1000, chains=1)

_ = pm.plot_trace(trace)
```

and, lastly, `trace.to_dict()["posterior"]["µ"].mean()` should evaluate to be close to 5. Inference on stick-breaking weights can be a tad more challenging, but it is still possible. Running the reverse stick-breaking function on weights that sum to 1, we can retrieve the concentration parameter as followed:

```python
rng = np.random.RandomState(seed=34)
betas = rng.beta(1., 5., size=[1000, 20])
weights = stick_breaking(betas)

recovered_betas = stick_glueing(weights)

with pm.Model() as model:
    α = pm.Uniform("α", 0., 10.)
    β = pm.Beta("β", 1., α, shape=(20,), observed=recovered_betas)
```

The notebook linked above combines both visual tests mentioned here. While it may seem redundant to obtain weights from betas and retrieve the latter back using stick_glueing, the two-level sampling below uses weights estimated using empirical frequencies of observed atoms. A preliminary first step was to get the tests here correct (which took me a while…).

However, when generating weights, the data-generating code above is prone to fail under small values of $$M$$. This is normal because the stick proportions that we break off are "bigger" and hence the remainder can easily fall below $$10^{-16}$$, which is were precision issues occur. See [this notebook](https://github.com/larryshamalama/pymc3-playground/blob/master/notebooks/shortcomings/replicate-precision-error.ipynb) in which I replicate this issue.

##### 1.B - Two-level sampling

Large K, small N_dp
No ordering
Larger values of K and smaller values

### 2 - Creating a StickBreakingWeights RandomVariable

### 3 - The DirichletProcess class: an ongoing effort…

See work-in-progress PR here.

### Final Comments

I am extremely grateful for this enriching opportunity and the support from my mentors Austin and Chris. Being a graduate student who’s primary goal is to understand statistical theory and do research, this experience has allowed me to strengthen my programming skills, but especially discover the plethora of opportunities in open source development. While my progress this summer was slow, I am confident that I will be able to get this submodule to work in the coming months.
