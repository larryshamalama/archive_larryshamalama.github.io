---
layout: page
title: Community Bonding Week 2
date: 2021-05-24 12:00:00-0400
description: 📚 Diving into the math...
---

January 2022 edit: Some parts have been edited following a change in the layout of the website.

My second week was spent mostly diving into textbooks to better understand what Dirichlet processes entail. To do so, I started with a brief revision of probability theory and reviewing Gaussian processes. Why? Given that I am should be studying for my comprehensive exams, I thought that I better understand what it means to posit priors on the space of probability measures. As recommended by my mentors, a good starting point would be to look at Gaussian processes (GP) since I found Dirichlet processes very difficult to wrap my head around. GPs are convenient priors for continuous functions and they seem to be slightly more easy to understand.

As an inspiration for the starting point of my learning, here one of my favorite Calvin and Hobbes strip.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
        {% responsive_image path: assets/img/calvin-hobbes.jpeg class: "img mx-auto d-block" style: "text-align: center"%}
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
    </div>
</div>
<div class="caption">
    A strip from my favorite comic, Calvin and Hobbes
</div>

As a side note, I am part of the organizing committee for the 2021 Canadian Statistics Student Conference which will be happening this Saturday, June 5. Quebec also just announced that the administration of second vaccination doses have been sped up. All lot is going on, and it’s all exciting!

### A bit about Bayesian nonparametrics...

When performing inference, we often posit parametric assumptions on the data-generating mechanism. However, this can be restrictive especially when we don’t know the functional form of such underlying mechanism. In these situations, we can turn to nonparametric methods which, as the name suggests, do not posit distributional assumptions and hence protects against model misspecification. In Bayesian nonparametrics (BNP), we can put priors on functions or probability distribution themselves. As Peter Müller once said when talking about ANOVA DDPs, a type of Dirichlet process: “A BNP is always right in the sense, no matter the true distribution, our prior always puts some probability mass in some neighborhood of the truth so we can learn about it. It has full support.” (see video here)

Here are some references that I used during the week:

- Bayesian Nonparametrics by N. Hjort et al. (2010);
- Bayesian Nonparametric Data Analysis by P. Müller et al. (2015);
- Machine Learning: a Probabilistic Perspective by K. Murphy (2012);
the probability theory textbooks mentioned above.
