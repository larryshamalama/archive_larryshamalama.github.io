---
layout: page
title: Merged my first distribution class 🎉
date: 2022-01-28 12:00:00-0400
description: The first step to Dirichlet processes
---

After a long (two month) process, my PR on adding a distribution class has been merged! 🙂 The whole process of taking the initiative to contribute to PyMC was rewarding, but definitely not easy. A small comment that, upon cleaning my computer, many of my 2021 GSoC posts have been deleted (I also happened to be in the process of reformatting my website), so this is why my 2021 summer blog may seem kind of empty...

### Revisiting Dirichlet Processes

A random distribution $$G$$ has a Dirichlet Process (DP) prior, denoted by $$G \sim \text{DP}(\alpha, G_0)$$ if any finite partition $$\{A_i\}_{i=1}^n$$ of the sample space $$\Theta$$ follows a Dirichlet distribution as such:

$$\Big(G(A_1), \dots, G(A_n) \Big) \sim \text{Dir}\Big(\alpha G_0(A_1), \dots, \alpha G_0(A_n) \Big)$$

where $$\alpha > 0$$ is the concentration parameter and $$G_0$$ is some base distribution, e.g. $$G_0 \equiv \mathcal{N}(0, 1)$$. The higher the value of $$\alpha$$, the smaller the weight values will be.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
        {% responsive_image path: assets/img/stick-breaking-weights.jpg class: "img-fluid rounded mx-auto d-block" %}
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
    </div>
</div>
<div class="caption">
    Here is a visual depiction of how the weights are obtained from a stick-breaking construction. At each iteration, a weight is taken from the remainder of a stick, initially to be one unit long. Figure taken from <a href="https://discovery.ucl.ac.uk/id/eprint/20467/1/20467.pdf">Sivakumar Murugiah's thesis</a>.
</div>

It can be initially difficult to understand why such a construction is useful, let alone nonparametric, but the idea is as followed. In Bayesian inference, we wish to perform inference by conditioning on observed data and looking at the posterior distribution of parameters of interest. However, by positing a DP prior on $$G$$, we can effectively perform inference on the distribution $$G$$ *without* positing any distributional assumption, hence rendering this construction nonparametric despite the need to specify $$G_0$$. It is already worth mentioning that a DP prior poses some "strong" restrictions, so a more common application of DPs are to posit them as priors in mixture modelling, but more on that at a later date...

When it comes to sampling, there are many schemas with desirable properties that provide nice (conditional)posterior distributions (see Chinese Restaurant Process and Polya Urn). However, an inherent challenge to build a DP functionality to PyMC is to leverage its default sampling methods which are primarily gradient-based, i.e. Hamiltonian Monte Carlo (HMC) or some of its extensions if I remember correctly. As such, the most useful construction of a DP is to represent it as an infinite linear combination of weights obtained via a stick-breaking process and atoms, represented as Dirac delta distributions:

$$G = \sum_{h=1}^\infty w_h \delta_{m_h}$$

where $$w_h = v_h \prod_{\ell < h} (1 - v_\ell)$$ where $$v_h \stackrel{\text{i.i.d.}}{\sim} \text{Beta}(1, \alpha)$$ and $$m_h \stackrel{\text{i.i.d.}}{\sim} G_0$$. This construction is particularly helpful since it is more easily vectorizable and plays into the strengths of HMC. Of course, as taught in undergraduate calculus, infinity is a concept, not a number, and our computers couldn't agree more. Instead of an infinite sum, we can express $$G$$ using some large truncation parameter $$K$$ and this finite approximation is justified by [Ishwaran and James (2001)](http://people.ee.duke.edu/~lcarin/Yuting3.3.06.pdf):

$$G = \sum_{h=1}^K w_h \delta_{m_h} \, .$$

From the same article, we have that the distribution of stick-breaking weights follows a generalized Dirichlet distribution, so that turns out to be the first step in implementing a submodule for DPs.

### `RandomVariable` classes in PyMC/Aesara

In PyMC, distributions are implemented as `Distribution` classes with `RandomVariable` instances `rv_op`. These `rv_op` have an `rng_fn` method that is able to generate samples from the prior distribution and it serves as the basis of `pm.sample_prior_predictive`.

```{python}
class StickBreakingWeightsRV(RandomVariable):
    name = "stick_breaking_weights"
    ndim_supp = 1
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("StickBreakingWeights", "\\operatorname{StickBreakingWeights}")

    def make_node(self, rng, size, dtype, alpha, K):

        alpha = at.as_tensor_variable(alpha)
        K = at.as_tensor_variable(intX(K))

        if alpha.ndim > 0:
            raise ValueError("The concentration parameter needs to be a scalar.")

        if K.ndim > 0:
            raise ValueError("K must be a scalar.")

        return super().make_node(rng, size, dtype, alpha, K)

    def _infer_shape(self, size, dist_params, param_shapes=None):
        alpha, K = dist_params

        size = tuple(size)

        return size + (K + 1,)

    @classmethod
    def rng_fn(cls, rng, alpha, K, size):
        if K < 0:
            raise ValueError("K needs to be positive.")

        if size is None:
            size = (K,)
        elif isinstance(size, int):
            size = (size,) + (K,)
        else:
            size = tuple(size) + (K,)

        betas = rng.beta(1, alpha, size=size)

        sticks = np.concatenate(
            (
                np.ones(shape=(size[:-1] + (1,))),
                np.cumprod(1 - betas[..., :-1], axis=-1),
            ),
            axis=-1,
        )

        weights = sticks * betas
        weights = np.concatenate(
            (weights, 1 - weights.sum(axis=-1)[..., np.newaxis]),
            axis=-1,
        )

        return weights
```

First, the distinction between `StickBreakingWeightsRV` and `StickBreakingWeights` is important as the latter is the one that will be used under a `pm.Model()` context manager. The `rng_fn` essentially follows the mathematical stick-breaking construction in that $$K$$ i.i.d. draws from a $$\text{Beta}(1, \alpha)$$ are used to construct $$w_1, \dots, w_{K}$$ with $$w_{K+1} = 1 - \sum_{\ell=1}^K w_\ell$$. An inherent challenge of this design is to abide by the existing OOP structure of the library since all the intricacies are not obvious. For instance, while `size` is used to specify the dimension of the observations that we want to sample (also in `StickBreakingWeights` below), we decided to provide the truncation parameter `K` as an explicit argument rather than include as part of `size` or `shape`.

### A distribution class for (truncated) stick-breaking weights

```{python}
class StickBreakingWeights(Continuous):
    # rv_op instance defined as: stickbreakingweights = StickBreakingWeights()
    rv_op = stickbreakingweights

    def __new__(cls, name, *args, **kwargs):
        kwargs.setdefault("transform", transforms.simplex)
        return super().__new__(cls, name, *args, **kwargs)

    @classmethod
    def dist(cls, alpha, K, *args, **kwargs):
        alpha = at.as_tensor_variable(floatX(alpha))
        K = at.as_tensor_variable(intX(K))

        assert_negative_support(alpha, "alpha", "StickBreakingWeights")
        assert_negative_support(K, "K", "StickBreakingWeights")

        return super().dist([alpha, K], **kwargs)

    def get_moment(rv, size, alpha, K):
        moment = (alpha / (1 + alpha)) ** at.arange(K)
        moment *= 1 / (1 + alpha)
        moment = at.concatenate([moment, [(alpha / (1 + alpha)) ** K]], axis=-1)
        if not rv_size_is_none(size):
            moment_size = at.concatenate(
                [
                    size,
                    [
                        K + 1,
                    ],
                ]
            )
            moment = at.full(moment_size, moment)

        return moment

    def logp(value, alpha, K):
        """
        Calculate log-probability of the distribution induced from the stick-breaking process
        at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        logp = -at.sum(
            at.log(
                at.cumsum(
                    value[..., ::-1],
                    axis=-1,
                )
            ),
            axis=-1,
        )
        logp += -K * betaln(1, alpha)
        logp += alpha * at.log(value[..., -1])

        logp = at.switch(
            at.or_(
                at.any(
                    at.and_(at.le(value, 0), at.ge(value, 1)),
                    axis=-1,
                ),
                at.or_(
                    at.bitwise_not(at.allclose(value.sum(-1), 1)),
                    at.neq(value.shape[-1], K + 1),
                ),
            ),
            -np.inf,
            logp,
        )

        return check_parameters(
            logp,
            alpha > 0,
            K > 0,
            msg="alpha > 0, K > 0",
        )
```

The length of this `Distribution` class may seem a bit daunting, but here is a summary of each method.

- `__new__`: Here, we suggest the default transformation that would help sampling. Because each weight is between 0 and 1, the range of the transformed "weight" is not $$\mathbb{R}$$ so the sampler cannot go into "danger" territories.
- `dist`: It is a bottleneck method which takes inputs that parametrize the distribution. At this point, inputs can be "anything", i.e. they have not necessarily been transformed into `aesara.tensor.var.TensorVariable` instances yet.
- `get_moment`: An approach that PyMC has recently taken is to initialize samplers at the mean of each distribution (with no observations). For that purpose, `get_moment` has been introduced and, given that stick-breaking weights are a product of linear transformations of i.i.d. Beta random variables, we have that:

$$\mathbb{E}\left[w_h\right] = \frac{1}{1 + \alpha}\left(\frac{\alpha}{1 + \alpha}\right)^{h - 1}$$

for all $$h = 1, \dots, K$$ and $$\mathbb{E}\left[w_{K+1}\right] = \left(\frac{\alpha}{1 + \alpha}\right)^{K}$$.

- `logp`: While it may seem naive, this is the most important method as this is exactly what allows `pm.sample()` to do its magic. Here, I provide the log distribution of a generalized Dirichlet distribution with $$b_h = 1$$ and $a_h = \alpha$ for all $$h$$ with respect to the density provided in the [Wiki article](https://en.wikipedia.org/wiki/Generalized_Dirichlet_distribution). Note that we assume that `value` can be of any dimension. Everything else in the method (what's in `at.switch` and `check_parameters`) are to ensure that inputs and parameters are all valid with respect to the distribution's constraints; errors would be raised if we provided something like `alpha = -2`, for instance.

A small comment that I added unit tests regarding this distribution: testing shapes, `logp`, `rng_fn` and their extensions to multidimensional samples. I do not talk about them here, but unit tests are quite important and, given that I didn't know what they were before, it was an interesting learning experience.

### Some final comments

As I am pursuing a PhD, this experience of contributing to open source has been extremely rewarding and educational. It was not an easy process, but it was enjoyable, as I was able to learn a lot about the process of "properly" contributing code (in quotation marks because I am no software engineer and who am I to say that this is the proper method although this is the by far most structure that I have ever experienced).

The next steps are to properly simulate data from a DP to better understand how we should go about building an API for it. The ultimate goal is to have a working submodule and a nice class for DP Mixtures, which will probably be the most useful extension of DPs in practice.

### Thanks

Firstly, a big thanks to you for reading this post! While this was a nice exercise for myself, it's nice to have people read on my work (and comment on it if you want!). I am proud to have coded everything, but none of this was possible without the help of these amazing people, who were always there to share their ideas and comments about my code:

- [Ricardo Vieira](https://github.com/ricardoV94/)
- [Austin Rochford](https://austinrochford.com/)
- [Chris Fonnesbeck](https://twitter.com/fonnesbeck)

Last but not least, a big thanks to the entire PyMC community for answering all my questions and all their ongoing efforts, whether they are noticeable or not 🚀
