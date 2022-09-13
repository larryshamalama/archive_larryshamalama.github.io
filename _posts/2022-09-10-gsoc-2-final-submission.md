---
layout: page
title: GSoC 2022 Work Product Submission
date: 2022-09-10 12:00:00-0400
description: The end of another summer of code
---

As part of my final work product submission, I summarize here what I have worked on and the progress on certain tasks: done, ongoing and for the future. While I have yet to address the issue of multivariate mixtures (AePPL [issue 106](https://github.com/aesara-devs/aeppl/issues/106)), I would say that I accomplished a very important goal of mine which was to learn more about Aesara and AePPL.

### **Graph rewrite for `Switch` mixture sub-graph [DONE-ish]**

A `Switch` takes three arguments: an index and two components. Akin to a `at.stack` that is being indexed, a `Switch` can be viewed as a mixture, where both components are measurable and we wish to obtain the log-probability of the component that is being indexed. Here is a working example:

```python
import aesara.tensor as at
from aeppl import joint_logprob

srng = at.random.RandomStream(seed=2320)

I_rv = srng.bernoulli(0.5, name="I")
X1_rv = srng.normal(loc=-5, scale=0.1, name="X1")
X2_rv = srng.normal(loc=5, scale=0.1, name="X2")

Z_rv = at.switch(I_rv, X1_rv, X2_rv)

z_vv = Z1_rv.clone()
i_vv = I_rv.clone()

logp = joint_logprob({Z_rv: z_vv, I_rv: i_vv})
logp.eval({z_vv: 5, i_vv: 0}), logp.eval({z_vv: 5, i_vv: 1})
# yields (array(-4999.30950062), array(0.69049938))
```

For more detailed information, I wrote a blogpost as part of my GSoC program [here](https://larrydong.com/gsoc/gsoc-2-part-2/).

### **Custom Python metaclass for dynamic type creation of unmeasurable `Op`s [DONE]**

See [AePPL PR](https://github.com/aesara-devs/aeppl/pull/158) and [this blogpost](https://larrydong.com/posts/2022-08-02-metaclass/) for more detail about this PR. Frankly, this was my contribution that I was most proud of. Copying a working example from the blogpost, below is an example of how dynamically created classes can be equal up to their hash value but are not inherently the same class object.

```python
import aesara.tensor as at

X_rv = at.random.normal(5., 3., name="X")
Y_rv = at.random.normal(-5., 3., name="Y")

hash(unmeasurable_X) == hash(unmeasurable_Y) # True: 4967640381975027986 == 4967640381975027986
id(unmeasurable_X) == id(unmeasurable_Y) # False: 6044493248 == 6044530000

unmeasurable_X = assign_custom_measurable_outputs(X_rv.owner).op
unmeasurable_Y = assign_custom_measurable_outputs(Y_rv.owner).op

hash(unmeasurable_X) == hash(unmeasurable_Y) # True
id(unmeasurable_X) == id(unmeasurable_Y) # False

unmeasurable_X == unmeasurable_Y # True, same hashes
unmeasurable_X is unmeasurable_Y # False, different ids
```

### **Graph rewrite for `IfElse` mixture sub-graphs [IN PROGRESS]**

See [PR 169](https://github.com/aesara-devs/aeppl/pull/169) of AePPL. Akin to the `Switch` working example, the `IfElse` `Op` takes the same three inputs: an binary condition and two components. A similar log-probability graph can be retrieved as with the `Switch` example above.

```python
import aesara.tensor as at
from aeppl import joint_logprob

srng = at.random.RandomStream(seed=2320)

I_rv = srng.bernoulli(0.5, name="I")
X1_rv = srng.normal(loc=-5, scale=0.1, name="X1")
X2_rv = srng.normal(loc=5, scale=0.1, name="X2")

Z_rv = at.ifelse.ifelse(I_rv, X1_rv, X2_rv)

z_vv = Z1_rv.clone()
i_vv = I_rv.clone()

logp = joint_logprob({Z_rv: z_vv, I_rv: i_vv})
logp.eval({z_vv: 5, i_vv: 0}), logp.eval({z_vv: 5, i_vv: 1})
# yields (array(-4999.30950062), array(0.69049938))
```

### **Dirichlet Process Mixtures in PyMC Experimental [IN PROGRESS]**

See [WIP PR 66](https://github.com/pymc-devs/pymc-experimental/pull/66).

### **Bug Fixes**

- Bug fix in Graphviz submodule (PyMC [PR 6011](https://github.com/pymc-devs/pymc/pull/6011))
- Fix `pm.Interpolated` moment (PyMC [PR 5986](https://github.com/pymc-devs/pymc/pull/5986))

I started an attempt to refactor the Latex representation for SymbolicDistributions (PyMC [PR 5793](https://github.com/pymc-devs/pymc/pull/5793)) and incorporating AePPL's `Cumsum` dispatch for the `GaussianRandomWalk` distribution (PyMC [PR 5814](https://github.com/pymc-devs/pymc/pull/5814)), but they were superceeded by an amazing PR that completely refactored [`SymbolicDistribution`s](https://github.com/pymc-devs/pymc/pull/6072).

### **Future work**

- Implement multivariate mixture models, particularly for mixture sub-graphs constructed via the `MakeVector` or `Join` `Op` as mentioned above.
- Check how `SymbolicDistribution`s show up in Graphviz. With the newly refactored `SymbolicRV`, chances are that they show up well, but it would be worth checking the `model_graph.py` file for any inconsistencies (r.f. PyMC issues [5303](https://github.com/pymc-devs/pymc/issues/5303) and [5766](https://github.com/pymc-devs/pymc/issues/5766)).
- Allow exclusion of model sub-graphs via "~" in front of variable names (PyMC [issue 5794](https://github.com/pymc-devs/pymc/issues/5794)).

### **Thanks**

Last but not least, I want to give my sincerest thanks to my mentors Brandon and Ricardo. They were patient and very helpful in guiding me through AePPL's sourcecode and the conceptual design of the codebase. However, I am most grateful for their mentorship. Thanks for everything.
