---
layout: page
title: GSoC 2022 Work Product Submission
date: 2022-09-10 12:00:00-0400
description: The end of another summer of code
---

As part of my final work product submission, I summarize here what I have worked on and the progress on certain tasks: done, ongoing and for the future.

### Bug Fixes

- Bug fix in Graphviz submodule (PyMC [PR 6011](https://github.com/pymc-devs/pymc/pull/6011))
- Fix `pm.Interpolated` moment (PyMC [PR 5986](https://github.com/pymc-devs/pymc/pull/5986))

I started an attempt to refactor the Latex representation for SymbolicDistributions (PyMC [PR 5793](https://github.com/pymc-devs/pymc/pull/5793)) and incorporating AePPL's `Cumsum` dispatch for the `GaussianRandomWalk` distribution (PyMC [PR 5814](https://github.com/pymc-devs/pymc/pull/5814)), but they were superceeded by an amazing PR that completely refactored [`SymbolicDistribution`s](https://github.com/pymc-devs/pymc/pull/6072).

### Graph rewrite for `Switch` mixture sub-graph [DONE-ish]

Here is a working example:

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

### Custom Python metaclass for dynamic type creation of unmeasurable `Op`s [DONE]

See [AePPL PR](https://github.com/aesara-devs/aeppl/pull/158).

### Graph rewrite for `IfElse` mixture sub-graphs [IN PROGRESS]

See [PR 169](https://github.com/aesara-devs/aeppl/pull/169) of AePPL.

Akin to the `Switch` working example, a similar log-probability graph can be retrieved.

### Dirichlet Process Mixtures in PyMC Experimental [IN PROGRESS]

See [WIP PR 66](https://github.com/pymc-devs/pymc-experimental/pull/66).

### Future work

- Implement multivariate mixture models, particularly for mixture sub-graphs constructed via the `MakeVector` or `Join` `Op` (AePPL [issue 106](https://github.com/aesara-devs/aeppl/issues/106))
- Check how `SymbolicDistribution`s show up in Graphviz. With the newly refactored `SymbolicRV`, chances are that they show up well, but it would be worth checking the `model_graph.py` file for any inconsistencies (r.f. PyMC issues [5303](https://github.com/pymc-devs/pymc/issues/5303) and [5766](https://github.com/pymc-devs/pymc/issues/5766)).
- Allow exclusion of model sub-graphs via "~" in front of variable names (PyMC [issue 5794](https://github.com/pymc-devs/pymc/issues/5794)).
