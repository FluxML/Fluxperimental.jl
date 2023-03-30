<img align="right" width="200px" src="https://github.com/FluxML/Optimisers.jl/raw/master/docs/src/assets/logo.png">

# Fluxperimental.jl

[![][action-img]][action-url]
[![][coverage-img]][coverage-url] 

[action-img]: https://github.com/FluxML/Fluxperimental.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Fluxperimental.jl/actions

[coverage-img]: https://codecov.io/gh/FluxML/Fluxperimental.jl/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/gh/FluxML/Fluxperimental.jl


The repository contains experimental features for [Flux.jl](https://github.com/FluxML/Flux.jl).
It needs to be loaded in addition to the main package:

```julia
using Flux, Fluxperimental
```

As an experiment, it only has discussion pages, not issues. Actual bugs reports are welcome,
as are comments that you think something is a great idea, or better ways achive the same goal,
or nice examples showing how it works.

Pull requests adding new features are also welcome. Ideally they should have at least some tests.
They should not alter existing functions (i.e. should not commit piracy)
to ensure that loading Fluxperimental won't affect other uses.
Prototypes for new versions of existing features should use a different name.

Features which break or are abandoned will be removed, in a minor (breaking) release.
As will any features which migrate to Flux itself.

## Current Features

* Layers [`Split` and `Join`](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/split_join.jl)
* More advanced [`train!` function](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/train.jl)
* Macro for [making custom layers](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/compact.jl) quickly
* Experimental [`apply(c::Chain, x)`](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/chain.jl) interface
