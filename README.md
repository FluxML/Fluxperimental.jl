<img align="right" width="200px" src="https://github.com/FluxML/Optimisers.jl/raw/master/docs/src/assets/logo.png">

# Fluxperimental.jl

[![][action-img]][action-url]
[![][coverage-img]][coverage-url]

[action-img]: https://github.com/FluxML/Fluxperimental.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Fluxperimental.jl/actions

[coverage-img]: https://codecov.io/gh/FluxML/Fluxperimental.jl/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/gh/FluxML/Fluxperimental.jl


This contains experimental features for [Flux.jl](https://github.com/FluxML/Flux.jl).
It needs to be loaded in addition to the main package:

```julia
using Flux, Fluxperimental
```

As an experiment, this repository only has discussion pages, not issues. Actual bugs reports are welcome,
as are comments that you think something is a great idea, or better ways achive the same goal,
or nice examples showing how it works.

Pull requests adding new features are also welcome. Ideally they should have at least some tests.
They should not alter existing functions (i.e. should not commit piracy)
to ensure that loading Fluxperimental won't affect other uses.
Prototypes for new versions of existing features should use a different name.

Features which break or are abandoned will be removed, in a minor (breaking) release.
As will any features which migrate to Flux itself.

## Current Features

There are no formal documentation pages, but these links to the source will show you docstrings
(which are also available at the REPL prompt).

* Layers [`Split` and `Join`](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/split_join.jl)
* More advanced [`train!` function](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/train.jl)
* *Two* macros for making custom layers quickly:
  [`@compact(kw...) do ...`](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/compact.jl), and
  [`@autostruct function Mine(d) ...`](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/autostruct.jl).
* Experimental [`apply(c::Chain, x)`](https://github.com/FluxML/Fluxperimental.jl/blob/master/src/chain.jl) interface
