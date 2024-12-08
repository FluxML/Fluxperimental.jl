"""
    Reactor(model)

Container for use with Reactant.jl. Stores the model alongside a compiled forward pass,
space for the gradient, and a compiled `Enzyme.autodiff` call.
These compiled functions are created and stored on first use.

Note that unlike `Duplicated(model)`, what is stored is a copy of the model.
All arrays are copied to Reactant's `ConcreteRArray` type.
We should make `|> cpu` copy back to ordinary `Array`s, but that doesn't work yet.
"""
mutable struct Reactor{M}
  model::M
  fwd_compiled
  fwd_input
  fwd_count::Int
  gradient::M
  grad_compiled
  grad_input
  grad_count::Int
end

Flux.@layer :expand Reactor

function Reactor(args...)
  if length(args)==1
    error("The method `Reactor(x)` is only available when Reactant.jl is loaded!")
  else
    error("The only legal method is `Reactor(x)`.")
  end
end

Optimisers.trainable(m::Reactor) = (; m.model)

(m::Reactor)(x...) = m.model(x...)
