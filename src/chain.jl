
import Flux: ChainRulesCore
# Some experiments with chain to start removing the need for recur to be mutable.
# As per the conversation in the recurrent network rework issue.

# Main difference between this and the _applychain function is we return a new chain
# with the internal state modified as well as the output of applying x to the chain.
function apply(chain::Flux.Chain, x)
  layers, out = _apply(chain.layers, x)
  Flux.Chain(layers), out
end

function _apply(layers::NamedTuple{NMS, TPS}, x) where {NMS, TPS}
  layers, out = _apply(Tuple(layers), x)
  NamedTuple{NMS}(layers), out
end

function _scan(layers::AbstractVector, x)
  new_layers = typeof(layers)(undef, length(layers))
  for (idx, f) in enumerate(layers)
    new_layers[idx], x = _apply_to_layer(f, x)
  end
  new_layers, x
end

function ChainRulesCore.rrule(::typeof(_scan), layers, x)
  function _scan_pullback(dy)
    error("_scan Pullback not implemented")
  end
  return _scan(layers, x), _scan_pullback
end

function _apply(layers::AbstractVector, x)  # type-unstable path, helps compile times
  _scan(layers, x)
end

# Generated function returns a tuple of args and the last output of the network.
@generated function _apply(layers::Tuple{Vararg{<:Any,N}}, x) where {N}
  x_symbols = vcat(:x, [gensym() for _ in 1:N])
  l_symbols = [gensym() for _ in 1:N]
  calls = [:(($(l_symbols[i]), $(x_symbols[i+1])) = _apply_to_layer(layers[$i], $(x_symbols[i]))) for i in 1:N]
  push!(calls, :(return tuple($(l_symbols...)), $(x_symbols[end])))
  Expr(:block, calls...)
end

_apply_to_layer(layer, x) = layer, layer(x)


