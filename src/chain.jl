
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
    new_layers[idx], x = _apply(f, x)
  end
  new_layers, x
end

# Reverse rule for _scan
# example pulled from https://github.com/mcabbott/Flux.jl/blob/chain_rrule/src/cuda/cuda.jl
function ChainRulesCore.rrule(cfg::ChainRulesCore.RuleConfig, ::typeof(_scan), layers, x)
  duo = accumulate(layers; init=((nothing, x), nothing)) do ((pl,  input), _), cur_layer
    out, back = ChainRulesCore.rrule_via_ad(cfg, _apply, cur_layer, input)
  end
  outs = map(first, duo)
  backs = map(last, duo)
  
  function _scan_pullback(dy)
    multi = accumulate(reverse(backs); init=(nothing, dy)) do (_, delta), back
      dapply, dlayer, din = back(delta)
      return dapply, (dlayer, din)
    end
    layergrads = reverse(map(first, multi))
    xgrad = last(multi[end])
    return (ChainRulesCore.NoTangent(), layergrads, xgrad)
  end
  return (map(first, outs), last(outs[end])), _scan_pullback
end

function _apply(layers::AbstractVector, x)  # type-unstable path, helps compile times
  _scan(layers, x)
end

# Generated function returns a tuple of args and the last output of the network.
@generated function _apply(layers::Tuple{Vararg{<:Any,N}}, x) where {N}
  x_symbols = vcat(:x, [gensym() for _ in 1:N])
  l_symbols = [gensym() for _ in 1:N]
  calls = [:(($(l_symbols[i]), $(x_symbols[i+1])) = _apply(layers[$i], $(x_symbols[i]))) for i in 1:N]
  push!(calls, :(return tuple($(l_symbols...)), $(x_symbols[end])))
  Expr(:block, calls...)
end

_apply(layer, x) = layer, layer(x)



"""
  NM_Recur

Non-mutating Recur. An experimental recur interface for the new chain api.
"""
struct NM_Recur{T,S}
  cell::T
  state::S
end

function _apply(m::NM_Recur, x)
  state, y = m.cell(m.state, x)
  return NM_Recur(m.cell, state), y
end

Flux.@functor NM_Recur
Flux.trainable(a::NM_Recur) = (; cell = a.cell)

Base.show(io::IO, m::NM_Recur) = print(io, "Recur(", m.cell, ")")

NM_RNN(a...; ka...) = NM_Recur(Flux.RNNCell(a...; ka...))
NM_Recur(m::Flux.RNNCell) = NM_Recur(m, m.state0)
