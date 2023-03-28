# This file implements the recurrent implementation details for the new chain interface.

"""
  NM_Recur

Non-mutating Recur. An experimental recur interface for the new chain api.
"""
struct NM_Recur{T,S}
  cell::T
  state::S
end

function _apply_to_layer(m::NM_Recur, x)
  state, y = m.cell(m.state, x)
  return NM_Recur(m.cell, state), y
end

# This is the same hacky way we do it from Flux.Recur
function _apply_to_layer(m::NM_Recur, x::AbstractArray{T, 3}) where T
  # h = [m(x_t) for x_t in eachlastdim(x)]
  l, h = _apply_to_layer(m, Flux.eachlastdim(x))
  sze = size(h[1])
  l, reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

Flux.@functor NM_Recur
Flux.trainable(a::NM_Recur) = (; cell = a.cell)

Base.show(io::IO, m::NM_Recur) = print(io, "Recur(", m.cell, ")")

NM_RNN(a...; ka...) = NM_Recur(Flux.RNNCell(a...; ka...))
NM_Recur(m::Flux.RNNCell) = NM_Recur(m, m.state0)

##
# Apply timeseries data in vectors to chains
##

"""
  _recur_scan

This method does a scan over the data. The rrule is note implemented yet, but should be similar to _scan above.
"""
function _recur_scan(layer, xs)
  ret = accumulate(xs; init=(layer, nothing)) do (l, y), x
    _apply_to_layer(l, x)
  end
  map(first, ret), map(last, ret)
end

# rrule for _dscan
function ChainRulesCore.rrule(cfg::ChainRulesCore.RuleConfig, ::typeof(_recur_scan), layer, xs)
  # Not implemented yet.

  duo = accumulate(xs; init=((layer, nothing), nothing)) do ((l, y), _), x
    out, back = ChainRulesCore.rrule_via_ad(cfg, _apply_to_layer, l, x)
  end
  outs = map(first, duo)
  backs = map(last, duo)

  function _recur_scan_pullback(dy)
    throw("_recur_scan_pullback Not Implemented Yet.")
  end
  return (map(first, outs), map(last, outs)), _recur_scan_pullback
end

# Specific details for working with time-series data.
function apply_timeseries_toplevel(chain::Flux.Chain, xs)
  _recur_scan(chain, xs)
end

# To apply the timeseries data at the layer levels, we can re-use the applies above.
# We only need to add a default method to _apply_to_layer, and then a method for
# stateful _apply_to_layer using the _dscan above.
function apply_timeseries_layerlevel(chain::Flux.Chain, xs)
  apply(chain, xs)
end

function _apply_to_layer(l, xs::Union{AbstractVector{<:AbstractArray}, Base.Generator})
  l, [l(x) for x in xs]
end

function _apply_to_layer(l::NM_Recur, xs::Union{AbstractVector{<:AbstractArray}, Base.Generator})
  ls, hs = _recur_scan(l, xs)
  ls[end], hs
end
