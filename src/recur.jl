# This file implements the recurrent implementation details for the new chain interface.

"""
  NM_Recur

Non-mutating Recur. An experimental recur interface for the new chain api.
"""
struct NM_Recur{RET_SEQUENCE, T, S}
  cell::T
  state::S
  function NM_Recur(cell, state; return_sequence::Bool=false)
    new{return_sequence, typeof(cell), typeof(state)}(cell, state)
  end
  function NM_Recur{true}(cell, state)
    new{true, typeof(cell), typeof(state)}(cell, state)
  end
  function NM_Recur{false}(cell, state)
    new{false, typeof(cell), typeof(state)}(cell, state)
  end
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

function _apply_to_layer(l::NM_Recur{false}, xs::Union{AbstractVector{<:AbstractArray}, Base.Generator})
  rnn = l.cell
  # carry = layer.stamte
  x_init, x_rest = Iterators.peel(xs)
  (carry, y) = rnn(l.state, x_init)
  for x in x_rest
    (carry, y) = rnn(carry, x)
  end
  NM_Recur{false}(rnn, carry), y
end

# From Lux.jl: https://github.com/LuxDL/Lux.jl/pull/287/
function _apply_to_layer(l::NM_Recur{true}, xs::Union{AbstractVector{<:AbstractArray}, Base.Generator})
  rnn = l.cell
  x_init, x_rest = Iterators.peel(xs)

  (carry, out_) = rnn(l.state, x_init)

  init = (typeof(out_)[out_], carry)

  function recurrence_op(input, (outputs, carry))
    carry, out = rnn(carry, input)
    return vcat(outputs, typeof(out)[out]), carry
  end

  results = foldr(recurrence_op, xs[(begin+1):end]; init)
  return NM_Recur{true}(rnn, results[2][end]), first(results)
end

Flux.@functor NM_Recur
Flux.trainable(a::NM_Recur) = (; cell = a.cell)

Base.show(io::IO, m::NM_Recur) = print(io, "Recur(", m.cell, ")")

NM_RNN(a...; return_sequence::Bool=false, ka...) = NM_Recur(Flux.RNNCell(a...; ka...); return_sequence=return_sequence)
NM_Recur(m::Flux.RNNCell; return_sequence::Bool=false) = NM_Recur(m, m.state0; return_sequence=return_sequence)

##
# Apply timeseries data in vectors to chains
##


# To apply the timeseries data at the layer levels, we can re-use the applies above.
# We only need to add a default method to _apply_to_layer, and then a method for
# stateful _apply_to_layer using the _dscan above.
function apply_timeseries_layerlevel(chain::Flux.Chain, xs)
  apply(chain, xs)
end

function _apply_to_layer(l, xs::Union{AbstractVector{<:AbstractArray}, Base.Generator})
  l, [l(x) for x in xs]
end

# This is temporary, and is in desparate need of a fix.
# It was adapted from how lux recommends using RNNs: https://lux.csail.mit.edu/stable/examples/generated/beginner/SimpleRNN/main/
# Issues:
# - Can't get access to each carry in the sequence
#   - problematic if wanting to regularize the hidden state of an RNN.
#   - Or wanting to add a loss to the hidden state sequence.
# - Only returns the output of the end of the sequence. Not the whole sequence.


function run_grad_test()
  cell = Flux.RNNCell(1, 1, identity)
  layer = Flux.Recur(cell)
  layer.cell.Wi .= 5.0
  layer.cell.Wh .= 4.0
  layer.cell.b .= 0.0f0
  layer.cell.state0 .= 7.0
  x = [[2.0f0], [3.0f0]]

  # theoretical primal gradients
  primal =
    layer.cell.Wh .* (layer.cell.Wh * layer.cell.state0 .+ x[1] .* layer.cell.Wi) .+
    x[2] .* layer.cell.Wi
  ∇Wi = x[1] .* layer.cell.Wh .+ x[2]
  ∇Wh = 2 .* layer.cell.Wh .* layer.cell.state0 .+ x[1] .* layer.cell.Wi
  ∇b = layer.cell.Wh .+ 1
  ∇state0 = layer.cell.Wh .^ 2

  nm_layer = Fluxperimental.NM_Recur(cell; return_sequence=true)

  ps = Flux.params(nm_layer)
  e, g = Flux.withgradient(ps) do
    # l, _ = Fluxperimental._apply_to_layer(nm_layer, x[1])
    # l2, out = Fluxperimental._apply_to_layer(l, x[2])
    _, out = Fluxperimental._apply_to_layer(nm_layer, x)
    sum(out[2])
  end
  
  @info primal[1] ≈ e
  @info ∇Wi, g[ps[1]]
  @info ∇Wi ≈ g[ps[1]]
  @info ∇Wh, g[ps[2]]
  @info ∇Wh ≈ g[ps[2]]
  @info ∇b, g[ps[3]]
  @info ∇b ≈ g[ps[3]]
  @info ∇state0, g[ps[4]]
  @info ∇state0 ≈ g[ps[4]]

end

