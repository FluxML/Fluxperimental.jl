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

function apply(m::NM_Recur, x)
  state, y = m.cell(m.state, x)
  return NM_Recur(m.cell, state), y
end

# This is the same way we do 3-tensers from Flux.Recur
function apply(m::NM_Recur{true}, x::AbstractArray{T, 3}) where T
  # h = [m(x_t) for x_t in eachlastdim(x)]
  l, h = apply(m, Flux.eachlastdim(x))
  sze = size(h[1])
  l, reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

function apply(m::NM_Recur{false}, x::AbstractArray{T, 3}) where T
  apply(m, Flux.eachlastdim(x))
end

function apply(l::NM_Recur{false}, xs::Union{AbstractVector{<:AbstractArray}, Base.Generator})
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
function apply(l::NM_Recur{true}, xs::Union{AbstractVector{<:AbstractArray}, Base.Generator})
  rnn = l.cell
  _xs = if xs isa Base.Generator
    collect(xs)  # TODO: Fix. I can't figure out how to get around this for generators.
  else
    xs
  end
  x_init, _ = Iterators.peel(_xs)

  (carry, out_) = rnn(l.state, x_init)

  init = (typeof(out_)[out_], carry)

  function recurrence_op(input, (outputs, carry))
    carry, out = rnn(carry, input)
    return vcat(outputs, typeof(out)[out]), carry
  end
  results = foldr(recurrence_op, _xs[(begin+1):end]; init)
  return NM_Recur{true}(rnn, results[1][end]), first(results)
end

Flux.@functor NM_Recur
Flux.trainable(a::NM_Recur) = (; cell = a.cell)

Base.show(io::IO, m::NM_Recur) = print(io, "Recur(", m.cell, ")")

NM_RNN(a...; return_sequence::Bool=false, ka...) = NM_Recur(Flux.RNNCell(a...; ka...); return_sequence=return_sequence)
NM_Recur(m::Flux.RNNCell; return_sequence::Bool=false) = NM_Recur(m, m.state0; return_sequence=return_sequence)

# Quick Reset functionality

struct RecurWalk <: Flux.Functors.AbstractWalk end
(::RecurWalk)(recurse, x) = x isa Fluxperimental.NM_Recur ? reset(x) : Flux.Functors.DefaultWalk()(recurse, x)

function reset(m::NM_Recur{SEQ}) where SEQ
  NM_Recur{SEQ}(m.cell, m.cell.state0)
end
reset(m) = m
function reset(m::Flux.Chain)
  ret = Flux.Functors.fmap((l)->l, m; walk=RecurWalk())
end


##
# Fallback apply timeseries data to other layers. Likely needs to be thoought through a bit more.
##

function apply(l, xs::Union{AbstractVector{<:AbstractArray}, Base.Generator})
  l, [l(x) for x in xs]
end
