

"""
  NewRecur
New Recur. An experimental recur interface for removing statefullness in recurrent architectures for flux.
"""
struct NewRecur{RET_SEQUENCE, T}
  cell::T
  # state::S
  function NewRecur(cell; return_sequence::Bool=false)
    new{return_sequence, typeof(cell)}(cell)
  end
  function NewRecur{true}(cell)
    new{true, typeof(cell)}(cell)
  end
  function NewRecur{false}(cell)
    new{false, typeof(cell)}(cell)
  end
end

# This is the same way we do 3-tensers from Flux.Recur
function (m::NewRecur{false})(x::AbstractArray{T, N}, carry) where {T, N}
  @assert N >= 3
  # h = [m(x_t) for x_t in eachlastdim(x)]

  cell = l.cell
  x_init, x_rest = Iterators.peel(xs)
  (carry, y) = cell(carry, x_init)
  for x in x_rest
    (carry, y) = cell(carry, x)
  end
  # carry, y
  y

end

function (l::NewRecur{false})(x::AbstractArray{T, 3}, carry=l.cell.state0) where T
  m(Flux.eachlastdim(x), carry)
end

function (l::NewRecur{false})(xs::Union{AbstractVector{<:AbstractArray}, Base.Generator},
                              carry=l.cell.state0)
  rnn = l.cell
  # carry = layer.stamte
  x_init, x_rest = Iterators.peel(xs)
  (carry, y) = rnn(carry, x_init)
  for x in x_rest
    (carry, y) = rnn(carry, x)
  end
  y
end

# From Lux.jl: https://github.com/LuxDL/Lux.jl/pull/287/
function (l::NewRecur{true})(xs::Union{AbstractVector{<:AbstractArray}, Base.Generator},
                             carry=l.cell.state0)
  rnn = l.cell
  _xs = if xs isa Base.Generator
    collect(xs)  # TODO: Fix. I can't figure out how to get around this for generators.
  else
    xs
  end
  x_init, _ = Iterators.peel(_xs)

  (carry, out_) = rnn(carry, x_init)

  init = (typeof(out_)[out_], carry)

  function recurrence_op(input, (outputs, carry))
    carry, out = rnn(carry, input)
    return vcat(outputs, typeof(out)[out]), carry
  end
  results = foldr(recurrence_op, _xs[(begin+1):end]; init)
  # return NewRecur{true}(rnn, results[1][end]), first(results)
  first(results)
end

Flux.@functor NewRecur
Flux.trainable(a::NewRecur) = (; cell = a.cell)

Base.show(io::IO, m::NewRecur) = print(io, "Recur(", m.cell, ")")

NewRNN(a...; return_sequence::Bool=false, ka...) = NewRecur(Flux.RNNCell(a...; ka...); return_sequence=return_sequence)

