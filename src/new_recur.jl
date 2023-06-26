

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

# assumes single timestep with batch=1.
function (l::NewRecur)(x_vec::AbstractVector{T},
                       init_carry=l.cell.state0) where T<:Number
  x_block = reshape(x_vec, :, 1, 1)
  l(x_block, init_carry)[:, 1, 1]
end

(l::NewRecur)(x_mat::AbstractMatrix, args...) = error("Matrix is ambiguous with NewRecur")

function (l::NewRecur{false})(x_block::AbstractArray{T, 3},
                              init_carry=l.cell.state0) where {T}
  xs = Flux.eachlastdim(x_block)
  cell = l.cell
  x_init, x_rest = Iterators.peel(xs)
  (carry, y) = cell(init_carry, x_init)
  for x in x_rest
    (carry, y) = cell(carry, x)
  end
  # carry, y
  y
end

# From Lux.jl: https://github.com/LuxDL/Lux.jl/pull/287/
function (l::NewRecur{true})(x_block::AbstractArray{T, 3},
                             init_carry=l.cell.state0) where {T}

  # Time index is always the last index.
  xs = Flux.eachlastdim(x_block)
  xs_ = if xs isa Base.Generator
    # This is because eachlastdim has different behavior in
    # a gradient environment vs outside a gradient environment.
    # Needs to be fixed....
    collect(xs)
  else
    xs
  end

  cell = l.cell
  x_init, x_rest = Iterators.peel(xs_)

  (carry, out_) = cell(init_carry, x_init)

  init = (typeof(out_)[out_], carry)

  function recurrence_op(input, (outputs, carry))
    carry, out = cell(carry, input)
    return vcat(outputs, typeof(out)[out]), carry
  end
  results = foldr(recurrence_op, xs_[(begin+1):end]; init)
  # return NewRecur{true}(rnn, results[1][end]), first(results)
  h = first(results)
  sze = size(h[1])
  reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

Flux.@functor NewRecur
Flux.trainable(a::NewRecur) = (; cell = a.cell)

Base.show(io::IO, m::NewRecur) = print(io, "Recur(", m.cell, ")")

NewRNN(a...; return_sequence::Bool=false, ka...) = NewRecur(Flux.RNNCell(a...; ka...); return_sequence=return_sequence)

