


##### Helper scan funtion which can likely be put into NNLib. #####
"""
  scan

Recreating jax.lax.scan functionality in julia.
"""
function scan_full(func, init_carry, xs::AbstractVector{<:AbstractArray})
  # xs = Flux.eachlastdim(x_block)
  x_init, x_rest = Iterators.peel(xs)

  (carry, out_) = func(init_carry, x_init)

  init = (typeof(out_)[out_], carry)

  function recurrence_op(input, (outputs, carry))
    carry, out = func(carry, input)
    return vcat(outputs, typeof(out)[out]), carry
  end
  results = foldr(recurrence_op, xs[(begin+1):end]; init)
  results[2], results[1]
end

function scan_full(func, init_carry, x_block)
  xs_ = Flux.eachlastdim(x_block)
  xs = if xs_ isa Base.Generator
    collect(xs_) # eachlastdim produces a generator in non-gradient environment
  else
    xs_
  end
  scan_full(func, init_carry, xs)
end

function scan_partial(func, init_carry, xs::AbstractVector{<:AbstractArray})
  x_init, x_rest = Iterators.peel(xs)
  (carry, y) = func(init_carry, x_init)
  for x in x_rest
    (carry, y) = func(carry, x)
  end
  # carry, y
  carry, y
end

function scan_partial(func, init_carry, x_block)
  xs_ = Flux.eachlastdim(x_block)
  xs = if xs_ isa Base.Generator
    collect(xs_) # eachlastdim produces a generator in non-gradient environment
  else
    xs_
  end
  scan_partial(func, init_carry, xs)
end


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

Flux.@functor NewRecur
Flux.trainable(a::NewRecur) = (; cell = a.cell)
Base.show(io::IO, m::NewRecur) = print(io, "Recur(", m.cell, ")")
NewRNN(a...; return_sequence::Bool=false, ka...) = NewRecur(Flux.RNNCell(a...; ka...); return_sequence=return_sequence)


(l::NewRecur)(init_carry, x_mat::AbstractMatrix) = MethodError("Matrix is ambiguous with NewRecur")
(l::NewRecur)(init_carry, x_mat::AbstractVector{T}) where {T<:Number} = MethodError("Vector is ambiguous with NewRecur")

(l::NewRecur)(xs) = l(l.cell.state0, xs)


function (l::NewRecur{false})(init_carry,
                              xs)
  results = scan_partial(l.cell, init_carry, xs)
  results[2]
end

# From Lux.jl: https://github.com/LuxDL/Lux.jl/pull/287/
function (l::NewRecur{true})(init_carry,
                             xs,)

  results = scan_full(l.cell, init_carry, xs)
  
  h = results[2]
  sze = size(h[1])
  reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end



