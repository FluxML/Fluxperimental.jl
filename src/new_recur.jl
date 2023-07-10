


##### Helper scan funtion which can likely be put into NNLib. #####
"""
  scan_full

Recreating jax.lax.scan functionality in julia. Takes a function, initial carry and a sequence,
then returns the output sequence and the final carry. 
"""
function scan_full(func, init_carry, xs::AbstractVector{<:AbstractArray})
  # get the first input to setup the initial state,
  # get the rest of the input to run the fold over.
  x_init, x_rest = Iterators.peel(xs)
  # the following does the same as peel, but doesn't produce correct gradients?
  ### x_init = first(xs)
  ### x_rest = xs[begin+1:end]

  # set up the initial state of the fold.
  (carry_, out_) = func(init_carry, x_init)
  init = (carry_, [out_])

  # recurrence operation used in the fold. Takes the state  of the
  # folde and the next input, returns the new state.
  function __recurrence_op((carry, outputs), input)
    carry, out = func(carry, input)
    return carry, vcat(outputs, [out])
  end
  # Fold left to right.
  foldl(__recurrence_op, x_rest; init)
end

function scan_full(func, init_carry, x_block)
  # x_block is an abstractarray and we want to scan over the last dimension.
  xs_ = Flux.eachlastdim(x_block)

  # this is needed due to a bug in eachlastdim which produces a vector in a
  # gradient context, but a generator otherwise.
  xs = if xs_ isa Base.Generator
    collect(xs_) # eachlastdim produces a generator in non-gradient environment
  else
    xs_
  end
  scan_full(func, init_carry, xs)
end


"""
  scan_partial

Recreating jax.lax.scan functionality in julia. Takes a function, initial carry and a sequence,
then returns the final output of the sequence and the final carry. 
"""
function scan_partial(func, init_carry, xs::AbstractVector{<:AbstractArray})
  x_init, x_rest = Iterators.peel(xs)
  (carry, y) = func(init_carry, x_init)
  for x in x_rest
    (carry, y) = func(carry, x)
  end
  carry, y
end

function scan_partial(func, init_carry, x_block)
  # x_block is an abstractarray and we want to scan over the last dimension.
  xs_ = Flux.eachlastdim(x_block)
  
  # this is needed due to a bug in eachlastdim which produces a vector in a
  # gradient context, but a generator otherwise.
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

function (l::NewRecur)(xs::AbstractArray)
  results = l(l.cell.state0, xs)
  results[2] # Only return the output here.
end

function (l::NewRecur{false})(init_carry, xs)
  results = scan_partial(l.cell, init_carry, xs)
  results[1], results[2]
end

function (l::NewRecur{true})(init_carry, xs)

  results = scan_full(l.cell, init_carry, xs)
  results[1], stack(results[2], dims=3)
end



