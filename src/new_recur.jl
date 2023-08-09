import Flux: ChainRulesCore
import Compat: stack

##### Helper scan funtion which can likely be put into NNLib. #####
"""
  scan_full

Recreating jax.lax.scan functionality in julia. Takes a function, initial carry and a sequence, then returns the full output of the sequence and the final carry. See `scan_partial` to only return the final output of the sequence. 
"""
function scan_full(func, init_carry, xs::AbstractVector{<:AbstractArray})
  # Recurrence operation used in the fold. Takes the state of the
  # fold and the next input, returns the new state.
  function recurrence_op((carry, outputs), input)
    carry, out = func(carry, input)
    return carry, vcat(outputs, [out])
  end
  # Fold left to right.
  return Base.mapfoldl_impl(identity, recurrence_op, (init_carry, empty(xs)), xs)
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

# Chain Rule for Base.mapfoldl_impl
function ChainRulesCore.rrule(
  config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
  ::typeof(Base.mapfoldl_impl),
  ::typeof(identity),
  op::G,
  init,
  x::Union{AbstractArray, Tuple};
) where {G}
  hobbits = Vector{Any}(undef, length(x))  # Unfornately Zygote needs this
  accumulate!(hobbits, x; init=(init, nothing)) do (a, _), b
    c, back = ChainRulesCore.rrule_via_ad(config, op, a, b)
  end
  y = first(last(hobbits))
  axe = axes(x)
  project = ChainRulesCore.ProjectTo(x)
  function unfoldl(dy)
    trio = accumulate(Iterators.reverse(hobbits); init=(0, dy, 0)) do (_, dc, _), (_, back)
      ds, da, db = back(dc)
    end
    dop = sum(first, trio)
    dx = map(last, Iterators.reverse(trio))
    d_init = trio[end][2]
    return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), dop, d_init, project(reshape(dx, axe)))
  end
  return y, unfoldl
end


"""
  scan_partial

Recreating jax.lax.scan functionality in julia. Takes a function, initial carry and a sequence, then returns the final output of the sequence and the final carry. See `scan_full` to return the entire output sequence.
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
New Recur. An experimental recur interface for removing statefullness in recurrent architectures for flux. This struct has two type parameters. The first `RET_SEQUENCE` is a boolean which determines whether `scan_full` (`RET_SEQUENCE=true`) or `scan_partial` (`RET_SEQUENCE=false`) is used to scan through the sequence. This structure has no internal state, and instead returns:

```julia
l = NewRNN(1,2)
xs # Some input array Input x BatchSize x Time
init_carry # the initial carry of the cell.
l(xs) # -> returns the output of the RNN, uses cell.state0 as init_carry.
l(init_carry, xs) # -> returns (final_carry, output), where the size ofoutput is determined by RET_SEQUENCE.
```
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
