module FluxMooncakeExt

using Flux, Fluxperimental, Optimisers, Functors, Mooncake
import Fluxperimental: _moonstrip
# using Flux: Const

function Fluxperimental.Moonduo(x)
  dx = Mooncake.zero_tangent(x)
  Moonduo(x, dx)
end

# Flux gradient etc.

"""
    Flux.gradient(f, args::Moonduo...)

This uses Mooncake.jl to compute the derivative,
which is both stored within `Moonduo` and returned.
Similar to the Enzyme.jl methods like `Flux.gradient(f, m::Duplicated)`.

# Example

```julia
julia> using Flux

julia> model = Chain(Dense([3.0;;]));

julia> Flux.gradient(model, [1]) do m, x  # computed using Zygote
         sum(abs2, m(x))
       end
((layers = ((weight = [6.0;;], bias = [6.0], σ = nothing),),), [18.0])

julia> using Fluxperimental, Mooncake

julia> dup_model = Moonduo(model);  # allocates space for gradient

julia> Flux.gradient(dup_model, Moonduo([1])) do m, x  # Mooncake, returns the same
         sum(abs2, m(x))
       end
((layers = ((weight = [6.0;;], bias = [6.0], σ = nothing),),), nothing)

julia> dup_model  # same gradient is also stored within Duplicated
Moonduo(
  Chain(
    Dense(1 => 1),                      # 2 parameters
  ),
  # norm(∇) ≈ 8.49
)

julia> Flux.destructure((weight = [6.0;;], bias = [6.0]))[1] |> norm
8.48528137423857

julia> Flux.gradient(dup_model, Moonduo([1]); zero=false) do m, x  # grad accumulation
         sum(abs2, m(x))
       end
((layers = ((weight = [12.0;;], bias = [12.0], σ = nothing),),), nothing)
```

!!! note
    At present there is no way to mark some arguments constant.
    Instead of `gradient(loss, Duplicated(model), Const(data))`,
    you can write `gradient(m -> loss(m, data), Moonduo(model))`.
"""
Flux.gradient(f, args::Moonduo...; zero::Bool=true) = _moon_withgradient(f, args...; zero).grad

"""
    Flux.withgradient(f, args::Moonduo...)

This should return the same answer as `withgradient(f, model, args...)`,
but it uses Mooncake.jl instead of Zygote.jl to compute the derivative.

# Example

```julia
julia> using Flux, Fluxperimental, Mooncake

julia> model = Chain(Embedding([1.1 2.2 3.3]), Dense([4.4;;]), only);

julia> model(3)
14.52

julia> Flux.withgradient(m -> m(3), model)  # this uses Zygote
(val = 14.52, grad = ((layers = ((weight = [0.0 0.0 4.4],), (weight = [3.3;;], bias = [1.0], σ = nothing), nothing),),))

julia> Flux.withgradient(m -> m(3), Moonduo(model))  # this uses Mooncake
(val = 14.52, grad = ((layers = ((weight = [0.0 0.0 4.4],), (weight = [3.3;;], bias = [1.0], σ = nothing), nothing),),))
```

!!! warning
    With Zygote, the function `f` may return Tuple or NamedTuple, with the loss as the first element.
    This feature is not supported here, for now.
"""
Flux.withgradient(f, args::Moonduo...; zero::Bool=true) = _moon_withgradient(f, args...; zero)

function _moon_withgradient(f, args::Moonduo...; zero)
  plain = map(x -> x.val, args)
  rule = Mooncake.build_rrule(f, plain...)

  for x in args
    _check_mutable(x)
    zero && Mooncake.set_to_zero!!(x.dval)
  end
  coduals = map(x -> Mooncake.CoDual(x.val, x.dval), args)
  val, _ = Mooncake.__value_and_gradient!!(rule, Mooncake.zero_codual(f), coduals...)

  grad = map(x -> _moongrad(x.dval), args)
  (; val, grad)
end

# _check_mutable(x::Const) = nothing
_check_mutable(x::Moonduo) = Functors.anymutable(x) || error(
    """`Flux.gradient(f, Moonduo(x), ...)` expects `x` to contain mutable parameter arrays."""
)

function _moongrad(dx)
  dx2 = _moonstrip(dx)  # remove all the weird types
  isnothing(dx2) && return
  return Flux.fmapstructure(identity, dx2; prune=nothing)
end

_moonstrip(dx::Mooncake.Tangent) = map(_moonstrip, dx.fields)
_moonstrip(dx::Mooncake.MutableTangent) = map(_moonstrip, dx.fields)
_moonstrip(dx::Mooncake.NoTangent) = nothing
_moonstrip(dx::Union{Tuple, NamedTuple, AbstractArray}) = map(_moonstrip, dx)
_moonstrip(dx::AbstractArray{Mooncake.NoTangent}) = nothing
_moonstrip(dx::AbstractArray{<:Number}) = dx
_moonstrip(dx::AbstractArray{<:Integer}) = nothing
_moonstrip(dx::Number) = nothing
function _moonstrip(dx)
  @warn "not sure what to do with this type" typeof(dx)
  dx
end

# Optimisers etc.

Flux.setup(rule::Optimisers.AbstractRule, m::Moonduo) = Flux.setup(rule, m.val)

function Flux.update!(opt_state, model::Moonduo)
  Flux.update!(opt_state, model.val, _moongrad(model.dval))
  nothing
end

### Flux.Train, for train!

_applyloss(loss, model, d...) = loss(model, d...)

"""
    train!(loss, Moonduo(model), data, opt_state)

This method uses Mooncake.jl instead of Zygote.jl to compute the gradients, but is otherwise the
same as `Flux.train!(loss, model, data, opt_state)`.
"""
function Flux.train!(loss, model::Moonduo, data, opt; cb=nothing, epochs::Int=1)
  isnothing(cb) || error("""train! does not support callback functions.
                            For more control use a loop with `gradient` and `update!`.""")
  Flux.Train.@withprogress for (i,d) in enumerate(Iterators.cycle(data, epochs))
    d_splat = d isa Tuple ? d : (d,)
    rule = Mooncake.build_rrule(f, model.val, d_splat...)  # perhaps not ideal to do this inside the loop?

    Mooncake.set_to_zero!!(model.dval)
    l, _ = Mooncake.__value_and_gradient!!(rule, Mooncake.zero_codual(f), model, map(Mooncake.zero_codual, d_splat)...)

    if !isfinite(l)
      throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
    end

    Flux.update!(opt, model)

    Flux.Train.@logprogress Base.haslength(data) ? i/(length(data)*epochs) : nothing
  end
end

end  # module
