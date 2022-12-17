using Flux, ChainRulesCore
using LinearAlgebra: mul!
using FastBroadcast: @..
using Strided

const NoT = NoTangent()

"""
    PreLayer(Dense(2 => 3, relu))

Stores, along with the layer, pre-allocated space for its output,
and all gradient components. Only works on layers it understands.
"""
struct PreLayer{L,G,V}
  layer::L
  grad::G  # same fixed sizes as layer
  fwd::V  # vector of dynamic length 
  rev::V
end

Flux.@functor PreLayer
Flux.trainable(p::PreLayer) = (; layer = p.layer)

"""
    model |> pre

Wrap as many layers as possible with `PreLayer`,
to store pre-allocated space for output & gradient.
Ignores layers it doesn't understand.
"""
pre(model) = fmap(PreLayer, model; exclude = x -> hasmethod(PreLayer, Tuple{typeof(x)}))

"""
    nopre(model)

Remove all `PreLayer`s & return the plain model.
"""
nopre(model) = fmap(x -> x.layer, model; exclude = x -> x isa PreLayer)


#####
#####  Dense
#####

function PreLayer(d::Dense)
  grad = _struct_sim(d)
  fwd, rev = similar(d.weight, 0), similar(d.weight, 0)
  PreLayer(d, grad, fwd, rev)
end

function (p::PreLayer{<:Dense})(x::AbstractMatrix{<:Real})
  y, dx = _pre_setup(p, x)
  _densecall!(y, p, x, dx)
end

function _pre_setup(p::PreLayer{<:Dense}, x)  # this function @nograd
  _, b = size(x)
  o, i = size(p.layer.weight)
  if o*b != length(p.fwd)
    resize!(p.fwd, o*b)
    resize!(p.rev, i*b)
  end
  y = _pre_reshape(p.fwd, (o,b))
  dx = _pre_reshape(p.rev, (i,b))
  (; y, dx)
end

function _densecall!(y, p, x, dx)
  y .= p.layer.bias
  mul!(y, p.layer.weight, x, true, true)
  act!(y, p.layer.σ)
  y
end

function ChainRulesCore.rrule(::typeof(_densecall!), y, p, x, dx)
  y = _densecall!(y, p, x, dx)
  function back(dy)
    dy = unthunk(dy)
    dy = ∇act!(y, dy, p.layer.σ)
    # layer
    weight = mul!(p.grad.weight, dy, x') 
    bias = ∇bias!(p.grad.bias, dy)
    tang = Tangent{Dense}(; weight, bias)
    # input
    dx = mul!(dx, p.layer.weight', dy)
    return (NoT, NoT, Tangent{PreLayer}(; layer = tang), dx, NoT)
  end
  y, back
end

#####
#####  Scale
#####

scale!(y, (scale, ds), (x, dx), (bias, db)) = y .= scale .* x .+ bias
# scale!(y, (scale, ds), (x, dx), (bias, db)) = @strided y .= scale .* x .+ bias

function ChainRulesCore.rrule(::typeof(scale!), y, (scale, ds), (x, dx), (bias, db))
  y = scale!(y, (scale, ds), (x, dx), (bias, db))
  function back(dy)
    dy = unthunk(dy)
    @strided dx .= dy .* scale
    @strided ds .= dy .* x
    dbias = ∇bias!(bias, db)
    return (NoT, NoT, (ds, NoT), (dx, NoT), (dbias, NoT))
  end
  y, back
end

#####
#####  Conv
#####

function PreLayer(c::Conv)
  grad = _struct_sim(c)
  fwd, rev = similar(c.weight, 0), similar(c.weight, 0)
  PreLayer(c, grad, fwd, rev)
end

function (p::PreLayer{<:Conv})(x::AbstractArray{<:Real})
  y, dx = _pre_setup(p, x)
  _convcall!(y, p, x, dx)
end

using Flux: conv_dims, conv_reshape_bias
using Flux.NNlib: fast_act, conv!, output_size, channels_out

function _pre_setup(p::PreLayer{<:Conv}, x)
  cdims = conv_dims(p.layer, x)
  ysize = (output_size(cdims)..., channels_out(cdims), size(x)[end])
  if prod(ysize) != length(p.fwd)
    resize!(p.fwd, prod(ysize))
    resize!(p.rev, length(x))
  end
  y = _pre_reshape(p.fwd, ysize)
  dx = _pre_reshape(p.rev, size(x))
  (; y, dx)
end

function _convcall!(y, p, x, dx)
  cdims = conv_dims(p.layer, x)
  conv!(y, x, p.layer.weight, cdims)
  if p.layer.bias isa AbstractArray
    y .+= conv_reshape_bias(p.layer)
  end
  act!(y, fast_act(p.layer.σ, x))
end

# function ChainRulesCore.rrule(::typeof(_convcall!), y, p, x, dx)
#   y = _densecall!(y, p, x, dx)
#   function back(dy)
#     dy = unthunk(dy)
#     dy = ∇act!(y, dy, p.layer.σ)
#     # layer
#     weight = mul!(p.grad.weight, dy, x') 
#     bias = ∇bias!(p.grad.bias, dy)
#     tang = Tangent{Dense}(; weight, bias)
#     # input
#     dx = mul!(dx, p.layer.weight', dy)
#     return (NoT, NoT, Tangent{PreLayer}(; layer = tang), dx, NoT)
#   end
#   y, back
# end



#####
#####  BatchNorm
#####

function PreLayer(bn::BatchNorm)
  grad = (β = similar(bn.β), γ = similar(bn.γ))  # only trainable fields
  fwd, rev = zeros(Float32, 0), zeros(Float32, 0)  # not ideal
  PreLayer(bn, grad, fwd, rev)
end

function (p::PreLayer{<:BatchNorm})(x::AbstractArray{<:Real})
  y, dx = _pre_setup(p, x)
  # _batchnormcall!(y, p, x, dx)

  # from (BN::BatchNorm)(x)
  N = ndims(x)
  reduce_dims = [1:N-2; N]
  affine_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
  _norm_layer_forward!(y, p, (x, dx); reduce_dims, affine_shape)
end

using Flux: _isactive, _track_stats!, hasaffine

function _norm_layer_forward!(y, p, (x, dx); reduce_dims, affine_shape)
  l = p.layer
  N = ndims(x)

  # This block verbatim from Flux. However, mean & var aren't in-place,
  # nor are their gradients... add more storage? 

  if !_isactive(l) && l.track_stats # testmode with tracked stats
    stats_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    μ = reshape(l.μ, stats_shape)
    σ² = reshape(l.σ², stats_shape)
  else # trainmode or testmode without tracked stats
    μ = mean(x; dims=reduce_dims)
    σ² = var(x; mean=μ, dims=reduce_dims, corrected=false)
    if l.track_stats
      _track_stats!(l, x, μ, σ², reduce_dims) # update moving mean/std
    end
  end

  y = _norm_layer_forward!(y, x, dx, μ, σ², l.ϵ)
  hasaffine(l) || return act!(y, l.λ)

  γ = reshape(l.γ, affine_shape)
  β = reshape(l.β, affine_shape)
  # return l.λ.(γ .* y .+ β)
  y2 = scale!(y, (γ, p.grad.γ), (x, dx), (β, p.grad.β))
  return act!(y2, l.λ)
end

_norm_layer_forward!(y, x, dx, μ, σ², ϵ) = y .= (x .- μ) ./ sqrt.(σ² .+ ϵ)
# _norm_layer_forward!(y, x, dx, μ, σ², ϵ) = @strided y .= (x .- μ) ./ sqrt.(σ² .+ ϵ)

function ChainRulesCore.rrule(::typeof(_norm_layer_forward!), y, x, dx, μ, σ², ϵ)
  y = _norm_layer_forward!(y, x, dx, μ, σ², ϵ)
  function back(dy)
    dx .= dy ./ sqrt.(σ² .+ ϵ)
    # TODO write gradients for mean & variance, these are WRONG!
    dμ = NoT
    dσ² = NoT
    return (NoT, NoT, dx, NoT, dμ, dσ², NoT)
  end
  y, back
end

#####
#####  softmax
#####

function PreLayer(::typeof(softmax))
  fwd, rev = zeros(Float32, 0), zeros(Float32, 0)  # not ideal, demands `model |> pre |> gpu` 
  PreLayer(softmax, nothing, fwd, rev)
end

function (p::PreLayer{typeof(softmax)})(x::AbstractArray{<:Real})
  y, dx = _pre_setup(p, x)  # generic version
  _softmaxcall!(y, p, x, dx)
end

_softmaxcall!(y, p, x, dx) = softmax!(y, x)

function ChainRulesCore.rrule(::typeof(_softmaxcall!), y, p, x, dx)
  y = _softmaxcall!(y, p, x, dx)
  function back(dy)
    # TODO: CHECK THIS!
    dx .= dy .* y
    dx .= dx .- y .* sum(dx; dims=1)  # could sum! into the end of rev
    return (NoT, NoT, NoT, dx, NoT)  # last one could be NotImplemented?
  end
  y, back
end


#####
#####  activation functions
#####

act!(y, ::typeof(identity)) = y
function act!(y, act::F) where F
  σ = Flux.NNlib.fast_act(act, y)
  # y .= σ.(y)
  # Unfortunately this hits  https://github.com/JuliaLang/julia/issues/43153
  # maybe you could patch Strided.jl to avoid it? Or use another package...
  # @strided y .= σ.(y)
  @.. y = σ(y)
end

# Piracy, disable @strided on CuArrays:
Strided.maybestrided(x::Flux.CuArray) = x

# For this rule, it's important to use what `act!` returns, not what it mutates
ChainRulesCore.rrule(::typeof(act!), y, f) = act!(y, f), dz -> (NoT, ∇act!(y, dy, f), NoT)

∇act!(y, dy, ::typeof(identity)) = dy
∇act!(y, dy, ::typeof(relu)) = @.. y = ifelse(y>0, dy, 0f0)
∇act!(y, dy, ::typeof(tanh)) = @.. y = (1 - y^2)
∇act!(y, dy, ::typeof(sigmoid)) = @.. y = y * (1 - y)


function PreLayer(::typeof(relu))
  fwd, rev = zeros(Float32, 0), zeros(Float32, 0)  # not ideal
  PreLayer(relu, nothing, fwd, rev)
end

function (p::PreLayer{typeof(relu)})(x::AbstractArray{<:Real})
  y, dx = _pre_setup(p, x)  # generic version
  _relucall!(y, p, x, dx)
end

_relucall!(y, p, x, dx) = y .= relu.(x)

function ChainRulesCore.rrule(::typeof(_relucall!), y, p, x, dx)
  y = _relucall!(y, p, x, dx)
  function back(dy)
    @. dx = ifelse(y>0, dy, 0f0)
    return (NoT, NoT, NoT, dx, NoT)
  end
  y, back
end

#####
#####  PreLayer utils
#####

_struct_sim(x) = Flux.fmapstructure(x) do x
    x isa AbstractArray{<:Real} ? similar(x) : nothing
end

function _pre_setup(p::PreLayer, x)  # generic version
  if length(x) != length(p.fwd)
    resize!(p.fwd, length(x))
    resize!(p.rev, length(x))
  end
  y = _pre_reshape(p.fwd, size(x))
  dx = _pre_reshape(p.rev, size(x))
  (; y, dx)
end
ChainRulesCore.@non_differentiable _pre_setup(::Any, ::Any)

# Cannot use reshape(::Array), as that prevents later resize!
_pre_reshape(x::Array, size::Tuple) = Base.ReshapedArray(x, size, ())
# _pre_reshape(x::Array, size::Tuple) = Base.__reshape((x, Base.IndexStyle(x)), size)  # what Base does, no better
# Must use reshape(::CuArray) as mul! rejects ReshapedArray
_pre_reshape(x::Flux.CuArray, size::Tuple) = reshape(x, size)
_pre_reshape(x, size::Tuple) = reshape(x, size)

# Base piracy! to prevent ReshapedArray from going missing
Base._reshape(R::Base.ReshapedArray, dims::Base.Dims) = Base.ReshapedArray(R.parent, dims, ())

∇bias!(::Bool, dx) = NoT
∇bias!(bias, dx) = sum!(bias, dx)

function Base.show(io::IO, p::PreLayer)
  show(io, p.layer)
  printstyled(io, " |> pre", color=:blue)
end

Flux._show_children(p::PreLayer) = Flux._show_children(p.layer)
