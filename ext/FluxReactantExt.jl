module FluxReactantExt

using Flux, Fluxperimental, Reactant, Enzyme
import Fluxperimental: Reactor

# mutable struct Reactor{M}
#   model::M
#   fwd_compiled
#   fwd_input
#   fwd_count::Int
#   grad_compiled
#   grad_input
#   grad_count::Int
#   gradient::M
# end

"""
    Reactor(model)(x)

Wrapper for use with Reactant.jl, which stores a Flux model,
and its compiled version which is saved on first call.

Like `Duplicated`, it also allocates space for an Enzyme.jl gradient, and
calling `Flux.gradient(loss, ::Reactor, ::Const...)` will compile the gradient calculation.

# Example
```julia-repl
julia> using Flux, Fluxperimental, Reactant, Enzyme

julia> img = rand32(28, 28, 1, 128);

julia> mlp = Chain(Flux.flatten, Dense(28^2 => 32, tanh), Dense(32 => 10));

julia> mlp(img)[1:3]  # plain Julia
3-element Vector{Float32}:
 -0.40599045
  0.48307753
  0.12329187

julia> re_mlp = Reactor(mlp);  # signal to use Reactant

julia> re_mlp(img)[1:3]
┌ Info: compiling forward pass
└   summary(xr) = "28×28×1×128 ConcreteRArray{Float32, 4}"
3-element ConcreteRArray{Float32, 1}:
 -0.4059906
  0.4830778
  0.12329111

julia> re_mlp(img)[1:3];  # does not recompile

julia> re_mlp  # after forward but not yet gradient
Reactor(
  Chain(
    Flux.flatten,
    Dense(784 => 32, tanh),             # 25_120 parameters
    Dense(32 => 10),                    # 330 parameters
  ),
  # compiled for 28×28×1×128 ConcreteRArray{Float32, 4}, run 2 times
  # norm(∇) ≈ 0.0
  # gradient not yet compiled
)         # Total: 4 trainable arrays, 25_450 parameters,
          # plus 4 non-trainable, 25_450 parameters, summarysize 705 bytes.
```
"""
function Reactor(model)
    mr = Reactant.to_rarray(model)
    Reactor{typeof(mr)}(mr)
end

### forward

function (m::Reactor)(x::AbstractArray)
    Flux.Zygote.isderiving() && error("can't use Reactor within Zygote!")
    xr = Reactant.to_rarray(x)
    input = _input_summary(xr)
    if input == m.fwd_input
        y = m.fwd_compiled(xr)
        m.fwd_count += 1
        return y
    else
        @info "compiling forward pass" summary(xr)
        fun = @compile m.model(xr)
        m.fwd_compiled = fun
        m.fwd_input = input
        y = fun(xr)
        m.fwd_count += 1
        return y
    end
end

# _input_summary(x::AbstractArray{T}) where T = (T, size(x...))
# Just use strings for now, although probably change later for performance:
_input_summary(x::AbstractArray) = summary(x)
_input_summary(xs::Const...) = map(x -> _input_summary(x.val), xs)
_input_summary(f, xs::Const...) = join((string(f), _input_summary(xs...)...), ", ")

### gradient

"""
    Flux.gradient(loss::Function, model::Reactor, args::Const...)

This exact signature uses Reactant to compile the Enzyme gradient call.

# Example

```julia-repl
julia> using Flux, Fluxperimental, Reactant, Enzyme

julia> img = rand32(28, 28, 1, 128);

julia> mlp = Chain(Flux.flatten, Dense(28^2 => 32, tanh), Dense(32 => 10));

julia> loss(m, x) = sum(abs2, m(x));

julia> Flux.gradient(loss, mlp, img)[1].layers[2].bias[1:3]  # uses Zygote
3-element Vector{Float32}:
  13.273897
   1.7208669
 -63.32639

julia> dup_mlp = Duplicated(mlp);

julia> Flux.gradient(loss, dup_mlp, Const(img))[1].layers[2].bias[1:3]  # uses Enzyme
3-element Vector{Float32}:
  13.273898
   1.7208662
 -63.3264

julia> re_mlp = Reactor(mlp);

julia> Flux.gradient(loss, re_mlp, Const(img))[1].layers[2].bias[1:3]  # uses Reactant
[ Info: compiling gradient(loss, ::Reactor, ::Const...)
ERROR: BoundsError: attempt to access ReverseMode{false, false, FFIABI, false, false} at index [1]
Stacktrace:
 [1] traced_getfield(obj::Any, field::Int64)
   @ Reactant.Compiler ~/.julia/packages/Reactant/sIJRJ/src/Compiler.jl:17
 [2] macro expansion
   @ ~/.julia/packages/Reactant/sIJRJ/src/Compiler.jl:771 [inlined]
 [3] (::Reactant.Compiler.Thunk{…})(::ReverseMode{…}, ::typeof(loss), ::Type{…}, ::Duplicated{…}, ::Const{…})
   @ Reactant.Compiler ~/.julia/packages/Reactant/sIJRJ/src/Compiler.jl:787
 [4] gradient(f::Function, m::Reactor{Chain{Tuple{typeof(Flux.flatten), Dense{…}, Dense{…}}}}, xs::Const{Array{Float32, 4}})
   @ FluxReactantExt ~/.julia/dev/Fluxperimental/ext/FluxReactantExt.jl:190
 [5] top-level scope
   @ REPL[95]:1
Some type information was truncated. Use `show(err)` to see complete types.

julia> Flux.gradient(loss, re_mlp, Const(img))[1].layers[2].bias[1:3]
3-element ConcreteRArray{Float32, 1}:
  13.273894
   1.7208662
 -63.3264

julia> re_mlp  # note aside that summarysize doesn't work, TODO
Reactor(
  Chain(
    Flux.flatten,
    Dense(784 => 32, tanh),             # 25_120 parameters
    Dense(32 => 10),                    # 330 parameters
  ),
  # call not yet compiled
  # norm(∇) ≈ 3590.0f0
  # ∇compiled for loss, 28×28×1×128 ConcreteRArray{Float32, 4}, and run 1 times
)         # Total: 4 trainable arrays, 25_450 parameters,
          # plus 4 non-trainable, 25_450 parameters, summarysize 711 bytes.

julia> dup_mlp  # Enzyme wrapper stores the same gradient
Duplicated(
  Chain(
    Flux.flatten,
    Dense(784 => 32, tanh),             # 25_120 parameters
    Dense(32 => 10),                    # 330 parameters
  ),
  # norm(∇) ≈ 3590.0f0
)         # Total: 4 trainable arrays, 25_450 parameters,
          # plus 4 non-trainable, 25_450 parameters, summarysize 199.391 KiB.
```
"""
function Flux.gradient(f::Function, m::Reactor, xs::Const...)
    if !isdefined(m, :gradient)
        @info "allocating shadow"
        m.gradient = Enzyme.make_zero(m.model)  # I think this may not be zero!
    end
    xrs = Reactant.to_rarray(xs)
    input = _input_summary(f, xrs...)
    dup = Duplicated(m.model, m.gradient)
    # _seed = Ref(0f0), Ref(1f0)  # MethodError: no method matching Float32(::Reactant.TracedRNumber{Float32})
    _seed = ([0f0], [1f0]) |> Reactant.to_rarray
    seed = Duplicated(_seed...)
    function _autodiff(seed, dup, xrs...)
        Enzyme.make_zero!(Ref(dup.dval))
        Enzyme.autodiff(Reverse, Const(_fun!), seed, Const(f), dup, xrs...)  # suggestion from @jumerckx to pass simpler arguments to the function seen by  @compile
    end
    if false
        # Enzyme.autodiff(Reverse, f, Active, dup, xrs...)  # just for testing, gives zero
        Enzyme.autodiff(Reverse, Const(_fun!), seed, Const(f), dup, xrs...)  # just for testing, gives zero
    elseif input == m.grad_input
        # m.grad_compiled(Reverse, f, Active, dup, xrs...)
        # m.grad_compiled(Reverse, Const(_fun!), seed, Const(f), dup, xrs...)
        m.grad_compiled(seed, dup, xrs...)
        m.grad_count += 1
    else
        @info "compiling gradient($f, ::Reactor, ::Const...)"
        # fun = @compile Enzyme.autodiff(Reverse, f, Active, dup, xrs...)  # this gives ERROR: "Unhandled type Type" above
        # fun = @compile Enzyme.autodiff(Reverse, Const(_fun!), seed, Const(f), dup, xrs...)  # this gives ERROR: type TypeVar has no field data
        fun = @compile _autodiff(seed, dup, xrs...)  # ERROR: BoundsError: attempt to access ReverseMode{false, false, FFIABI, false, false} at index [1]
        m.grad_compiled = fun
        m.grad_input = _input_summary(f, xrs...)
        fun(Reverse, f, Active, dup, xrs...)
        m.grad_count += 1
    end
    map(_grad_or_nothing, (dup, xrs...))
end

@inline _fun!(out, f::F, args...) where F = begin out[] = f(args...); nothing end

# This function strips the returned gradient to be Zygote-like:
_grad_or_nothing(dup::Duplicated) = Flux.fmapstructure(_grad_or_nothing, dup.dval; prune=nothing)
_grad_or_nothing(re::Reactor) = Flux.fmapstructure(_grad_or_nothing, re.gradient; prune=nothing)
_grad_or_nothing(::Const) = nothing
_grad_or_nothing(x) = Optimisers.isnumeric(x) ? x : nothing

### Optimisers etc.

Optimisers.trainable(m::Reactor) = (; m.model)
Flux.Functors.@functor Reactor (model,)

Flux.setup(rule::Optimisers.AbstractRule, m::Reactor) = Flux.setup(rule, m.model)

function Flux.update!(opt_state, m::Reactor)
  Flux.update!(opt_state, m.model, _grad_or_nothing(m))
  nothing
end

### Flux.Train, for train!

# _applyloss(loss, model, d...) = loss(model, d...)
#
# """
#     train!(loss, Reactor(model), data, opt_state)

# This method uses ... instead of Zygote.jl to compute the gradients, but is otherwise the
# same as `Flux.train!(loss, model, data, opt_state)`.
# """
# function Flux.train!(loss, model::Reactor, data, opt; cb=nothing, epochs::Int=1)
#   isnothing(cb) || error("""train! does not support callback functions.
#                             For more control use a loop with `gradient` and `update!`.""")
#   Flux.Train.@withprogress for (i,d) in enumerate(Iterators.cycle(data, epochs))
#     d_splat = d isa Tuple ? d : (d,)
#     rule = Mooncake.build_rrule(f, model.val, d_splat...)  # perhaps not ideal to do this inside the loop?

#     Mooncake.set_to_zero!!(model.dval)
#     l, _ = Mooncake.__value_and_gradient!!(rule, Mooncake.zero_codual(f), model, map(Mooncake.zero_codual, d_splat)...)

#     if !isfinite(l)
#       throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
#     end

#     Flux.update!(opt, model)

#     Flux.Train.@logprogress Base.haslength(data) ? i/(length(data)*epochs) : nothing
#   end
# end

### Model state & loading

Flux.state(x::Reactor) = Flux.state(x.model)

function Flux.loadmodel!(dst::Reactor, src::Reactor; kw...)
   Flux.loadmodel!(dst.model, src.model; kw...)
   dst
end
function Flux.loadmodel!(dst::Reactor, src; kw...)
    Flux.loadmodel!(dst.model, src; kw...)
    dst
end

### show

function Flux._show_pre_post(m::Reactor)
    # Forward
    if m.fwd_input === nothing
        post = "  # call not yet compiled\n"
    else
        # inp = Base.dims2string(m.input.size) * " " * string(m.input.size)
        inp = m.fwd_input
        n = m.fwd_count
        post = "  # compiled for $inp, run $n times\n"
    end

    # Gradient
    if isdefined(m, :gradient)
        nrm = Flux.norm(Optimisers.destructure(_grad_or_nothing(m))[1])
        str = repr(round(nrm; sigdigits=3))
        post *= "  # norm(∇) ≈ $str\n"
    else
        post *= "  # no shadow\n"
    end

    if m.grad_input === nothing
        post *= "  # gradient not yet compiled\n"
    else
        rev = m.grad_input
        n = m.grad_count
        post *= "  # ∇compiled for $rev, and run $n times\n"
    end

    pre = "Reactor("
    post *= ") "
    return pre, post
end

end  # module
