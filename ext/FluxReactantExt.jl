module FluxReactantExt

using Flux, Fluxperimental, Reactant, Enzyme
import Fluxperimental: Fluxactor

# mutable struct Fluxactor{M}
#   model::M
#   fwd_compiled
#   fwd_input
#   gradient::M
#   grad_compiled
#   grad_input
# end

"""

# Example
```julia
julia> using Flux, Fluxperimental, Reactant, Enzyme

julia> img = rand32(28, 28, 1, 128);

julia> loss(m, x) = sum(abs2, m(x));

julia> mlp = Chain(Flux.flatten, Dense(28^2 => 32, tanh), Dense(32 => 10));

julia> mlp(img)[1:3]  # plain Julia
3-element Vector{Float32}:
  0.22694848
 -0.72605485
  0.57976365

julia> re_mlp = Fluxactor(mlp);  # uses Reactant

julia> re_mlp(img)[1:3]
┌ Info: compiling...
└   summary(xr) = "28×28×1×128 ConcreteRArray{Float32, 4}"
3-element ConcreteRArray{Float32, 1}:
  0.22694828
 -0.72605544
  0.57976323

julia> re_mlp  # after forward but not yet gradient
Fluxactor(
  Chain(
    Flux.flatten,
    Dense(784 => 32, tanh),             # 25_120 parameters
    Dense(32 => 10),                    # 330 parameters
  ),
  # compiled for 28×28×1×128 ConcreteRArray{Float32, 4}
  # norm(∇) ≈ 0.0f0
  # ∇compiled for nothing
)         # Total: 4 trainable arrays, 25_450 parameters,
          # plus 4 non-trainable, 25_450 parameters, summarysize 689 bytes.

julia> Flux.gradient(loss, mlp, img)[1].layers[2].bias[1:3]  # uses Zygote
3-element Vector{Float32}:
   90.490005
 -208.77806
   28.711397

julia> Flux.gradient(loss, Duplicated(mlp), Const(img))[1].layers[2].bias[1:3]  # uses Enzyme
3-element Vector{Float32}:
   90.490005
 -208.77806
   28.711397

julia> Flux.gradient(loss, re_mlp, Const(img))[1].layers[2].bias[1:3]  # uses Reactant
[ Info: compiling gradient(loss, ::Fluxactor)...
ERROR: "Unhandled type Type"
Stacktrace:
  [1] traced_type(::Type{Type}, seen::Tuple{Val{Core.TypeName}, Val{Core.TypeName}}, mode::Val{Reactant.ConcreteToTraced})
    @ Reactant ~/.julia/packages/Reactant/m1CaM/src/Tracing.jl:104
  [2] traced_type(::Type{Core.TypeName}, seen::Tuple{}, mode::Val{Reactant.ConcreteToTraced})
    @ Reactant ~/.julia/packages/Reactant/m1CaM/src/Tracing.jl:132
  [3] make_tracer(seen::Reactant.OrderedIdDict{…}, prev::Core.TypeName, path::Any, mode::Reactant.TraceMode; toscalar::Bool, tobatch::Nothing, kwargs::@Kwargs{})
    @ Reactant ~/.julia/packages/Reactant/m1CaM/src/Tracing.jl:242
  [4] make_tracer(seen::Reactant.OrderedIdDict{…}, prev::Type{…}, path::Any, mode::Reactant.TraceMode; toscalar::Bool, tobatch::Nothing, kwargs::@Kwargs{})
    @ Reactant ~/.julia/packages/Reactant/m1CaM/src/Tracing.jl:254
  [5] make_tracer(seen::Reactant.OrderedIdDict{…}, prev::TypeVar, path::Any, mode::Reactant.TraceMode; toscalar::Bool, tobatch::Nothing, kwargs::@Kwargs{})
    @ Reactant ~/.julia/packages/Reactant/m1CaM/src/Tracing.jl:254
  [6] make_tracer(seen::Reactant.OrderedIdDict{…}, prev::Type{…}, path::Any, mode::Reactant.TraceMode; toscalar::Bool, tobatch::Nothing, kwargs::@Kwargs{…})
    @ Reactant ~/.julia/packages/Reactant/m1CaM/src/Tracing.jl:277
  [7] (::Reactant.var"#21#31"{Bool, Bool, Tuple{…}, Bool, Reactant.OrderedIdDict{…}})(i::Int64)
    @ Reactant ~/.julia/packages/Reactant/m1CaM/src/utils.jl:69
  [8] ntuple
    @ ./ntuple.jl:19 [inlined]
  [9] make_mlir_fn(f::Function, args::Tuple{…}, kwargs::Tuple{}, name::String, concretein::Bool; toscalar::Bool, return_dialect::Symbol, no_args_in_result::Bool, construct_function_without_args::Bool, do_transpose::Bool)
    @ Reactant ~/.julia/packages/Reactant/m1CaM/src/utils.jl:68
 [10] make_mlir_fn
    @ ~/.julia/packages/Reactant/m1CaM/src/utils.jl:36 [inlined]
 [11] #10
    @ ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:283 [inlined]
 [12] block!(f::Reactant.Compiler.var"#10#15"{typeof(autodiff), Tuple{…}}, blk::Reactant.MLIR.IR.Block)
    @ Reactant.MLIR.IR ~/.julia/packages/Reactant/m1CaM/src/mlir/IR/Block.jl:201
 [13] #9
    @ ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:282 [inlined]
 [14] mmodule!(f::Reactant.Compiler.var"#9#14"{…}, blk::Reactant.MLIR.IR.Module)
    @ Reactant.MLIR.IR ~/.julia/packages/Reactant/m1CaM/src/mlir/IR/Module.jl:93
 [15] compile_mlir!(mod::Reactant.MLIR.IR.Module, f::Function, args::Tuple{…}; optimize::Bool)
    @ Reactant.Compiler ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:279
 [16] compile_mlir!
    @ ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:278 [inlined]
 [17] (::Reactant.Compiler.var"#34#36"{Bool, typeof(autodiff), Tuple{…}})()
    @ Reactant.Compiler ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:726
 [18] context!(f::Reactant.Compiler.var"#34#36"{Bool, typeof(autodiff), Tuple{…}}, ctx::Reactant.MLIR.IR.Context)
    @ Reactant.MLIR.IR ~/.julia/packages/Reactant/m1CaM/src/mlir/IR/Context.jl:76
 [19] compile_xla(f::Function, args::Tuple{…}; client::Nothing, optimize::Bool)
    @ Reactant.Compiler ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:723
 [20] compile_xla
    @ ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:718 [inlined]
 [21] compile(f::Function, args::Tuple{…}; client::Nothing, optimize::Bool, sync::Bool)
    @ Reactant.Compiler ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:750
 [22] macro expansion
    @ ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:485 [inlined]
 [23] gradient(f::Function, m::Fluxactor{Chain{Tuple{…}}}, xs::Const{Array{Float32, 4}})
    @ Main ./REPL[91]:11
 [24] top-level scope
    @ REPL[97]:1

later...
julia> Flux.gradient(loss, re_mlp, Const(img))[1].layers[2].bias[1:3]  # uses Reactant
[ Info: compiling gradient(loss, ::Fluxactor)...
ERROR: type TypeVar has no field data
Stacktrace:
 [1] getproperty(x::TypeVar, f::Symbol)
   @ Base ./Base.jl:37
 [2] macro expansion
   @ ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:799 [inlined]
 [3] (::Reactant.Compiler.Thunk{…})(::ReverseMode{…}, ::typeof(loss), ::Type{…}, ::Duplicated{…}, ::Const{…})
   @ Reactant.Compiler ~/.julia/packages/Reactant/m1CaM/src/Compiler.jl:815
 [4] gradient(f::Function, m::Fluxactor{Chain{Tuple{…}}}, xs::Const{Array{Float32, 4}})
   @ FluxReactantExt ~/.julia/dev/Fluxperimental/ext/FluxReactantExt.jl:182
 [5] top-level scope
   @ REPL[15]:1
Some type information was truncated. Use `show(err)` to see complete types.
```

"""
function Fluxactor(model)
    mr = Reactant.to_rarray(model)
    gr = Reactant.to_rarray(Enzyme.make_zero(model))  # the other way isn't zero!
    Fluxactor(mr, nothing, nothing, gr, nothing, nothing)
end

### forward

function (m::Fluxactor)(x::AbstractArray)
    Flux.Zygote.isderiving() && error("can't use Fluxactor within Zygote!")
    xr = Reactant.to_rarray(x)
    input = _input_summary(xr)
    if input == m.fwd_input
        return m.fwd_compiled(xr)
    else
        @info "compiling..." summary(xr)
        fun = @compile m.model(xr)
        m.fwd_compiled = fun
        m.fwd_input = input
        return fun(xr)
    end
end

# _input_summary(x::AbstractArray{T}) where T = (T, size(x...))
# Just use strings for now, although probably change later for performance:
_input_summary(x::AbstractArray) = summary(x)
_input_summary(xs::Const...) = map(x -> _input_summary(x.val), xs)
_input_summary(f, xs::Const...) = join((string(f), _input_summary(xs...)...), ", ")

### gradient

"""
    Flux.gradient(loss::Function, model::Fluxactor, args::Const...)

This exact signature uses Reactant to compile the Enzyme gradient call.
"""
function Flux.gradient(f::Function, m::Fluxactor, xs::Const...)
    xrs = Reactant.to_rarray(xs)
    input = _input_summary(f, xrs...)
    dup = Duplicated(m.model, m.gradient)
    _seed = Ref(0f0), Ref(1f0)  # MethodError: no method matching Float32(::Reactant.TracedRNumber{Float32})
    _seed = ([0f0], [1f0]) |> Reactant.to_rarray
    seed = Duplicated(_seed...)
    if false
        # Enzyme.autodiff(Reverse, f, Active, dup, xrs...)  # just for testing, gives zero
        Enzyme.autodiff(Reverse, Const(_fun!), seed, Const(f), dup, xrs...)  # just for testing, gives zero
    elseif input == m.grad_input
        # m.grad_compiled(Reverse, f, Active, dup, xrs...)
        m.grad_compiled(Reverse, Const(_fun!), seed, Const(f), dup, xrs...)
    else
        @info "compiling gradient($f, ::Fluxactor)..."
        # fun = @compile Enzyme.autodiff(Reverse, f, Active, dup, xrs...)  # this gives ERROR: "Unhandled type Type" above
        fun = @compile Enzyme.autodiff(Reverse, Const(_fun!), seed, Const(f), dup, xrs...)  # this gives ERROR: type TypeVar has no field data
        m.grad_compiled = fun
        m.grad_input = _input_summary(f, xrs...)
        fun(Reverse, f, Active, dup, xrs...)
    end
    map(_grad_or_nothing, (dup, xrs...))
end

@inline _fun!(out, f::F, args...) where F = begin out[] = f(args...); nothing end

# This function strips the returned gradient to be Zygote-like:
_grad_or_nothing(dup::Duplicated) = Flux.fmapstructure(_grad_or_nothing, dup.dval; prune=nothing)
_grad_or_nothing(::Const) = nothing
_grad_or_nothing(x) = Optimisers.isnumeric(x) ? x : nothing

### Optimisers etc.

Flux.setup(rule::Optimisers.AbstractRule, m::Fluxactor) = Flux.setup(rule, m.model)

function Flux.update!(opt_state, m::Fluxactor)
  Flux.update!(opt_state, m.model, _grad_or_nothing(m))
  nothing
end

### Flux.Train, for train!

# _applyloss(loss, model, d...) = loss(model, d...)
#
# """
#     train!(loss, Fluxactor(model), data, opt_state)

# This method uses ... instead of Zygote.jl to compute the gradients, but is otherwise the
# same as `Flux.train!(loss, model, data, opt_state)`.
# """
# function Flux.train!(loss, model::Fluxactor, data, opt; cb=nothing, epochs::Int=1)
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

Flux.state(x::Fluxactor) = Flux.state(x.model)

function Flux.loadmodel!(dst::Fluxactor, src::Fluxactor; kw...)
   Flux.loadmodel!(dst.model, src.model; kw...)
   dst
end
function Flux.loadmodel!(dst::Fluxactor, src; kw...)
    Flux.loadmodel!(dst.model, src; kw...)
    dst
end

### show

function Flux._show_pre_post(m::Fluxactor)
    # inp = Base.dims2string(m.input.size) * " " * string(m.input.size)
    inp = m.fwd_input

    nrm = Flux.norm(Optimisers.destructure(_grad_or_nothing(m))[1])
    str = repr(round(nrm; sigdigits=3))

    rev = m.grad_input

    # "Fluxactor(", "  # compiled for $inp\n  # norm(∇) ≈ $str\n) "
    "Fluxactor(", "  # compiled for $inp\n  # norm(∇) ≈ $str\n  # ∇compiled for $rev\n) "
end

end  # module
