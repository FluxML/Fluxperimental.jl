using Flux, Fluxperimental, Reactant, Enzyme
using Test
@testset "Reactant + Flux" begin

@testset "simple forwards" begin
    img = rand32(28, 28, 1, 2)
    mlp = Chain(Flux.flatten, Dense(28^2 => 32, tanh), Dense(32 => 10))
    y1 = mlp(img)
    @test y1 isa Matrix

    re_mlp = Reactor(mlp)  # signal to use Reactant
    y2 = re_mlp(img)
    @test y2 isa ConcreteRArray
    @test y1 ≈ Array(y2)

    y3 = re_mlp(img)
    @test y1 ≈ Array(y3)
    @test re_mlp.fwd_count == 2  # re-used without recompilation

    img10 = rand32(28, 28, 1, 10)
    y10 = mlp(img10)
    y11 = re_mlp(img10)  # re-compiles for the new size
    @test y10 ≈ Array(y11)
end

@testset "simple gradient" begin
    img = rand32(28, 28, 1, 2)
    mlp = Chain(Flux.flatten, Dense(28^2 => 32, tanh), Dense(32 => 10))
    loss1(m, x) = sum(abs2, m(x))

    g1 = Flux.gradient(loss1, mlp, img)[1].layers[2].bias
    @test g1 isa Vector

    re_mlp = Reactor(mlp)
    dup_mlp = Duplicated(mlp);
    g2 = Flux.gradient(loss1, dup_mlp, Const(img))[1].layers[2].bias  # Enzyme
    @test g2 ≈ g1
    @test g2 isa Vector

    re_mlp = Reactor(mlp);
    g3 = Flux.gradient(loss1, re_mlp, Const(img))[1].layers[2].bias
    @test Array(g3) ≈ g1
    g4 = Flux.gradient(loss1, re_mlp, Const(img))[1].layers[2].bias
    @test Array(g4) ≈ g1
    @test re_mlp.grad_count == 2  # re-used without recompilation
end

#=

simple gradient: Error During Test at REPL[59]:1
  Got exception outside of a @test
  Constant memory is stored (or returned) to a differentiable variable.
  As a result, Enzyme cannot provably ensure correctness and throws this error.
  This might be due to the use of a constant variable as temporary storage for active memory (https://enzyme.mit.edu/julia/stable/faq/#Runtime-Activity).
  If Enzyme should be able to prove this use non-differentable, open an issue!
  To work around this issue, either:
   a) rewrite this variable to not be conditionally active (fastest, but requires a code change), or
   b) set the Enzyme mode to turn on runtime activity (e.g. autodiff(set_runtime_activity(Reverse), ...) ). This will maintain correctness, but may slightly reduce performance.
  Mismatched activity for:   store i8* %17, i8* addrspace(11)* %.repack, align 8, !dbg !165, !tbaa !118, !alias.scope !121, !noalias !166 const val:   %17 = load i8*, i8* addrspace(11)* %16, align 8, !dbg !112, !tbaa !118, !alias.scope !121, !noalias !122, !enzyme_type !123, !enzymejl_byref_BITS_VALUE !0, !enzymejl_source_type_Ptr\7BFloat32\7D !0
   value=Unknown object of type Ptr{Float32}
   llvalue=  %17 = load i8*, i8* addrspace(11)* %16, align 8, !dbg !112, !tbaa !118, !alias.scope !121, !noalias !122, !enzyme_type !123, !enzymejl_byref_BITS_VALUE !0, !enzymejl_source_type_Ptr\7BFloat32\7D !0

  Stacktrace:
   [1] reshape
     @ ./reshapedarray.jl:60
   [2] reshape
     @ ./reshapedarray.jl:129
   [3] reshape
     @ ./reshapedarray.jl:128
   [4] flatten
     @ ~/.julia/packages/MLUtils/LmmaQ/src/utils.jl:504
   [5] flatten
     @ ~/.julia/dev/Flux/src/layers/stateless.jl:105
   [6] macro expansion
     @ ~/.julia/dev/Flux/src/layers/basic.jl:68
   [7] _applychain
     @ ~/.julia/dev/Flux/src/layers/basic.jl:68

  Stacktrace:
    [1] reshape
      @ ./reshapedarray.jl:60 [inlined]
    [2] reshape
      @ ./reshapedarray.jl:129 [inlined]
    [3] reshape
      @ ./reshapedarray.jl:128 [inlined]
    [4] flatten
      @ ~/.julia/packages/MLUtils/LmmaQ/src/utils.jl:504 [inlined]
    [5] flatten
      @ ~/.julia/dev/Flux/src/layers/stateless.jl:105 [inlined]
    [6] macro expansion
      @ ~/.julia/dev/Flux/src/layers/basic.jl:68 [inlined]
    [7] _applychain
      @ ~/.julia/dev/Flux/src/layers/basic.jl:68
    [8] Chain
      @ ~/.julia/dev/Flux/src/layers/basic.jl:65 [inlined]
    [9] loss1
      @ ./REPL[59]:4 [inlined]
   [10] loss1
      @ ./REPL[59]:0 [inlined]
   [11] diffejulia_loss1_99632_inner_54wrap
      @ ./REPL[59]:0
   [12] macro expansion
      @ ~/.julia/packages/Enzyme/haqjK/src/compiler.jl:5204 [inlined]
   [13] enzyme_call
      @ ~/.julia/packages/Enzyme/haqjK/src/compiler.jl:4750 [inlined]
   [14] CombinedAdjointThunk
      @ ~/.julia/packages/Enzyme/haqjK/src/compiler.jl:4622 [inlined]
   [15] autodiff(::ReverseMode{false, false, FFIABI, false, false}, ::Const{var"#loss1#11"}, ::Type{Active}, ::Duplicated{Chain{Tuple{typeof(Flux.flatten), Dense{typeof(tanh), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}}, ::Const{Array{Float32, 4}})
      @ Enzyme ~/.julia/packages/Enzyme/haqjK/src/Enzyme.jl:503
   [16] _enzyme_gradient(::Function, ::Duplicated{Chain{Tuple{typeof(Flux.flatten), Dense{typeof(tanh), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}}, ::Vararg{Union{Const, Duplicated}}; zero::Bool)
      @ FluxEnzymeExt ~/.julia/dev/Flux/ext/FluxEnzymeExt/FluxEnzymeExt.jl:49
   [17] _enzyme_gradient
      @ ~/.julia/dev/Flux/ext/FluxEnzymeExt/FluxEnzymeExt.jl:44 [inlined]
   [18] gradient(::Function, ::Duplicated{Chain{Tuple{typeof(Flux.flatten), Dense{typeof(tanh), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}}, ::Const{Array{Float32, 4}})
      @ Flux ~/.julia/dev/Flux/src/gradient.jl:122
   [19] macro expansion
      @ REPL[59]:11 [inlined]
   [20] macro expansion
      @ /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/Test/src/Test.jl:1700 [inlined]
   [21] top-level scope
      @ REPL[59]:2
   [22] eval
      @ ./boot.jl:430 [inlined]
   [23] eval_user_input(ast::Any, backend::REPL.REPLBackend, mod::Module)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:226
   [24] repl_backend_loop(backend::REPL.REPLBackend, get_module::Function)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:323
   [25] start_repl_backend(backend::REPL.REPLBackend, consumer::Any; get_module::Function)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:308
   [26] run_repl(repl::REPL.AbstractREPL, consumer::Any; backend_on_current_task::Bool, backend::Any)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:464
   [27] run_repl(repl::REPL.AbstractREPL, consumer::Any)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:450
   [28] (::Base.var"#1138#1140"{Bool, Symbol, Bool})(REPL::Module)
      @ Base ./client.jl:446
   [29] #invokelatest#2
      @ ./essentials.jl:1054 [inlined]
   [30] invokelatest
      @ ./essentials.jl:1051 [inlined]
   [31] run_main_repl(interactive::Bool, quiet::Bool, banner::Symbol, history_file::Bool, color_set::Bool)
      @ Base ./client.jl:430
   [32] repl_main
      @ ./client.jl:567 [inlined]
   [33] _start()
      @ Base ./client.jl:541
Test Summary:   | Pass  Error  Total  Time
simple gradient |    1      1      2  1.3s
ERROR: Some tests did not pass: 1 passed, 0 failed, 1 errored, 0 broken.

=#

@testset "simple train!" begin
    X = repeat(hcat(digits.(0:3, base=2, pad=2)...), 1, 32)
    Y = Flux.onehotbatch(xor.(eachrow(X)...), 0:1)
    # data = Flux.DataLoader((X, Y); batchsize=16, shuffle=true)
    data = Flux.DataLoader((X .+ 0f0, Y .+ 0f0); batchsize=16, shuffle=true)  # this avoids some erros from conversion

    model = Chain(Dense(2 => 3, sigmoid), BatchNorm(3), Dense(3 => 2)) |> Reactor
    state = Flux.setup(Adam(0.1, (0.7, 0.95)), model)  # Note that I'm doing this after |> Reactor, ideally before would work too?

    Flux.train!(model, data, state; epochs=100) do m, x, y
        Flux.logitcrossentropy(m(x), y)
    end

    @test all((softmax(model(X)) .> 0.5) .== Y)
end

#=

[ Info: compiling
simple train!: Error During Test at REPL[57]:1
  Got exception outside of a @test
  type Array has no field data
  Stacktrace:
    [1] getproperty
      @ ./Base.jl:49 [inlined]
    [2] macro expansion
      @ ~/.julia/packages/Reactant/sIJRJ/src/Compiler.jl:771 [inlined]
    [3] (::Reactant.Compiler.Thunk{Symbol("##_step!_reactant#1017757")})(::var"#9#10", ::Duplicated{ConcreteRArray{Float32, 1}}, ::Chain{Tuple{Dense{typeof(σ), ConcreteRArray{Float32, 2}, ConcreteRArray{Float32, 1}}, BatchNorm{typeof(identity), ConcreteRArray{Float32, 1}, Float32, ConcreteRArray{Float32, 1}}, Dense{typeof(identity), ConcreteRArray{Float32, 2}, ConcreteRArray{Float32, 1}}}}, ::Tuple{Matrix{Float32}, Matrix{Float32}}, ::@NamedTuple{layers::Tuple{@NamedTuple{weight::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 2}, ConcreteRArray{Float32, 2}, Tuple{Float32, Float32}}}, bias::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 1}, ConcreteRArray{Float32, 1}, Tuple{Float32, Float32}}}, σ::Tuple{}}, @NamedTuple{λ::Tuple{}, β::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 1}, ConcreteRArray{Float32, 1}, Tuple{Float32, Float32}}}, γ::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 1}, ConcreteRArray{Float32, 1}, Tuple{Float32, Float32}}}, μ::Tuple{}, σ²::Tuple{}, ϵ::Tuple{}, momentum::Tuple{}, affine::Tuple{}, track_stats::Tuple{}, active::Tuple{}, chs::Tuple{}}, @NamedTuple{weight::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 2}, ConcreteRArray{Float32, 2}, Tuple{Float32, Float32}}}, bias::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 1}, ConcreteRArray{Float32, 1}, Tuple{Float32, Float32}}}, σ::Tuple{}}}})
      @ Reactant.Compiler ~/.julia/packages/Reactant/sIJRJ/src/Compiler.jl:787
    [4] macro expansion
      @ ~/.julia/dev/Fluxperimental/ext/FluxReactantExt.jl:332 [inlined]
    [5] macro expansion
      @ ~/.julia/packages/ProgressLogging/6KXlp/src/ProgressLogging.jl:328 [inlined]
    [6] train!(loss::Function, m::Reactor{Chain{Tuple{Dense{typeof(σ), ConcreteRArray{Float32, 2}, ConcreteRArray{Float32, 1}}, BatchNorm{typeof(identity), ConcreteRArray{Float32, 1}, Float32, ConcreteRArray{Float32, 1}}, Dense{typeof(identity), ConcreteRArray{Float32, 2}, ConcreteRArray{Float32, 1}}}}}, data::MLUtils.DataLoader{Tuple{Matrix{Float32}, Matrix{Float32}}, Random.TaskLocalRNG, Val{nothing}}, opt_state::@NamedTuple{layers::Tuple{@NamedTuple{weight::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 2}, ConcreteRArray{Float32, 2}, Tuple{Float32, Float32}}}, bias::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 1}, ConcreteRArray{Float32, 1}, Tuple{Float32, Float32}}}, σ::Tuple{}}, @NamedTuple{λ::Tuple{}, β::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 1}, ConcreteRArray{Float32, 1}, Tuple{Float32, Float32}}}, γ::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 1}, ConcreteRArray{Float32, 1}, Tuple{Float32, Float32}}}, μ::Tuple{}, σ²::Tuple{}, ϵ::Tuple{}, momentum::Tuple{}, affine::Tuple{}, track_stats::Tuple{}, active::Tuple{}, chs::Tuple{}}, @NamedTuple{weight::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 2}, ConcreteRArray{Float32, 2}, Tuple{Float32, Float32}}}, bias::Optimisers.Leaf{Adam, Tuple{ConcreteRArray{Float32, 1}, ConcreteRArray{Float32, 1}, Tuple{Float32, Float32}}}, σ::Tuple{}}}}; epochs::Int64)
      @ FluxReactantExt ~/.julia/dev/Fluxperimental/ext/FluxReactantExt.jl:324
    [7] macro expansion
      @ REPL[57]:10 [inlined]
    [8] macro expansion
      @ /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/Test/src/Test.jl:1700 [inlined]
    [9] top-level scope
      @ REPL[57]:2
   [10] eval
      @ ./boot.jl:430 [inlined]
   [11] eval_user_input(ast::Any, backend::REPL.REPLBackend, mod::Module)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:226
   [12] repl_backend_loop(backend::REPL.REPLBackend, get_module::Function)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:323
   [13] start_repl_backend(backend::REPL.REPLBackend, consumer::Any; get_module::Function)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:308
   [14] run_repl(repl::REPL.AbstractREPL, consumer::Any; backend_on_current_task::Bool, backend::Any)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:464
   [15] run_repl(repl::REPL.AbstractREPL, consumer::Any)
      @ REPL /Applications/Julia-1.11.app/Contents/Resources/julia/share/julia/stdlib/v1.11/REPL/src/REPL.jl:450
   [16] (::Base.var"#1138#1140"{Bool, Symbol, Bool})(REPL::Module)
      @ Base ./client.jl:446
   [17] #invokelatest#2
      @ ./essentials.jl:1054 [inlined]
   [18] invokelatest
      @ ./essentials.jl:1051 [inlined]
   [19] run_main_repl(interactive::Bool, quiet::Bool, banner::Symbol, history_file::Bool, color_set::Bool)
      @ Base ./client.jl:430
   [20] repl_main
      @ ./client.jl:567 [inlined]
   [21] _start()
      @ Base ./client.jl:541
Test Summary: | Error  Total   Time
simple train! |     1      1  14.3s
ERROR: Some tests did not pass: 0 passed, 0 failed, 1 errored, 0 broken.

(jl_smIYmq) pkg> st
Status `/private/var/folders/yq/4p2zwd614y59gszh7y9ypyhh0000gn/T/jl_smIYmq/Project.toml`
  [587475ba] Flux v0.16.1 `~/.julia/dev/Flux`
  [3102ee7a] Fluxperimental v0.2.3 `~/.julia/dev/Fluxperimental`
  [3c362404] Reactant v0.2.10

=#

end  # @testset "Reactant + Flux"
