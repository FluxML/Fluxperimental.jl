using Flux, Fluxperimental, Mooncake

@testset "gradient, withgradient, Moonduo" begin
    # Tests above are about how Enzyme digests Flux layers.
    # Tests here are just the interface Flux.gradient(f, Moonduo(model)) etc.
    m1 = Moonduo(Dense(3=>2))
    @test m1 isa Moonduo
    g1 = Flux.gradient(m -> sum(m.bias), m1) |> only
    @test iszero(g1.weight)
    @test g1.bias == [1, 1]
    @test m1.dval.fields.bias == [1, 1]

    g2 = Flux.withgradient((m,x) -> sum(m(x)), m1, Moonduo([1,2,3f0]))  # would prefer Const
    @test g2.val ≈ sum(m1([1,2,3f0]))
    @test g2.grad[1].weight ≈ [1 2 3; 1 2 3]
    @test_skip g2.grad[2] === nothing  # implicitly Const

    # g3 = Flux.withgradient(Moonduo([1,2,4.], zeros(3))) do x
    #           z = 1 ./ x
    #           sum(z), z  # here z is an auxillary output
    #        end
    # @test g3.grad[1] ≈ [-1.0, -0.25, -0.0625]
    # @test g3.val[1] ≈ 1.75
    # @test g3.val[2] ≈ [1.0, 0.5, 0.25]
    # g4 = Flux.withgradient(Moonduo([1,2,4.], zeros(3))) do x
    #           z = 1 ./ x
    #           (loss=sum(z), aux=string(z))
    #        end
    # @test g4.grad[1] ≈ [-1.0, -0.25, -0.0625]
    # @test g4.val.loss ≈ 1.75
    # @test g4.val.aux == "[1.0, 0.5, 0.25]"

    # setup understands Moonduo:
    @test Flux.setup(Adam(), m1) == Flux.setup(Adam(), m1.val)

    # # At least one Moonduo is required:
    # @test_throws ArgumentError Flux.gradient(m -> sum(m.bias), Const(m1.val))
    # @test_throws ArgumentError Flux.gradient((m,x) -> sum(m(x)), Const(m1.val), [1,2,3f0])
    # @test_throws ArgumentError Flux.withgradient(m -> sum(m.bias), Const(m1.val))
    # @test_throws ArgumentError Flux.withgradient((m,x) -> sum(m(x)), Const(m1.val), [1,2,3f0])
    # # Active is disallowed:
    # @test_throws ArgumentError Flux.gradient((m,z) -> sum(m.bias)/z, m1, Active(3f0))
    # @test_throws ArgumentError Flux.gradient((m,z) -> sum(m.bias)/z, m1.val, Active(3f0))
    # @test_throws ArgumentError Flux.gradient((m,z) -> sum(m.bias)/z, Const(m1.val), Active(3f0))
end
