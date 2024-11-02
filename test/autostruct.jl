Fluxperimental.DEFINE |> empty!

@autostruct function New1(a::Int)
    A = Dense(a => 2a)
    New1(A)
end

(m::New1)(x) = one.(m.A(x))

@testset "simple case" begin
    m1 = New1(3)
    @test m1 isa New1
    @test Flux.state(m1).A.bias == zeros(Float32, 6)
    @test m1([1,2,3]) == ones(Float32, 6)
end

id1 = string(New1)  # something like "var\"##New1#265\""

@autostruct function New1(a::Int)  # re-definition of constructor, same struct!
    A = Dense(2a => a, tanh)
    New1(A)
end

(m::New1)(x) = one.(m.A(x)) .+ 2  # re-definition of forward pass, same struct!

@testset "re-defined" begin
    @test string(New1) == id1
    m2 = New1(2)
    @test m2 isa New1
    @test Flux.state(m2).A.bias == zeros(Float32, 2)
    @test m2([1,2,3,4]) == [3f0, 3f0]
end

@autostruct :expand function New1(a, b=3)  # new struct, both for :expand and for b argument
    A = Dense(a => b)
    New1(A::Dense)
end

@testset "new defn" begin
    @test string(New1) != id1
    m3 = New1(3)
    @test m3 isa New1
    @test Flux.state(m3).A.bias == zeros(Float32, 3)
    # pretty printing
    @test contains(repr("text/plain", m3), "New1(\n")
    @test contains(repr("text/plain", m3), "Dense(3 => 3)")
end

@autostruct One(field::Dense)

@testset "no-function" begin
    m1 = One(Dense(2=>3))
    @test Flux.state(m1).field.bias == zeros(Float32, 3)
    @test_throws MethodError One(3)
    # pretty printing
    @test contains(repr("text/plain", m1), "One(\n")
    @test contains(repr("text/plain", m1), "Dense(2 => 3)")
end
