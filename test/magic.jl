import Fluxperimental: @compact

@testset "@compact" begin

  r = @compact(w = [1, 5, 10]) do x
    sum(w .* x)
  end
  @test Flux.params(r) == Flux.Params([[1, 5, 10]])
  @test r([1, 1, 1]) == 1 + 5 + 10
  @test r([1, 2, 3]) == 1 + 2 * 5 + 3 * 10
  @test r(ones(3, 3)) == 3 * (1 + 5 + 10)

  # Test gradients:
  @test gradient(r, [1, 1, 1])[1] == [1, 5, 10]

  d = @compact(in = 5, out = 7, W = randn(out, in), b = zeros(out), act = relu) do x
    y = W * x
    act.(y .+ b)
  end

  @test size.(Flux.params(d)) == [(7, 5), (7,)]

  @test size(d(ones(5, 10))) == (7, 10)
  @test all(d(randn(5, 10)) .>= 0)

  # Test gradients:
  y, ∇ = Flux.withgradient(Flux.params(d)) do
    input = randn(5, 32)
    desired_output = randn(7, 32)
    prediction = d(input)
    sum((prediction - desired_output) .^ 2)
  end
  @test typeof(y) == Float64
  grads = ∇.grads
  @test typeof(grads) <: IdDict
  @test length(grads) == 3
  @test Set(size.(values(grads))) == Set([(7, 5), (), (7,)])


  # MLP:
  n_in = 1
  n_out = 1
  nlayers = 3

  model = @compact(
    w1 = Dense(n_in, 128),
    w2 = [Dense(128, 128) for i = 1:nlayers],
    w3 = Dense(128, n_out),
    act = relu
  ) do x
    embed = act(w1(x))
    for w in w2
      embed = act(w(embed))
    end
    out = w3(embed)
    return out
  end

  @test size.(Flux.params(model)) == [
    (128, 1),
    (128,),
    (128, 128),
    (128,),
    (128, 128),
    (128,),
    (128, 128),
    (128,),
    (1, 128),
    (1,),
  ]
  @test size(model(randn(n_in, 32))) == (1, 32)

  # Test string representations:
  model = @compact(w=randn(32, 32)) do x, y
    tmp = sum(w .* x)
    return tmp + y
  end
  @test string(model) == """@compact(
  w = randn(32, 32),
) do x, y
    tmp = sum(w .* x)
    return tmp + y
end"""
  model = @compact(w=randn(32, 32), name="Linear(...)") do x, y
    tmp = sum(w .* x)
    return tmp + y
  end
  @test string(model) == "Linear(...)"

end
