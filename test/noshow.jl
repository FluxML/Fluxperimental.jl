
@testset "NoShow" begin
  d23 = Dense(2 => 3)
  d34 = Dense(3 => 4, tanh)
  d35 = Dense(3 => 5, relu)
  d910 = Dense(9 => 10)

  model = Chain(d23, Parallel(vcat, d34, d35), d910)
  m_no = Chain(d23, NoShow(Parallel(vcat, d34, NoShow("zzz", d35))), d910)

  @test sum(length, Flux.params(model)) == sum(length, Flux.params(m_no))

  xin = randn(Float32, 2, 7)
  @test model(xin) ≈ m_no(xin)

  # gradients
  grad = gradient(m -> m(xin)[1], model)[1]
  g_no = gradient(m -> m(xin)[1], m_no)[1]

  @test grad.layers[2].layers[1].bias ≈ g_no.layers[2].layer.layers[1].bias
  @test grad.layers[2].layers[2].bias ≈ g_no.layers[2].layer.layers[2].layer.bias

  # printing -- see also compact.jl for another test
  @test !contains(string(model), "NoShow(...)")
  @test contains(string(m_no), "NoShow(...)")
  @test !contains(string(m_no), "3 => 4")
end

