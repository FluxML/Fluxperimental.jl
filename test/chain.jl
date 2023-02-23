
import Flux, Fluxperimental

@testset "Applying the Chain!" begin

  x = rand(Float32, 10, 1)
  l1 = Flux.Dense(10, 10)
  l2 = Flux.Dense(10, 1)
  truth = l2(l1(x))
  
  t_c = Flux.Chain(l1, l2) # tuple Chain
  new_t_c, out = Fluxperimental.apply(t_c, x)
  @test new_t_c[1] === l1 && new_t_c[2] === l2
  @test all(out .== truth)
  
  
  nt_c = Flux.Chain(l1=l1, l2=l2) # namedtuple Chain
  new_nt_c, out = Fluxperimental.apply(nt_c, x)
  @test new_nt_c[:l1] === l1 && new_nt_c[:l2] === l2
  @test all(out .== truth)

  
  v_c = Flux.Chain([l1, l2]) # vector Chain
  new_v_c, out = Fluxperimental.apply(nt_c, x)
  @test new_v_c.layers[1] === l1 && new_v_c.layers[2] === l2
  @test all(out .== truth)
  
    
end
