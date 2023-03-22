# Checking if the two grad structures are equal. Simplifies tests below.
function _grads_equal(grads1, grads2)
  if length(keys(grads1)) != length(keys(grads2))
    return false
  end
  ret = true
  for weights in keys(grads1)
    if grads1[weights] isa AbstractArray
      ret = ret && all(grads1[weights] .== grads2[weights])
    elseif isnothing(grads1[weights])
      ret = ret && isnothing(grads2[weights])
    else
      throw("Grad returned type $(typeof(grads1[weights]))")
    end
  end
  return ret
end

@testset "Applying the Chain!" begin
  @testset "Forward pass" begin
    x = rand(Float32, 3, 1)
    l1 = Flux.Dense(3, 4)
    l2 = Flux.Dense(4, 1)
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
    new_v_c, out = Fluxperimental.apply(v_c, x)
    @test new_v_c.layers[1] === l1 && new_v_c.layers[2] === l2
    @test all(out .== truth)
  end # @testset "Forward Pass"

  @testset "Backward pass" begin
    x = rand(Float32, 3, 1)
    l1 = Flux.Dense(3, 4)
    l2 = Flux.Dense(4, 1)
    
    @test begin # Test Tuple Chain Gradients
      t_c = Flux.Chain(l1, l2) # tuple Chain
      grads_truth = Flux.gradient(Flux.params(t_c)) do
        sum(t_c(x))
      end

      grads_tuple = Flux.gradient(Flux.params(t_c)) do
        sum(Fluxperimental.apply(t_c, x)[end])
      end
      
      _grads_equal(grads_tuple, grads_truth)
    end

    @test begin # Test Named Tuple's Gradients
      nt_c = Flux.Chain(l1=l1, l2=l2) # named tuple Chain
      grads_truth = Flux.gradient(Flux.params(nt_c)) do
        sum(nt_c(x))
      end

      grads_tuple = Flux.gradient(Flux.params(nt_c)) do
        sum(Fluxperimental.apply(nt_c, x)[end])
      end
      
      _grads_equal(grads_tuple, grads_truth)
    end
  end # @testset "Backward Pass"
end # @testset "Applying the Chain!"
