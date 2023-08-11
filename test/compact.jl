import Fluxperimental: @compact, CompactLayer

# Strip both strings of spaces, and then test:
function similar_strings(s1, s2)
  s1 = replace(s1, r"\s" => "")
  s2 = replace(s2, r"\s" => "")

  # We also remove any instances of, e.g.,
  # 17.057 KiB (or any other number)
  # because this depends on indentation in this file.
  s1 = replace(s1, r"\d+\.\d+KiB" => "")
  s2 = replace(s2, r"\d+\.\d+KiB" => "")

  # Display any differences:
  if s1 != s2
    println(stderr, "s1: ", s1)
    println(stderr, "s2: ", s2)
  end
  return s1 == s2
end

function get_model_string(model)
  io = IOBuffer()
  show(io, MIME"text/plain"(), model)
  String(take!(io))
end

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
  model = @compact(w=Dense(32 => 32)) do x, y
    tmp = sum(w(x))
    return tmp + y
  end
  expected_string = """@compact(
    w = Dense(32=>32), #1_056 parameters
  ) do x, y
    tmp = sum(w(x))
    return tmp + y
  end"""
  @test similar_strings(get_model_string(model), expected_string)

  # Custom naming:
  model = @compact(w=Dense(32, 32), name="Linear(...)") do x, y
    tmp = sum(w(x))
    return tmp + y
  end
  expected_string = "Linear(...)         # 1_056 parameters"
  @test similar_strings(get_model_string(model), expected_string)

  # Hierarchical models should work too:
  model1 = @compact(w1=Dense(32=>32, relu), w2=Dense(32=>32, relu)) do x
    w2(w1(x))
  end
  model2 = @compact(w1=model1, w2=Dense(32=>32, relu)) do x
    w2(w1(x))
  end
  expected_string = """@compact(
    w1 = @compact(
      w1 = Dense(32 => 32, relu),         # 1_056 parameters
      w2 = Dense(32 => 32, relu),         # 1_056 parameters
    ) do x
      w2(w1(x))
  end,
    w2 = Dense(32 => 32, relu),           # 1_056 parameters
  ) do x
      w2(w1(x))
  end                  # Total: 6 arrays, 3_168 parameters, 13.271 KiB."""
  @test similar_strings(get_model_string(model2), expected_string)

  # With array params:
  model = @compact(x=randn(32), w=Dense(32=>32)) do s
    w(x .* s)
  end
  expected_string = """@compact(
    x = randn(32),                        # 32 parameters
    w = Dense(32 => 32),                  # 1_056 parameters
  ) do s 
    w(x .* s)
  end                  # Total: 3 arrays, 1_088 parameters, 4.734 KiB."""
  @test similar_strings(get_model_string(model), expected_string)

  # Hierarchy with inner model named:
  model = @compact(
    w1=@compact(w1=randn(32, 32), name="Model(32)") do x
      w1 * x
    end,
    w2=randn(32, 32),
    w3=randn(32),
  ) do x
    w2 * w1(x)
  end
  expected_string = """@compact(
    Model(32),                            # 1_024 parameters
    w2 = randn(32, 32),                   # 1_024 parameters
    w3 = randn(32),                       # 32 parameters
  ) do x 
      w2 * w1(x)
  end                  # Total: 3 arrays, 2_080 parameters, 17.089 KiB."""
  @test similar_strings(get_model_string(model), expected_string)

  # Hierarchy with outer model named:
  model = @compact(
    w1=@compact(w1=randn(32, 32)) do x
      w1 * x
    end,
    w2=randn(32, 32),
    w3=randn(32),
    name="Model(32)"
  ) do x
    w2 * w1(x)
  end
  expected_string = """Model(32)                  # Total: 3 arrays, 2_080 parameters, 17.057KiB."""
  @test similar_strings(get_model_string(model), expected_string)

end

@testset "Dispatch using symbols" begin
  model1 = @compact(W=randn(32)) do x
    W .* x
  end
  model2 = @compact(MyCustomLayer, W=randn(32)) do x
    W .* x
  end
  @eval my_custom_function(::CompactLayer{:Default}) = :Default
  @eval my_custom_function(::CompactLayer{:MyCustomLayer}) = :MyCustomLayer

  @test model1 isa CompactLayer{:Default}
  @test model2 isa CompactLayer{:MyCustomLayer}
  @test my_custom_function(model1) == :Default
  @test my_custom_function(model2) == :MyCustomLayer

  expected_string2 = """@compact(
    MyCustomLayer,
    W = randn(32),                        # 32 parameters
  ) do x 
      W .* x
  end"""
  @test similar_strings(get_model_string(model2), expected_string2)

  @testset "Nested symbolic layers" begin
    num_features = 1
    num_out = 2
    d_attn = 4
    d_value = 12
    num_heads = 3

    model3 = @compact(
        SelfAttention,
        out = Dense(num_heads * d_value => num_out),
        heads = [
          @compact(
            Head,
            K = Dense(num_features => d_attn),
            V = Dense(num_features => d_value),
            Q = Dense(num_features => d_attn)
          ) do x
            k, v, q = K(x), V(x), Q(x)
            x = sum(k .* q; dims=1) ./ sqrt(d_attn)
            softmax(x; dims=2) .* v
          end for _ in 1:num_heads
        ]
    ) do x
        out(vcat([h(x) for h in heads]...))
    end
    @test model3 isa CompactLayer{:SelfAttention}
    @test all(t -> isa(t, CompactLayer{:Head}), model3.variables.heads)

    expected_string3 = """@compact(
      SelfAttention,
      out = Dense(36 => 2),                 # 74 parameters
      heads = Array(
        @compact(
          Head,
          K = Dense(1 => 4),                # 8 parameters
          V = Dense(1 => 12),               # 24 parameters
          Q = Dense(1 => 4),                # 8 parameters
        ) do x 
            (k, v, q) = (K(x), V(x), Q(x))
            x = sum(k .* q; dims = 1) ./ sqrt(d_attn)
            softmax(x; dims = 2) .* v
        end,
        @compact(
          Head,
          K = Dense(1 => 4),                # 8 parameters
          V = Dense(1 => 12),               # 24 parameters
          Q = Dense(1 => 4),                # 8 parameters
        ) do x 
            (k, v, q) = (K(x), V(x), Q(x))
            x = sum(k .* q; dims = 1) ./ sqrt(d_attn)
            softmax(x; dims = 2) .* v
        end,
        @compact(
          Head,
          K = Dense(1 => 4),                # 8 parameters
          V = Dense(1 => 12),               # 24 parameters
          Q = Dense(1 => 4),                # 8 parameters
        ) do x 
            (k, v, q) = (K(x), V(x), Q(x))
            x = sum(k .* q; dims = 1) ./ sqrt(d_attn)
            softmax(x; dims = 2) .* v
        end,
      ),
    ) do x 
        out(vcat([h(x) for h = heads]...))
    end                  # Total: 20 arrays, 194 parameters, 3.515 KiB."""
    @test similar_strings(get_model_string(model3), expected_string3)
  end
end
