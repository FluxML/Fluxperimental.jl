"""
    @Magic(forward::Function; name=nothing, construct...)

Creates a layer by specifying some code to construct the layer, run immediately,
and (usually as a `do` block) a function for the forward pass.
You may think of `construct` as a `let` block creating local variables.
Their names may be used within the body of the `forward` function.

Here is a linear model:

```
r = @Magic(w = rand(3)) do x
  w .* x
end
r([1, 1, 1])  # x is set to [1, 1, 1].
```

Here is a linear model with bias and activation:

```
d = @Magic(in=5, out=7, W=randn(out, in), b=zeros(out), act=relu) do x
  y = W * x
  act.(y .+ b)
end
d(ones(5, 10))  # 7×10 Matrix as output.
```

Finally, here is a simple MLP:

```
using Flux

n_in = 1
n_out = 1
nlayers = 3

model = @Magic(
  w1=Dense(n_in, 128),
  w2=[Dense(128, 128) for i=1:nlayers],
  w3=Dense(128, n_out),
  act=relu
) do x
  embed = act(w1(x))
  for w in w2
    embed = act(w(embed))
  end
  out = w3(embed)
  return out
end

model(randn(n_in, 32))  # 1×32 Matrix as output.
```

We can train this model just like any `Chain`:

```
data = [([x], 2x-x^3) for x in -2:0.1f0:2]
optim = Flux.setup(Adam(), model)

for epoch in 1:1000
  Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, optim)
end
```

You may also specify a `name` for the model, which will
be used instead of the default printout, which gives a verbatim
representation of the code used to construct the model:

```
model = @Magic(w=rand(3), name="Linear(3 => 1)") do x
  sum(w .* x)
end
println(model)  # "Linear(3 => 1)"
```

This can be useful when using `@Magic` to hierarchically construct
complex models to be used inside a `Chain`.
"""
macro Magic(fex, kwexs...)
  # check input
  Meta.isexpr(fex, :(->)) || error("expects a do block")
  isempty(kwexs) && error("expects keyword arguments")
  all(ex -> Meta.isexpr(ex, :kw), kwexs) || error("expects only keyword argumens")

  # check if user has named layer:
  name = findfirst(ex -> ex.args[1] == :name, kwexs)
  name_str = ""
  if name !== nothing && kwexs[name].args[2] !== nothing
    length(kwexs) == 1 && error("expects keyword arguments")
    name_str = string(kwexs[name].args[2])
    # remove name from kwexs (a tuple)
    kwexs = (kwexs[1:name-1]..., kwexs[name+1:end]...)
  end

  # make strings
  layer = "@Magic"
  setup = join(map(ex -> string("  ", ex.args[1], " = ", ex.args[2]), kwexs), ",\n")
  input = join(fex.args[1].args, ", ")
  block = string(Base.remove_linenums!(fex).args[2])

  if name !== nothing
    # make strings
    layer = name_str
    setup = ""
    input = ""
    block = ""
  end

  # edit expressions
  vars = map(ex -> ex.args[1], kwexs)
  assigns = map(ex -> Expr(:(=), ex.args...), kwexs)
  @gensym self
  pushfirst!(fex.args[1].args, self)
  addprefix!(fex, self, vars)

  # assemble
  return esc(quote
    let
      $(assigns...)
      $MagicLayer($fex, ($layer, $setup, $input, $block); $(vars...))
    end
  end)
end

function addprefix!(ex::Expr, self, vars)
  for i = 1:length(ex.args)
    if ex.args[i] in vars
      ex.args[i] = :($self.$(ex.args[i]))
    else
      addprefix!(ex.args[i], self, vars)
    end
  end
end
addprefix!(not_ex, self, vars) = nothing

struct MagicLayer{F,NT<:NamedTuple}
  fun::F
  strings::NTuple{4,String}
  variables::NT
end
MagicLayer(f::Function, str::Tuple; kw...) = MagicLayer(f, str, NamedTuple(kw))
(m::MagicLayer)(x...) = m.fun(m.variables, x...)
MagicLayer(args...) = error("MagicLayer is meant to be constructed by the macro")
Flux.@functor MagicLayer

function Base.show(io::IO, m::MagicLayer)
  layer, setup, input, block = m.strings
  print(io, layer)
  setup != "" && print(io, "(\n", setup, ",\n)")
  input != "" && print(io, " do ", input)
  block != "" && print(io, block[6:end])
  return nothing
end
