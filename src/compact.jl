import Flux: _big_show

"""
    @compact(forward::Function; name=nothing, parameters...)

Creates a layer by specifying some `parameters`, in the form of keywords,
and (usually as a `do` block) a function for the forward pass.
You may think of `@compact` as a specialized `let` block creating local variables 
that are trainable in Flux.
Declared variable names may be used within the body of the `forward` function.

Here is a linear model:

```
r = @compact(w = rand(3)) do x
  w .* x
end
r([1, 1, 1])  # x is set to [1, 1, 1].
```

Here is a linear model with bias and activation:

```
d = @compact(in=5, out=7, W=randn(out, in), b=zeros(out), act=relu) do x
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

model = @compact(
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
model = @compact(w=rand(3), name="Linear(3 => 1)") do x
  sum(w .* x)
end
println(model)  # "Linear(3 => 1)"
```

This can be useful when using `@compact` to hierarchically construct
complex models to be used inside a `Chain`.
"""
macro compact(fex, _kwexs...)
  # check inputs
  Meta.isexpr(fex, :(->)) || error("expects a do block")
  isempty(_kwexs) && error("expects keyword arguments")
  all(ex -> Meta.isexpr(ex, (:kw,:(=),:parameters)), _kwexs) || error("expects only keyword arguments")

  # process keyword arguments
  if Meta.isexpr(_kwexs[1], :parameters) # handle keyword arguments provided after semicolon
    kwexs1 = map(ex -> ex isa Symbol ? Expr(:kw, ex, ex) : ex, _kwexs[1].args) 
    _kwexs = _kwexs[2:end]
  else
    kwexs1 = ()
  end
  kwexs2 = map(ex -> Expr(:kw, ex.args...), _kwexs) # handle keyword arguments provided before semicolon
  kwexs = (kwexs1..., kwexs2...)

  # check if user has named layer:
  name = findfirst(ex -> ex.args[1] == :name, kwexs)
  if name !== nothing && kwexs[name].args[2] !== nothing
    length(kwexs) == 1 && error("expects keyword arguments")
    name_str = kwexs[name].args[2]
    # remove name from kwexs (a tuple)
    kwexs = (kwexs[1:name-1]..., kwexs[name+1:end]...)
    name = name_str
  end

  # make strings
  layer = "@compact"
  setup = NamedTuple(map(ex -> Symbol(string(ex.args[1])) => string(ex.args[2]), kwexs))
  input = join(fex.args[1].args, ", ")
  block = string(Base.remove_linenums!(fex).args[2])

  # edit expressions
  vars = map(ex -> ex.args[1], kwexs)
  @gensym self
  pushfirst!(fex.args[1].args, self)
  addprefix!(fex, self, vars)

  # assemble
  return esc(quote
    let
      $CompactLayer($fex, $name, ($layer, $input, $block), $setup; $(kwexs...))
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

struct CompactLayer{F,NT1<:NamedTuple,NT2<:NamedTuple}
  fun::F
  name::Union{String,Nothing}
  strings::NTuple{3,String}
  setup_strings::NT1
  variables::NT2
end
CompactLayer(f::Function, name::Union{String,Nothing}, str::Tuple, setup_str::NamedTuple; kw...) = CompactLayer(f, name, str, setup_str, NamedTuple(kw))
(m::CompactLayer)(x...) = m.fun(m.variables, x...)
CompactLayer(args...) = error("CompactLayer is meant to be constructed by the macro")
Flux.@functor CompactLayer

Flux._show_children(m::CompactLayer) = m.variables

function Base.show(io::IO, ::MIME"text/plain", m::CompactLayer)
  if get(io, :typeinfo, nothing) === nothing  # e.g., top level of REPL
    Flux._big_show(io, m)
  elseif !get(io, :compact, false)  # e.g., printed inside a Vector, but not a matrix
    Flux._layer_show(io, m)
  else
    show(io, m)
  end
end

function Flux._big_show(io::IO, obj::CompactLayer, indent::Int=0, name=nothing)
  setup_strings = obj.setup_strings
  local_name = obj.name
  has_explicit_name = local_name !== nothing
  if has_explicit_name
    if indent != 0 || length(Flux.params(obj)) <= 2
      _just_show_params(io, local_name, obj, indent)
    else  # indent == 0
      print(io, local_name)
      Flux._big_finale(io, obj)
    end
  else  # no name, so print normally
    layer, input, block = obj.strings
    pre, post = ("(", ")")
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", layer, pre)
    for k in keys(obj.variables)
      v = obj.variables[k]
      if Flux._show_leaflike(v)
        # If the value is a leaf, just print verbatim what the user wrote:
        str = String(k) * " = " * setup_strings[k]
        _just_show_params(io, str, v, indent+2)
      else
        Flux._big_show(io, v, indent+2, String(k))
      end
    end
    if indent == 0  # i.e. this is the outermost container
      print(io, rpad(post, 1))
    else
      print(io, " "^indent, post)
    end

    input != "" && print(io, " do ", input)
    if block != ""
      block_to_print = block[6:end]
      # Increase indentation of block according to `indent`:
      block_to_print = replace(block_to_print, r"\n" => "\n" * " "^(indent))
      print(io, " ", block_to_print)
    end
    if indent == 0
      Flux._big_finale(io, obj)
    else
      println(io, ",")
    end
  end
end

# Modified from src/layers/show.jl
function _just_show_params(io::IO, str::String, layer, indent::Int=0)
  print(io, " "^indent, str, indent==0 ? "" : ",")
  if !isempty(Flux.params(layer))
    print(io, " "^max(2, (indent==0 ? 20 : 39) - indent - length(str)))
    printstyled(io, "# ", Flux.underscorise(sum(length, Flux.params(layer))), " parameters"; color=:light_black)
    nonparam = Flux._childarray_sum(length, layer) - sum(length, Flux.params(layer))
    if nonparam > 0
      printstyled(io, ", plus ", Flux.underscorise(nonparam), indent==0 ? " non-trainable" : ""; color=:light_black)
    end
    Flux._nan_show(io, Flux.params(layer))
  end
  indent==0 || println(io)
end
