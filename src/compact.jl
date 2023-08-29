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
d_in = 5
d_out = 7
d = @compact(W = randn(d_out, d_in), b = zeros(d_out), act = relu) do x
  y = W * x
  act.(y .+ b)
end
d(ones(5, 10)) # 7×10 Matrix as output.
d([1,2,3,4,5]) ≈ Dense(d.variables.W, zeros(7), relu)([1,2,3,4,5]) # Equivalent to a dense layer
``` 
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
To specify a custom printout for the model, you may find [`NoShow`](@ref) useful.
"""
macro compact(_exs...)
  # check inputs, extracting function expression fex and unprocessed keyword arguments _kwexs
  isempty(_exs) && error("expects at least two expressions: a function and at least one keyword")
  if Meta.isexpr(_exs[1], :parameters)
    length(_exs) >= 2 || error("expects an anonymous function")
    fex = _exs[2]
    _kwexs = (_exs[1], _exs[3:end]...)
  else
    fex = _exs[1]
    _kwexs = _exs[2:end]
  end
  Meta.isexpr(fex, :(->)) || error("expects an anonymous function")
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

  # make strings
  input =
      try
          fex_args = fex.args[1]
          isa(fex_args, Symbol) ? string(fex_args) : join(fex_args.args, ", ")
      catch e 
        @warn "Function stringifying does not yet handle all cases. Falling back to empty string for input arguments"
        ""
      end
  block = string(Base.remove_linenums!(fex).args[2])

  # edit expressions
  vars = map(ex -> ex.args[1], kwexs)
  fex = supportself(fex, vars)

  # assemble
  return esc(:($CompactLayer($fex, ($input, $block); $(kwexs...))))
end

function supportself(fex::Expr, vars)
  @gensym self
  @gensym curried_f
  # To avoid having to manipulate fex's arguments and body explicitly, we form a curried function first
  # that wraps the full fex expression, and then uncurry it programatically rather than syntactically.
  let_exprs = map(var -> :($var = $self.$var), vars)
  return quote
    $curried_f = ($self) -> let $(let_exprs...) 
        $fex
    end
    ($self, args...; kwargs...) -> $curried_f($self)(args...; kwargs...)
  end
end

struct CompactLayer{F<:Function, NT<:NamedTuple}
  fun::F
  strings::NTuple{2,String}
  variables::NT
end
CompactLayer(f::Function, str::Tuple; kw...) = CompactLayer(f, str, NamedTuple(kw))
CompactLayer(args...) = error("CompactLayer is meant to be constructed by the macro @compact")

Flux.@functor CompactLayer

(m::CompactLayer)(x...) = m.fun(m.variables, x...)

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
    input, block = obj.strings
    pre, post = ("(", ")")
    println(io, " "^indent, "@compact", pre)
    for k in keys(obj.variables)
      v = obj.variables[k]
      if false # Flux._show_leaflike(v)
        # If the value is a leaf, just print verbatim what the user wrote:
        # str = String(k) * " = " * summary(v)
        str = String(k) * " isa " * string(typeof(v))
        _just_show_params(io, str, v, indent+2)
        # Flux._layer_show(io::IO, str, indent+2, nothing)  # doesn't work
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

# Temporarily fixing things via piracy, but would be an easy change in Flux
using Flux: params, underscorise, _childarray_sum, _nan_show
function Flux._layer_show(io::IO, layer::AbstractArray, indent::Int=0, name=nothing)
  _str = isnothing(name) ? "" : "$name = "
  # str = _str * sprint(show, layer, context=io)  # before
  # str = _str * String(typeof(layer).name.name)  # print Array
  str = _str * summary(layer)  # print size too, sometimes too long... trim it?
  print(io, " "^indent, str, indent==0 ? "" : ",")
  if !isempty(params(layer))
    print(io, " "^max(2, (indent==0 ? 20 : 39) - indent - length(str)))
    printstyled(io, "# ", underscorise(sum(length, params(layer); init=0)), " parameters"; 
color=:light_black)
    nonparam = _childarray_sum(length, layer) - sum(length, params(layer), init=0)
    if nonparam > 0
      printstyled(io, ", plus ", underscorise(nonparam), indent==0 ? " non-trainable" : ""; color=:light_black)
    end
    _nan_show(io, params(layer))
  end
  indent==0 || println(io)
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
