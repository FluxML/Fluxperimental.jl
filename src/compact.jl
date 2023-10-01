import Flux: _big_show

"""
    @compact(forward::Function; name=nothing, parameters...)

Creates a layer by specifying some `parameters`, in the form of keywords,
and a function for the forward pass (often as a `do` block).

You may think of `@compact` as a specialized `let` block creating local variables 
that are trainable in Flux.
Declared variable names may be used within the body of the `forward` function.

# Examples

Here is a linear model, equivalent to `Flux.Scale`:

```
using Flux, Fluxperimental

w = rand(3)
sc = @compact(x -> x .* w; w)

sc([1 10 100])  # 3×3 Matrix as output.
ans ≈ Flux.Scale(w)([1 10 100])  # equivalent Flux layer
```

Here is a linear model with bias and activation, equivalent to Flux's `Dense` layer.
The forward pass function is now written as a do block, instead of `x -> begin y = W * x; ...`

```
d_in = 3
d_out = 7
layer = @compact(W = randn(d_out, d_in), b = zeros(d_out), act = relu) do x
  y = W * x
  act.(y .+ b)
end

den = Dense(layer.variables.W, zeros(7), relu)([1,2,3])  # equivalent Flux layer
layer(ones(3, 10)) ≈ layer(ones(3, 10))  # 7×10 Matrix as output.
``` 

Finally, here is a simple MLP, equivalent to a `Chain` with 5 `Dense` layers:

```
d_in = 1
nlayers = 3

model = @compact(
  lay1 = Dense(d_in => 64),
  lay234 = [Dense(64 => 64) for i=1:nlayers],
  wlast = rand32(64),
) do x
  y = tanh.(lay1(x))
  for lay in lay234
    y = relu.(lay(y))
  end
  return wlast' * y
end

model(randn(Float32, d_in, 8))  # 1×8 array as output.
```

We can train this model just like any `Chain`, for example:

```
data = [([x], [2x-x^3]) for x in -2:0.1f0:2]
optim = Flux.setup(Adam(), model)

for epoch in 1:1000
  Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, optim)
end
```
To specify a custom printout for the model, you may find [`NoShow`](@ref) useful.
"""
macro compact(_exs...)
  _compact(_exs...) |> esc
end

function _compact(_exs...)
  # check inputs, extracting function expression fex and unprocessed keyword arguments _kwexs
  isempty(_exs) && error("@compact expects at least two expressions: a function and at least one keyword")
  if Meta.isexpr(_exs[1], :parameters)
    length(_exs) >= 2 || error("@compact expects an anonymous function")
    fex = _exs[2]
    _kwexs = (_exs[1], _exs[3:end]...)
  else
    fex = _exs[1]
    _kwexs = _exs[2:end]
  end
  Meta.isexpr(fex, :(->)) || error("@compact expects an anonymous function")
  isempty(_kwexs) && error("@compact expects keyword arguments")
  all(ex -> Meta.isexpr(ex, (:kw,:(=),:parameters)), _kwexs) || error("@compact expects only keyword arguments")

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
        @warn """@compact's function stringifying does not yet handle all cases. Falling back to "?" """ maxlog=1
        "?"
      end
  block = string(Base.remove_linenums!(fex).args[2])  # TODO make this remove macro comments

  # edit expressions
  vars = map(ex -> ex.args[1], kwexs)
  fex = _supportself(fex, vars)

  # assemble
  return :($CompactLayer($fex, ($input, $block); $(kwexs...)))
end

function _supportself(fex::Expr, vars)
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

    print(io, " do ", input)
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
