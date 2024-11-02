
"""
    @autostruct function MyLayer(d); ...; MyLayer(f1, f2, ...); end

This is a macro for easily defining new layers.

Recall that Flux layer is a callable `struct` which may contain parameter arrays.
Usually, the steps to define a new one are:
1. Define a `struct MyLayer` with the desired fields,
   and tell Flux to look inside with `@layer MyLayer` (or on earlier versions, `@functor`).
2. Define a constructor function like `MyLayer(d::Int)`,
   which initialises the parameters (say to `randn32(d, d)`)
   and returns an instance of the `struct`, some `m::MyLayer`.
3. Define the forward pass, by making the struct callable: `(m::MyLayer)(x) = ...`

Given the function in step 2, this macro handles step 1. You still do step 3.

If you change the name or types of the fields, then the `struct` definition is
automatically replaced, without re-starting Julia.
This works because this definition uses an auto-generated name, which is `== MyLayer`.
(But existing instances of the old `struct` are not changed in any way!)

Writing `@autostruct :expand function MyLayer(d)` will use `@layer :expand MyLayer`,
and result in container-style pretty-printing.

## Examples

```julia
@autostruct function MyModel(d::Int)
  alpha, beta = [Dense(d=>d, tanh) for _ in 1:2]    # arbitrary code here, not just keyword-like
  beta.bias[:] .= 1/d
  return MyModel(alpha, beta)  # this must be very simple, no = signs allowed (return optional)
end

function (m::MyModel)(x)  # forward pass looks just like a normal struct
  y = m.alpha(x)
  z = m.beta(y)
  (x .+ y .+ z)./3
end

Flux.trainable(m::MyModel) = (; m.alpha)  # if necessary, restrict which fields are trainable

Base.show(io::IO, m::MyModel) =  # if desired, replace default printing "MyModel(...)"
  print(io, "MyModel(", size(m.alpha.weight, 1), ")")

MyModel(2) isa MyModel  # true
```

The `struct` defined by the macro here is something like this:

```julia
struct MyModel001{T1, T2}
  alpha::T1
  beta::T2
end
```

This can hold any objects, even `MyModel("hello", "world")`.
As you can see by looking `methods(MyModel)`, there should never be an ambiguity
between the `struct`'s own constructor, and your `MyModel(d::Int)`.

You can also restrict the types allowed in the struct:
```
@autostruct :expand function MyOtherModel(d1, d2, act=identity)
  gamma = Embedding(128 => d1)
  delta = Dense(d1 => d2, act)
  MyOtherModel(gamma::Embedding, delta::Dense)  # struct will only hold these types
end

(m::MyOtherModel)(x) = softmax(m.delta(m.gamma(x)))  # forward pass

methods(MyOtherModel)  # will show 3 methods
```

This creates a struct like this:

```julia
struct MyOtherModel001{T1 <: Embedding, T2 <: Dense}
  gamma::T1
  delta::T2
end
```

If you need to add additional constructor methods, the obvious syntax will not work.
But you can add them to the type, like this:

```julia
MyModel(str::String) = MyModel(parse(Int, str))
# ERROR: cannot define function MyModel; it already has a value

(::Type{MyModel})(str::String) = MyModel(parse(Int, str))
MyModel("4")  # this works
```

## Compared to `@compact`

For comparison, the use of `@compact` to do much the same thing looks like this -- shorter,
but further from being ordinary Julia code.

```julia
function MyModel2(d::Int)
    alpha, beta = [Dense(d=>d, tanh) for _ in 1:2]
    beta.bias[:] .= 1/d
    @compact(; alpha, beta) do x
        y = alpha(x)
        z = beta(y)
        (x .+ y .+ z)./3
    end
end

MyModel2(2) isa Fluxperimental.CompactLayer  # no easy struct type

MyOtherModel2(d1, d2, act=identity) =
  @compact(gamma = Embedding(128 => d1), delta=Dense(d1 => d2, act)) do x
    softmax(delta(gamma(x)))
  end
```
"""
macro autostruct(ex)
    esc(_autostruct(ex))
end

macro autostruct(ex1, ex2)
    (ex1 isa QuoteNode && ex1.value == :expand) || throw("Expected either `@autostruct function` or `@autostruct :expand function`")
    esc(_autostruct(ex2; expand=true))
end

const DEFINE = Dict{UInt, Tuple}()

function _autostruct(expr; expand::Bool=false)
    # Check first & last line of the input expression:
    Meta.isexpr(expr, :function) || throw("Expected a function definition, like `@autostruct function MyStruct(...); ...`")
    fun = expr.args[1].args[1]
    ret = expr.args[2].args[end]
    if Meta.isexpr(ret, :return)
        ret = only(ret.args)
    end
    Meta.isexpr(ret, :call) || throw("Last line of `@autostruct function $fun` must return `$fun(field1, field2, ...)`")
    ret.args[1] === fun || throw("Last line of `@autostruct function $fun` must return `$fun(field1, field2, ...)`")
    for ex in ret.args
        ex isa Symbol && continue
        Meta.isexpr(ex, :(::)) && continue
        throw("Last line of `@autostruct function $fun` must return `$fun(field1, field2, ...)` or `$fun(field1::T1, field2::T2, ...)`, but got $ex")
    end
    funargs = expr.args[1].args[2:end]
    retargs = ret.args[2:end]
    @show funargs retargs
    if length(retargs) == length(funargs) && all(ex -> ex isa Symbol, retargs) && all(ex -> ex isa Symbol, funargs)
        # This check only catches cases like MyFun(a) -> MyFun(A), not MyFun(as...) or MyFun(a, b=1) or MyFun(a; b=1)
        @warn "Function $(expr.args[1]) will be ambiguous with struct $ret. " *
            "Please add some type restrictions to the function, or to the return line (which sets struct fields)"
    end

    # If the last line is new, construct struct definition:
    name, defex = get!(DEFINE, hash(ret, UInt(expand))) do
        name = gensym(fun)
        fields = map(enumerate(ret.args[2:end])) do (i, ex)
            field = ex isa Symbol ? ex : ex.args[1]  # we allow `return MyModel(alpha, beta::Chain)`
            type = Symbol("T#", i)
            :($field::$type)
        end
        types = map(fields, ret.args[2:end]) do ft, ex
            if ex isa Symbol  # then no type spec on return line
                ft.args[2]
            else
                Expr(:(<:), ft.args[2], ex.args[2])
            end
        end
        layer = if !expand
            :($Flux.@layer $name)
        else
            str = "$fun("
            quote
                $Flux.@layer :expand $name
                Flux._show_pre_post(::$name) = $str, ")"  # needs https://github.com/FluxML/Flux.jl/pull/2344
            end
        end
        str = "$fun(...)"
        ex = quote
            struct $name{$(types...)}
                $(fields...)
            end
            $layer
            $Base.show(io::IO, _::$name) = $print(io, $str)
            $fun = $name
        end
        (name, ex)
    end

    # Change first line to use the struct's name:
    expr.args[1].args[1] = name
    quote
        $(defex.args...)  # struct definition
        $expr  # constructor function
    end
end
