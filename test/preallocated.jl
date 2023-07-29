
m1 = Chain(Dense(784 => 32, relu), Dense(32 => 10), softmax)
m2 = m1 |> pre

x = randn(Float32, 784, 64);

@test m1(x) ≈ m2(x)

g1 = gradient((m,x) -> m(x)[1], m1, x)
g2 = gradient((m,x) -> m(x)[1], m2, x)

@test g1[1].layers[1].bias ≈ g2[1].layers[1].layer.bias
@test g1[2] ≈ g2[2]


#=

julia> @btime gradient((m,x) -> m(x)[1], $m1, $x);
  min 50.167 μs, mean 88.796 μs (58 allocations, 355.41 KiB)

julia> @btime gradient((m,x) -> m(x)[1], $m2, $x);
  min 57.792 μs, mean 66.050 μs (115 allocations, 17.75 KiB)



let data = [(x,) for _ in 1:1000]
    o1 = Flux.setup(Adam(), m1)
    @btime Flux.train!((m,x) -> m(x)[1], $m1, $data, $o1)

    o2 = Flux.setup(Adam(), m2)
    @btime Flux.train!((m,x) -> m(x)[1], $m2, $data, $o2)

    nothing
end

# Yesterday:
#  min 1.799 s, mean 1.802 s (177001 allocations, 352.94 MiB)
#  min 146.713 ms, mean 251.041 ms (295001 allocations, 25.71 MiB)

# Today, wtf? Maybe threading changes have hurt.
#  min 244.235 ms, mean 251.582 ms (177001 allocations, 352.94 MiB)
#  min 224.760 ms, mean 227.594 ms (301001 allocations, 26.02 MiB)


m1cu = m1 |> gpu
m2cu = m2 |> gpu
xcu = x |> gpu


let data = [(xcu,) for _ in 1:1000]
    o1 = Flux.setup(Adam(), m1cu)
    CUDA.@time Flux.train!((m,x) -> sum(m(x)), m1cu, data, o1)

    o2 = Flux.setup(Adam(), m2cu)
    CUDA.@time Flux.train!((m,x) -> sum(m(x)), m2cu, data, o2)

    nothing
end
#  1.280640 seconds (1.86 M CPU allocations: 111.723 MiB, 10.99% gc time) (17.00 k GPU allocations: 340.008 MiB, 8.80% memmgmt time)
#  1.327849 seconds (1.73 M CPU allocations: 112.376 MiB, 6.70% gc time)   (3.00 k GPU allocations: 2.689 MiB, 2.29% memmgmt time)


=#


m3 = Chain(Dense(784 => 1024, tanh), BatchNorm(1024), Dense(1024 => 10), softmax)
m4 = m3 |> pre

x = randn(Float32, 784, 64);

@test m3(x) ≈ m4(x)

@btime $m3($x);
@btime $m4($x);

#=

julia> @btime $m3($x);
  min 318.000 μs, mean 7.944 ms (31 allocations, 1.01 MiB)

julia> @btime $m4($x);
  min 410.459 μs, mean 440.106 μs (57 allocations, 3.55 KiB)

=#

x4 = randn(Float32, 28, 28, 1, 13);

m5 = @autosize (size(x4)...,) Chain(
        Conv((3,3), 1 => 7, relu, stride=2, pad=1), 
        Conv((3,3), _ => 9, relu, stride=2),
        Conv((3,3), _ => 5, tanh, stride=2, bias=false),
        Flux.flatten, 
        Dense(_ => 10),
    )
m6 = m5 |> pre

@test m5(x4) ≈ m6(x4)

#=

julia> @btime $m5($x4);
  min 139.125 μs, mean 191.653 μs (179 allocations, 262.73 KiB)

julia> @btime $m6($x4);
  min 140.125 μs, mean 196.337 μs (160 allocations, 86.39 KiB)

=#


using Metalhead
m50 = Metalhead.ResNet(50)  # 100MB
m50pre = m50 |> pre  # 200BM


# First run

julia> @time m50(randn(Float32, 100,100,3,32)) |> size
  5.543590 seconds (6.11 M allocations: 1.963 GiB, 14.14% gc time, 96.22% compilation time)
(1000, 32)

julia> @time m50pre(randn(Float32, 100,100,3,32)) |> size
 16.098089 seconds (15.84 M allocations: 2.576 GiB, 62.26% gc time, 69.06% compilation time)
(1000, 32)

# Later


julia> @time m50(randn(Float32, 100,100,3,32)) |> size
 11.541100 seconds (4.40 k allocations: 1.570 GiB, 85.73% gc time)
(1000, 32)

julia> @time m50pre(randn(Float32, 100,100,3,32)) |> size
  4.664626 seconds (4.09 k allocations: 381.454 MiB, 61.15% gc time)
(1000, 32)


m50pre  # now 1.340 GiB

