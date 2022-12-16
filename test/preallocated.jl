
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
  min 52.167 μs, mean 2.519 ms (58 allocations, 355.41 KiB)

julia> @btime gradient((m,x) -> m(x)[1], $m2, $x);
  min 58.750 μs, mean 190.440 μs (109 allocations, 17.44 KiB)



let data = [(x,) for _ in 1:1000]
    o1 = Flux.setup(Adam(), m1)
    @btime Flux.train!((m,x) -> m(x)[1], $m1, $data, $o1)

    o2 = Flux.setup(Adam(), m2)
    @btime Flux.train!((m,x) -> m(x)[1], $m2, $data, $o2)

    nothing
end

#  min 1.799 s, mean 1.802 s (177001 allocations, 352.94 MiB)
#  min 146.713 ms, mean 251.041 ms (295001 allocations, 25.71 MiB)


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

