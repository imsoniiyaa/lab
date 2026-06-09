include("../src/unfold.jl")
include("../src/mode.product.jl")
include("../src/tucker3.jl")
include("../src/tucker_3.jl")
include("../src/todense.jl")
include("../src/tucker_inner.jl")
using LinearAlgebra

A_dense = randn(10, 10, 10)
B_dense = randn(10, 10, 10)

TA = Tucker3(A_dense, 1e-14)
TB = Tucker3(B_dense, 1e-14)

ip_tucker = tucker_inner(TA, TB)
ip_dense  = sum(todense(TA) .* todense(TB))
ip_exact  = sum(A_dense .* B_dense)

println("tucker_inner:  ", ip_tucker)
println("dense inner:   ", ip_dense)
println("exact inner:   ", ip_exact)
println("error vs exact: ", abs(ip_tucker - ip_exact))
@assert abs(ip_tucker - ip_exact) < 1e-10
println("tucker_inner passed")