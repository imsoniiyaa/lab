include("LLRSVD.jl")
include("todense.jl")
include("toelem.jl")
include("truncate.jl")

using .LLRSVDModule
using .ToDenseModule
using .ToElemModule
using LinearAlgebra

# Verification test
W = randn(30, 20)
A = LLRSVD(W, 0.0)
err = norm(W - todense(A))

println(err)
println("Pass? ", err < 1e-12)
