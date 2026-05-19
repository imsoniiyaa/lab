using Test
using LinearAlgebra
include("../src/LLRSVD.jl")
include("../src/todense.jl")

using .LLRSVDModule: LLRSVD
using .ToDenseModule: todense

@testset "LLRSVD Exercise 1" begin
    W = randn(30, 20)
    A = LLRSVD(W, 0.0)
    err = norm(W - todense(A))
    @test err < 1e-12
end
