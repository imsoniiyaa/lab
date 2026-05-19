using Test
using LinearAlgebra
include("../src/LLRSVD.jl")
include("../src/todense.jl")
include("../src/toelem.jl")

using .LLRSVDModule: LLRSVD, truncate
using .ToDenseModule: todense

@testset "Exercise 2: Truncation Verification" begin
    W = randn(50, 40)
    A = LLRSVD(W, 0.0)

    tols = [1e-1, 1e-3, 1e-6, 1e-10]
    
    for TOL in tols
        A2 = LLRSVDModule.truncate(A, TOL)
        err = norm(todense(A) - todense(A2))
        @test err <= TOL
    end
end