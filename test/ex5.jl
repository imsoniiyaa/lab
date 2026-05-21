using Test
using LinearAlgebra

include("../src/LLRSVD.jl")
using .LLRSVDModule: LLRSVD
include("../src/todense.jl")
using .ToDenseModule: todense
include("../src/trunc_sum.jl")

@testset "Exercise 5 - TruncSum" begin
    u1 = rand(3)
    u2 = rand(3)
    v1 = rand(2)
    v2 = rand(2)
    
    denseA = u1 * v1'
    denseB = u2 * v2'
    denseC = denseA + denseB

    a = LLRSVD(reshape(u1, 3, 1), [1.0], reshape(v1, 2, 1))
    b = LLRSVD(reshape(u2, 3, 1), [1.0], reshape(v2, 2, 1))
    c = trunc_sum([a, b], 1e-10)

    C = todense(c)
    @test norm(C - denseC) < 1e-10
end