include("../src/unfold.jl")
include("../src/mode.product.jl")
include("../src/tucker3.jl")
include("../src/tucker_3.jl")
include("../src/todense.jl")
include("../src/tucker_inner.jl")
include("../src/tucker_add.jl")
include("../src/tucker_sum.jl")

using Test
using LinearAlgebra

@testset "tucker_sum test" begin
    U1 = Matrix(I, 10, 10)[:, 1:2]
    U2 = Matrix(I, 10, 10)[:, 1:2]
    U3 = Matrix(I, 10, 10)[:, 1:2]

    Ga = ones(2,2,2)           
    Gb = reshape(1:8, 2,2,2)      

    A = Tucker3(Ga, U1, U2, U3)
    B = Tucker3(Gb, U1, U2, U3)

    dense_exact = todense(A) + todense(B)

    Tsum = tucker_sum([A, B], 1e-12)
    dense_tucker = todense(Tsum)

    err = norm(dense_tucker - dense_exact) / norm(dense_exact)

    @test err < 1e-10
end