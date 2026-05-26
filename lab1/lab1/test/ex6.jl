using LinearAlgebra
using Test

include("../src/LLRSVD.jl")
using .LLRSVDModule: LLRSVD
include("../src/todense.jl")
using .ToDenseModule: todense
include("../src/trunc_sum.jl")
include("../src/secondOrder.jl")
include("../src/applyL.jl")

@testset "Exercise 6: applyL" begin
    m, n = 64, 64
    x = range(0, 2*pi, length=m)
    y = range(0, 2*pi, length=n)
    W0_dense = sin.(x) * sin.(y)'

    Dx, Dy = diff(x, y)

    #LLRSVD from rank-1 factors
    W0 = LLRSVD(reshape(sin.(x), m, 1), [1.0], reshape(sin.(y), n, 1))

    #Dense reference
    Wx_dense = Dx * W0_dense
    Wy_dense = W0_dense * Dy'
    L_dense = Wx_dense + Wy_dense

    Low = applyL(W0, Dx, Dy, 1e-10)
    Low_dense = todense(Low)

    err = norm(L_dense - Low_dense)

    @test err < 1e-10
    println("Error: ", err)

end