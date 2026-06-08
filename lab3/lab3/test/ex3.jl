include("../src/unfold.jl")
include("../src/mode.product.jl")
include("../src/tucker3.jl")     
include("../src/tucker_3.jl")   
include("../src/todense.jl")
include("../src/hosvd.jl")
include("../src/tucker_round.jl") 

using Test
using LinearAlgebra

@testset "Tucker_Round Test" begin
    n1 = n2 = n3 = 30
    r1 = r2 = r3 = 20

    U1 = randn(n1, r1)
    U2 = randn(n2, r2)
    U3 = randn(n3, r3)
    G  = randn(r1, r2, r3)

    A = mode_product(mode_product(mode_product(G, U1, 1), U2, 2), U3, 3)

    T = Tucker3(A, 1e-12)

    tols = [1e-2, 1e-4, 1e-8]

    for tol in tols
        Tr = tucker_round(T, tol)

        r = (size(Tr.G,1), size(Tr.G,2), size(Tr.G,3))

        err = norm(todense(Tr) - A) / norm(A)

        @test err < 10 * tol  
        @test all(ri <= 20 for ri in r)

        println("tol = $tol   ranks = $r   rel error = $err")
    end
end


@testset "Gaussian Tucker Rank Test" begin
    n = 100
    xs = range(-3, 3, length=n)
    ys = range(-3, 3, length=n)
    zs = range(-3, 3, length=n)

    A = Array{Float64,3}(undef, n, n, n)
    for i in 1:n, j in 1:n, k in 1:n
        A[i,j,k] = exp(-(xs[i]^2 + ys[j]^2 + zs[k]^2))
    end

    T = Tucker3(A, 1e-12)

    r = (size(T.G,1), size(T.G,2), size(T.G,3))
    println("Multilinear rank assigned by HOSVD = ", r)

    @test r[1] >= 1
    @test r[2] >= 1
    @test r[3] >= 1

    err = norm(todense(T) - A) / norm(A)
    println("Error = ", err)

    @test err < 1e-10
end
