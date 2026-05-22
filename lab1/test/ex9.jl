using Test
using LinearAlgebra
include("../src/LLRSVD.jl")
using .LLRSVDModule: LLRSVD
include("../src/trunc_sum.jl")
include("../src/applyL_sbr.jl")
include("../src/todense.jl")
using .ToDenseModule: todense
@testset "Exercise 9" begin
    m = n = 64
    x = range(-6, 6, length=m)
    y = range(-6, 6, length=n)

    a = 1.0
    b = 0.5

    u0 = [exp(-(xi^2)/(2a^2) - (yj^2)/(2b^2)) for xi in x, yj in y]

    ux = [-(xi/a^2) * u0[i,j] for (i,xi) in enumerate(x), (j,yj) in enumerate(y)]
    uy = [-(yj/b^2) * u0[i,j] for (i,xi) in enumerate(x), (j,yj) in enumerate(y)]

    L_ref = [ y[j]*ux[i,j] - x[i]*uy[i,j] for i in 1:m, j in 1:n ]

    function D(N, h)
        L = N * h
        scale = 2*pi / L
        M = zeros(Float64, N, N)
        for i in 1:N
            for j in 1:N
                if i != j
                    M[i,j] = 0.5 * (-1)^(i-j) * cot(pi*(i-j)/N) * scale
                end
            end
        end
        return M
    end

    Dx = D(m, step(x))
    Dy = D(n, step(y))

    X = Diagonal(x)
    Y = Diagonal(y)
    W0 = LLRSVD(u0, 1e-12)

    W_lr = applyL_sbr(W0, Dx, Dy, X, Y, 1e-12)
    L_lr = todense(W_lr)
    err = norm(L_lr - L_ref)

    @test err < 1e-6
    @info "$err"
end
