using Test
using LinearAlgebra
include("../src/LLRSVD.jl")
using .LLRSVDModule: LLRSVD
include("../src/trunc_sum.jl")
include("../src/applyL_sbr.jl")
include("../src/todense.jl")
using .ToDenseModule: todense
include("../src/taylor_step_sbr.jl")
include("../src/sat_sbr.jl")

function D(N, h)
    L = N * h
    scale = 2*pi / L
    M = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        if i != j
            M[i,j] = 0.5 * (-1)^(i-j) * cot(pi*(i-j)/N) * scale
        end
    end
    return M
end

@testset "Exercise 10" begin
    m = n = 128
    x = range(-6, 6, length=m)
    y = range(-6, 6, length=n)

    a = 1.0
    b = 0.5
    u0 = [exp(-(xi^2)/(2a^2) - (yj^2)/(2b^2)) for xi in x, yj in y]

    Dx = D(m, step(x))
    Dy = D(n, step(y))
    X = Diagonal(x)
    Y = Diagonal(y)

    W0 = LLRSVD(u0, 1e-12)

    h = step(x)
    dts = [0.2, 0.1, 0.05]
    p = 3
    tol = 1e-12
    prev_ratio = nothing

    for dt in dts
        WN = step_and_truncate_sbr(W0, Dx, Dy, X, Y, dt, p, tol)
        err = h * norm(todense(WN) - u0)
        ratio = err / dt^p

        @info "$dt → error = $err"

        if prev_ratio !== nothing
            @test abs(ratio - prev_ratio) / prev_ratio < 0.5
        end
        prev_ratio = ratio
    end
end
