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

function D(N, L)
    h = L / N
    x = (-N/2:N/2-1) * h
    M = zeros(Float64, N, N)
    for j in 1:N
        for k in 1:N
            if j != k
                M[j,k] = 0.5 * (-1)^(j-k) / tan((x[j] - x[k]) * pi / L)
            end
        end
    end
    return (2*pi/L) * M
end

@testset "Exercise 10" begin
    m = n = 128
    x = range(-6, 6, length=m+1)[1:end-1]  
    y = range(-6, 6, length=n+1)[1:end-1] 

    a = 1.0
    b = 0.5
    u0 = [exp(-(xi^2)/(2a^2) - (yj^2)/(2b^2)) for xi in x, yj in y]

    Dx = D(m, 12)
    Dy = D(n, 12)
    X = Diagonal(x)
    Y = Diagonal(y)

    W0 = LLRSVD(u0, 1e-12)

    h = 12.0 / m
    dts = [0.01, 0.005, 0.0025]
    p = 3
    prev_ratio = nothing

    for dt in dts
        local tol = dt^(p+1)
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