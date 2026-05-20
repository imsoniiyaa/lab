include("../src/grid.jl")
include("../src/secondOrder.jl")

using Test
using LinearAlgebra
using Plots

@testset "Exercise 4-1" begin
    x, y, W0 = grid()
    Dx, Dy = diff(x, y)
    dense(W, Dx, Dy) = Dx * W + W * Dy'
    Wk = [W0]
    for k in 1:7
        push!(Wk, dense(Wk[end], Dx, Dy))
    end

    @test length(Wk) == 8
    dt = 0.1
    Tk = [ (dt^k / factorial(k)) * Wk[k+1] for k in 0:7 ]
    @test length(Tk) == 8
    #singular values
    sv = [ svd(Tk[k+1]).S for k in 0:7 ]
    #plot the singular values
    p = plot(yscale=:log10, legend=:outertopright)

    for k in 1:8
        plot!(p, sv[k], label="T$(k-1)")
    end

    display(p)
end

@testset "Exercise 4-2" begin
    x, y, W0 = grid()
    Dx, Dy = diff(x, y)

    dense(W, Dx, Dy) = Dx * W + W * Dy'
    Wk = [W0]
    for k in 1:7
        push!(Wk, dense(Wk[end], Dx, Dy))
    end

    dt = 0.1
    Tk = [ (dt^k / factorial(k)) * Wk[k+1] for k in 0:7 ]

    sv = [ svd(Tk[k+1]).S for k in 0:7 ]

    norms = [ sqrt(sum(sv[k+1].^2)) for k in 0:7 ]

    @test length(norms) == 8

    global TK_NORMS = norms
end

@testset "Exercise 4-3" begin
    x, y, W0 = grid()
    Dx, Dy = diff(x, y)
    dense(W, Dx, Dy) = Dx * W + W * Dy'

    Wk = [W0]
    for k in 1:7
        push!(Wk, dense(Wk[end], Dx, Dy))
    end

    dt = 0.1
    p = 7
    Wref = [ sin(x[i] - dt) * sin(y[j] - dt) for i in 1:length(x), j in 1:length(y) ]


    alphas = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    results = Dict{Float64,Float64}()

    for alpha in alphas
        Tk_trunc = []
        for k in 0:p
            Tk = (dt^k / factorial(k)) * Wk[k+1]
            U, S, V = svd(Tk)
            if alpha == 0.0
                r = length(S)
            else
                tol = alpha * dt^(p+1-k)
                r = sum(S .> tol)
                if r == 0
                    r = 1
                end
            end
            push!(Tk_trunc, U[:,1:r] * Diagonal(S[1:r]) * V[:,1:r]')
        end

        W_TS = sum(Tk_trunc)
        err = norm(vec(W_TS - Wref))
        println("alpha=", alpha, "  err=", err)
        results[alpha] = err
    end

    global ERR_43 = results

    minerr = minimum(values(results))
    @show minerr
    @test minerr < 10.0
end

alpha = 1e-6

@testset "Exercise 4-4" begin
    x, y, W0 = grid()
    Dx, Dy = diff(x, y)
    dense(W, Dx, Dy) = Dx * W + W * Dy'

    Wk = [W0]
    for k in 1:7
        push!(Wk, dense(Wk[end], Dx, Dy))
    end

    dt = 0.1
    p = 7
    alpha = 1e-6

    ranks_Tk = Int[]
    Tk_trunc = []

    for k in 0:p
        Tk = (dt^k / factorial(k)) * Wk[k+1]
        U, S, V = svd(Tk)

        tol = alpha * dt^(p+1-k)
        r = sum(S .> tol)
        if r == 0
            r = 1
        end

        push!(ranks_Tk, r)
        push!(Tk_trunc, U[:,1:r] * Diagonal(S[1:r]) * V[:,1:r]')
    end

    W_TS = sum(Tk_trunc)
    Usum, Ssum, Vsum = svd(W_TS)

    tol_sum = alpha * dt^(p+1)
    r_eff = sum(Ssum .> tol_sum)

    println("k   rank(Tk after truncation)")
    for k in 0:p
        println("k = ", k, "   rank = ", ranks_Tk[k+1])
    end
    println("Effective rank of final sum W_TS: ", r_eff)

    global RANKS_44 = (ranks_Tk = ranks_Tk, r_eff = r_eff)

    @test r_eff <= maximum(ranks_Tk)
end