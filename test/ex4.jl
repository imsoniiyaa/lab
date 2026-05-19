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