using Test
using LinearAlgebra

include("../src/solver.jl")

@testset "low-rank" begin

    N = 300

    dt = 0.001

    T = 0.05

    u0 = rand(N)

    D = rand(N, N)

    a = rand(N)

    r_list = [2, 4, 8, 16, 32]

    println("\nRunning dense solver")

    X_dense =
        run_dense_solver(
            u0,
            D,
            a,
            dt,
            T
        )

    results = Dict()

    println("\nRunning low-rank solvers")

    for r in r_list

        X_lr, time_per_step = run_lowrank_solver(u0, D, a, dt, T, r)

        err = rel_error(X_lr, X_dense)

        results[r] = (error=err, time_per_step=time_per_step)

        println(
            "r = $r, ",
            "time/step = $(round(time_per_step, digits=4)), ",
            "error = $(err)"
        )

        @test isfinite(err)

        @test err >= 0

        @test rank(X_lr) <= r
    end

    errs =[results[r].error for r in r_list]

    println("\nErrors:")
    println(errs)

end