using Test
using LinearAlgebra
using Plots

include("../src/solver.jl")

function dense_history(u0, D, a, dt, T)
    N = length(u0)
    X = zeros(N, N)
    u = copy(u0)

    steps = Int(round(T/dt))
    hist = Vector{Matrix}(undef, steps+1)
    hist[1] = copy(X)

    for k in 1:steps
        u, X = step_rk4_dense(u, X, D, a, dt)
        hist[k+1] = copy(X)
    end

    return hist
end

function lowrank_history(u0, D, a, dt, T, r)
    N = length(u0)
    X0 = zeros(N, N)
    U, S, V = truncate_rank(X0, r)

    u = copy(u0)
    steps = Int(round(T/dt))

    hist = Vector{Matrix}(undef, steps+1)
    hist[1] = U * S * V'

    for k in 1:steps
        u, U, S, V = step_and_truncate_rk4(u, U, S, V, D, a, dt, r)
        hist[k+1] = U * S * V'
    end

    return hist
end

function compute_error_curves(u0, D, a, dt, T, r_list)
    Xdense_hist = dense_history(u0, D, a, dt, T)
    steps = length(Xdense_hist)

    errors = Dict{Int, Vector{Float64}}()

    for r in r_list
        Xlr_hist = lowrank_history(u0, D, a, dt, T, r)
        err = zeros(steps)

        for k in 1:steps
            err[k] = norm(Xlr_hist[k] - Xdense_hist[k]) / norm(Xdense_hist[k])
        end

        errors[r] = err
    end

    return errors
end

function plot_error_curves(errors, dt)
    steps = length(first(values(errors)))
    t = (0:steps-1) .* dt

    plt = plot(yscale=:log10, xlabel="t", ylabel="‖X_lr - X_dense‖F",
               title="Relative Error vs Time", legend=:topright)

    for r in sort(collect(keys(errors)))
        plot!(plt, t, errors[r], label="r = $r")
    end

    display(plt)
end

@testset "Exercise 3-3: fixed-rank step-and-truncate" begin
    N  = 300
    dt = 0.001
    T  = 0.05
    u0 = rand(N)
    D  = rand(N, N)
    a  = rand(N)
    r_list = [2, 4, 8, 16, 32]

    println("\nRunning dense reference history")
    Xdense_hist = dense_history(u0, D, a, dt, T)
    steps = length(Xdense_hist)
    tgrid = (0:steps-1) .* dt

    results = Dict{Int,Any}()

    println("\nRunning low-rank solvers")
    for r in r_list
    
        t_accum = 0.0
        Xlr_hist = Vector{Matrix}(undef, steps)
    
        X0 = zeros(N, N)
        U, S, V = truncate_rank(X0, r)
        u = copy(u0)
        Xlr_hist[1] = U * S * V'

        for k in 1:steps-1
            t_step = @elapsed begin
                u, U, S, V = step_and_truncate_rk4(u, U, S, V, D, a, dt, r)
            end
            t_accum += t_step
            Xlr_hist[k+1] = U * S * V'
        end

        time_per_step = t_accum / (steps-1)

        X_dense_T = Xdense_hist[end]
        X_lr_T    = Xlr_hist[end]
        final_err = norm(X_lr_T - X_dense_T) / norm(X_dense_T)
        err_curve = zeros(steps)
        for k in 1:steps
            err_curve[k] = norm(Xlr_hist[k] - Xdense_hist[k]) / norm(Xdense_hist[k])
        end

        results[r] = (time_per_step=time_per_step,
                      final_err=final_err,
                      err_curve=err_curve)

        println("r = $r, time/step = $(round(time_per_step, digits=4)), final error = $(final_err)")

        @test isfinite(final_err)
        @test final_err >= 0
    end

    plt = plot(yscale=:log10, xlabel="t", ylabel="‖X_lr - X_dense‖F",
               title="Exercise 3-3: Relative Error vs Time", legend=:topright)
    for r in r_list
        plot!(plt, tgrid, results[r].err_curve, label="r = $r")
    end
    savefig(plt, "error_curves.png")
end