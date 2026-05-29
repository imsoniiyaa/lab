using LinearAlgebra
using SummationByPartsOperators
using Printf
using Plots

include("solver.jl")  

xmin, xmax, N = -5.0, 5.0, 500
a_true_fun(x) = 1.0 - 0.3 * exp(-36 * ((x - 2.5)/(2.5))^2)
u0_fun(x) = exp(-10*x^2) + exp(-10*(x-10)^2) + exp(-10*(x+10)^2)

Du_op = upwind_operators(periodic_derivative_operator;
                         derivative_order=1, accuracy_order=4,
                         xmin=xmin, xmax=xmax, N=N)
D = Du_op.minus
x = SummationByPartsOperators.grid(D)
Dm = Matrix(D)
u0 = u0_fun.(x)

h = step(x)
T = 10.0

a_true = a_true_fun.(x)
dt_true = h / maximum(abs.(a_true))
Nt_true = ceil(Int, T / dt_true)
dt_true = T / Nt_true

function solve_u(u0, Dm, a, T)
    dt = step(SummationByPartsOperators.grid(D)) / maximum(abs.(a))
    Nt = ceil(Int, T / dt)
    dt = T / Nt
    u = copy(u0)
    for n in 1:Nt
        k1 = -(a .* (Dm * u))
        k2 = -(a .* (Dm * (u + 0.5dt*k1)))
        k3 = -(a .* (Dm * (u + 0.5dt*k2)))
        k4 = -(a .* (Dm * (u + dt*k3)))
        u = u + (dt/6)*(k1 + 2k2 + 2k3 + k4)
    end
    return u
end

function solve_u_and_X_lr(u0, Dm, a, T, r)
    dt = h / maximum(abs.(a))
    Nt = ceil(Int, T / dt)
    dt = T / Nt
    u = copy(u0)
    U = zeros(N, r); S = Diagonal(zeros(r)); V = zeros(N, r)
    for n in 1:Nt
        u, U, S, V = step_and_truncate_rk4(u, U, S, V, Dm, a, dt, r)
    end
    return u, U, S, V
end

J_func(u, u_target) = 0.5 * norm(u - u_target)^2

function grad_J(u, U, S, V, u_target)
    diff = u - u_target
    return V * (Matrix(S)' * (U' * diff))
end

u_target = solve_u(u0, Dm, a_true, T)


function J_of(a_vec)
    uT = solve_u(u0, Dm, a_vec, T)
    return 0.5 * norm(uT - u_target)^2
end

function gradJ_of(a_vec, r)
    uT, U, S, V = solve_u_and_X_lr(u0, Dm, a_vec, T, r)
    return grad_J(uT, U, S, V, u_target)
end

function line_search(a_k, d_k, Jk; eta0=1.0, rho=0.5, c1=1e-4, r=32)
    eta = eta0
    gk = gradJ_of(a_k, r)
    gTd = dot(gk, d_k)

    while true
        a_trial = a_k + eta * d_k
        if J_of(a_trial) <= Jk + c1 * eta * gTd
            return eta
        end
        eta *= rho
        if eta < 1e-12
            return eta
        end
    end
end

maxiter = 20
r_lr = 32
a_k = fill(1.0, N)   
J_history = zeros(maxiter)


function run()
    maxiter = 20
    r_lr = 32
    a_k = fill(1.0, N)
    J_history = zeros(maxiter)


    for k in 1:maxiter
        Jk = J_of(a_k)
        gk = gradJ_of(a_k, r_lr)
        d_k = -gk

        eta_k = line_search(a_k, d_k, Jk; r=r_lr)
        a_k = a_k + eta_k * d_k

        J_history[k] = Jk

        @printf("iter %2d: J = %.6e, step = %.3e, gradnorm = %.3e\n",
                k, Jk, eta_k, norm(gk))
    end

    return a_k, J_history
end

# run it
a_opt, J_hist = run()

println("$(norm(u_target))")
println("Final J = $(J_of(a_k))")

