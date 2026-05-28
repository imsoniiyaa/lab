using LinearAlgebra
using SummationByPartsOperators
using Printf

include("solver.jl")

# Setup
xmin, xmax, N = -5.0, 5.0, 500
a_fun(x) = 1.0 - 0.3 * exp(-36 * ((x - 2.5)/(2.5))^2)
u0_fun(x,a) = exp(-a*x^2) + exp(-a*(x-10)^2) + exp(-a*(x+10)^2)

Du_op = upwind_operators(periodic_derivative_operator; 
                         derivative_order=1, accuracy_order=4, 
                         xmin=xmin, xmax=xmax, N=N)
D = Du_op.minus
x = SummationByPartsOperators.grid(D)
Dm = Matrix(D)
a = a_fun.(x)
u0 = u0_fun.(x, 10)

h = step(x)
dt = h / maximum(abs.(a))
Nt = ceil(Int, 10.0 / dt)
dt = 10.0 / Nt
T = 10.0

#Dense reference solver
function run_dense(u0, Dm, a, dt, Nt)
    N = length(u0)
    u = copy(u0)
    X = zeros(N, N)
    ux = similar(u); DS = similar(X)
    k1u=similar(u); k2u=similar(u); k3u=similar(u); k4u=similar(u); utmp=similar(u)
    k1S=similar(X); k2S=similar(X); k3S=similar(X); k4S=similar(X); Stmp=similar(X)

    function rhs!(du, dS, u, S)
        mul!(ux, Dm, u)
        @. du = -a * ux
        for j in axes(S,2); mul!(view(DS,:,j), Dm, view(S,:,j)); end
        @. dS = -a * DS
        for i in eachindex(ux); dS[i,i] -= ux[i]; end
    end

    for n in 1:Nt
        rhs!(k1u,k1S,u,X)
        @. utmp=u+0.5dt*k1u; @. Stmp=X+0.5dt*k1S
        rhs!(k2u,k2S,utmp,Stmp)
        @. utmp=u+0.5dt*k2u; @. Stmp=X+0.5dt*k2S
        rhs!(k3u,k3S,utmp,Stmp)
        @. utmp=u+dt*k3u; @. Stmp=X+dt*k3S
        rhs!(k4u,k4S,utmp,Stmp)
        @. u=u+(dt/6)*(k1u+2k2u+2k3u+k4u)
        @. X=X+(dt/6)*(k1S+2k2S+2k3S+k4S)
    end
    return u, X
end

#low rank solver
function run_lowrank(u0, Dm, a, dt, Nt, r)
    N = length(u0)
    u = copy(u0)
    # Initialize X=0 as rank-r
    U = zeros(N, r)
    S = Diagonal(zeros(r))
    V = zeros(N, r)

    for n in 1:Nt
        u, U, S, V = step_and_truncate_rk4(u, U, S, V, Dm, a, dt, r)
    end
    return u, U, S, V
end

println("Running dense solver...")
u_dense, X_dense = run_dense(u0, Dm, a, dt, Nt)
gradJ_dense = X_dense' * (u_dense - u0)
println("$(norm(gradJ_dense))")


for r in [2, 4, 8, 16, 32]
    println("Running low-rank r=$r...")
    u_lr, U_lr, S_lr, V_lr = run_lowrank(u0, Dm, a, dt, Nt, r)

    # Cheap gradient
    diff = u_lr - u0
    gradJ_lr = V_lr * (Matrix(S_lr)' * (U_lr' * diff))

    X_lr_full = U_lr * Matrix(S_lr) * V_lr'
    rel_err_X = norm(X_lr_full - X_dense) / norm(X_dense)
    rel_err_g = norm(gradJ_lr - gradJ_dense) / norm(gradJ_dense)

    @printf("r=%2d, X err = %.4e, gradJ err = %.4e\n", r, rel_err_X, rel_err_g)
end