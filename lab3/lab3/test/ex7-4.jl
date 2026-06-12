include("../src/unfold.jl")
include("../src/mode.product.jl")
include("../src/tucker3.jl")
include("../src/tucker_3.jl")
include("../src/todense.jl")
include("../src/tucker_inner.jl")
include("../src/tucker_add.jl")
include("../src/tucker_sum.jl")
include("../src/applyL_tucker.jl")
using LinearAlgebra
using Printf

function D_spectral(n)
    h = 2*pi / n
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        if i != j
            D[i,j] = 0.5 * (-1)^(i-j) * cot((i-j)*pi/n)
        end
    end
    return D * (2*pi / (n*h))
end

function taylor_step_tucker(W, Dx, Dy, Dz, dt, p, tol)
    terms = [tucker_round(W, tol)]
    Wk = tucker_round(W, tol)
    for k in 1:p
        Wk = applyL_tucker(Wk, Dx, Dy, Dz, tol)
        scale = dt^k / factorial(k)
        push!(terms, Tucker3(Wk.G * scale, Wk.U1, Wk.U2, Wk.U3))
    end
    return tucker_sum(terms, tol)
end

n = 32  
h = 2*pi / n
x = range(0, 2*pi - h, length=n)
Dx = D_spectral(n)
Dy = D_spectral(n)
Dz = D_spectral(n)

u0_dense = [sin(xi)*sin(yj)*sin(zk) for xi in x, yj in x, zk in x]
u_ref = copy(u0_dense)  # exact solution at T=2π equals u0

p = 4
prev_ratio = nothing

println("Convergence test (n=$n, p=$p)")
println("="^55)
println("dt       | error        | dt^p        | ratio")
println("-"^55)

for dt in [0.2, 0.1, 0.05]
    tol = dt^(p+1)
    N_steps = round(Int, 2*pi/dt)

    W = Tucker3(u0_dense, tol)
    t0 = time()

    for step in 1:N_steps
        W = taylor_step_tucker(W, Dx, Dy, Dz, dt, p, tol)
    end

    err = h^(3/2) * norm(todense(W) - u_ref)
    ratio = err / dt^p
    elapsed = round(time()-t0, digits=1)

    @printf("dt=%.3f | err=%.4e | dt^p=%.4e | ratio=%.4f  (%.1fs)\n",
            dt, err, dt^p, ratio, elapsed)

    global prev_ratio = ratio
end