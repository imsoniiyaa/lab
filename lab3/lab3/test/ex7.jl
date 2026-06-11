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
# 7-3 step and truncate 
n = 32
h = 2 * pi/n
x = range(0, 2*pi - h, length=n)
function D_spectral(n)
    h = 2*pi / n
    x = range(0, 2*pi - h, length=n)
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        if i != j
            D[i,j] = 0.5 * (-1)^(i-j) * cot((i-j)*pi/n)
        end
    end
    return D * (2*pi / (n*h))
end
Dx = D_spectral(n)
Dy = D_spectral(n)
Dz = D_spectral(n)

function taylor_step_tucker(W, Dx, Dy, Dz, dt, p, tol)
    terms = [tucker_round(W, tol)]  # 초기 W도 truncate
    Wk = tucker_round(W, tol)
    for k in 1:p
        Wk = applyL_tucker(Wk, Dx, Dy, Dz, tol)
        scale = dt^k / factorial(k)
        push!(terms, Tucker3(Wk.G * scale, Wk.U1, Wk.U2, Wk.U3))
    end
    return tucker_sum(terms, tol)
end

function run_step_and_truncate()
    u0_dense = [sin(xi)*sin(yj)*sin(zk)
                for xi in x, yj in x, zk in x]

    dt = 0.1
    p = 4
    T = 2*pi
    N_steps = round(Int, T/dt)
    tol = dt^(p+1)

    W = Tucker3(u0_dense, tol)
    rank_history = Tuple{Int,Int,Int}[]
    t0 = time()

    for step in 1:N_steps
        W = taylor_step_tucker(W, Dx, Dy, Dz, dt, p, tol)
        push!(rank_history, (size(W.U1,2), size(W.U2,2), size(W.U3,2)))
        if step % 10 == 0
            elapsed = round(time()-t0, digits=1)
            println("step=$step/$(N_steps), ranks=$(rank_history[end]), t=$(elapsed)s")
        end
    end

    println("Finished $N_steps steps")
    for (step, ranks) in enumerate(rank_history)
        println("step=$step ranks=$ranks")
    end

    return rank_history
end

run_step_and_truncate()


