include("../src/LLRSVD.jl")
using .LLRSVDModule: LLRSVD
include("../src/trunc_sum.jl")
include("../src/applyL_sbr.jl")
include("../src/todense.jl")
using .ToDenseModule: todense
include("../src/taylor_step_sbr.jl")
include("../src/sat_sbr.jl")
using LinearAlgebra
using Plots

function D(N, L)
    h = L / N
    x = (-N/2:N/2-1) * h
    M = zeros(Float64, N, N)
    for j in 1:N, k in 1:N
        if j != k
            M[j,k] = 0.5 * (-1)^(j-k) / tan((x[j] - x[k]) * π / L)
        end
    end
    return (2*pi/L) * M
end

m = n = 128
x = collect(range(-6, 6, length=m+1)[1:end-1])
y = collect(range(-6, 6, length=n+1)[1:end-1])
a, b = 1.0, 0.5
u0 = [exp(-(xi^2)/(2a^2) - (yj^2)/(2b^2)) for xi in x, yj in y]
Dx = D(m, 12); Dy = D(n, 12)
X = Diagonal(x); Y = Diagonal(y)
W0 = LLRSVD(u0, 1e-12)

dt = 0.005
p = 3

snapshot_times = [0.0, pi/4, pi/2, pi, 2*pi]
snapshot_steps = [round(Int, t/dt) for t in snapshot_times]

W = W0
plots_list = []

step = 0
for target in snapshot_steps
    # ensure assignments inside this loop refer to the top-level variables
    global W, step
    while step < target
        local tol = dt^(p+1)
        terms = taylor_step_sbr(W, Dx, Dy, X, Y, dt, p, 0.0)
        W = trunc_sum(terms, tol)
        step += 1
    end
    t = step * dt
    u = todense(W)
    pl = contourf(x, y, u', 
                  title="t = $(round(t, digits=3))",
                  xlabel="x", ylabel="y",
                  color=:viridis, levels=20,
                  aspect_ratio=:equal)
    push!(plots_list, pl)
    println("t=$(round(t,digits=3)), rank=$(W.r)")
end

p_final = plot(plots_list..., layout=(1,5), size=(1400, 300))
savefig(p_final, "snapshots_ex10.png")
println("Saved: snapshots_ex10.png")