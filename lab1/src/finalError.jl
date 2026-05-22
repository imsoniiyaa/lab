using LinearAlgebra
include("LLRSVD.jl")
using .LLRSVDModule: LLRSVD
include("trunc_sum.jl")

function trunc_sum(terms::Vector{LLRSVD}, TOL::Float64)::LLRSVD
    a = terms[1]
    for i in 2:length(terms)
        b = terms[i]
        u = [a.U b.U]
        v = [a.V b.V]
        s = Diagonal(vcat(a.S, b.S))

        qu, ru = qr(u)
        qv, rv = qr(v)

        mat = ru * s * rv'
        us, ss, vs = svd(mat)

        r = sum(ss .> TOL)
        r = max(r, 1)

        a = LLRSVD(qu * us[:, 1:r], ss[1:r], qv * vs[:, 1:r])
    end
    return a
end

include("applyL.jl")
include("taylor_step.jl")
include("time.jl")
include("todense.jl")
using .ToDenseModule: todense
using Printf

function D(N,h)
    L = N * h
    scale = 2*pi / L
    p = zeros(Float64, N, N)

    for i in 1:N
        for j in 1:N
            if i != j
                p[i, j] = 0.5 * (-1)^(i - j) * cot(pi * (i - j) / N) * scale
            end
        end
    end

    return p
end

m = n = 28

x = range(0, 2*pi, length=m+1)[1:end-1]
y = range(0, 2*pi, length=n+1)[1:end-1]

W0_dense = [sin(xi)*sin(yj) for xi in x, yj in y]
W0 = LLRSVD(W0_dense, 1e-12)

Dt_steps = Int(round(2*pi / 0.1))
T = Dt_steps * 0.1

Wref = begin
    T = 2π
    ex = exp(T * second_derivative_matrix(m))
    ey = exp(T * second_derivative_matrix(n))
    ex * W0_dense * ey'
end

Dx = D(length(x), step(x))
Dy = D(length(y), step(y))


dt = 0.1
p = 7
tol = 1e-8
h = 2*pi / m

W_A, times_A, ranks_A = step_and_truncate(W0, Dx, Dy, dt, p, tol; strategy=:A)
W_B, times_B, ranks_B = step_and_truncate(W0, Dx, Dy, dt, p, tol; strategy=:B)

WA_dense = todense(W_A)
WB_dense = todense(W_B)

err_A = h * norm(WA_dense - Wref)
err_B = h * norm(WB_dense - Wref)

println("Strategy A error = ", err_A)
println("Strategy B error = ", err_B)

t = (1:length(times_A)) .* dt
using Printf

WA_dense = todense(W_A)
WB_dense = todense(W_B)

println("\nFinal Error Table")
println("------------------------------------------------------")
@printf("%-15s | %15.8e\n", "Strategy A", err_A)
@printf("%-15s | %15.8e\n", "Strategy B", err_B)
println("------------------------------------------------------")


