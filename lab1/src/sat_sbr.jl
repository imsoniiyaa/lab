using .LLRSVDModule: LLRSVD
include("trunc_sum.jl")
include("applyL_sbr.jl")

function step_and_truncate_sbr(W0::LLRSVD, Dx, Dy, X, Y, dt, p, tol)
    T = 2*pi
    N = round(Int, T/dt)
    dt = T/N
    W = W0

    for n in 1:N
        terms = taylor_step_sbr(W, Dx, Dy, X, Y, dt, p, tol)
        W = trunc_sum(terms, dt^(p+1)) 
    end

    return W
end
