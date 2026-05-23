using .LLRSVDModule: LLRSVD
include("trunc_sum.jl")
include("applyL_sbr.jl")

function taylor_step_sbr(W::LLRSVD, Dx, Dy, X, Y, dt, p, tol)
    T = Vector{LLRSVD}(undef, p+1)
    T[1] = LLRSVD(W.U, W.S, W.V)

    Wk = W
    for k in 1:p
        # compute kth derivative
        Wk = applyL_sbr(Wk, Dx, Dy, X, Y, tol)

        # scale 
        scale = dt^k / factorial(k)

        # store taylor term
        T[k+1] = LLRSVD(Wk.U, scale .* Wk.S, Wk.V)
    end

    return T
end
