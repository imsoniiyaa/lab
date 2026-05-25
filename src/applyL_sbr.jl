using LinearAlgebra
using .LLRSVDModule: LLRSVD
include("trunc_sum.jl")

function applyL_sbr(W::LLRSVD, Dx, Dy, X, Y, TOL::Float64)::LLRSVD
    U = W.U
    S = W.S
    V = W.V

    U1 = Dx * U
    V1 = Y * V
    A = LLRSVD(U1, S, V1)

    U2 = X * U
    V2 = Dy * V
    B = LLRSVD(-U2, S, V2)

    return trunc_sum([A, B], TOL)  
end