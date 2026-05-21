function applyL(W::LLRSVD, Dx, Dy, TOL::Float64)::LLRSVD
    U = W.U
    V = W.V
    S = W.S
    U1 = Dx * U
    a = LLRSVD(U1, S, V)

    V1 = Dy * V
    b = LLRSVD(U, S, V1)

    return trunc_sum([a, b], TOL)
end
