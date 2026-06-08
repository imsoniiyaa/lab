function tucker_round(T::Tucker3, tol::Float64)::Tucker3
    G  = T.G
    U1 = T.U1
    U2 = T.U2
    U3 = T.U3

    F1 = svd(unfold(G, 1))
    F2 = svd(unfold(G, 2))
    F3 = svd(unfold(G, 3))

    threshold = tol / sqrt(3) * norm(G)

    r1 = max(1, sum(F1.S .>= threshold))
    r2 = max(1, sum(F2.S .>= threshold))
    r3 = max(1, sum(F3.S .>= threshold))

    Ub1 = F1.U[:, 1:r1]
    Ub2 = F2.U[:, 1:r2]
    Ub3 = F3.U[:, 1:r3]

    Gnew = mode_product(G, Matrix(Ub1'), 1)
    Gnew = mode_product(Gnew, Matrix(Ub2'), 2)
    Gnew = mode_product(Gnew, Matrix(Ub3'), 3)

    U1new = U1 * Ub1
    U2new = U2 * Ub2
    U3new = U3 * Ub3

    return Tucker3(Gnew, U1new, U2new, U3new)
end
