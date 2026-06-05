function Tucker3(A::Array{Float64,3}, tol::Float64)::Tucker3
    threshold = tol / sqrt(3) * norm(A)

    F1 = svd(unfold(A, 1))
    F2 = svd(unfold(A, 2))
    F3 = svd(unfold(A, 3))

    r1 = max(1, sum(F1.S .>= threshold))
    r2 = max(1, sum(F2.S .>= threshold))
    r3 = max(1, sum(F3.S .>= threshold))

    U1 = F1.U[:, 1:r1]
    U2 = F2.U[:, 1:r2]
    U3 = F3.U[:, 1:r3]

    # Core tensor
    G = mode_product(A, Matrix(U1'), 1)
    G = mode_product(G, Matrix(U2'), 2)
    G = mode_product(G, Matrix(U3'), 3)


    return Tucker3(G, U1, U2, U3)
end