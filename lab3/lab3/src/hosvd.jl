function hosvd(A::Array{Float64,3}, r::Tuple{Int,Int,Int})::Tucker3
    r1, r2, r3 = r

    U1 = svd(unfold(A, 1)).U[:, 1:r1]
    U2 = svd(unfold(A, 2)).U[:, 1:r2]
    U3 = svd(unfold(A, 3)).U[:, 1:r3]

    G = mode_product(A, U1', 1)
    G = mode_product(G, U2', 2)
    G = mode_product(G, U3', 3)

    return Tucker3(G, U1, U2, U3)
end