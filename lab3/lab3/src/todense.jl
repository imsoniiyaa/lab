function todense(T::Tucker3)::Array{Float64,3}
    A = mode_product(T.G, T.U1, 1)
    A = mode_product(A, T.U2, 2)
    A = mode_product(A, T.U3, 3)
    return A
end