#5-2
function tucker_inner(A::Tucker3, B::Tucker3)::Float64
    M1 = A.U1' * B.U1
    M2 = A.U2' * B.U2
    M3 = A.U3' * B.U3

    Hc = mode_product(B.G, M1, 1)
    Hc = mode_product(Hc, M2, 2)
    Hc = mode_product(Hc, M3, 3)

    return sum(A.G .* Hc)
end

#5-3
function tucker_norm(A::Tucker3)::Float64
    return sqrt(tucker_inner(A, A))
end
