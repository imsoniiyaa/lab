function tucker_add(A::Tucker3, B::Tucker3)::Tucker3
    U1_new = hcat(A.U1, B.U1)   
    U2_new = hcat(A.U2, B.U2)
    U3_new = hcat(A.U3, B.U3)  

    r1A, r2A, r3A = size(A.G)
    r1B, r2B, r3B = size(B.G)

    G_new = zeros(r1A+r1B, r2A+r2B, r3A+r3B)
    G_new[1:r1A, 1:r2A, 1:r3A] = A.G
    G_new[r1A+1:end, r2A+1:end, r3A+1:end] = B.G

    return Tucker3(G_new, U1_new, U2_new, U3_new)
end

