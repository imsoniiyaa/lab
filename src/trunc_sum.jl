function trunc_sum(terms::Vector{LLRSVD}, TOL::Float64)::LLRSVD
    a = terms[1]
    for i in 2:length(terms)
        b = terms[i]
        #Concatenate the matrices
        u = [a.U b.U]
        v = [a.V b.V]
        s = Diagonal(vcat(a.S, b.S))
        #QR decomposition
        qu, ru = qr(u)
        qv, rv = qr(v)
        #Form the matrix to small
        mat = ru * s * rv'
        us, ss, vs = svd(mat)

        r = sum(ss .> TOL)
        r = max(r, 1)

        a = LLRSVD(qu[:, 1:r] * us[:, 1:r], ss[1:r], qv[:, 1:r] * vs[:, 1:r], a.m, a.n, r)
    end
    return a
end