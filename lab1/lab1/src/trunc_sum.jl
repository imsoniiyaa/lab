function trunc_sum(terms::Vector{LLRSVD}, TOL::Float64)::LLRSVD
    if length(terms) == 1
        return terms[1]
    end

    U_cat = hcat([t.U for t in terms]...)
    V_cat = hcat([t.V for t in terms]...)
    S_cat = vcat([t.S for t in terms]...)

    qu, ru = qr(U_cat)
    qv, rv = qr(V_cat)

    mat = ru * Diagonal(S_cat) * rv'

    if any(isnan, mat) || any(isinf, mat)
        return terms[1]
    end

    us, ss, vs = svd(mat)

    r = max(sum(ss .> TOL), 1)
    m = terms[1].m
    n = terms[1].n

    return LLRSVD(Matrix(qu) * us[:, 1:r], ss[1:r], Matrix(qv) * vs[:, 1:r], m, n, r)
end