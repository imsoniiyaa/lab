module LLRSVDModule
using LinearAlgebra

export LLRSVD, truncate

mutable struct LLRSVD
    U::Matrix{Float64}
    S::Vector{Float64}
    V::Matrix{Float64}
    m::Int
    n::Int
    r::Int
end

# Convenience constructor
function LLRSVD(U::Matrix{Float64}, S::Vector{Float64}, V::Matrix{Float64})
    m, r1 = size(U)
    r2 = length(S)
    n, r3 = size(V)
    @assert r1 == r2 == r3
    return LLRSVD(U, S, V, m, n, r1)
end

# Constructor W
function LLRSVD(W::Matrix{Float64}, TOL::Float64)
    F = svd(W)
    U, S, V = F.U, F.S, F.V
    s1 = S[1]
    keep = findall(s -> s >= TOL * s1, S)

    if isempty(keep)
        keep = [1]
    end

    return LLRSVD(U[:, keep], S[keep], V[:, keep],
                  size(W,1), size(W,2), length(keep))
end

# Truncate the SVD to a given tolerance
function truncate(A::LLRSVD, TOL::Float64)::LLRSVD
    S = A.S
    t = sum(S.^2)
    k = 0.0
    r = 0

    for i in 1:length(S)
        k += S[i]^2
        tail = t - k
        if tail <= TOL
            r = i
            break
        end
    end

    if r == 0
        r = length(S)
    end

    return LLRSVD(A.U[:,1:r], A.S[1:r], A.V[:,1:r], A.m, A.n, r)
end

end 
