function mode_product(A::Array{Float64,3}, M::Matrix{Float64}, n::Int)::Array{Float64,3}
    n1, n2, n3 = size(A)
    s = size(M, 1)
    
    An = unfold(A, n)
    B_unfolded = M * An
    
    if n == 1
        return permutedims(reshape(B_unfolded, s, n2, n3), [1,2,3])
    elseif n == 2
        return permutedims(reshape(B_unfolded, s, n1, n3), [2,1,3])
    else
        return permutedims(reshape(B_unfolded, s, n1, n2), [2,3,1])
    end
end