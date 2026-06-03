function unfold(A::Array{Float64,3}, n::Int)::Matrix{Float64}
    n1, n2, n3 = size(A)
    if n == 1
        return reshape(permutedims(A, [1,2,3]), n1, n2*n3)
    elseif n == 2
        return reshape(permutedims(A, [2,1,3]), n2, n1*n3)
    else
        return reshape(permutedims(A, [3,1,2]), n3, n1*n2)
    end
end
