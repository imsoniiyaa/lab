include("tucker_add.jl")
include("tucker_round.jl")

function tucker_sum(terms::Vector{Tucker3}, tol::Float64)::Tucker3
    result = terms[1]
    for i in 2:length(terms)
        result = tucker_add(result, terms[i])
    end
    return tucker_round(result, tol)
end