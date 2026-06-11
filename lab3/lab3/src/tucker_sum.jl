include("tucker_add.jl")
include("tucker_round.jl")

function tucker_sum(terms::Vector{Tucker3}, tol::Float64)::Tucker3
    result = tucker_round(terms[1], tol)
    for i in 2:length(terms)
        result = tucker_round(tucker_add(result, terms[i]), tol)
    end
    return result
end