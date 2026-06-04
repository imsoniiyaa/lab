mutable struct Tucker3
    G::Array{Float64,3}  # r1 x r2 x r3 core tensor
    U1::Matrix{Float64}  # n1 x r1
    U2::Matrix{Float64}  # n2 x r2
    U3::Matrix{Float64}  # n3 x r3
end