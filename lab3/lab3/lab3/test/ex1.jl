include("../src/unfold.jl")

A = rand(4, 5, 6)
A1 = unfold(A, 1)
A2 = unfold(A, 2)
A3 = unfold(A, 3)

println("unfold(A,1) size: ", size(A1))
println("unfold(A,2) size: ", size(A2))
println("unfold(A,3) size: ", size(A3))
@assert size(A1) == (4, 30)
@assert size(A2) == (5, 24)
@assert size(A3) == (6, 20)
println("All passed for unfold function.")