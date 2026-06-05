include("../src/unfold.jl")
include("../src/mode.product.jl")
include("../src/tucker3.jl")
include("../src/tucker_3.jl")
include("../src/todense.jl")
include("../src/hosvd.jl")


using LinearAlgebra

A = randn(30, 30, 30)
T = Tucker3(A, 1e-12)

err = norm(todense(T) - A)
println("norm of A - todense(Tucker3(A, 1e-12))= ", err)
println("1e-10 * norm(A)= ", 1e-10 * norm(A))
@assert err < 1e-10 * norm(A)
println("Verified that the error is within the specified tolerance.")