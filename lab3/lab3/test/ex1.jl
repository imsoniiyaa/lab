include("../src/unfold.jl")
include("../src/mode.product.jl")

using LinearAlgebra

A = rand(4, 5, 6)
A1 = unfold(A, 1)
A2 = unfold(A, 2)
A3 = unfold(A, 3)

#Ex 1-1
println("unfold(A,1) size: ", size(A1))
println("unfold(A,2) size: ", size(A2))
println("unfold(A,3) size: ", size(A3))
@assert size(A1) == (4, 30)
@assert size(A2) == (5, 24)
@assert size(A3) == (6, 20)
println("All passed for unfold function.")

#Ex 1-2
M1 = rand(3, 4)
B = mode_product(A, M1, 1)
println("mode_product(A, M1, 1) size: ", size(B))
@assert size(B) == (3, 5, 6)
println("passed for mode_product with n=1.")

#Ex 1-3
M2 = rand(7, 5)

C1 = mode_product(mode_product(A, M1, 1), M2, 2)
C2 = mode_product(mode_product(A, M2, 2), M1, 1)

err = maximum(abs.(C1 - C2))
println("commutativity error: ", err)
@assert err < 1e-12
println("passed for commutativity of mode_product.")

#Ex 1-4
n = 20
pts = range(0, pi, length=n)

f1 = sin.(pts)
f2 = cos.(2 .* pts)
f3 = exp.(.-pts)

# A[i,j,k] = f1(xi) * f2(yj) * f3(zk)
A_sep = [f1[i] * f2[j] * f3[k] for i in 1:n, j in 1:n, k in 1:n]

A1 = unfold(A_sep, 1)
A2 = unfold(A_sep, 2)
A3 = unfold(A_sep, 3)

sv1 = svdvals(A1)
sv2 = svdvals(A2)
sv3 = svdvals(A3)

tol = 1e-10
r1 = sum(sv1 .> tol)
r2 = sum(sv2 .> tol)
r3 = sum(sv3 .> tol)

println("rank of mode-1 unfolding: ", r1)
println("rank of mode-2 unfolding: ", r2)
println("rank of mode-3 unfolding: ", r3)