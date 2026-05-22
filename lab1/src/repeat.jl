dts = [0.2, 0.1, 0.05]
errA = Float64[]
errB = Float64[]

for dt_val in dts
    global dt = dt_val
    println("dt = $dt")

    include("finalError.jl")

    push!(errA, err_A)
    push!(errB, err_B)
end

# Convergence
for i in 1:2
    pA = log(errA[i] / errA[i+1]) / log(2)
    pB = log(errB[i] / errB[i+1]) / log(2)
    @printf(" Strategy A order p ≈ %.4f\n", pA)
    @printf(" Strategy B order p ≈ %.4f\n\n", pB)
end