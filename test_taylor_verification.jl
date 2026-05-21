using LinearAlgebra
include("src/LLRSVD.jl")
using .LLRSVDModule
include("src/trunc_sum.jl")
include("src/applyL.jl")
include("src/taylor_step.jl")

# Create a simple test case
println("Testing taylor_step implementation...")

# Create a simple matrix and convert to LLRSVD
W0_dense = randn(10, 10)
W0 = LLRSVD(W0_dense, 1e-10)

# Create simple differential operators
Dx = randn(10, 10) * 0.1
Dy = randn(10, 10) * 0.1

# Test parameters
dt = 0.1
p = 3
tol = 1e-6

# Test taylor_step with strategy A
println("\nTesting taylor_step with strategy A (truncated)...")
try
    T_A = taylor_step(W0, Dx, Dy, dt, p, tol; strategy=:A)
    println("✓ taylor_step(:A) executed successfully")
    println("  - Number of terms: $(length(T_A))")
catch e
    println("✗ Error in taylor_step(:A): $e")
end

# Test taylor_step with strategy B
println("\nTesting taylor_step with strategy B (untruncated)...")
try
    T_B = taylor_step(W0, Dx, Dy, dt, p, tol; strategy=:B)
    println("✓ taylor_step(:B) executed successfully")
    println("  - Number of terms: $(length(T_B))")
catch e
    println("✗ Error in taylor_step(:B): $e")
end

# Test taylor_stepA directly
println("\nTesting taylor_stepA directly...")
try
    T_A_direct = taylor_stepA(W0, Dx, Dy, dt, p, tol)
    println("✓ taylor_stepA executed successfully")
catch e
    println("✗ Error in taylor_stepA: $e")
end

# Test taylor_stepB directly
println("\nTesting taylor_stepB directly...")
try
    T_B_direct = taylor_stepB(W0, Dx, Dy, dt, p, tol)
    println("✓ taylor_stepB executed successfully")
catch e
    println("✗ Error in taylor_stepB: $e")
end

println("\n✓ All tests completed!")
