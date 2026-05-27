using Test
using LinearAlgebra
include("../src/sat_rk4.jl")

@testset "RK4 step and truncate" begin
    N = 100
    r = 5
    u_0 = rand(N)
    U0 = zeros(N, r)
    S0 = zeros(r, r)
    V0 = zeros(N, r)
    D = rand(N, N)
    a = rand(N)
    dt = 0.01

    u_new, U_new, S_new, V_new = step_and_truncate_rk4(u_0, U0, S0, V0, D, a, dt, r)

    @test size(u_new) == (N,)
    @test size(U_new) == (N, r)
    @test size(S_new) == (r, r)
    @test size(V_new) == (N, r)

    @test rank(U_new * S_new * V_new') <= r
end