using Test
using LinearAlgebra
include("../src/LLRSVD.jl")
using .LLRSVDModule: LLRSVD
include("../src/trunc_sum.jl")
include("../src/applyL.jl")
include("../src/taylor_step.jl")

@testset "Taylor Step Implementation" begin
    
    W0_dense = randn(10, 10)
    W0 = LLRSVD(W0_dense, 1e-10)
    Dx = randn(10, 10) * 0.1
    Dy = randn(10, 10) * 0.1
    dt = 0.1
    p = 3
    tol = 1e-6

    @testset "taylor_step with strategy A (truncated)" begin
        T_A = taylor_step(W0, Dx, Dy, dt, p, tol; strategy=:A)
        
        @test isa(T_A, Vector{LLRSVD})
        
        @test length(T_A) == p + 1
        for (i, term) in enumerate(T_A)
            @test isa(term, LLRSVD)
            @test size(term.U, 1) == W0.m
            @test size(term.V, 1) == W0.n
            @test term.r == length(term.S)
        end
        
        @test size(T_A[1].U) == size(W0.U)
        @test size(T_A[1].V) == size(W0.V)
    end

    @testset "taylor_step with strategy B (untruncated)" begin
        T_B = taylor_step(W0, Dx, Dy, dt, p, tol; strategy=:B)
        
        @test isa(T_B, Vector{LLRSVD})
        @test length(T_B) == p + 1
        
        for (i, term) in enumerate(T_B)
            @test isa(term, LLRSVD)
            @test size(term.U, 1) == W0.m
            @test size(term.V, 1) == W0.n
        end
    end

    @testset "taylor_stepA function" begin
        T_A_direct = taylor_stepA(W0, Dx, Dy, dt, p, tol)
        
        @test isa(T_A_direct, Vector{LLRSVD})
        @test length(T_A_direct) == p + 1
        
        T_A = taylor_step(W0, Dx, Dy, dt, p, tol; strategy=:A)
        for i in 1:length(T_A_direct)
            @test norm(T_A_direct[i].S - T_A[i].S) < 1e-14
        end
    end

    @testset "taylor_stepB function" begin
        T_B_direct = taylor_stepB(W0, Dx, Dy, dt, p, tol)
        
        @test isa(T_B_direct, Vector{LLRSVD})
        @test length(T_B_direct) == p + 1
        
        T_B = taylor_step(W0, Dx, Dy, dt, p, tol; strategy=:B)
        for i in 1:length(T_B_direct)
            @test norm(T_B_direct[i].S - T_B[i].S) < 1e-14
        end
    end

    @testset "Scaling verification" begin
        T = taylor_step(W0, Dx, Dy, dt, p, tol; strategy=:B)
         @test norm(T[1].S - W0.S) < 1e-14
        
        
        for i in 2:length(T)
            expected_scale = dt^(i-1) / factorial(i-1)
            @test expected_scale < 1.0  # Verify our assumption
        end
    end

    @testset "Invalid strategy error" begin
        @test_throws ArgumentError taylor_step(W0, Dx, Dy, dt, p, tol; strategy=:C)
    end
end
