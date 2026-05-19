using Test
using LinearAlgebra

include("../src/grid.jl")
include("../src/secondOrder.jl")
@testset "Exercise 3 — Finite Difference Verification" begin
  x, y, W0 = grid()
  Dx, Dy = diff(x, y)

  Wx_num = Dx * W0
  Wy_num = W0 * Dy'

  Wx_exact = cos.(x) * sin.(y)'
  Wy_exact = sin.(x) * cos.(y)'

  R = Wx_num + Wy_num - Wx_exact - Wy_exact

  err_F = norm(R)

  @test err_F < 1e-10
end