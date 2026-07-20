# pml_fd.jl
# Finite-difference Schrödinger with PML in x (and optionally y)
#
#   i ∂_t u = - v_x ∂_x (v_x ∂_x u) - v_y ∂_y (v_y ∂_y u)
#
# Grid: interior points only, size (Nx-1) × (Ny-1)
# Dirichlet boundary: u = 0 outside [0,1]²

using LinearAlgebra

"""
    make_pml_v(x, x0_left, x0_right, d, σ_max, p, γ)

Build the 1D PML coefficient v(x) for grid points `x`.

- Left PML:  x ∈ [0, x0_left],  σ(x) = σ_max * ((x0_left - x)/d)^p
- Right PML: x ∈ [x0_right, 1], σ(x) = σ_max * ((x - x0_right)/d)^p
- Interior:  σ(x) = 0

Returns complex vector `v` with
    v[j] = 1 / (1 + exp(im*γ) * σ[j])
"""
function make_pml_v(x, x0_left, x0_right, d, σ_max, p, γ)
    σ = zeros(Float64, length(x))
    for (j, xj) in enumerate(x)
        if xj < x0_left
            σ[j] = σ_max * ((x0_left - xj) / d)^p
        elseif xj > x0_right
            σ[j] = σ_max * ((xj - x0_right) / d)^p
        end
    end
    return @. 1.0 / (1.0 + exp(im*γ) * σ)
end

"""
    tridiagonal_solve(dl, d, du, b)

Thomas algorithm for tridiagonal system.
"""
function tridiagonal_solve(dl, d, du, b)
    n = length(b)
    c  = copy(du)
    d_ = copy(d)
    b_ = copy(b)

    for i in 2:n
        m = dl[i] / d_[i-1]
        d_[i] -= m * c[i-1]
        b_[i] -= m * b_[i-1]
    end

    x = zeros(ComplexF64, n)
    x[n] = b_[n] / d_[n]
    for i in n-1:-1:1
        x[i] = (b_[i] - c[i]*x[i+1]) / d_[i]
    end
    return x
end

"""
    cn_step_pml(U, vx, vy, dt, Nx, Ny)

One Crank–Nicolson step for
    i ∂_t u = - v_x ∂_x (v_x ∂_x u) - v_y ∂_y (v_y ∂_y u)

- `U`  : complex matrix of size (Nx-1) × (Ny-1)
- `vx` : PML coefficient in x (length Nx-1)
- `vy` : PML coefficient in y (length Ny-1)
- `dt` : time step
- `Nx`, `Ny`: total grid points including boundaries
"""
function cn_step_pml(U, vx, vy, dt, Nx::Int, Ny::Int)
    hx = 1.0 / Nx
    hy = 1.0 / Ny
    nx = Nx - 1
    ny = Ny - 1

    U_half = similar(U)
    for k in 1:ny
        rhs = zeros(ComplexF64, nx)
        dl  = zeros(ComplexF64, nx)
        d_  = zeros(ComplexF64, nx)
        du  = zeros(ComplexF64, nx)

        for j in 1:nx
            vph = j < nx ? 0.5 * (vx[j] + vx[j+1]) : vx[j]
            vmh = j > 1  ? 0.5 * (vx[j] + vx[j-1]) : vx[j]

            α = dt / (2 * hx^2)

            d_[j] = im - α * vx[j] * (-vph - vmh)
            if j < nx
                du[j] = -α * vx[j] * vph
            end
            if j > 1
                dl[j] = -α * vx[j] * vmh
            end

            uj  = U[j, k]
            ujp = j < nx ? U[j+1, k] : 0.0 + 0im
            ujm = j > 1  ? U[j-1, k] : 0.0 + 0im

            rhs[j] = (im + α * vx[j] * (-vph - vmh)) * uj +
                     α * vx[j] * vph * ujp +
                     α * vx[j] * vmh * ujm
        end

        U_half[:, k] = tridiagonal_solve(dl, d_, du, rhs)
    end

    U_new = similar(U_half)
    for j in 1:nx
        rhs = zeros(ComplexF64, ny)
        dl  = zeros(ComplexF64, ny)
        d_  = zeros(ComplexF64, ny)
        du  = zeros(ComplexF64, ny)

        for k in 1:ny
            vph = k < ny ? 0.5 * (vy[k] + vy[k+1]) : vy[k]
            vmh = k > 1  ? 0.5 * (vy[k] + vy[k-1]) : vy[k]

            α = dt / (2 * hy^2)

            d_[k] = im - α * vy[k] * (-vph - vmh)
            if k < ny
                du[k] = -α * vy[k] * vph
            end
            if k > 1
                dl[k] = -α * vy[k] * vmh
            end

            uk  = U_half[j, k]
            ukp = k < ny ? U_half[j, k+1] : 0.0 + 0im
            ukm = k > 1  ? U_half[j, k-1] : 0.0 + 0im

            rhs[k] = (im + α * vy[k] * (-vph - vmh)) * uk +
                     α * vy[k] * vph * ukp +
                     α * vy[k] * vmh * ukm
        end

        U_new[j, :] = tridiagonal_solve(dl, d_, du, rhs)
    end

    return U_new
end
