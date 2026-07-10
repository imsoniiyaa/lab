# schrodinger.jl
# Spectral Crank-Nicolson solver for the 2D Schrödinger equation
#   i ∂_t ψ = -Δψ  on (0,1)²,  ψ|_{∂Ω} = 0
#
# PROVIDED: DST toolkit and apply_laplacian helper.
# YOUR TASK: implement the functions marked TODO below.
#
# Install dependencies (run once in the Julia REPL):
#   using Pkg; Pkg.add(["FFTW", "Plots"])

using FFTW

# ─────────────────────────────────────────────────────────────────────────────
#  Discrete Sine Transform (DST-I, orthogonal normalization)
# ─────────────────────────────────────────────────────────────────────────────
#
# We use the normalization S_{jp} = sqrt(2/N) sin(πjp/N), which makes S an
# orthogonal matrix satisfying S = S^{-1} = Sᵀ.  Forward and inverse
# transforms are therefore the same operation — there is nothing extra to
# remember.
#
# FFTW.RODFT00 computes the unnormalized DST-I along dimension `dims`:
#   output[p,...] = 2 Σ_j A[j,...] sin(πjp/N),   p = 1,…,N-1
# Dividing by sqrt(2N) gives (S applied along that dimension).
#
# Since ψ is complex-valued, both methods below handle complex arrays by
# splitting into real and imaginary parts before applying the real-to-real
# FFTW transform.

"""
    dst1d(A, N; dims=1)

Normalized DST-I applied along dimension `dims` of array `A`.

- For a vector of length `N-1`, this is exactly `S * A`.
- For a matrix, `dims=1` transforms each column (x-direction),
  `dims=2` transforms each row (y-direction).
- For higher-dimensional arrays, `dims=k` applies the 1D transform
  along the k-th axis — useful for generalizing to 3D or tensor problems.

The normalization is chosen so that `S = S^{-1}`: applying `dst1d` twice
returns the original array (up to floating-point rounding).
"""
function dst1d(A::AbstractArray{<:Real}, N::Int; dims::Int=1)
    return FFTW.r2r(A, FFTW.RODFT00, dims) ./ sqrt(2N)
end

# ψ is complex: apply the real DST to Re(ψ) and Im(ψ) separately.
function dst1d(A::AbstractArray{<:Complex}, N::Int; dims::Int=1)
    return dst1d(real(A), N; dims=dims) .+ im .* dst1d(imag(A), N; dims=dims)
end

"""
    dst2d(U, Nx, Ny)

Normalized 2D DST of a matrix `U` of size `(Nx-1) × (Ny-1)`, implemented
as two successive 1D transforms:
  1. `dst1d(...; dims=1)` along x (each column),
  2. `dst1d(...; dims=2)` along y (each row).

Since `S = S^{-1}`, this function also serves as the inverse 2D DST —
applying it twice returns the original matrix.

Generalizing to 3D is straightforward: add a third call with `dims=3`.
"""
function dst2d(U, Nx::Int, Ny::Int)
    V = dst1d(U, Nx; dims=1)    # DST along x (each column)
    return dst1d(V, Ny; dims=2)  # DST along y (each row)
end

# Forward and inverse are the same operation (S = S^{-1}).
const idst2d = dst2d

"""
    apply_laplacian(U, Γ, Nx, Ny)

Apply the spectral 2D Laplacian to grid matrix U:
    Δ_h U = S_x (Γ ⊙ S_x U S_y) S_y = idst2d(Γ .* dst2d(U))

Requires `make_eigenvalues` (Exercise 1) to be implemented first.
"""
function apply_laplacian(U, Γ, Nx::Int, Ny::Int)
    return idst2d(Γ .* dst2d(U, Nx, Ny), Nx, Ny)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Exercise 1 — Eigenvalues of the spectral 2D Laplacian
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_eigenvalues(Nx, Ny)

Return an `(Nx-1) × (Ny-1)` real matrix Γ with entries
    Γ[p, q] = -(pπ)² - (qπ)²,   p = 1,…,Nx-1,  q = 1,…,Ny-1.

These are the eigenvalues of the spectral Laplacian Δ_h on the unit
square with zero Dirichlet boundary conditions.

Hint: build a column vector `p = 1:Nx-1` and a row vector `q = (1:Ny-1)'`,
then use broadcasting to form the matrix without explicit loops.
"""
function make_eigenvalues(Nx::Int, Ny::Int)
    p = (1:Nx-1) .* π
    q = (1:Ny-1)' .* π
    Γ = @. -(p^2) - (q^2)
    return Γ
end

    

# ─────────────────────────────────────────────────────────────────────────────
#  Exercise 2 — Crank-Nicolson time step
# ─────────────────────────────────────────────────────────────────────────────

"""
    cn_step(U, Γ, dt, Nx, Ny)

Advance the complex-valued solution `U` by one Crank-Nicolson step of
size `dt`, using the eigenvalue matrix `Γ` from `make_eigenvalues`.

In transform space, each mode evolves by the unimodular factor
    ρ[p,q] = (1 + im·dt/2·Γ[p,q]) / (1 − im·dt/2·Γ[p,q]),
so |ρ[p,q]| = 1 and the discrete L² norm is exactly preserved.

Algorithm (3 operations per step):
1. Forward 2D DST:    Û  ← dst2d(U, Nx, Ny)
2. Pointwise update:  Û  ← ρ .* Û     (ρ computed inline)
3. Inverse 2D DST:    U_new ← idst2d(Û, Nx, Ny)   (same as dst2d)
"""
function cn_step(U, Γ, dt::Real, Nx::Int, Ny::Int)
    Uhat = dst2d(U, Nx, Ny)   
    ρ = (1 .+ im * dt/2 .* Γ) ./ (1 .- im * dt/2 .* Γ)
    U_new = ρ .* Uhat
    return idst2d(U_new, Nx, Ny)
end



# ─────────────────────────────────────────────────────────────────────────────
#  Exercise 3 — Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate(ψ0, Γ, dt, nsteps, Nx, Ny; save_every=1)

Run the Crank-Nicolson simulation starting from the complex initial
condition `ψ0` for `nsteps` steps of size `dt`.

Returns `(snapshots, times)` where `snapshots[k]` is the solution matrix
at time `times[k]`.  The initial condition is always `snapshots[1]` at
`times[1] = 0`.

Arguments
- `ψ0`         : complex `(Nx-1)×(Ny-1)` matrix (interior grid values)
- `Γ`          : eigenvalue matrix from `make_eigenvalues`
- `dt`         : time step size
- `nsteps`     : total number of time steps
- `save_every` : record a snapshot every this many steps (default: 1)
"""
function simulate(ψ0, Γ, dt::Real, nsteps::Int, Nx::Int, Ny::Int;
                  save_every::Int = 1)
    ψ = complex(copy(ψ0))       
    snapshots = [copy(ψ)]      
    times     = [0.0]        

    for step in 1:nsteps
        ψ = cn_step(ψ, Γ, dt, Nx, Ny)   
        if step % save_every == 0
            push!(snapshots, copy(ψ))
            push!(times, step * dt)
        end
    end
    #only last one

    return snapshots, times
end

# ─────────────────────────────────────────────────────────────────────────────
#  Utility: discrete L² norm
# ─────────────────────────────────────────────────────────────────────────────

"""
    l2norm(U, Nx, Ny)

Approximate the L² norm of ψ from its interior grid values:
    ‖ψ‖_{L²} ≈ sqrt( Σ_{j,k} |U[j,k]|² · h_x · h_y ),   h_x=1/Nx, h_y=1/Ny.
"""
function l2norm(U, Nx::Int, Ny::Int)
    hx, hy = 1/Nx, 1/Ny
    return sqrt(sum(abs2.(U)) * hx * hy)
end
