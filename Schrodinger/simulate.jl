# simulate.jl  — Complete reference solution
#
# Self-contained script that runs the Crank-Nicolson simulation and
# produces the target figures.  Compare your output from schrodinger.jl
# against the figures saved in figures/.
#
# Usage (from the Schro/ directory):
#   julia simulate.jl
#
# Install dependencies once:
#   using Pkg; Pkg.add(["FFTW", "Plots"])

using FFTW
using Plots
gr()

mkpath("figures")

# ─────────────────────────────────────────────────────────────────────────────
#  DST toolkit  (orthogonal normalization: S = S^{-1})
# ─────────────────────────────────────────────────────────────────────────────
#
# ψ is complex-valued.  Since FFTW.RODFT00 is a real-to-real transform,
# we apply it to Re(ψ) and Im(ψ) separately (complex dispatch below).

function dst1d(A::AbstractArray{<:Real}, N::Int; dims::Int=1)
    return FFTW.r2r(A, FFTW.RODFT00, dims) ./ sqrt(2N)
end

function dst1d(A::AbstractArray{<:Complex}, N::Int; dims::Int=1)
    return dst1d(real(A), N; dims=dims) .+ im .* dst1d(imag(A), N; dims=dims)
end

# 2D DST as two successive 1D transforms — explicit to aid generalization.
function dst2d(U, Nx::Int, Ny::Int)
    V = dst1d(U, Nx; dims=1)    # DST along x (each column)
    return dst1d(V, Ny; dims=2)  # DST along y (each row)
end

const idst2d = dst2d   # same operation (S = S^{-1})

# ─────────────────────────────────────────────────────────────────────────────
#  Core routines
# ─────────────────────────────────────────────────────────────────────────────

function make_eigenvalues(Nx::Int, Ny::Int)
    p = (1:Nx-1)
    q = (1:Ny-1)'
    return @. -(p * π)^2 - (q * π)^2
end

function cn_step(U, Γ, dt::Real, Nx::Int, Ny::Int)
    Uhat = dst2d(U, Nx, Ny)
    ρ = (1 .+ im*dt/2 .* Γ) ./ (1 .- im*dt/2 .* Γ)
    Uhat_new = ρ .* Uhat
    return idst2d(Uhat_new, Nx, Ny)
end


function simulate(ψ0, Γ, dt::Real, nsteps::Int, Nx::Int, Ny::Int;
                  save_every::Int = 1)
    snapshots = [copy(ψ0)]
    times     = [0.0]
    ψ = copy(ψ0)
    for n in 1:nsteps
        ψ = cn_step(ψ, Γ, dt, Nx, Ny)
        if n % save_every == 0
            push!(snapshots, copy(ψ))
            push!(times, n * dt)
        end
    end
    return snapshots, times
end

l2norm(U, Nx, Ny) = sqrt(sum(abs2.(U)) / (Nx * Ny))

# ─────────────────────────────────────────────────────────────────────────────
#  Grid and initial condition
# ─────────────────────────────────────────────────────────────────────────────

const Nx = 128
const Ny = 128
const xs = (1:Nx-1) ./ Nx
const ys = (1:Ny-1) ./ Ny
const dt = 5e-4

const Γ = make_eigenvalues(Nx, Ny)

X = xs .* ones(1, Ny-1)
Y = ones(Nx-1) .* ys'

# Gaussian wave packet with initial momentum k⃗ = (kx, ky)
x0, y0  = 0.30, 0.30
σ       = 0.04
kx, ky  = 15π, 7π

ψ0 = @. exp(-((X-x0)^2 + (Y-y0)^2) / (2σ^2)) * exp(im*(kx*X + ky*Y))
ψ0 ./= l2norm(ψ0, Nx, Ny)

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1: probability density snapshots, t ∈ [0, 0.075]
# ─────────────────────────────────────────────────────────────────────────────

# 8 snapshots at t = 0, 0.005, 0.010, …, 0.030, 0.035
total_steps   = 70                      # t_max = 0.035  (70 × 5e-4)
save_stride   = 10                      # every 10 steps → t = 0.005, 0.010, …, 0.035

snaps, times = simulate(ψ0, Γ, dt, total_steps, Nx, Ny; save_every=save_stride)

vmax = maximum(abs2.(snaps[1]))
vmin = vmax * 1e-4   # four decades of dynamic range (log scale)

plt1 = plot(layout=(2, 4), size=(1120, 580), dpi=150)
for (i, (s, t)) in enumerate(zip(snaps, times))
    heatmap!(plt1[i], xs, ys, log10.(clamp.(abs2.(s)', vmin, vmax)),
             title="t = $(round(t; digits=3))",
             xlabel="x", ylabel="y",
             clim=(log10(vmin), log10(vmax)),
             color=:inferno, colorbar=false, aspect_ratio=1)
end
savefig(plt1, "figures/wavepacket_snapshots.png")
println("Saved figures/wavepacket_snapshots.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2: norm conservation (run longer to show stability)
# ─────────────────────────────────────────────────────────────────────────────

snaps_n, times_n = simulate(ψ0, Γ, dt, 4000, Nx, Ny; save_every=8)
norms = [l2norm(s, Nx, Ny) for s in snaps_n]

plt2 = plot(times_n, norms,
            xlabel="t", ylabel="‖ψⁿ‖_{L²}",
            title="Discrete L² norm  (N=128, Δt = 5×10⁻⁴)",
            lw=1.8, color=:steelblue, legend=false,
            ylim=(0.9999985, 1.0000015), size=(700, 360), dpi=150)
hline!(plt2, [1.0], ls=:dash, color=:firebrick, lw=1.2)
savefig(plt2, "figures/norm_conservation.png")
println("Saved figures/norm_conservation.png")

println("Initial norm: ", l2norm(ψ0, Nx, Ny))
