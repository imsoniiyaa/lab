include("schrodinger.jl") 
# EX4

##(a)
using FFTW

Nx, Ny = 8, 8  

x = (1:Nx-1) ./ Nx
y = (1:Ny-1) ./ Ny

U = [sin(pi*xi) * sin(2*pi*yi) for xi in x, yi in y]

Γ = make_eigenvalues(Nx, Ny)
LU = apply_laplacian(U, Γ, Nx, Ny)
expected = -5 * pi^2 .* U

println("Max error in eigenvalue check: ", maximum(abs.(LU .- expected)))

##(b)
using Plots
Nx, Ny = 64, 64
Γ = make_eigenvalues(Nx, Ny)

x = (1:Nx-1) ./ Nx
y = (1:Ny-1) ./ Ny
X = repeat(x, 1, Ny-1)
Y = repeat(y', Nx-1, 1)

ψ0 = @. exp(-50*((X-0.3)^2 + (Y-0.5)^2)) * exp(im*20*X)  # 파라미터 조정
ψ0 ./= l2norm(ψ0, Nx, Ny)

dt = 5e-4
nsteps = 2000
snapshots, times = simulate(ψ0, Γ, dt, nsteps, Nx, Ny; save_every=1)

norms = [l2norm(s, Nx, Ny) for s in snapshots]
println("Max norm drift: ", maximum(abs.(norms .- 1.0)))

plot(times, norms,
     label="‖ψ‖₂",
     xlabel="time", ylabel="norm",
     ylims=(0.999, 1.001),
     legend=:bottomright)
hline!([1.0], linestyle=:dash, color=:red, label="expected")
savefig("figures/norm_conservation.png")

# Ex5

##(a)

##(b)

# Ex6