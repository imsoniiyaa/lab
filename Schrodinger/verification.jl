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
p0, q0 = 2, 3
lambda = -(p0*pi)^2 - (q0*pi)^2   

Nx, Ny = 64, 64
x = (1:Nx-1) ./ Nx
y = (1:Ny-1) ./ Ny
X = repeat(x, 1, Ny-1)
Y = repeat(y', Nx-1, 1)

ψ0 = complex.(sin.(p0*pi*X) .* sin.(q0*pi*Y))
ψ0 ./= l2norm(ψ0, Nx, Ny)

hx, hy = 1/Nx, 1/Ny
inner(u, v) = hx * hy * sum(conj.(u) .* v)

##(a) 
Γ = make_eigenvalues(Nx, Ny)
dt = 5e-4
nsteps = 200
snapshots, times = simulate(ψ0, Γ, dt, nsteps, Nx, Ny; save_every=1)

density_errors = [maximum(abs.(abs2.(s) .- abs2.(ψ0))) for s in snapshots]
println("Max |ψ|² drift over time: ", maximum(density_errors))

plot(times, density_errors,
     xlabel="time", ylabel="max |ψₙ|² − |ψ₀|²",
     label="density error", legend=:topright)
savefig("figures/density_error.png")


##(b) 
dts = [1e-2, 5e-3, 1e-3, 5e-4]
max_phase_errors = Float64[]

for dt_test in dts
    T = 0.01
    n = round(Int, T / dt_test)
    snaps, ts = simulate(ψ0, Γ, dt_test, n, Nx, Ny; save_every=1)

    θ_num = [angle(inner(ψ0, s)) for s in snaps]
    θ_ex  = [lambda * t for t in ts]   

    push!(max_phase_errors, maximum(abs.(θ_num .- θ_ex)))
end

println("\ntime step t scaling:")
println("dt\t\tmax phase error\t\t\trate")

for i in eachindex(dts)
    rate = i > 1 ? log(max_phase_errors[i]/max_phase_errors[i-1]) /
                   log(dts[i]/dts[i-1]) : NaN
    println(dts[i], "\t\t", max_phase_errors[i], "\t\t", round(rate, digits=2))
end

plot(log10.(dts), log10.(max_phase_errors),
     xlabel="log10(∆t)", ylabel="log₁₀(max phase error)",
     label="numerical", marker=:circle, legend=:topleft)
plot!(log10.(dts), 2*log10.(dts) .+ (log10.(max_phase_errors[1]) - 2*log10.(dts[1])),
     linestyle=:dash, label="O(∆t²) reference")
savefig("figures/time_step_scaling.png")



# Ex6