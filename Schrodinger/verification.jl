include("schrodinger.jl") 
# EX4

##(a)
using FFTW

Nx, Ny = 8, 8  

x = (1:Nx-1) ./ Nx
y = (1:Ny-1) ./ Ny

U = [sin(pi*xi) * sin(2*pi*yi) for xi in x, yi in y]

Γ = make_eigenvalues(Nx, Ny)
LU = idst2d(Γ .* dst2d(U, Nx, Ny), Nx, Ny)
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

ψ0 = @. exp(-200*((X-0.3)^2 + (Y-0.5)^2)) * exp(im*40*X)
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
using LinearAlgebra

function exact_solution(psi0, Gamma, T, Nx, Ny)
    psi0hat = dst2d(psi0, Nx, Ny)
    E       = exp.(-im .* Gamma .* T)   
    return idst2d(E .* psi0hat, Nx, Ny)
end

function ex6()
    Nx, Ny = 128, 128
    x = (1:Nx-1) ./ Nx
    y = (1:Ny-1) ./ Ny
    X = repeat(x, 1, Ny-1)
    Y = repeat(y', Nx-1, 1)

    x0, y0 = 0.3, 0.3
    σ       = 0.04
    kx, ky  = 15*pi, 7*pi            

    ψ0 = @. exp(-((X-x0)^2 + (Y-y0)^2)/(2*σ^2)) * exp(im*(kx*X + ky*Y))
    ψ0 ./= l2norm(ψ0, Nx, Ny)

    Γ = make_eigenvalues(Nx, Ny)
    T = 0.05

    dts    = [0.02, 0.01, 0.005, 0.0025]
    errors = Float64[]

    println("\n--- Running Exercise 6 Refined Convergence Test ---")

    dt_ref = 1e-5
    n_ref = round(Int, T / dt_ref)
    println("Computing fine reference solution (this may take a few seconds)...")
    snaps_ref, _ = simulate(ψ0, Γ, dt_ref, n_ref, Nx, Ny; save_every=n_ref)
    U_ref = snaps_ref[end]

    for dt_test in dts
        n = ceil(Int, T / dt_test)   
        final_time = n * dt_test 
        snaps_cur, _ = simulate(ψ0, Γ, dt_test, n, Nx, Ny; save_every=n)
        U_cur = snaps_cur[end]
        
        U_ref = exact_solution(ψ0, Γ, final_time, Nx, Ny)
        
        err = l2norm(U_cur .- U_ref, Nx, Ny)
        push!(errors, err)
    end

    println("\nConvergence table:")
    println("dt\t\tError\t\t\tRate")
    for i in eachindex(dts)
        rate = i > 1 ? log(errors[i]/errors[i-1]) / log(dts[i]/dts[i-1]) : NaN
        println(dts[i], "\t\t", errors[i], "\t\t", round(rate, digits=2))
    end

    plot(log10.(dts), log10.(errors),
         xlabel="log₁₀(∆t)", ylabel="log₁₀(‖U−U_ref‖)",
         label="numerical error", marker=:circle, legend=:topleft)
    plot!(log10.(dts), 2*log10.(dts) .+ (log10.(errors[1]) - 2*log10.(dts[1])),
         linestyle=:dash, color=:red, label="O(∆t²) reference")
    savefig("figures/convergence.png")
end

ex6()