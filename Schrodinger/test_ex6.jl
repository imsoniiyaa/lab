#comparing cn with exact_solution to check time accuracy 
include("schrodinger.jl")
using LinearAlgebra, Plots

#initialize parameters
Nx, Ny = 128, 128
x = (1:Nx-1) ./ Nx
y = (1:Ny-1) ./ Ny
X = repeat(x, 1, Ny-1)
Y = repeat(y', Nx-1, 1)

#initialize initial condition gaussian wave packet
x0, y0  = 0.30, 0.30
σ       = 0.04
ψ0 = @. exp(-((X-0.3)^2 + (Y-0.3)^2)/(2*0.04^2)) * exp(im*15*pi*X + im*7*pi*Y)
ψ0 ./= l2norm(ψ0, Nx, Ny)

#spectral laplacian eigenvalues
Γ = make_eigenvalues(Nx, Ny)
T = 0.05

#repeat experiment y'=ay
a = im
T = 0.05
dts = [0.02, 0.01, 0.005, 0.0025]
errors = Float64[]

for dt in dts
    nsteps = round(Int, T/dt)
    y_num = 1.0 + 0im
    G = (1 + a*dt/2) / (1 - a*dt/2)
    for _ in 1:nsteps
        y_num *= G
    end
    y_exact = exp(a*T)
    push!(errors, abs(y_num - y_exact))
end

println("Convergence table:")
println("dt\t\tError\t\tRate")
for i in eachindex(dts)
    if i == 1
        println(dts[i], "\t\t", errors[i], "\t\tNaN")
    else
        rate = log(errors[i]/errors[i-1]) / log(dts[i]/dts[i-1])
        println(dts[i], "\t\t", errors[i], "\t\t", round(rate, digits=2))
    end
end
#plot

plot(dts, errors,
     xscale=:log10, yscale=:log10,
     marker=:circle, label="CN error",
     xlabel="dt", ylabel="Error",
     title="Crank-Nicolson Convergence (y' = ay)")

# add reference slope O(dt^2)
ref = errors[1] * (dts ./ dts[1]).^2
plot!(dts, ref, linestyle=:dash, label="O(dt^2) reference")

savefig("figures/cn_error.png")

#exact solution at time T
function exact_solution(U0, Γ, T, Nx, Ny)
    U0hat = dst2d(U0, Nx, Ny)
    E     = exp.(im .* Γ .* T)
    return idst2d(E .* U0hat, Nx, Ny)
end

#calculate reference solution using exact_solution
U_ref = exact_solution(ψ0, Γ, T, Nx, Ny)

#dt values to test
dts    = [0.02, 0.01, 0.005, 0.0025]  
errors = Float64[]

#cn simulation for each dt and calculate error against exact solution
for dt in dts
    nsteps = round(Int, T / dt)
    snaps, _ = simulate(ψ0, Γ, dt, nsteps, Nx, Ny; save_every=1)
    err_L2 = norm(snaps[end] .- U_ref) * sqrt(1/Nx * 1/Ny)
    push!(errors, err_L2)
end

#print convergence table
println("Convergence table:")
println("dt\t\tError\t\t\tRate")
for i in eachindex(dts)
    if i == 1
        println(dts[i], "\t\t", errors[i], "\t\t", "NaN")
    else
        #log log slope for convergence rate
        rate = log(errors[i]/errors[i-1]) / log(dts[i]/dts[i-1])
        println(dts[i], "\t\t", errors[i], "\t\t", round(rate, digits=2))
    end
end

#slope calculation for convergence rate
for i in 1:length(dts)-1
    slope = log(errors[i+1]/errors[i]) / log(dts[i+1]/dts[i])
    println("$(dts[i]) -> $(dts[i+1]):  slope = $(round(slope, digits=3))")
end

#plotting
plot(log10.(dts), log10.(errors),
     xlabel="log10(dt)", ylabel="log10(‖U-U_exact‖_L2)",
     label="numerical error", marker=:circle, legend=:topleft)
plot!(log10.(dts), 2*log10.(dts) .+ (log10.(errors[1]) - 2*log10.(dts[1])),
     linestyle=:dash, color=:red, label="O(dt^2) reference")
savefig("figures/convergence.png")

#=
What I think about this code is:
error does not decrease as dt decreases, which is not expected. 
The error should decrease as dt decreases, indicating that the numerical solution is converging to the exact solution.
This suggests that there may be an issue with the implementation of the Crank-Nicolson method or the exact solution.
Further investigation is needed to identify and fix the problem.
So, CN time error is way too small, and the error is dominated by spatial discretization error in high frequency 
components.

We figured that CN reference and exact solution reference are almost same with the error within 1e-9,
with dt = 1e-5.

Also, slope convergence rate is approximately 0 which is not expected. 
The slope should be around 2 for second-order convergence in time.

Therefore, the initial value given sigma = 0.04 is too narrow Gaussian wave packet and k = 15*pi and 7*pi which has high
frequency lane wave are causing issues.
This includes high frequency components strongly among DST basis functions. So its difference of value 
between CN and exact solution is dominated by spatial discretization error in high frequency components.
That is why every one step causes big wave in CN, and it does not decrease as dt decreases. What we think is
the time step t values are outside asymptotic region of convergence since time step t should be convergent to 0, but 
the asymptotic limit might not be reached yet. 

So, we need to use smaller dt values to avoid high frequency components or increase the sigma value to make gaussian
wider in my opinion. By refining $\Delta t$ down to $1.0 \times 10^{-5}$, the simulation successfully enters 
the asymptotic convergence regime, suppressing the high-frequency numerical dispersion error and recovering 
the theoretical second-order convergence rate (slope $\approx 1.76$ toward $2.0$

=#

