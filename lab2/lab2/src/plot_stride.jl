using LinearAlgebra
using SummationByPartsOperators
using Plots

xmin, xmax, N = -5.0, 5.0, 500
a_fun(x) = 1.0 - 0.3 * exp(-36 * ((x - 2.5)/(2.5))^2)
u0_fun(x,a) = exp(-a*(x)^2) + exp(-a*(x-10)^2) + exp(-a*(x+10)^2)

Du = upwind_operators(periodic_derivative_operator; derivative_order=1, accuracy_order=4, xmin=xmin, xmax=xmax, N=N)
D = Du.minus
x = SummationByPartsOperators.grid(D)
a = a_fun.(x)
u0 = u0_fun.(x, 10)

h = step(x)
dt = h / maximum(abs.(a))
Nt = ceil(Int, 10.0 / dt)
dt = 10.0 / Nt
PLOT_STRIDE = 10

function rhs!(du, dS, u, S, D, a, ux, DS)
    mul!(ux, D, u)
    @. du = -a * ux
    for j in axes(S, 2); mul!(view(DS,:,j), D, view(S,:,j)); end
    @. dS = -a * DS
    for i in eachindex(ux); dS[i,i] -= ux[i]; end
end

u = copy(u0)
S = zeros(N, N)
ux = similar(u); DS = similar(S)
k1u=similar(u); k2u=similar(u); k3u=similar(u); k4u=similar(u); utmp=similar(u)
k1S=similar(S); k2S=similar(S); k3S=similar(S); k4S=similar(S); Stmp=similar(S)

epsilons = [1e-4, 1e-6, 1e-8, 1e-10]
t_log = Float64[]
rank_log = [Int[] for _ in epsilons]

for n in 1:Nt
    rhs!(k1u,k1S,u,S,D,a,ux,DS)
    @. utmp=u+0.5*dt*k1u; @. Stmp=S+0.5*dt*k1S
    rhs!(k2u,k2S,utmp,Stmp,D,a,ux,DS)
    @. utmp=u+0.5*dt*k2u; @. Stmp=S+0.5*dt*k2S
    rhs!(k3u,k3S,utmp,Stmp,D,a,ux,DS)
    @. utmp=u+dt*k3u; @. Stmp=S+dt*k3S
    rhs!(k4u,k4S,utmp,Stmp,D,a,ux,DS)
    @. u=u+(dt/6)*(k1u+2k2u+2k3u+k4u)
    @. S=S+(dt/6)*(k1S+2k2S+2k3S+k4S)

    if n % PLOT_STRIDE == 0
        sv = svdvals(S)
        push!(t_log, n*dt)
        for (i, eps) in enumerate(epsilons)
            push!(rank_log[i], sum(sv ./ sv[1] .> eps))
        end
        println("t=$(round(n*dt, digits=2)): ranks = $(last.(rank_log))")
    end
end

# Plot
plt = plot(title="Effective rank", xlabel="t", ylabel="rε", yscale=:log10, legend=:topright)
colors = [:blue, :red, :green, :purple]
for (i, eps) in enumerate(epsilons)
    plot!(plt, t_log, rank_log[i], label="ε=1e$(Int(log10(eps)))", lw=2, color=colors[i])
end
savefig(plt, "rank.png")
display(plt)
println("Saved")