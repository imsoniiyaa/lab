include("schrodinger.jl")
using Plots
gr()
mkpath("figures")

Nx, Ny = 128, 128
x = (1:Nx-1) ./ Nx
y = (1:Ny-1) ./ Ny
X = repeat(x, 1, Ny-1)
Y = repeat(y', Nx-1, 1)

ψ0 = @. exp(-((X-0.3)^2 + (Y-0.3)^2)/(2*0.04^2)) * exp(im*15*pi*X + im*7*pi*Y)
ψ0 ./= l2norm(ψ0, Nx, Ny)

Γ = make_eigenvalues(Nx, Ny)
dt = 5e-4

total_steps = 70        
save_stride = 10        

snapshots, times = simulate(ψ0, Γ, dt, total_steps, Nx, Ny; save_every=save_stride)

vmax = maximum(abs2.(snapshots[1]))
vmin = vmax * 1e-4  

plt = plot(layout=(2,4), size=(1200,600), dpi=150)

for (i, (s, t)) in enumerate(zip(snapshots, times))
    density = abs2.(s)'  
    log_density = log10.(clamp.(density, vmin, vmax))
    
    heatmap!(plt[i],
             x, y, log_density,
             title = "t = $(round(t, digits=3))",
             xlabel = "x", ylabel = "y",
             clim = (log10(vmin), log10(vmax)),
             color = :inferno,
             colorbar = false,
             aspect_ratio = 1)
end

savefig(plt, "figures/wavepacket_plot.png")
println("Saved figures/wavepacket_plot.png")