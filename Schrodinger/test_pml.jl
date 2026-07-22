let
    include("schrodinger.jl")
    include("pml_fd.jl")

    using LinearAlgebra, Plots

    Nx, Ny = 64, 64
    x = (1:(Nx-1)) ./ Nx
    y = (1:(Ny-1)) ./ Ny
    X = repeat(x, 1, Ny-1)
    Y = repeat(y', Nx-1, 1)

    T = 0.1
    dt = 0.001
    nsteps = round(Int, T/dt)

    ψ0 = @. exp(-((X-0.3)^2 + (Y-0.3)^2)/(2*0.04^2)) * exp(im*15*pi*X + im*7*pi*Y)
    ψ0 ./= l2norm(ψ0, Nx, Ny)

    γ_pml = π/4
    σ_max = 20.0
    d_pml = 0.1
    p_poly = 4

    x0_left = d_pml
    x0_right = 1.0 - d_pml

    vx_pml = make_pml_v(x, x0_left, x0_right, d_pml, σ_max, p_poly, γ_pml)
    vy_pml = ones(Ny-1)

    ψ_pml = copy(ψ0)
    for _ in 1:nsteps
        ψ_pml = cn_step_pml(ψ_pml, vx_pml, vy_pml, dt, Nx, Ny)
    end

    p1 = heatmap(x, y, abs.(ψ_pml) .^ 2', title="With PML")
    p2 = heatmap(x, y, abs.(ψ0) .^ 2', title="Initial Condition")
    plot(p1, p2, layout=(1, 2))

    contour(x, y, abs.(ψ_pml) .^ 2',
        xlabel="x", ylabel="y",
        title="|ψ|^2 with PML (contour)",
        aspect_ratio=1)

    savefig("figures/pml_effect.png")
    println("Saved figure: pml_effect.png")
end
