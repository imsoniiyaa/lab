using LinearAlgebra
include("../src/LLRSVD.jl")
using .LLRSVDModule: LLRSVD
include("../src/trunc_sum.jl")
include("../src/applyL.jl")
include("../src/taylor_step.jl")
include("../src/time.jl")
using Plots

W0_dense = randn(10, 10)
W0 = LLRSVD(W0_dense, 1e-10)
Dx = randn(10, 10) * 0.1
Dy = randn(10, 10) * 0.1

dt = 0.1
p = 3
tol = 1e-6

# Run both strategies and record rank/time at every step
W_A, times_A, ranks_A = step_and_truncate(W0, Dx, Dy, dt, p, tol; strategy=:A)
W_B, times_B, ranks_B = step_and_truncate(W0, Dx, Dy, dt, p, tol; strategy=:B)

# Time grid
t = (1:length(times_A)) .* dt

# Plot rank vs time
p_rank = plot(
    t, ranks_A,
    label = "Strategy A",
    xlabel = "Time",
    ylabel = "Rank",
    lw = 2,
    title = "Rank vs Time"
)
plot!(p_rank,
    t, ranks_B,
    label = "Strategy B",
    lw = 2
)
savefig(p_rank, "snapshots_ex8_rank.png")
println("Saved: snapshots_ex8_rank.png")


# Plot cumulative truncation time
cum_A = cumsum(times_A)
cum_B = cumsum(times_B)
p_cum = plot(
    t, cum_A,
    label = "Strategy A",
    xlabel = "Time",
    ylabel = "Cumulative Time (s)",
    lw = 2,
    title = "Cumulative trunc_sum Time vs Time"
)
plot!(p_cum,
    t, cum_B,
    label = "Strategy B",
    lw = 2
)
savefig(p_cum, "snapshots_ex8_cumtime.png")
println("Saved: snapshots_ex8_cumtime.png")
