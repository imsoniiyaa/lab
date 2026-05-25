function step_and_truncate(W0::LLRSVD, Dx, Dy, dt, p, tol; strategy=:A)
    W = W0
    TOL = dt^(p+1)
    nt = Int(round(2*pi/dt))

    ranks = zeros(Int, nt)
    times = zeros(Float64, nt)

    for i in 1:nt
        list = taylor_step(W, Dx, Dy, dt, p, tol; strategy=strategy)
        t0 = time_ns()
        W = trunc_sum(list, TOL)
        t1 = time_ns()

        ranks[i] = W.r
        times[i] = (t1 - t0) / 1e9
    end
    return W, times, ranks
end