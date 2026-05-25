function step_and_truncate(W0::LLRSVD, Dx, Dy, dt, p, tol; strategy=:A)
    W = W0
    TOL = dt^(p+1)
    nt = Int(round(2*pi/dt))
    for i in 1:nt
        list = taylor_step(W, Dx, Dy, dt, p, tol; strategy=strategy)
        W = trunc_sum(list, TOL)
    end
    return W
end

function step_and_truncateA(W0::LLRSVD, Dx, Dy, dt, p, tol)
    return step_and_truncate(W0, Dx, Dy, dt, p, tol; strategy=:A)
end

function step_and_truncateB(W0::LLRSVD, Dx, Dy, dt, p, tol)
    return step_and_truncate(W0, Dx, Dy, dt, p, tol; strategy=:B)
end