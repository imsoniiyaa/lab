function taylor_stepA(W::LLRSVD, Dx, Dy, dt, p, tol)
    T = Vector{LLRSVD}(undef, p+1)
    Wk = W
    T[1] = LLRSVD(Wk.U, Wk.S * 1.0, Wk.V)

    for i in 1:p
        Wk = applyL(Wk, Dx, Dy, tol * dt^(-i))
        Sk = Wk.S .* dt^i / factorial(i)
        T[i+1] = LLRSVD(Wk.U, Sk, Wk.V)
    end
    return T
end

function taylor_stepB(W::LLRSVD, Dx, Dy, dt, p, tol)
    T = Vector{LLRSVD}(undef, p+1)
    Wk = W
    T[1] = LLRSVD(Wk.U, Wk.S * 1.0, Wk.V)

    for i in 1:p
        Wk = applyL(Wk, Dx, Dy, 0.0)
        Sk = Wk.S .* dt^i / factorial(i)
        T[i+1] = LLRSVD(Wk.U, Sk, Wk.V)
    end
    return T
end

function taylor_step(W0::LLRSVD, Dx, Dy, dt, p, tol; strategy::Symbol=:A)
    T = Vector{LLRSVD}(undef, p+1)
    Wk = W0
    T[1] = LLRSVD(Wk.U, Wk.S * 1.0, Wk.V)

    for i in 1:p
        if strategy == :A
            Wk = applyL(Wk, Dx, Dy, tol * dt^(-i))
        elseif strategy == :B
            Wk = applyL(Wk, Dx, Dy, 0.0)
        else
            throw(ArgumentError("Invalid strategy. Use :A or :B."))
        end
        Sk = Wk.S .* dt^i / factorial(i)
        T[i+1] = LLRSVD(Wk.U, Sk, Wk.V)
    end
    return T
end
