using LinearAlgebra

function truncate_rank(X, r)

    F = svd(X)

    k = min(r, length(F.S))

    U = F.U[:, 1:k]
    S = Diagonal(F.S[1:k])
    V = F.Vt[1:k, :]'

    return U, S, V
end


function step_rk4_dense(u, X, D, a, dt)

    f_u(u) = -(a .* (D * u))

    function f_X(u, X)
        Du = D * u
        return -diagm(Du) - diagm(a) * D * X
    end

    # RK4

    k1u = f_u(u)
    k1X = f_X(u, X)

    k2u = f_u(u + 0.5dt * k1u)
    k2X = f_X(
        u + 0.5dt * k1u,
        X + 0.5dt * k1X
    )

    k3u = f_u(u + 0.5dt * k2u)
    k3X = f_X(
        u + 0.5dt * k2u,
        X + 0.5dt * k2X
    )

    k4u = f_u(u + dt * k3u)
    k4X = f_X(
        u + dt * k3u,
        X + dt * k3X
    )

    u_new =
        u + (dt / 6) * (
            k1u + 2k2u + 2k3u + k4u
        )

    X_new =
        X + (dt / 6) * (
            k1X + 2k2X + 2k3X + k4X
        )

    return u_new, X_new
end


function step_and_truncate_rk4(u, U, S, V, D, a, dt, r)
    X = U * S * V'

    f_u(u) = -(a .* (D * u))
    f_X(u, X) = -diagm(D * u) - diagm(a) * D * X

    k1u = f_u(u)
    k1X = f_X(u, X)

    k2u = f_u(u + 0.5dt * k1u)
    k2X = f_X(u + 0.5dt * k1u, X + 0.5dt * k1X)

    k3u = f_u(u + 0.5dt * k2u)
    k3X = f_X(u + 0.5dt * k2u, X + 0.5dt * k2X)

    k4u = f_u(u + dt * k3u)
    k4X = f_X(u + dt * k3u, X + dt * k3X)

    u_new = u + (dt/6) * (k1u + 2k2u + 2k3u + k4u)
    X_new = X + (dt/6) * (k1X + 2k2X + 2k3X + k4X)

    
    U_new, S_new, V_new = truncate_rank(X_new, r)

    return u_new, U_new, S_new, V_new
end



function run_lowrank_solver(u0, D, a, dt, T, r)

    N = length(u0)

    X0 = zeros(N, N)

    U, S, V = truncate_rank(X0, r)

    u = copy(u0)

    steps = Int(round(T / dt))

    total_time = 0.0

    for n in 1:steps

        t = @elapsed begin

            u, U, S, V = step_and_truncate_rk4(u, U, S, V, D, a, dt, r)

        end

        total_time += t
    end

    X_lr = U * S * V'

    return X_lr, total_time / steps
end


function run_dense_solver(u0, D, a, dt, T)

    N = length(u0)

    X = zeros(N, N)

    u = copy(u0)

    steps = Int(round(T / dt))

    for n in 1:steps

        u, X = step_rk4_dense(u, X, D, a, dt)

    end

    return X
end



function rel_error(X_lr, X_dense)

    return norm(X_lr - X_dense) / norm(X_dense)

end