using LinearAlgebra

function step_and_truncate_rk4(u, U, S, V, D, a, dt, r)
# One coupled RK4 step of
# du/dt = -a .* (D u)
  # dX/dt = -diag (D u) - diag (a) D X with X = U S V’
    function truncate(U, S, V, r)
        M = S
        U_s, sigma, V_s = svd(M)
        k = min(r, length(sigma))
        return U * U_s[:, 1:k], Diagonal(sigma[1:k]), V * V_s[:, 1:k]
    end

    # --- helper: compute low-rank slope kX for given (U,S,V,u) ---
    function compute_kX(U, S, V, u, D, a, r)
    # For each stage k = 1..4:
    # 1. form stage state u_k = u + c_k * dt * ku_ {k -1}
    # and stage factors (U_k , S_k , V_k ) from previous
    # stage slope
    # 2. ku_k = -a .* (D u_k ) ( vector slope )
      # 3. kX_k slope for X:
        # a. left action -diag (a) D U_k . ( updates U)
        DU = D * U
        U_left = -(a .* DU)        

        # b. add diagonal forcing -diag (D u_k ) (rank -N term )
        Du = D * u                  
        diagF = -Du                 

        U2 = diagF                   
        S2 = ones(1, 1)         
        V2 = ones(length(u))      

        # c. concatenate factors , thin -QR on left and right , small SVD
        U_cat = [U_left  U2]         
        V_cat = [V  V2]              
        S_cat = [S  zeros(size(S,1),1);
                 zeros(1,size(S,2))  1]

        # d. TRUNCATE to exactly rank r ( keep top -r singular triplets )
        F1 = qr(U_cat)
        Q1 = Matrix(F1.Q)
        R1 = F1.R

        F2 = qr(V_cat)
        Q2 = Matrix(F2.Q)
        R2 = F2.R 

        # Combine the four ( ku_k ) into u_new ( standard RK4 weights ).
        M = R1 * S_cat * R2'
        U_s, sigma, V_s = svd(M)

        # Combine the four low - rank ( kX_k ) slopes into X_new (low - rank addition )
        k = min(r, length(sigma))
        U_new = Q1 * U_s[:, 1:k]
        S_new = Diagonal(sigma[1:k])
        V_new = Q2 * V_s[:, 1:k]

        # and re - truncate the result to rank r
        return U_new, S_new, V_new
    end
    function fX(u, X, D, a)
        return -diagm(D*u) - diagm(a) * D * X
    end

    ku1 = similar(u); ku2 = similar(u); ku3 = similar(u); ku4 = similar(u)
    ku1 .= -(a .* (D * u))
    kU1, kS1, kV1 = compute_kX(U, S, V, u, D, a, r)

    u2 = u + (1/2)*dt*ku1
    U2 = U + (1/2)*dt*kU1
    S2 = S + (1/2)*dt*kS1
    V2 = V + (1/2)*dt*kV1
    U2, S2, V2 = truncate(U2, S2, V2, r)

    ku2 .= -(a .* (D * u2))
    kU2, kS2, kV2 = compute_kX(U2, S2, V2, u2, D, a, r)

    u3 = u + (1/2)*dt*ku2
    U3 = U + (1/2)*dt*kU2
    S3 = S + (1/2)*dt*kS2
    V3 = V + (1/2)*dt*kV2
    U3, S3, V3 = truncate(U3, S3, V3, r)

    ku3 .= -(a .* (D * u3))
    kU3, kS3, kV3 = compute_kX(U3, S3, V3, u3, D, a, r)

    u4 = u + dt*ku3
    U4 = U + dt*kU3
    S4 = S + dt*kS3
    V4 = V + dt*kV3
    U4, S4, V4 = truncate(U4, S4, V4, r)

    ku4 .= -(a .* (D * u4))
    kU4, kS4, kV4 = compute_kX(U4, S4, V4, u4, D, a, r)

    u_new = u + (dt/6) * (ku1 + 2*ku2 + 2*ku3 + ku4)

    U_new = U + (dt/6) * (kU1 + 2*kU2 + 2*kU3 + kU4)
    S_new = S + (dt/6) * (kS1 + 2*kS2 + 2*kS3 + kS4)
    V_new = V + (dt/6) * (kV1 + 2*kV2 + 2*kV3 + kV4)

    U_new, S_new, V_new = truncate(U_new, S_new, V_new, r)

    return u_new, U_new, S_new, V_new
end

