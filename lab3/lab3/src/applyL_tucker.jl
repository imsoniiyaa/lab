include("tucker_sum.jl")

function applyL_tucker(W::Tucker3, Dx, Dy, Dz, tol)
    T1 = Tucker3(W.G, Dx * W.U1, W.U2, W.U3)
    T2 = Tucker3(W.G, W.U1, Dy * W.U2, W.U3)
    T3 = Tucker3(W.G, W.U1, W.U2, Dz * W.U3)
    return tucker_sum([T1, T2, T3], max(tol, 1e-14)) 
end
