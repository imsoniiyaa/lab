module ToElemModule

import ..LLRSVDModule: LLRSVD

export toelem

function toelem(A::LLRSVD, i::Int, j::Int)::Float64
    return dot(A.U[i, :], A.S .* A.V[j, :])
end

end 
