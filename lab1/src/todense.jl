module ToDenseModule

using LinearAlgebra
import ..LLRSVDModule: LLRSVD

export todense

function todense(A::LLRSVD)::Matrix{Float64}
    return A.U * Diagonal(A.S) * A.V'
end

end
