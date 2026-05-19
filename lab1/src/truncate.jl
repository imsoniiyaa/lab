
module LLRSVDModule

using .LLRSVDModule

function truncate(A::LLRSVD, TOL::Float64)::LLRSVD
  S = A.S
  t = sum(S.^2)
  k = 0.0
  r = 0

  for i in 1:length(S)
       k += S[i]^2
       tail = t-k

       if tail <= TOL
         r = i
         break
       end
 end
 if r == 0
     r = length(S)
 end
 return LLRSVD(A.U[:,1:r], A.S[1:r], A.V[:,1:r], A.m,A.n,r)
end

end