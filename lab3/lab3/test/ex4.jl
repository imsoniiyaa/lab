using Test
using LinearAlgebra
using Plots

# helper function to unfold a 3D array along mode n
function unfold(A, n)
    if n == 1
        return reshape(A, size(A,1), :)
    elseif n == 2
        return reshape(permutedims(A, (2,1,3)), size(A,2), :)
    else
        return reshape(permutedims(A, (3,1,2)), size(A,3), :)
    end
end

# singular values of mode-n unfolding
function mode_svals(A)
    s1 = svd(unfold(A,1)).S
    s2 = svd(unfold(A,2)).S
    s3 = svd(unfold(A,3)).S
    return s1, s2, s3
end

@testset "Exercise 4: Singular Value Decay" begin
    n = 50

    #1
    xs = range(0, 2π, length=n)
    ys = range(0, 2π, length=n)
    zs = range(0, 2π, length=n)

    A1 = [sin(x+y+z) for x in xs, y in ys, z in zs]
    s1a, s2a, s3a = mode_svals(A1)

    r1a = sum(s1a .> 1e-6)
    r2a = sum(s2a .> 1e-6)
    r3a = sum(s3a .> 1e-6)

    println("\n(1) sin(x+y+z) ranks = ", (r1a, r2a, r3a))

    #2
    eps = 0.1
    xs = range(0, 1, length=n)
    ys = range(0, 1, length=n)
    zs = range(0, 1, length=n)

    A2 = [1/sqrt(x^2 + y^2 + z^2 + eps^2) for x in xs, y in ys, z in zs]
    s1b, s2b, s3b = mode_svals(A2)

    r1b = sum(s1b .> 1e-6)
    r2b = sum(s2b .> 1e-6)
    r3b = sum(s3b .> 1e-6)

    println("(2) 1/sqrt(x^2+y^2+z^2+eps^2) ranks = ", (r1b, r2b, r3b))

    #3
    xs = range(-1, 1, length=n)
    ys = range(-1, 1, length=n)
    zs = range(-1, 1, length=n)

    A3 = [tanh(10*(x^2 + y^2 - z)) for x in xs, y in ys, z in zs]
    s1c, s2c, s3c = mode_svals(A3)

    r1c = sum(s1c .> 1e-6)
    r2c = sum(s2c .> 1e-6)
    r3c = sum(s3c .> 1e-6)

    println("(3) tanh(10(x^2+y^2−z)) ranks = ", (r1c, r2c, r3c))

    #plotting singular value decay
    p1 = plot(s1a, yscale=:log10, label="mode-1", title="sin(x+y+z)")
    plot!(p1, s2a, label="mode-2")
    plot!(p1, s3a, label="mode-3")
    savefig(p1, "p1.png")
    display(p1)

    p2 = plot(s1b, yscale=:log10, label="mode-1", title="1/sqrt(x^2+y^2+z^2+eps^2)")
    plot!(p2, s2b, label="mode-2")
    plot!(p2, s3b, label="mode-3")
    savefig(p2, "p2.png")
    display(p2)

    p3 = plot(s1c, yscale=:log10, label="mode-1", title="tanh(10(x^2+y^2-z))")
    plot!(p3, s2c, label="mode-2")
    plot!(p3, s3c, label="mode-3")

    savefig(p3, "p3.png")
    display(p3)

    plot(p1, p2, p3, layout=(3,1), size=(800,1200))
end
