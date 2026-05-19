function D(N,h)
    L = N * h
    scale = 2*pi / L
    p = zeros(Float64, N, N)

    for i in 1:N
        for j in 1:N
            if i != j
                p[i, j] = 0.5 * (-1)^(i - j) * cot(pi * (i - j) / N) * scale
            end
        end
    end

    return p
end

function diff(x, y)
    Dx = D(length(x), step(x))
    Dy = D(length(y), step(y))
    return Dx, Dy
end
