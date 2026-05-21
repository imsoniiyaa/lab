function grid()
    m = 64
    n = 64

    x = range(0, 2*pi, length=m+1)[1:end-1]
    y = range(0, 2*pi, length=n+1)[1:end-1]

    u_x = sin.(x)
    u_y = sin.(y)

    W0 = u_x * u_y'

    return x, y, W0
end