function linear_interpolation(x1, xT, T)
    X = Vector{typeof(x1)}(undef, T)
    Δx = xT - x1
    Δt = T - 1
    for t = 1:T
        X[t] = x1 + Δx * (t - 1) / Δt
    end
    return X
end
