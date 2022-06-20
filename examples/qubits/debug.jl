using DirectTrajectoryOptimization
using LinearAlgebra
using CairoMakie

X = [
    0 1;
    1 0
]

Y = [
    0 -im;
    im 0
]

Z = [
    1 0;
    0 -1
]

f = 0.5

H_drift = f * Z

H_drive = X

ψ0 = [1.0, 0.0]
ψ1 = [0.0, 1.0]

gate = X
ψi = ψ0
ψf = X * ψi

ket_to_iso(ψ) = [real(ψ[1]), imag(ψ[1]), real(ψ[2]), imag(ψ[2])]
iso_to_ket(ψ̃) = [ψ̃[1] + im * ψ̃[2], ψ̃[3] + im * ψ̃[4]]

Id2 = I(2)
Im2 = [
    0 -1;
    1  0
]

⊗(A, B) = kron(A, B)

G(H) = Id2 ⊗ imag(H) - Im2 ⊗ real(H)

G_drift = G(H_drift)
G_drive = G(H_drive)

schroedinger(x, u, w) = (G_drift + u[1] * G_drive) * x

h = 0.01
Id = I(4)

# general implicit midpoint method
function midpoint_implicit(y, x, u, w)
    return y - (x + h * schroedinger(0.5 * (x + y), u, w))
end

# analytic solution of midpoint equation for schroedinger dynamics
function pade_schroedinger(y, x, u, w)
    G = G_drift + u[1] * G_drive
    return y - inv(Id - h / 2 * G) * (Id + h / 2 * G) * x
end

ψ̃i = ket_to_iso(ψi)
ψ̃f = ket_to_iso(ψf)

function cost1(ψ̃, ψ̃f)
    ψ = iso_to_ket(ψ̃)
    ψf = iso_to_ket(ψ̃f)
    amp = ψ'ψf
    return min(abs(1 - amp), abs(1 + amp))
end

function cost2(ψ̃, ψ̃f)
    ψ = iso_to_ket(ψ̃)
    ψf = iso_to_ket(ψ̃f)
    amp = ψ'ψf
    return min(abs(1 - real(amp)), abs(1 + real(amp)))
end

function cost3(ψ̃, ψ̃f)
    ψ = iso_to_ket(ψ̃)
    ψf = iso_to_ket(ψ̃f)
    amp = ψ'ψf
    return abs(1 - abs(real(amp)) + abs(imag(amp)))
end

function cost4(ψ̃, ψ̃f)
    return min(abs(1 - dot(ψ̃, ψ̃f)), abs(1 + dot(ψ̃, ψ̃f)))
end

costi(ψ̃) = ψ̃ - ψ̃i
costf(ψ̃) = cost4(ψ̃, ψ̃f)

T = 1000
num_state = 4
num_action = 1
eval_hess=true

d_t = Dynamics(
    # pade_schroedinger,
    midpoint_implicit,
    num_state,
    num_state,
    num_action,
    evaluate_hessian=eval_hess
)

dynamics = [d_t for t = 1:T]

ot = (x, u, w) -> costf(x) .+ 0.1 * dot(u, u)
oT = (x, u, w) -> costf(x)
ct = Cost(ot, num_state, num_action; evaluate_hessian=eval_hess)
cT = Cost(oT, num_state, num_action; evaluate_hessian=eval_hess)
objective = [[ct for t = 1:T-1]; cT]

bnd1 = Bound(num_state, num_action)
bndt = Bound(num_state, num_action)
bndT = Bound(num_state, 0)
bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

con1 = Constraint((x, u, w) -> costi(x), num_state, num_action, evaluate_hessian=eval_hess)
conT = Constraint((x, u, w) -> [costf(x)], num_state, num_action, evaluate_hessian=eval_hess)
constraints = [con1, [Constraint() for t = 2:T-1]..., conT]

solver = Solver(dynamics, objective, constraints, bounds)

u_guess = [0.1 * randn(num_action) for t = 1:T-1];
initialize_controls!(solver, u_guess)

solve!(solver)

function plot_wfn(X, U)
    ψs = hcat(X...)
    fig = Figure(resolution=(1200, 500))
    ax, _ = series(fig[1,1], ψs, labels=[L"\psi_0^R", L"\psi_0^I", L"\psi_1^R", L"\psi_1^I"])
    axislegend(ax, position=:cb)
    us = vec(hcat(U...))
    ax, _ = lines(fig[1,2], 0:T-1, us, label=L"u(t)")
    axislegend(ax)
    return fig
end

X, U = get_trajectory(solver)

fig = plot_wfn(X, U)

save("test.png", fig)
