struct Bound{T}
    state_lower::Vector{T}
    state_upper::Vector{T}
    action_lower::Vector{T}
    action_upper::Vector{T}
end

function Bound(
    num_state::Int=0,
    num_action::Int=0;
    state_lower=fill(-Inf, num_state),
    state_upper=fill(Inf, num_state),
    action_lower=fill(-Inf, num_action),
    action_upper=fill(Inf, num_action)
)
    return Bound(
        state_lower,
        state_upper,
        action_lower,
        action_upper
    )
end

const Bounds{T} = Vector{Bound{T}}
