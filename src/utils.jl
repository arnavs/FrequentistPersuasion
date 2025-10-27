# Learning rules 
abstract type AbstractLearningRule end 
struct EmpiricalLearningRule <: AbstractLearningRule
    # No parameters needed for empirical
end

function (rule::EmpiricalLearningRule)(
    sample::Matrix{Int},   # row is message, column is state, value is count observed in sample 
    m::Int,                # message received
    mu_0::Vector{Float64}, # prior over states. Index is state, value is prior prob
)
    M, N = size(sample)

    belief_prob_dist = zeros(Float64, N)
    all_states = 1:N
    # Find count in sample that correspond to message m
    msg_count = sum(sample[m, :])
    # @show(msg_count)
    # @show(sample)
    if msg_count == 0
        # Message unseen 
        belief_prob_dist = mu_0 
        return belief_prob_dist
    end

    states_with_msg_idx = findall(y -> sample[m, y]>0, all_states) # states (presented as indices in mu_0) that map to message m in sample
    states_with_msg = sample[m, states_with_msg_idx] 
    states_with_msg_count = sum(states_with_msg)

    for (i, state_idx) in enumerate(states_with_msg_idx)
        belief_prob_dist[state_idx] = states_with_msg[i] / states_with_msg_count
    end

    return belief_prob_dist
end

# Sender, receiver objects 
struct Receiver 
    U::Matrix{Float64}    # payoff matrix. Row is action, Col is state. 
    learning_rule::AbstractLearningRule
end

struct Sender 
    U::Matrix{Float64}    # payoff matrix. Row is action, Col is state.
    mu_0::Vector{Float64} # prior over states 
end

function br(
    receiver::Receiver,
    sender::Sender,
    sample::Matrix{Int},
    m::Int, # Message 
    mu_0::Vector{Float64}
    )
    belief = receiver.learning_rule(sample, m, mu_0)

    # Get best action based on expected utility 
    receiver_exp_U = receiver.U * belief
    max_utility = maximum(receiver_exp_U)
    candidate_actions = findall(x -> x == max_utility, receiver_exp_U)
    if length(candidate_actions) > 1
        # Multiple best actions, use sender's payoff to break tie
        sender_utilities = sender.U[candidate_actions, :] * belief
        best_action_idx = argmax(sender_utilities)
        best_action = candidate_actions[best_action_idx]
    else
        best_action = candidate_actions[1]
    end

    return best_action
end

function samples_matrix(K::Int, M::Int, N::Int)
    total_cells = M * N
    results = Vector{Matrix{Int}}()

    # recursive backtracking over flattened cells (length total_cells)
    # ordering: we fill a vector `v` of length total_cells, then reshape(v, M, N).
    # reshape in Julia is column-major: element order maps to (m, θ) with m varying fastest
    function backtrack!(sofar::Vector{Int}, remaining::Int, pos::Int)
        if pos == total_cells
            push!(results, reshape(vcat(sofar, remaining), M, N)) # M rows, N cols
            return
        end
        # assign 0..remaining to current cell and recurse
        for x in 0:remaining
            backtrack!(vcat(sofar, x), remaining - x, pos + 1)
        end
    end

    backtrack!(Int[], K, 1)
    return results
end
