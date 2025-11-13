# Learning rules 
abstract type AbstractLearningRule end 
struct EmpiricalLearningRule <: AbstractLearningRule
    # No parameters needed for empirical
end

function (rule::EmpiricalLearningRule)(
    sample::Matrix{Int},   # row is state, column is message, value is count observed in sample 
    m::Int,                # message received
    mu_0::Vector{Float64}, # prior over states. Index is state, value is prior prob
)
    N, M = size(sample)

    belief_prob_dist = zeros(Float64, N)
    all_states = 1:N
    # Find count in sample that correspond to message m
    msg_count = sum(sample[:, m])
    # @show(msg_count)
    # @show(sample)
    if msg_count == 0
        # Message unseen 
        belief_prob_dist = mu_0 
        return belief_prob_dist
    end

    states_with_msg_idx = findall(y -> sample[y, m]>0, all_states) # states (presented as indices in mu_0) that map to message m in sample
    states_with_msg = sample[states_with_msg_idx, m] 
    states_with_msg_count = sum(states_with_msg)

    for (i, state_idx) in enumerate(states_with_msg_idx)
        belief_prob_dist[state_idx] = states_with_msg[i] / states_with_msg_count
    end

    return belief_prob_dist
end

# Sender, receiver objects 
struct Receiver 
    U::Matrix{Float64}    # payoff matrix. Row is state, Col is action. 
    learning_rule::AbstractLearningRule
end

struct Sender 
    U::Matrix{Float64}    # payoff matrix. Row is state, Col is action.
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
    receiver_exp_U = receiver.U' * belief
    max_utility = maximum(receiver_exp_U)
    candidate_actions = findall(x -> x == max_utility, receiver_exp_U)
    if length(candidate_actions) > 1
        # Multiple best actions, use sender's payoff to break tie
        sender_utilities = sender.U[:, candidate_actions]' * belief
        best_action_idx = argmax(sender_utilities)
        best_action = candidate_actions[best_action_idx]
    else
        best_action = candidate_actions[1]
    end

    return best_action
end

# Enumerate all nonnegative integer vectors of length D that sum to K,
# then reshape to N×M with column-major ordering.
function samples_matrix(K::Int, N::Int, M::Int)
    D = N * M
    results = Vector{Matrix{Int}}()

    # Backtrack over the flattened vector of length D.
    function backtrack_flat!(sofar::Vector{Int}, remaining::Int, pos::Int)
        if pos == D
            # append the final element and reshape in column-major order to N×M
            push!(results, reshape(vcat(sofar, remaining), N, M))
            return
        end
        for x in 0:remaining
            backtrack_flat!(vcat(sofar, x), remaining - x, pos + 1)
        end
    end

    backtrack_flat!(Int[], K, 1)
    return results
end

# function samples_matrix(K::Int, N::Int, M::Int)
#     total_cells = N * M 
#     results = Vector{Matrix{Int}}()

#     # recursive backtracking over flattened cells (length total_cells)
#     # ordering: we fill a vector `v` of length total_cells, then reshape(v, M, N).
#     # reshape in Julia is column-major: element order maps to (m, θ) with m varying fastest
#     function backtrack!(sofar::Vector{Int}, remaining::Int, pos::Int)
#         if pos == total_cells
#             push!(results, reshape(vcat(sofar, remaining), N, M)) # N rows, M cols
#             return
#         end
#         # assign 0..remaining to current cell and recurse
#         for x in 0:remaining
#             backtrack!(vcat(sofar, x), remaining - x, pos + 1)
#         end
#     end

#     backtrack!(Int[], K, 1)
#     return results
# end
