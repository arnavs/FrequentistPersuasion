using Distributions, Random, LinearAlgebra
Random.seed!(42)

# # import utils 
# include("utils.jl")

# TODO as of Oct 24: 
# 1. DONE: Remove sigma from receiver, make sample matrix instead of vector
# 2. DONE: Tie-breaking rule
# 3. DONE: Mixed strategies 
# 4. Other learning rules


# epsilon = 0.01
# badpayoff = -1
# U = zeros(N+2, N) 
# U[diagind(U)] .= 1 # match the state
# U[1, :] .= 0 
# U[2, :] .= 1 - epsilon 

# pure strategies
# Sender always knows what message he's gna send for a particular state 
# Later on: sender wants to lie (mixed strategies)
    # σ_revelation = collect(1:N) # send message i in state i
    # σ_pool = ones(Int, N)
    # σ_pool[end] = N 
    
# value function for deterministic sigma 
function value_function(
    sender::Sender,
    receiver::Receiver,
    σ::Matrix{Float64}, # Signal rule
    K::Int
)
    mu_0 = sender.mu_0 
    U_sender = sender.U 
    U_receiver = receiver.U 
    M, N = size(σ) # num messages, num states

    # All possible samples for K draws over N states 
    # multinomial distribution over sample, which is now a matrix (rows: messages, cols: states)
    # Build the joint probability matrix
    P = σ .* reshape(mu_0, 1, N)  # (M×N) with P[m,θ] = mu_0[θ] * σ[m,θ] = mu_0[θ] * P(m|θ)

    sample_dist = Multinomial(K, vec(P)) # multinomial distribution for K trials with joint probability vector P
    possible_samples = samples_matrix(K, M, N)
    expected_val = 0.0

    for sample_matrix in possible_samples
        # Get probability of each sample 
        sample_vec = vec(sample_matrix) # flatten to vector for pdf calculation
        prob_sample = pdf(sample_dist, sample_vec) # Probability of this sample under the multinomial dist
        # @show sample_matrix, prob_sample

        # Compute sender's expected payoff given this sample
        vals = zeros(Float64, N)

        for state in 1:N
            msgs_in_state = σ[:, state]
            # If sigma deterministic (only 1 non-zero entry per col)
            # m = findfirst(x -> x==1.0, msgs_in_state) # Message assigned by signal rule based on true state 
            # best_response = br(receiver, sender, sample_matrix, m, mu_0) # Receiver's best action given sample and message (and the belief it formed)
            # vals[state] = U_sender[best_response, state] # Sender's payoff given true state and receiver takes this best_response action  

            # If sigma mixed (multiple non-zero entries per col, representing probability) 
            expected_over_msgs = 0.0 
            for m in 1:M
                prob_m_given_state = msgs_in_state[m]
                if prob_m_given_state == 0.0
                    continue
                end
                best_response = br(receiver, sender, sample_matrix, m, mu_0) # Receiver's best action given sample and message (and the belief it formed)
                expected_over_msgs += prob_m_given_state * U_sender[best_response, state]
            end
            vals[state] = expected_over_msgs
        end
        val_weighed_by_prior = dot(mu_0, vals)
        expected_val += prob_sample * val_weighed_by_prior 
    end

    return expected_val 
end

# mu_0 = [1/4, 1/4, 1/4, 1/4]
# U_R = [0 0 0 0; -2 0.99 0.99 0.99; 1 -1 -1 -1; -1 1 -1 -1]
# U_S = copy(U_R)
# σ = Matrix{Float64}(I, 4, 4)
# Do a small test case to check if samples are being generated correctly
mu_0 = [1/3, 1/3, 1/3]
U_R = [0 0 0; -2 0.99 0.99; 1 -1 -1]
U_S = copy(U_R)
σ = Matrix{Float64}(I, 3, 3) # Identity matrix as signal rule 

learning_rule = EmpiricalLearningRule()
receiver = Receiver(U_R, learning_rule)
sender = Sender(U_S, mu_0)
println("V = ", value_function(sender, receiver, σ, 3))


# function v(σ, K, mu_0, U)
#     # sigma is strategy, K sample size, mu_0 prior over states, U payoff matrix 
#     # Finite-sample simulation of sender payoff under a given deterministic experiment
#     # For each possible sample (size K):
#     # simulate best response of receiver given message and sample frequencies 
#     # compute expected sender utility = prob-weighted avg of payoffs (multinomial distribution)
#     # mixed strategy: sometimes send msg 1, sometimes send msg 2
#     # for each state, sigma assigns it a distinct message (signal rule)
#     # pooling: bundled 
#     a_0 = br(mu_0, U)
#     @show a_0
#     sample_dist = Multinomial(K, mu_0)
#     possible_samples = samples(K, N)
    
#     strat_val = 0 
#     for sample in possible_samples
#         brs = zeros(Int, N)
#         for θ in 1:N  # iterate over state 
#             m = σ[θ] # msg assigned by sigma 
#             brs[θ] = br(m, a_0, σ::Vector, sample) # best response based on that msg + state being iterated over + the sample we have 
#         end 
#         vals = [U[brs[θ], θ] for θ in 1:N]
#         val = dot(mu_0, vals) # across states taking expectation 
#         strat_val += pdf(sample_dist, sample) * val # across all samples taking expectation 
#     end    
#     return strat_val
# end 

# strategy: matrix of states and message probabilities. start as 1. once that works can change to sample for mixed 
# basic logic of game
# once this is done, throw it an optimizer 
# learning rule: has state and message, throws back action, also puts in best response. 
# global variables are fine 
# package called Parameters to create NamedTuple (does type inference for you)