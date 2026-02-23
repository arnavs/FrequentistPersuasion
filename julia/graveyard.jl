# old stuff, code, notes, etc. 

# import utils 
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