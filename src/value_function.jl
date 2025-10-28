# value function  
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