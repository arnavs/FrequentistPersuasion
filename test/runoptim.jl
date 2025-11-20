using Revise 
using FrequentistPersuasion, Test, Random, LinearAlgebra, Optimization, OptimizationOptimJL


# prosecutor-judge game
U_S = [1 1 1; 1 0 0; 0 1 1]  # Sender payoff: states × actions
U_R = [1 0 0; 0 1 1; 1 0 0]  # Receiver payoff: states × actions
learning_rule = EmpiricalLearningRule()
mu_0 = [1/3, 1/3, 1/3]  # Prior over states
sender = Sender(U_S, mu_0)
receiver = Receiver(U_R, learning_rule)
K = 3  # num samples

# Test with a specific signal rule first
println("=" ^ 60)
println("Testing value function with a specific σ:")
println("=" ^ 60)
σ = [0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]  # Row-stochastic: rows sum to 1
println("Signal rule σ (states × messages):")
display(σ)
println()
V = value_function(sender, receiver, σ, K)
println("Value: $V")

# Optimize
println("\n" * "=" ^ 60)
println("Optimizing signal rule:")
println("=" ^ 60)
σ_opt, V_opt = optimize_sigma(sender, receiver, K)

println("\n" * "=" ^ 60)
println("FINAL RESULTS:")
println("=" ^ 60)
println("Optimal σ:")
display(σ_opt)
println("\nOptimal sender payoff: $V_opt")