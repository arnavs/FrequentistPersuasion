using FrequentistPersuasion, Test, Random, LinearAlgebra
Random.seed!(8554)

@testset "Correctness: Prosecutor-Judge" begin 
    @test 1 == 1 # stuff
end

@testset "Regression: Empirical" begin 
    # random game 
    mu_0 = [1/3, 1/3, 1/3]
    U_R = [0 -2 1; 0 0.99 -1; 0 0.99 -1]
    # U_R = [0 0 0; -2 0.99 0.99; 1 -1 -1] # states on rows, actions on cols
    U_S = copy(U_R)
    σ = Matrix{Float64}(I, 3, 3) # Identity matrix as signal rule 
    learning_rule = EmpiricalLearningRule()
    receiver = Receiver(U_R, learning_rule)
    sender = Sender(U_S, mu_0)
    V = value_function(sender, receiver, σ, 3)
    @test V ≈ 0.699012345679012

    # prosecutor judge 
    # U_S = [1 1; 0 0]
    # U_R = [1 0; 0 1]
    # U_S = [1 0; 1 0]
    # U_R = [1 0; 0 1]
    # learning_rule = EmpiricalLearningRule()
    # mu_0 = [1/3, 2/3]
    # sender = Sender(U_S, mu_0)
    # receiver = Receiver(U_R, learning_rule)
    # σ = [1 0; 0.25 0.75] # mixed strategy 
    # K = 3 
    # V = value_function(sender, receiver, σ, 3)
    # @test V ≈ 0.4074074074074074
end 

@testset "Regression: Kernel" begin
    # random game, Nadaraya-Watson learning
    mu_0 = [1/3, 1/3, 1/3]
    U_R = [0 -2 1; 0 0.99 -1; 0 0.99 -1]
    # U_R = [0 0 0; -2 0.99 0.99; 1 -1 -1] # states on rows, actions on cols
    U_S = copy(U_R)
    σ = Matrix{Float64}(I, 3, 3) # Identity matrix as signal rule 
    gaussian_kernel(x) = exp(-0.5 * x^2) / sqrt(2π)
    learning_rule = NadarayaWatsonLearningRule(gaussian_kernel, 1.0)
    receiver = Receiver(U_R, learning_rule)
    sender = Sender(U_S, mu_0)
    V = value_function(sender, receiver, σ, 3)
    @show(V)
end 