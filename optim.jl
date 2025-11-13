using Optimization, OptimizationOptimJL, DifferentiationInterface

"""
Objective function for optimization. Takes a flattened signal rule matrix and model parameters.

# Output 
- Negative of the sender's expected value (we minimize, so negate to maximize)

# Notes
- σ states on rows, messages on columns
- x is flattened in column-major order (Julia default)
"""
function objective(x::Vector{Float64}, model::NamedTuple)
    sender, receiver, K, N, M = model.sender, model.receiver, model.K, model.N, model.M
    
    # Reshape flattened vector back to matrix (N states × M messages) (column-major)
    σ = reshape(x, N, M)
    
    # Sender's expected payoff
    val = value_function(sender, receiver, σ, K)
    
    # Return negative to maximize sender payoff
    return -1*val
end


"""
Finds the optimal signal rule σ that maximizes the sender's expected payoff.

# Output
- `σ_opt::Matrix{Float64}`: Optimal signal rule (N×M, row-stochastic)
- `V_opt::Float64`: Optimal sender payoff

# Signal Rule constraints so that each row forms a probability distribution 
The signal rule σ must be row-stochastic:
1. All elements are non-negative: σ[θ,m] ≥ 0
2. Each row sums to 1: Σₘ σ[θ,m] = 1 for all states θ

"""
function optimize_sigma(
    sender::Sender, 
    receiver::Receiver, 
    K::Int;
    initial_sigma::Union{Nothing, Matrix{Float64}}=nothing,
    algorithm=LBFGS()
)  
    
    # ========================================
    # Problem dimensions 
    N = length(sender.mu_0)           # Number of states
    M = size(sender.U, 2)             # Number of messages (= number of actions)
    
    println("Problem dimensions:")
    println("\tN (states) = $N")
    println("\tM (messages) = $M")
    println("\tK (samples) = $K")

    # ========================================
    # Initial guess
    # If none provided, use uniform distribution over messages for each state
    if initial_sigma === nothing
        σ_init = ones(Float64, N, M) ./ M  # Each row sums to 1/M * M = 1
    else
        σ_init = initial_sigma
    end
    
    # Flatten to vector for optimizer (column-major order)
    x0 = vec(σ_init)  # N*M
    println("\nInitial guess: uniform distribution (each state sends each message with prob 1/$M)")
    
    # ========================================
    # Box constraints (bounds)
    # All elements in [0, 1] because they are probabilities
    lb = zeros(Float64, N * M)   
    ub = ones(Float64, N * M)    
    println("\nBox constraints: 0 ≤ σ[θ,m] ≤ 1 for all (θ,m)")
    
    # ========================================
    # Row-stochastic constraints
    # For each state θ: Σₘ σ[θ,m] = 1
    # Represent as linear equality constraint: A*vec(σ) = b where b = ones(N)
    A = zeros(Float64, N, N * M) # Constraint matrix A (N rows × N*M columns)
    b = ones(Float64, N)
    
    for θ in 1:N
        # For state θ, we want to sum over all messages m
        # In flattened vector x, element σ[θ,m] is at index: (m-1)*N + θ
        # because reshape uses column-major ordering
        for m in 1:M
            idx = (m - 1) * N + θ  # Index in flattened vector
            A[θ, idx] = 1.0
        end
    end

    # Create linear equality constraint: A*vec(σ) = b
    lcon = b
    ucon = b

    println("\nRow-stochastic constraints: Σₘ σ[θ,m] = 1 for each state θ")
    println("\tImplemented as $N linear equality constraints")
    
    # ========================================
    # Model params, optimization function and optimization problem
    model = (
        sender = sender,
        receiver = receiver,
        K = K,
        N = N,
        M = M
    )
    # Use forward autodiff 
    optf = OptimizationFunction(objective, DifferentiationInterface.AutoSimpleFiniteDiff())
    prob = OptimizationProblem(
        optf,           # Objective function
        x0,             # Initial guess
        model;          # Parameters passed to objective
        lb = lb,        # Lower bounds on variables
        ub = ub,        # Upper bounds on variables
        lcons = lcon,   # Lower bounds on constraints
        ucons = ucon    # Upper bounds on constraints 
    )
    
    println("\nOptimization problem created with:")
    println("\tVariables: $(N*M)")
    println("\tBox constraints: $(N*M) lower bounds, $(N*M) upper bounds")
    println("\tEquality constraints: $N (row sums)")

    # ========================================
    # Solve and get results 
    println("\nSolving with algorithm: $algorithm")
    sol = solve(prob, algorithm)
    
    println("\nOptimization complete!")
    println("\tConverged: $(sol.retcode)")
    println("\tObjective value (negative): $(sol.objective)")
    
    # Reshape solution back to matrix 
    σ_opt = reshape(sol.u, N, M)
    
    # Flip sign to get actual value 
    V_opt = -sol.objective
    
    println("\tSender's optimal payoff: $V_opt")
    println("\nOptimal signal rule σ:")
    display(σ_opt)
    println()
    
    # Verify tgat each row sums to 1 
    row_sums = sum(σ_opt, dims=2)
    println("\nRow sums (should all be ≈1.0):")
    display(row_sums)
    println()
    
    return σ_opt, V_opt
end