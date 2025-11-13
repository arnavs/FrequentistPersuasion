# Optimization Guide for Frequentist Persuasion

## Overview

This guide explains the optimization setup in `optim.jl` for finding optimal signal rules in the frequentist persuasion game.

## Problem Setup

### Goal
Maximize the sender's expected payoff by choosing an optimal signal rule σ.

### Variables
- **σ** (signal rule): An N×M matrix where:
  - N = number of states
  - M = number of messages (= number of actions)
  - σ[θ,m] = probability of sending message m when in state θ
  
### Constraints
The signal rule must be **row-stochastic**:
1. **Non-negativity**: σ[θ,m] ≥ 0 for all θ,m
2. **Row sums to 1**: Σₘ σ[θ,m] = 1 for each state θ

This means each row of σ is a probability distribution over messages.

## Matrix Orientation

**IMPORTANT**: Throughout the codebase:
- **Signal rule σ**: States on rows, messages on columns
- **Sample matrix**: States on rows, messages on columns  
- **Payoff matrices U**: States on rows, actions on columns

## Optimization Framework

### 1. Objective Function

```julia
function objective(x, model)
    σ = reshape(x, N, M)  # Unflatten to matrix
    val = value_function(sender, receiver, σ, K)
    return -val  # Minimize negative = maximize positive
end
```

**Why negative?** Standard optimizers minimize functions. To maximize the sender's payoff, we minimize its negative.

### 2. Parameter Passing

We use a named tuple `model` containing:
- `sender`: Sender object with payoff matrix and prior
- `receiver`: Receiver object with payoff matrix and learning rule
- `K`: Number of samples
- `N`: Number of states
- `M`: Number of messages

This is cleaner than using global variables and makes the objective function self-contained.

### 3. Flattening/Reshaping

**Julia uses column-major ordering** (like FORTRAN, unlike C/Python):
- `vec(σ)` flattens matrix to vector: [σ[:,1]; σ[:,2]; ...; σ[:,M]]
- `reshape(x, N, M)` converts back: columns filled first

For element σ[θ,m], its position in flattened vector x is:
```
index = (m - 1) * N + θ
```

## Constraint Implementation

### Box Constraints (Element-wise bounds)

```julia
lb = zeros(N * M)  # All elements ≥ 0
ub = ones(N * M)   # All elements ≤ 1
```

Simple bounds on each variable.

### Linear Equality Constraints (Row sums)

For each state θ, we need: Σₘ σ[θ,m] = 1

This is expressed as: **A·x = b** where:
- A is N × (N·M) matrix
- x is the flattened σ vector
- b = [1, 1, ..., 1] (length N)

**Building A:**
```julia
A = zeros(N, N*M)
for θ in 1:N
    for m in 1:M
        idx = (m - 1) * N + θ  # Position in flattened vector
        A[θ, idx] = 1.0        # Sum over messages for this state
    end
end
```

Each row of A has M ones at positions corresponding to σ[θ,:] for that state θ.

## Step-by-Step Optimization Process

### Step 1: Determine Dimensions
```julia
N = length(sender.mu_0)    # Number of states
M = size(sender.U, 2)      # Number of messages/actions
```

### Step 2: Initial Guess
```julia
σ_init = ones(N, M) ./ M   # Uniform: each message equally likely
x0 = vec(σ_init)           # Flatten for optimizer
```

### Step 3: Set Bounds
```julia
lb = zeros(N * M)   # Non-negativity
ub = ones(N * M)    # Upper bound for probabilities
```

### Step 4: Build Constraints
```julia
A = zeros(N, N * M)        # Constraint matrix
for θ in 1:N
    for m in 1:M
        A[θ, (m-1)*N + θ] = 1.0
    end
end
lcon = ucon = ones(N)      # Equality: A*x = 1
```

### Step 5: Create Optimization Problem
```julia
optf = OptimizationFunction(objective, Optimization.AutoForwardDiff())
prob = OptimizationProblem(
    optf, x0, model;
    lb=lb, ub=ub,
    lcons=lcon, ucons=ucon
)
```

**AutoForwardDiff**: Automatically computes gradients using forward-mode automatic differentiation.

### Step 6: Solve
```julia
sol = solve(prob, algorithm)
```

Default algorithm is `LBFGS()` (limited-memory BFGS), a quasi-Newton method.

### Step 7: Extract Results
```julia
σ_opt = reshape(sol.u, N, M)  # Reshape solution
V_opt = -sol.objective         # Negate to get true value
```

## Algorithm Choices

The function accepts an `algorithm` parameter. Good choices:

1. **LBFGS()** (default): Fast, memory-efficient quasi-Newton method
   - Good for smooth problems
   - Uses gradient information

2. **IPNewton()**: Interior-point Newton method
   - Handles constraints well
   - May be slower but more robust

3. **NelderMead()**: Gradient-free simplex method
   - Doesn't need gradients
   - Slower, but robust for non-smooth problems

## Usage Example

```julia
using Optimization, OptimizationOptimJL
include("optim.jl")

# Set up game
sender = Sender(U_sender, mu_0)
receiver = Receiver(U_receiver, learning_rule)
K = 3

# Optimize with default algorithm (LBFGS)
σ_opt, V_opt = optimize_sigma(sender, receiver, K)

# Or specify algorithm
σ_opt, V_opt = optimize_sigma(sender, receiver, K; algorithm=IPNewton())

# Or provide initial guess
σ_init = [0.7 0.3; 0.2 0.8]  # Custom starting point
σ_opt, V_opt = optimize_sigma(sender, receiver, K; initial_sigma=σ_init)
```

## Verification

The function prints row sums at the end to verify constraints:
```julia
row_sums = sum(σ_opt, dims=2)  # Should all be ≈1.0
```

If row sums deviate significantly from 1.0, the optimization may not have converged properly.

## Required Packages

Add to `Project.toml`:
```julia
[deps]
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
```

Install with:
```julia
using Pkg
Pkg.add(["Optimization", "OptimizationOptimJL"])
```

## Common Issues

### Issue 1: "Optimization not defined"
**Solution**: Add `using Optimization` at the top of the file.

### Issue 2: Row sums not equal to 1
**Solution**: 
- Check constraint matrix A is built correctly
- Try a different algorithm (e.g., IPNewton)
- Tighten convergence tolerances

### Issue 3: Optimization doesn't improve value
**Solution**:
- Check that objective function is correct (negative sign?)
- Verify value_function is computing correctly
- Try different initial guesses
- Check if constraints are too restrictive

### Issue 4: Algorithm fails to converge
**Solution**:
- Try a different algorithm
- Adjust algorithm parameters (max iterations, tolerance)
- Check that the problem is well-posed (feasible region non-empty)

## Mathematical Background

The optimization problem is:

```
maximize    V(σ) = E[U_sender(θ, a*(σ, sample))]
subject to  σ[θ,m] ≥ 0           ∀θ,m
            Σₘ σ[θ,m] = 1        ∀θ
```

where:
- V(σ) is the sender's expected payoff under signal rule σ
- a*(σ, sample) is the receiver's best response given the signal rule and observed sample
- The expectation is over:
  1. True state θ ~ mu_0
  2. Message m ~ σ[θ,:]
  3. Sample ~ Multinomial(K, σ ⊙ mu_0)

This is a **constrained nonlinear optimization** problem:
- **Nonlinear**: value_function is nonlinear in σ
- **Smooth**: value_function is differentiable (via automatic differentiation)
- **Constrained**: box constraints + linear equality constraints
