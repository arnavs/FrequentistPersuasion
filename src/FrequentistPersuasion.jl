module FrequentistPersuasion

# external deps 
using Distributions, LinearAlgebra, Optimization, OptimizationOptimJL, ForwardDiff

# internal files 
include("utils.jl")
include("value_function.jl")

# exports
export AbstractLearningRule, EmpiricalLearningRule
export Sender, Receiver, br
export value_function


end 