module FrequentistPersuasion

# external deps 
using Distributions, LinearAlgebra, Optimization, OptimizationOptimJL, OptimizationNLopt

# internal files 
include("utils.jl")
include("value_function.jl")
include("optim.jl")

# exports
export AbstractLearningRule, EmpiricalLearningRule, NadarayaWatsonLearningRule
export Sender, Receiver, br
export value_function
export optimize_sigma


end 