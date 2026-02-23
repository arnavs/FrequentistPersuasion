using FrequentistPersuasion, Plots, ProgressMeter

# model primitives 
mu_0 = [2/3, 1/3]
S = Sender([0 1; 0 1], mu_0)
R = Receiver([1 0; 0 1], EmpiricalLearningRule())

# capture solutions
Ks = 1:100 
sols = []
@showprogress for k in Ks 

end

sols = [optimize_sigma(S, R, k) for k in Ks]