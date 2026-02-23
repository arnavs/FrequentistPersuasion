# test_comparison.jl
#
# Compares the Julia value_function against the JS computeCK / VK
# for the binary prosecutor-judge case.
#
# JS logic (fp-zK.html):
#   pX = mu0                    Pr[theta=guilty,  m=guilty]
#   pY = (1-mu0)*z              Pr[theta=innocent, m=guilty]
#   pZ = (1-mu0)*(1-z)          Pr[theta=innocent, m=acquit]
#   (X,Y,Z) ~ Multinomial(K; pX, pY, pZ)
#   c_K(z) = Pr[X >= Y, X+Y > 0]
#   V_K(z) = pi(z) * c_K(z)    where pi(z) = mu0 + (1-mu0)*z
#
# Julia setup that matches this:
#   N=2 states  (1=innocent, 2=guilty)
#   M=2 messages (1=acquit,  2=guilty)
#   mu_0 = [1-mu0, mu0]
#   sigma = [1-z  z ]   row 1: innocent -> Pr[acquit]=1-z, Pr[guilty]=z
#           [0    1 ]   row 2: guilty   -> Pr[acquit]=0,   Pr[guilty]=1
#   U_S[state, action]: sender wants conviction regardless of state
#       = [0 1]   (acquit=0, convict=1) for both rows
#   U_R[state, action]: judge wants to match state
#       = [1 0]   (innocent: acquit correct)
#         [0 1]   (guilty:   convict correct)

using Distributions, LinearAlgebra
include("../src/utils.jl")
include("../src/value_function.jl")

# ── Direct JS-equivalent computation in Julia ─────────────────────────────────

function js_multinomial_pmf(x, y, zz, K, pX, pY, pZ)
    # K! / (x! y! zz!) * pX^x * pY^y * pZ^zz
    return pdf(Multinomial(K, [pX, pY, pZ]), [x, y, zz])
end

function js_cK(z, mu0, K)
    pX = mu0
    pY = (1 - mu0) * z
    pZ = (1 - mu0) * (1 - z)
    cK = 0.0
    for x in 0:K
        for y in 0:(K - x)
            x + y == 0 && continue   # defective: default acquit
            x < y     && continue   # judge acquits
            zz = K - x - y
            cK += js_multinomial_pmf(x, y, zz, K, pX, pY, pZ)
        end
    end
    return cK
end

function js_VK(z, mu0, K)
    z < 1e-9 && return 0.0
    piZ = mu0 + (1 - mu0) * z
    return piZ * js_cK(z, mu0, K)
end

# ── Julia value_function setup ────────────────────────────────────────────────

function julia_VK(z, mu0, K)
    # U_S: rows=states, cols=actions. Sender always wants conviction (action 2).
    U_S = [0.0 1.0;   # innocent: acquit=0, convict=1
           0.0 1.0]   # guilty:   acquit=0, convict=1

    # U_R: rows=states, cols=actions. Judge wants to match state.
    U_R = [1.0 0.0;   # innocent: acquit correct
           0.0 1.0]   # guilty:   convict correct

    mu_0 = [1 - mu0, mu0]   # [Pr[innocent], Pr[guilty]]

    sigma = [1-z  z ;        # innocent: Pr[acquit]=1-z, Pr[guilty]=z
             0.0  1.0]       # guilty:   Pr[acquit]=0,   Pr[guilty]=1

    learning_rule = EmpiricalLearningRule()
    receiver = Receiver(U_R, learning_rule)
    sender   = Sender(U_S, mu_0)

    return value_function(sender, receiver, sigma, K)
end

# ── Run comparison ────────────────────────────────────────────────────────────

mu0_vals = [0.2, 0.3, 0.5, 0.7]
z_vals   = [0.1, 0.3, 0.5, 0.8]
K_vals   = [1, 3, 5, 10, 20]

println("Comparing JS V_K(z) vs Julia value_function for prosecutor-judge\n")
println(rpad("mu0",  6), rpad("z",    6), rpad("K",    5),
        rpad("JS V_K",    12), rpad("Julia V_K",  12), "match?")
println(repeat("-", 50))

all_pass = true
for mu0 in mu0_vals, z in z_vals, K in K_vals
    js  = js_VK(z, mu0, K)
    jl  = julia_VK(z, mu0, K)
    ok  = abs(js - jl) < 1e-9
    all_pass = all_pass && ok
    flag = ok ? "✓" : "✗  <-- MISMATCH"
    println(rpad(round(mu0, digits=2), 6),
            rpad(round(z,   digits=2), 6),
            rpad(K,                    5),
            rpad(round(js,  digits=8), 12),
            rpad(round(jl,  digits=8), 12),
            flag)
end

println()
println(all_pass ? "All tests passed." : "FAILURES detected — see above.")
