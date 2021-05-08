using LinearAlgebra
using Distributions
include("Optimization.jl")
include("Plotting.jl")


function pathLength(R)
    n = length(R)
    segments = [R[i+1] - R[i] for i in 1:n-1]
    sum(norm.(segments))
end

function Action(c,B,Δt,m)
    r = B*c
    Δr = [r[i+1]-r[i] for i in 1:length(r)-1]
    return Δt*(KEsum(Δr,Δt,m) - sum(V.(r,m)))
end

KEsum(Δr,Δt,m) = (m/2) * (Δr'Δr/Δt^2) 
V(r,m) = Vgrav(r,m)

Vgrav(r,m) = m*9.8*last(r)

Ms = [[rand(Uniform(0.1,4)),rand(Uniform(-1,6))] for i in 1:60]     

function VmudMs(r,Ms)
    v = -25*sum(norm(1/((r-Ms[i]))) for i in 1:length(Ms))
    if v < -1000
        v = -1000
    end
    return v
end

Vmud = r -> VmudMs(r,Ms)
     
y(θ,r0,rf,B,Δt,m) = L(θ,r0,rf,B,Δt,m) + rand(Normal(0,1))

function L(θ,r0,rf,B,Δt,m)
    c = [r0,θ...,rf]
    return Action(c,B,Δt,m)
end

function Bezier(i,t,t0,tf,n)
    b = binomial(n,i)
    return b*((t-t0)^i)*((tf-t)^(n-i))/((tf-t0)^n)
end

function Path(c,r0,rf,B)
    return B*[r0,c...,rf]
end

begin #setup
    N = 23
    m = 5
    r0 = [0,0]; rf = [3,0]; t0 = 0; tf = 2;
    t = range(t0,stop=tf,length=N)
    c0 = [r0,[1,0],[1.5,0],[4,0],[1,1],[1,4],[3.5,0],rf]
    θ₀ = c0[begin+1:end-1]
    n = length(c0);
    Δt = (tf-t0) / (N-1)
    B = [Bezier(i,t[j],t[begin],t[end],n-1) for i in 0:n-1, j in 1:N]'
    Binv = pinv(B)
    path0=B*c0
    pathMin = [(rf - r0)*t[i]/tf + r0 for i in 1:length(t)]

    LPz = θ -> L(θ,r0,rf,B,Δt,m)
    yPz = θ -> y(θ,r0,rf,B,Δt,m)

    L₀ = LPz(θ₀)
    vx0 = rf[1]/tf
    vy0 = 9.8
    xreal = vx0 .* t
    yreal = vy0 .* t - (1/2)*(9.8)*(t.^2)
    realPath = [[xreal[i],yreal[i]] for i in 1:length(t)]
    Lstar = LPz((Binv*realPath)[begin+1:end-1])

    normErrDiv = norm(path0 - realPath)
    LDiv = Lstar - L₀
    
    aₖ(k,a,A,α) = a/(k+1+A)^α
    cₖ(k,c,γ) = c/(k+1)^γ
    a = 0.2; A = 25; α = 0.602;
    c = 2; γ = 0.101;
    N_RDSA = 100
    #precompute coefficients
    ak = aₖ.(1:N_RDSA,a,A,α)
    ck = cₖ.(1:N_RDSA,c,γ)
    Δ(p) = [rand((-1,1),2) for i in 1:p]
    #Δ(p) = [rand(MvNormal(2,1)) for i in 1:p]

    N_SAN = N_RDSA * 2
    T₀ = 0.1
    N_TEMP = 50
    λ = 0.98
    dk(p) = [rand(Normal(0,1),2) for i in 1:p]
    #dk(p) = [rand(MvNormal(2,1)) for i in 1:p]
    #dk(p) = [rand((-1,1),2) for i in 1:p]
end

begin
    Nreps = 5

    RANDcs = [localizedRandomSearch(yPz,θ₀,dk,N_SAN) for i in 1:Nreps]
    RANDPaths = [Path.(RANDcs[i],Ref(r0),Ref(rf),Ref(B)) for i in 1:length(RANDcs)]
    RANDnormErr = mean(getNormErr(RANDPaths[i],realPath,normErrDiv) for i in 1:length(RANDPaths))
    RANDLs = mean(LPz.(RANDcs[i]) for i in 1:Nreps)

    SANcs = [SAN(yPz,θ₀,dk,T₀,λ,N_SAN,N_TEMP) for i in 1:Nreps]
    SANPaths = [Path.(SANcs[i],Ref(r0),Ref(rf),Ref(B)) for i in 1:length(SANcs)]
    SANnormErr = mean(getNormErr(SANPaths[i],realPath,normErrDiv) for i in 1:length(SANPaths))
    SANLs = mean(LPz.(SANcs[i]) for i in 1:Nreps)

    RDSAcs = [RDSA(yPz,θ₀,Δ,ak,ck,N_RDSA) for i in 1:Nreps]
    RDSAPaths = [Path.(RDSAcs[i],Ref(r0),Ref(rf),Ref(B)) for i in 1:length(RDSAcs)]
    RDSAnormErr = mean(getNormErr(RDSAPaths[i],realPath,normErrDiv) for i in 1:length(RDSAPaths))
    RDSALs = mean(LPz.(RDSAcs[i]) for i in 1:Nreps)

    println("RAND_Lterm: " * string(RANDLs[end]))
    println("SAN_Lterm: " * string(SANLs[end]))
    println("RDSA_LAterm: " * string(RDSALs[end]))
    #=
    println("RAND_ErrTerm: " * string(RANDnormErr[end]))
    println("SAN_ErrTerm: " * string(SANnormErr[end]))
    println("RDSA_ErrTerm: " * string(RDSAnormErr[end]))=#
end

begin #compute means and CIs
    RANDnorms = [getNormErr(RANDPaths[i],realPath,normErrDiv) for i in 1:length(SANPaths)]
    RANDnormTerms = last.(RANDnorms)
    mean(RANDnormTerms)
    tCI(RANDnormTerms)

    SANnorms = [getNormErr(SANPaths[i],realPath,normErrDiv) for i in 1:length(SANPaths)]
    SANnormTerms = last.(SANnorms)
    mean(SANnormTerms)
    tCI(SANnormTerms)

    RDSAnorms = [getNormErr(RDSAPaths[i],realPath,normErrDiv) for i in 1:length(SANPaths)]
    RDSAnormTerms = last.(RDSAnorms)
    mean(RDSAnormTerms)
    tCI(RDSAnormTerms)


    RANDsall = [(((LPz.(RANDcs[i]))/L₀)) for i in 1:Nreps]
    RANDterms = last.(RANDsall)
    mean(RANDterms)
    tCI(RANDterms)

    SANLsall = [(((LPz.(SANcs[i]))/L₀)) for i in 1:Nreps]
    SANLterms = last.(SANLsall)
    mean(SANLterms)
    tCI(SANLterms)

    RDSALsall = [(((LPz.(RDSAcs[i]))/L₀)) for i in 1:Nreps]
    RDSALterms = last.(RDSALsall)
    mean(RDSALterms)
    tCI(RDSALterms)
end

begin #plots loss values
    x = 1:1:N_SAN
    plot(x,RANDLs, label = "RAND B", 
                  title = "Normalized Loss values",
                  legend=(0.8,0.9),
                  xlabel = "Number of Action Measurements",
                  ylabel = "Lnorm")
    plot!(x,SANLs, label = "SAN")
    x = 1:2:N_SAN+1
    plot!(x,RDSALs, label = "SPSA")
end

begin #plot norm Err
    x = 1:1:N_SAN
    plot(x,RANDnormErr,label = "RAND", 
                      title = "Normalized Error",
                      legend=(0.8,0.9),
                      xlabel = "Number of Action Measurements",
                      ylabel = "Normalized Error")
    plot!(x,SANnormErr, label = "SAN")
    x = 1:2:N_SAN+1
    plot!(x,RDSAnormErr, label = "SPSA")
end

begin
    xV = -0.2:0.05:3.5
    yV = -0.2:0.05:5.5
    data = [V([j,i],m) for i∈-1:0.05:6,j∈-1:0.05:6]
    heatmap(xV,yV, data,
        c=cgrad([:white,:lightblue]),
        left_margin = 5mm,
        legend=(0.81,0.92),
        label = "V(R)",
        xlabel="x", ylabel="y",
        title="Terminal X,Y Paths RAND vs. SAN vs. SPSA")
end

begin
    plotPathXY!(path0,:blue, "θ₀")
    plotPathXY!(realPath,:darkorange, "θ*")

    RANDPath = mean(last.(RANDPaths))
    plotPathXY!(RANDPath,:black, "RAND B")

    SANPath = mean(last.(SANPaths))
    plotPathXY!(SANPath,:red, "SAN")

    RDSAPath = mean(last.(RDSAPaths))
    plotPathXY!(RDSAPath,:green,"SPSA")
end
