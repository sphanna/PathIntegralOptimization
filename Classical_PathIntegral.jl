#=
Code by Scott Hanna
5/6/21
Final Project - 625.743 SOC - Johns Hopkins University

Least action principle and path integral optimization
for classical mechanics.
=#


using LinearAlgebra
using Plots
using Distributions
Plots.plotlyjs()
using Plots.PlotMeasures

function tCI(x,conf_level=0.95)
    N = length(x)
    alpha = (1 - conf_level)
    tstar = quantile(TDist(N-1), 1 - alpha/2)
    r = tstar * std(x)/sqrt(N)
    s = mean(x)
    return [s - r, s + r]
end

plotPath!(f,θ,l,col = :green) = plot3d!(first.(θ),last.(θ),f.(θ), label=l, c = col)
function plotPathXY(R)
    r = first.(R);
    x = first.(r); y = last.(r)
    plot(x,y)
    scatter!(x,y)
end

function plotPathXY!(R, col = :blue, lbl = "")
    r = first.(R);
    x = first.(r); y = last.(r)
    plot!(x,y, c = col, label = lbl)
    scatter!(x,y, c = col, label = "", markersize = 2, 
                                       markershape = :diamond)
end

function plotPathXT(R)
    r = first.(R); t = last.(R)
    x = first.(r)
    plot(t,x)
    scatter!(t,x)
end

function plotSurface(f;xrange = [-1,1],yrange = [-1,1],l=500)
    xmin,xmax = xrange
    ymin,ymax = yrange
    x=range(xmin,stop=xmax,length=l)
    y=range(ymin,stop=ymax,length=l)

    plot(x,y,f,st = :surface,
                c = cgrad(:thermal, 12, categorical = true, scale = :exp),
                xlabel = "t1",
                ylabel = "t2", 
                zlabel = "L(θ)")
end

sqr(θ) = sum(θ.^2)

function pathLength(R)
    n = length(R)
    segments = [R[i+1] - R[i] for i in 1:n-1]
    sum(norm.(segments))
end

function unpack(v)
    n = length(v)
    r = Array{Float64,1}()
    for i in 1:n
        append!(r,v[i])
    end
    return r
end

function pack(v,p)
    r = Array{Array{Float64,1},1}()
    n = length(v)
    for i in 1:2:n
        append!(r,[[v[i+j-1] for j in 1:p]])
    end
    return r
end

function RDSA(y,θ₀,Δ,aₖ::AbstractArray,cₖ::AbstractArray,N)
    θ = fill(θ₀,N+1)
    θ[1] = θ₀
    p = length(θ₀)
    Δₖ = [Δ(p) for k in 1:N]
    [θ[k+1] = θ[k] - aₖ[k]*gRDSA(y,θ[k],Δₖ[k],cₖ[k]) for k in 1:N]
    return θ
end

function gRDSA(y,θₖ,Δ,cₖ)
    ckΔ = cₖ*Δ
    (1/(2*cₖ))*(y(θₖ+ckΔ) - y(θₖ-ckΔ)) * Δ
end

function localizedRandomSearch(L,θ₀,dₖ,N)
    θ = fill(θ₀,N)
    p = length(θ₀)
    for k in 1:N-1
        θ[k+1] = θ[k] + dₖ(p)
        L(θ[k+1]) >= L(θ[k]) && (θ[k+1] = θ[k])
    end
    return θ
end

function SAN(L,θ₀,dₖ,T₀,λ,Nmeas,NtempIter)
    N = Nmeas
    p = length(θ₀)
    θ = [θ₀]; θcurr = θ₀;
    T = T₀; k = 1;
    Lcurr = L(θ₀); N = N - 1;
    while N > 0
        θnew = θcurr + dₖ(p)
        Lnew = L(θnew); N = N - 1
        δ = Lnew - Lcurr
        if δ < 0
            append!(θ,[θnew])
            θcurr = θnew; Lcurr = Lnew;
        else
            if rand() < exp(-δ/T)
                append!(θ,[θnew])
                θcurr = θnew; Lcurr = Lnew;
            else                
                append!(θ,[θcurr])
            end
        end
        if(k % NtempIter == 0)
            T = λ*T
        end
        k=k+1
    end
    return θ
end


y(θ,r0,rf,t) = L(θ,r0,rf,t) + rand(Normal(0,1))

function L(θ,r0,rf,t)
    c = [first(r0),θ...,first(rf)]
    return Action(Path(c,t))
end

function Action(R)
    r = first.(R); t = last.(R);
    p = length(r)
    sum(((5*sqr((r[k+1] - r[k])/(t[k+1]-t[k]))/2) - V(r[k+1]))*(t[k+1]-t[k]) for k in 1:p-1)
end


function KEtrad(r,t)
    p = length(r)
    sum((5*sqr((r[k+1] - r[k])/(t[k+1]-t[k]))/2) for k in 1:p-1)
end

ActionC(c,DB,B,Δt,m) = Δt *(KEsum(c,DB,Δt,m) - Vsum(c,B,m))
function KEsum(c,DB,Δt,m)
    r = DB*c
    (m/2) * (1/Δt^2) * r'r
end

Vsum(c,B,m) = m*9.8*sum(last.(B*c))

function Bezier(i,t,t0,tf,n)
    b = binomial(n,i)
    return b*((t-t0)^i)*((tf-t)^(n-i))/((tf-t0)^n)
end

X(t,c,t0,tf) = sum(c[i+1] * Bezier(i,t,t0,tf,length(c)-1) for i in 0:length(c)-1)

function Path(c,t)
    x = X.(t,Ref(c),t[begin],t[end])
    return [(x[i],t[i]) for i in 1:length(x)]
end

function PathC(c,t,r0,rf)
    fullC = [first(r0),c...,first(rf)]
    return Path(fullC,t)
end

function getNormErr(PathList,realPath,normErrDiv)
    SANnormErr = [(norm(first.(PathList[i]) - first.(realPath)))/normErrDiv for i in 1:length(PathList)]
end

begin
    g0 = maximum((norm(gRDSA(LPz,θ₀,Δ(length(θ₀)),1)[1]) for i in 1:10))
    b = pathLength(first.(path0)) / length(θ₀)
    A0 = (1+A)^α
    a = (b / (g0)) * A0
end

begin
    Npath = 23
    r0 = ([0,0],0); rf = ([3,0],2);
    t = range(last(r0),stop=last(rf),length=Npath)
    c0 = [r0[1],[1,0],[1.5,0],[4,0],[1,3],[2,0],[2.5,0],rf[1]]
    θ₀ = c0[begin+1:end-1]
    path0 = PathC(θ₀,t,r0,rf)
    pathMin = [(rf[1] - r0[1])*t[i]/rf[2] + r0[1] for i in 1:length(t)]
    L0 = L(θ₀,r0,rf,t)

    LPz = θ -> L(θ,r0,rf,t)
    yPz = θ -> y(θ,r0,rf,t)
    #V(r) = r[2]
    vx0 = rf[1][1]/rf[2]
    vy0 = 9.8

    xreal = vx0 .* t
    yreal = vy0 .* t - (1/2)*(9.8)*(t.^2)
    realPath = [([xreal[i],yreal[i]],t[i]) for i in 1:length(t)]
    Lstar = Action(realPath)

    normErrDiv = norm(first.(path0) - first.(realPath))
    LDiv = Lstar - L0
    
    aₖ(k,a,A,α) = a/(k+1+A)^α
    cₖ(k,c,γ) = c/(k+1)^γ
    a = 0.124; A = 25; α = 0.602;
    c = 2; γ = 0.101;
    N_RDSA = 250
    #precompute coefficients
    ak = aₖ.(1:N_RDSA,a,A,α)
    ck = cₖ.(1:N_RDSA,c,γ)
    Δ(p) = [rand((-1,1),2) for i in 1:p]
    #Δ(p) = [rand(MvNormal(2,1)) for i in 1:p]

    N_SAN = N_RDSA * 2
    T₀ = 1
    dk(p) = [rand(Normal(0,0.5),2) for i in 1:p]
    #dk(p) = [rand(MvNormal(2,1)) for i in 1:p]
    #dk(p) = [rand((-1,1),2) for i in 1:p]
    N_TEMP = 50
    λ = 0.95
end

begin
    Nreps = 20

    RANDcs = [localizedRandomSearch(yPz,θ₀,dk,N_SAN) for i in 1:Nreps]
    RANDPaths = [PathC.(RANDcs[i],Ref(t),Ref(r0),Ref(rf)) for i in 1:length(RANDcs)]
    RANDnormErr = mean(getNormErr(RANDPaths[i],realPath,normErrDiv) for i in 1:length(RANDPaths))
    RANDLs = mean((((LPz.(RANDcs[i])))) for i in 1:Nreps)

    SANcs = [SAN(yPz,θ₀,dk,T₀,λ,N_SAN,N_TEMP) for i in 1:Nreps]
    SANPaths = [PathC.(SANcs[i],Ref(t),Ref(r0),Ref(rf)) for i in 1:length(SANcs)]
    SANnormErr = mean(getNormErr(SANPaths[i],realPath,normErrDiv) for i in 1:length(SANPaths))
    SANLs = mean((((LPz.(SANcs[i])))) for i in 1:Nreps)

    RDSAcs = [RDSA(yPz,θ₀,Δ,ak,ck,N_RDSA) for i in 1:Nreps]
    RDSAPaths = [PathC.(RDSAcs[i],Ref(t),Ref(r0),Ref(rf)) for i in 1:length(RDSAcs)]
    RDSAnormErr = mean(getNormErr(RDSAPaths[i],realPath,normErrDiv) for i in 1:length(RDSAPaths))
    RDSALs = mean((((LPz.(RDSAcs[i])))) for i in 1:Nreps)

    println("RAND_Lterm: " * string(RANDLs[end]))
    println("SAN_Lterm: " * string(SANLs[end]))
    println("RDSA_LAterm: " * string(RDSALs[end]))
    #=
    println("RAND_ErrTerm: " * string(RANDnormErr[end]))
    println("SAN_ErrTerm: " * string(SANnormErr[end]))
    println("RDSA_ErrTerm: " * string(RDSAnormErr[end]))=#
end

begin
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


    RANDsall = [(((LPz.(RANDcs[i]))/L0)) for i in 1:Nreps]
    RANDterms = last.(RANDsall)
    mean(RANDterms)
    tCI(RANDterms)

    SANLsall = [(((LPz.(SANcs[i]))/L0)) for i in 1:Nreps]
    SANLterms = last.(SANLsall)
    mean(SANLterms)
    tCI(SANLterms)

    RDSALsall = [(((LPz.(RDSAcs[i]))/L0)) for i in 1:Nreps]
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
    function Vl(r,ls) 
        #=
        v = -25*sum(norm(1/((r-ls[i]))) for i in 1:length(ls))
        if v < -1000
            v = -1000
        end
        return v
        =#
        return 9.8 * 5 * r[2]
    end
    ls = [[rand(Uniform(0.1,4)),rand(Uniform(-1,6))] for i in 1:60]     
    V = r -> Vl(r,ls)
end

begin
    xV = -0.2:0.05:3.5
    yV = -0.2:0.05:5.5
    data = [V([j,i]) for i∈-1:0.05:6,j∈-1:0.05:6]
    heatmap(xV,yV, data,
        c=cgrad([:white,:lightblue]),
        left_margin = 5mm,
        legend=(0.81,0.92),
        label = "V(R)",
        xlabel="x", ylabel="y",
        title="Terminal X,Y Paths RAND vs. SAN vs. SPSA")
end

begin
    #plotPathXY!(path0,:blue, "θ₀")
    #plotPathXY!(realPath,:darkorange, "θ*")

    RANDterm = RANDcs[end][end]
    RANDtermFull = [first(r0),RANDterm...,first(rf)]
    RANDPath = Path(RANDtermFull,t)
    plotPathXY!(RANDPath,:black, "RAND B")

    SANterm = SANcs[end][end]
    SANtermFull = [first(r0),SANterm...,first(rf)]
    SANPath = Path(SANtermFull,t)
    plotPathXY!(SANPath,:red, "SAN")

    RDSAterm = RDSAcs[end][end]
    RDSAtermFull = [first(r0),RDSAterm...,first(rf)]
    RDSAPath = Path(RDSAtermFull,t)
    plotPathXY!(RDSAPath,:green,"SPSA")
end

c0
n = length(c0); N = Npath
Δt = (t[end]-t[begin]) / (N-1)
B = [Bezier(i,t[j],t[begin],t[end],n-1) for i in 0:n-1, j in 1:N]'
r0=B*c0
R = [([r0[i][1],r0[i][2]],t[i]) for i in 1:length(t)]
Action(R)

begin
    D = Bidiagonal(fill(-1,N-1),fill(1,N-2),:U)
    f = zeros(N-1); f[end] = 1;
    D = hcat(D,f)
end

DB = D*B

KEsum(c0,DB,Δt,5)
Vsum(c0,B,5)

@time ActionC(c0,DB,B,Δt,5)


