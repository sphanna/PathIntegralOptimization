


function tCI(x,conf_level=0.95)
    N = length(x)
    alpha = (1 - conf_level)
    tstar = quantile(TDist(N-1), 1 - alpha/2)
    r = tstar * std(x)/sqrt(N)
    s = mean(x)
    return [s - r, s + r]
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

function getNormErr(PathList,realPath,normErrDiv)
    SANnormErr = [(norm(first.(PathList[i]) - first.(realPath)))/normErrDiv for i in 1:length(PathList)]
end