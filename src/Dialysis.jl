module Dialysis

using DataFrames
import CSV, ZipFile
using LinearAlgebra: det, Diagonal, UniformScaling
using Distributions: pdf, MvNormal, pdf!
using ForwardDiff
using Statistics: std, var, mean, cov
using FixedEffectModels
import ShiftedArrays

export panellag, locallinear, loaddata,
  partiallinearIV,
  errors_gm, objective_gm, polyreg,
  partiallinear, clustercov


include("data.jl")
include("partiallinear.jl")


"""
    errors_gm(y::Symbol, k::Symbol, l::Symbol, Φ::Symbol,
                   id::Symbol, t::Symbol, data::DataFrame;
                   npregress::Function=polyreg)

Returns functions that given β calculate ω(β) and η(β).

# Arguments

 - `y` Symbol for y variable in data
 - `k` Symbol for k variable in data
 - `l` Symbol for l variable in data
 - `q` Symbol for q variable in data
 - `Φ` Symbol for Φ variable in data
 - `id`  Symbol for id variable in data
 - `t` Symbol for t variable in data
 - `data` DataFrame containing variables
 - `α` estimate of α

# Returns

 - ωfunc(β) computes ω given β for the `data` and α passed in as
   input. `length(ωfunc(β)) == nrow(data)` ωfunc(β) will contain missings
   if the data does.
 - ηfunc(β) computes η given β. for the `data` and α passed in as
   input. `length(ηfunc(β)) == nrow(data)` ηfunc(β) will contain
   missings.

Warning: this function is not thread safe!
"""
function errors_gm(y::Symbol,  k::Symbol, l::Symbol, q::Symbol,
                   Φ::Symbol, id::Symbol, t::Symbol, data::DataFrame,
                   α::Real;
                   npregress::Function=polyreg, degree=1)
  function Ω(β::AbstractVector)
    data[!,Φ] - data[!,k]*β[1] - data[!,l] * β[2];
  end
  df = deepcopy(data) # modifying df in Η make this not thread safe
  function Η(β::AbstractVector)
    df[!,:ω] = Ω(β);
    df[!,:ωlag] = panellag(:ω, df, id, t);
    df[!,:ytilde] = df[!,y] - α*df[!,q] -df[!,k]*β[1] - df[!,l]*β[2];
    inc = completecases(df[:,[:ωlag, :ytilde]])
    X = reshape(disallowmissing(df[inc,:ωlag]), sum(inc), 1)
    Y = reshape(disallowmissing(df[inc,:ytilde]), sum(inc),1)
    ηi = Y - npregress(X,X,Y, degree=degree)
    η = Array{Union{Missing, eltype(ηi)}, 1}(undef, nrow(df))
    η .= missing
    η[inc] = ηi
    return(η)
  end
  return(ωfunc=Ω, ηfunc=Η)
end

"""
    objective_gm(y::Symbol,  k::Symbol, l::Symbol, q::Symbol,
                      Φ::Symbol, id::Symbol, t::Symbol,
                      instruments::Array{Symbol,1}, data::DataFrame
                      ; W=UniformScaling(1.),
                      npregress::Function=(xp,xd,yd)->polyreg(xp,xd,yd,degree=1))



"""
function objective_gm(y::Symbol,  k::Symbol, l::Symbol, q::Symbol,
                      Φ::Symbol, id::Symbol, t::Symbol,
                      instruments::Array{Symbol,1}, data::DataFrame
                      ; W=UniformScaling(1.),
                      npregress::Function=(xp,xd,yd)->polyreg(xp,xd,yd,degree=1,
                                                              deriv=true),
                      clusterid=nothing)
  z = Matrix(data[!,instruments])
  dta = deepcopy(data[:,unique([y, k, l, q, Φ, id, t, instruments...,
                                :eq, :ez, :ey])]);
  dta[!,:eqlag] = panellag(:eq, dta, id, t, 1)
  dta[!,:eylag] = panellag(:ey, dta, id, t, 1)
  dta[!,:ezlag] = panellag(:ez, dta, id, t, 1)
  function createparts(datain,β,α)
    df = deepcopy(datain)
    df[!,:ω] = df[!,Φ] - df[!,k]*β[1] - df[!,l] * β[2];
    df[!,:ωlag] = panellag(:ω, df, id, t);
    df[!,:eΦ] = df[!,y] - α*df[!,q] - df[!,Φ];
    df[!,:eΦlag] = panellag(:eΦ, df, id, t);
    df[!,:ytilde] = df[!,y] - α*df[!,q] -df[!,k]*β[1] - df[!,l]*β[2];
    return(df)
  end
  dta = createparts(dta, [0.1,0.1], 0.01)
  inc = completecases(dta[:,[:ωlag, :ytilde, :eΦlag, instruments...]])
  function momenti(β::AbstractVector, α::Real)
    df = createparts(dta,β,α)
    X = reshape(disallowmissing(df[!,:ωlag][inc]), sum(inc), 1)
    Y = reshape(disallowmissing(df[!,:ytilde][inc]), sum(inc),1)
    Q = reshape(disallowmissing(df[!,:quality][inc]), sum(inc),1)
    Z = disallowmissing(z[inc,:])
    YQZ = hcat(Y,Q,Z)
    (EYQZ, dEYQZ) = npregress(X,X,YQZ)
    resid = YQZ - EYQZ
    eΦ = disallowmissing(df[!,:eΦlag][inc])
    η = resid[:,1] - eΦ.*dEYQZ[:,1]
    ez = resid[:,3:end]
    gi = η.*ez

    eqlag = disallowmissing(df[inc,:eqlag])
    dGα = mean( (-resid[:,2] + eqlag.*dEYQZ[:,1]).*ez, dims=1)
    gi = gi -
      ((df[inc,:eΦlag].*df[inc,:ezlag])/mean(skipmissing(df[inc,:eqlag].*df[inc,:ezlag]))
       )*dGα
  end
  function obj(β::AbstractVector, α::Real)
    m = momenti(β,α)
    M = mean(m, dims=1)
    (M*W*M')[1]
  end
  if (clusterid===nothing)
    N = size(gi,1)
  else
    N = length(unique(data[inc,clusterid]))
  end
  function Σ(gi)
    if (clusterid===nothing)
      V = cov(gi)
    else
      V = clustercov(gi, data[inc,clusterid])
    end
    return(V, N)
  end

  function cue(β::AbstractVector, α::Real)
    gi = momenti(β,α)
    (V, n) = Σ(gi)
    Wn = inv(V)
    M = sum(gi,dims=1)/n
    n*(M*Wn*M')[1]
  end
  return(obj=obj, momenti=momenti, cue=cue, Σ=Σ)
end

"""
    clustercov(x::Array{Real,2}, clusterid)

Compute clustered, heteroskedasticity robust covariance matrix
estimate for Var(∑ x). Assumes that observations with different
clusterid are independent, and observations with the same clusterid
may be arbitrarily correlated. Uses number of observations - 1 as the
degrees of freedom.

# Arguments

- `x` number of observations by dimension of x matrix
- `clusterid` number of observations length vector

# Returns
- `V` dimension of x by dimension of x matrix
"""
function clustercov(x::Array{<:Number,2}, clusterid)
  ucid = unique(clusterid)
  V = zeros(eltype(x),size(x,2),size(x,2))
  e = x .- mean(x, dims=1)
  for c in ucid
    idx = findall(clusterid.==c)
    V+= e[idx,:]'*e[idx,:]
  end
  V = V./(length(ucid)-1)
end

function clustercov(x::Array{<:Any,1}, clusterid)
  clustercov(reshape(x,length(x),1),clusterid)
end

function clustercov(x::Array{<:Union{Missing,<:Number},2},
                    clusterid)
  df = hcat(DataFrame(x), DataFrame([clusterid],[:cid]))
  inc = findall(completecases(df))
  clustercov(Matrix(disallowmissing(df[inc,1:size(x,2)])),
             disallowmissing(df[:cid][inc]))
end

end # module
