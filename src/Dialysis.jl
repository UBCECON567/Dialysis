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


"""
    panellag(x::Symbol, data::AbstractDataFrame, id::Symbol, t::Symbol,
              lags::Integer=1)

Create lags of variables in panel data.

# Arguments

- `x` variable to create lag of
- `data` DataFrame containing `x`, `id`, and `t`
- `id` cross-section identifier
- `t` time variable
- `lags` number of lags. Can be negative, in which cause leads will
   be created

# Returns

- A vector containing lags of data[x]. Will be missing for `id` and
  `t` combinations where the lag is not contained in `data`.

"""
function panellag(x::Symbol, data::AbstractDataFrame, id::Symbol, t::Symbol,
                   lags::Integer=1)
  if (!issorted(data, [id, t]))
    @warn "data is not sorted, panellag() will be more efficient with a sorted DataFrame"
    p = sortperm(data, [id, t])
    df = data[p,:]
  else
    p = nothing
    df = data
  end
  idlag= ShiftedArrays.lag(df[!,id], lags)
  tlag = ShiftedArrays.lag(df[!,t], lags)
  xlag = ShiftedArrays.lag(df[!,x], lags)
  xlag = copy(xlag)
  xlag[ ismissing.(tlag) .|
        (tlag .!= df[!,t].-lags) .|
        (idlag .!= df[!,id]) ] .= missing
  if (p == nothing)
    return(xlag)
  else
    xout = similar(xlag)
    xout[p] .= xlag
    return(xout)
  end
end

function panellag(x::AbstractArray, id::AbstractVector, t::AbstractVector,
                  lags::Integer=1)
  idlag= ShiftedArrays.lag(id, lags)
  tlag = ShiftedArrays.lag(t, lags)
  xlag = copy(ShiftedArrays.lag(x, lags))
  xlag[ ismissing.(tlag) .|
        (tlag .!= t.-lags) .|
        (idlag .!= id) ] .= missing
  return(xlag)
end


"""

    locallinear(xpred::AbstractMatrix,
                xdata::AbstractMatrix,
                ydata::AbstractMatrix)

Computes local linear regression of ydata on xdata. Returns predicted
y at x=xpred. Uses Scott's rule of thumb for the bandwidth and a
Gaussian kernel. xdata should not include an intercept.

# Arguments

- `xpred` x values to compute fitted y
- `xdata` observed x
- `ydata` observed y, must have `size(y)[1] == size(xdata)[1]`
- `bandwidth_multiplier` multiply Scott's rule of thumb bandwidth by
   this number

# Returns

- Estimates of `f(xpred)`
"""
function locallinear(xpred::AbstractMatrix,
                     xdata::AbstractMatrix,
                     ydata::AbstractMatrix;
                     bandwidth_multiplier=1.0)

  (n, d) = size(xdata)
  ypred = Array{eltype(xpred), 2}(undef, size(xpred,1), size(ydata,2))
  # use Scott's rule of thumb
  rootH = n^(-1/(d+4))*vec(std(xdata;dims=1))*bandwidth_multiplier
  dist = MvNormal(rootH)
  X = hcat(ones(n), xdata)
  w = Array{eltype(xpred), 1}(undef, n)
  dx = Array{eltype(xdata), 2}(undef, d,n)
  @inbounds for i in 1:size(xpred)[1]
    @views dx .= (xdata' .- xpred[i,:])
    pdf!(w, dist, dx)
    w .= sqrt.(w)
    @views ypred[i,:] = X[i,:]'* ((X'*Diagonal(w)*X) \ (X'*Diagonal(w)*ydata))
  end
  return(ypred)
end


"""

      polyreg(xpred::AbstractMatrix,
              xdata::AbstractMatrix,
              ydata::AbstractMatrix; degree=1)

Computes  polynomial regression of ydata on xdata. Returns predicted
y at x=xpred.

# Arguments

- `xpred` x values to compute fitted y
- `xdata` observed x
- `ydata` observed y, must have `size(y)[1] == size(xdata)[1]`
- `degree`
- `deriv` whether to also return df(xpred). Only implemented when
   xdata is one dimentional

# Returns

- Estimates of `f(xpred)`
"""
function polyreg(xpred::AbstractMatrix,
                 xdata::AbstractMatrix,
                 ydata::AbstractMatrix;
                 degree=1, deriv=false)
  function makepolyx(xdata, degree, deriv=false)
    X = ones(size(xdata,1),1)
    dX = nothing
    for d in 1:degree
      Xnew = Array{eltype(xdata), 2}(undef, size(xdata,1), size(X,2)*size(xdata,2))
      k = 1
      for c in 1:size(xdata,2)
        for j in 1:size(X,2)
          @views Xnew[:, k] = X[:,j] .* xdata[:,c]
          k += 1
        end
      end
      X = hcat(X,Xnew)
    end
    if (deriv)
      if (size(xdata,2) > 1)
        error("polyreg only supports derivatives for one dimension x")
      end
      dX = zeros(eltype(X), size(X))
      for c in 2:size(X,2)
        dX[:,c] = (c-1)*X[:,c-1]
      end
    end
    return(X, dX)
  end
  X = makepolyx(xdata,degree)
  (Xp, dXp) = makepolyx(xpred,degree, deriv)
  coef = (X \ ydata)
  ypred = Xp * coef
  if (deriv)
    dy = dXp * coef
    return(ypred, dy)
  else
    return(ypred)
  end
end


include("data.jl")

"""
    loaddata()

Loads "dialysisFacilityReports.rda". Returns a DataFrame.
"""
function loaddata()
  rdafile=normpath(joinpath(dirname(Base.pathof(Dialysis)),"..","data","dialysisFacilityReports.rda"))
  dt = RData.load(rdafile,convert=true)
  dt["dialysis"]
end

"""
function partiallinear(y::Symbol, x::Array{Symbol, 1},
                       controls::Array{Symbol,1},
                       data::DataFrame;
                       npregress::Function=polyreg,
                       clustervar::Symbol=Symbol())

Estimates a partially linear model. That is, estimate

```math
  y = xβ + f(controls) + ϵ
```

Assuming that E[ϵ|x, controls] = 0.

# Arguments

- `y` symbol specificying y variable
- `x`
- `controls` list of control variables entering f
- `data` DataFrame where all variables are found
- `npregress` function for estimating E[w|x] nonparametrically. Used
   to partial out E[y|controls] and E[x|controls] and E[q|controls].
- `clustervar` symbol specifying categorical variable on which to
   cluster when calculating standard errors

# Returns

- regression output from FixedEffectModels.jl

# Details

Uses orthogonal (with respect to f) moments to estimate β. In particular, it uses

```math
0 = E[(y - E[y|controls]) - (x - E[x|controls])β)*(x - E[x|controls])]
```

to estimate β. In practice this can be done by regressing
(y - E[y|controls]) on (x - E[x|controls]). FixedEffectModels is used
for this regression. Due to the orthogonality of the moment condition
the standard errors on β will be the same as if
E[y|controls] and E[x|controls] were observed (i.e. FixedEffectModels
will report valid standard errors)
"""
function partiallinear(y::Symbol, x::Array{Symbol, 1},
                       controls::Array{Symbol,1},
                       data::DataFrame;
                       npregress::Function=polyreg,
                       clustervar::Symbol=Symbol())
  vars = [y, x..., controls...]
  inc = completecases(data[!,vars])
  YX = disallowmissing(convert(Matrix, data[!,[y, x...]][inc,:]))
  W = disallowmissing(convert(Matrix, data[!,controls][inc,:]))
  fits = npregress(W,W,YX)
  Resid = YX - fits
  df = DataFrame(Resid, [y, x...])
  if (clustervar==Symbol())
    est=reg(df, @eval @formula($(y) ~ $(Meta.parse(String(reduce((a,b) -> "$a + $b",
                                                          x)))) ))
  else
    df[!,clustervar] = data[!,clustervar][inc]
    est=reg(df, @eval @formula($(y) ~ $(Meta.parse(String(reduce((a,b) -> "$a + $b", x)))))
            , Vcov.cluster(clustervar) )
  end
  eyex = Matrix{Union{Missing,eltype(fits)}}(undef, nrow(data), length(x)+1)
  eyex .= missing
  eyex[inc,:] .= fits
  return(est, eyex)
end

"""
      partiallinearIV(y::Symbol, q::Symbol, z::Symbol,
                      controls::Array{Symbol,1}, data::DataFrame;
                      npregress::Function)

Estimates a partially linear model using IV. That is, estimate

y = αq + Φ(controls) + ϵ

using z as an instrument for q with first stage

q = h(z,controls) + u

It assumes that E[ϵ|z, controls] = 0.

Uses orthogonal (wrt Φ and other nuisance functions) moments for
estimating α. In particular, it uses

0 = E[(y - E[y|controls] - α(q - E[q|controls]))*(E[q|z,controls] - E[q|controls])]

See section 4.2 (in particular footnote 8) of Chernozhukov,
Chetverikov, Demirer, Duflo, Hansen, Newey, and Robins (2018) for more
information.

In practice α can be estimated by an iv regression
of (y - E[y|controls]) on (q - E[q|controls]) using (E[q|z,controls] -
E[q|controls]) as an instrument. FixedEffectModels is used
for this regression. Due to the orthogonality of the moment condition,
the standard error on α will be the same as if
E[y|controls] and E[q|controls] were observed (i.e. FixedEffectModels
will report valid standard errors)


# Arguments
- `y` symbol specificying y variable
- `q`
- `z` list of instruments
- `controls` list of control variables entering Φ
- `data` DataFrame where all variables are found
- `npregress` function for estimating E[w|x] nonparametrically. Used
   to partial out E[y|controls], E[q|z,controls], and E[q|controls]. Syntax
   should be the same as `locallinear` or `polyreg`

# Returns
- `α` estimate of α
- `Φ` estimate of Φ(controls)
- `regest` regression output with standard error for α
"""
function partiallinearIV(y::Symbol, q::Symbol, z::Array{Symbol,1},
                         controls::Array{Symbol,1}, data::DataFrame;
                         npregress::Function=locallinear,
                         parts=false)
  # drop missing observations
  vars = [y, q, z..., controls...]
  inc = completecases(data[:,vars])
  Y = disallowmissing(data[inc,y])
  Q = disallowmissing(data[inc,q])
  Z = disallowmissing(convert(Matrix, data[inc,z]))
  X = disallowmissing(convert(Matrix, data[inc,controls]))
  XZ = hcat(X,Z)
  qhat = npregress(XZ,XZ,reshape(Q,length(Q),1))
  fits = npregress(X,X,hcat(Y,qhat))
  ey = Y - fits[:,1]
  eq = Q - fits[:,2]
  ez = qhat - fits[:,2]
  α = (ey'*ez)/(eq'*ez)
  df = DataFrame([ey,eq,vec(ez)], [:y, :q, :z])
  est = reg(df, @formula(y ~ (q ~ z)))
  #Φi = npregress(X, X, reshape(Y - α*Q, length(Y),1))
  Φi = fits[:,1] - α*fits[:,2]

  # put missings back in
  if (sum(inc) != length(inc))
    Φ = Array{Union{Missing, eltype(Φi)},1}(undef, length(data[y]))
    Φ .= missing
    Φ[inc] = Φi
    if (parts)
      eyqz = Array{Union{Missing, eltype(ey)},2}(undef,
                                                 length(data[y]),3)
      eyqz .= missing
      eyqz[inc,1] = ey
      eyqz[inc,2] = eq
      eyqz[inc,3] = ez
    else
      eyqz= Array{eltype(ey),2}(undef,0,0)
    end
  else
    Φ = Φi
    if (parts)
      eyqz = hcat(ey,eq,ez)
    else
      eyqz = Array{eltype(ey),2}(undef,0,0)
    end
  end
  return(α=α, Φ=Φ, regest=est, eyqz=eyqz)
end

function partiallinearIV(y::Symbol, q::Symbol, z::Symbol,
                         controls::Array{Symbol,1}, data::DataFrame;
                         npregress::Function=locallinear,
                         parts=false)
  partiallinearIV(y, q, [z], controls, data, npregress=npregress, parts=parts)
end

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
  z = convert(Matrix,data[!,instruments]);
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
  if (clusterid==nothing)
    N = size(gi,1)
  else
    N = length(unique(data[inc,clusterid]))
  end
  function Σ(gi)
    if (clusterid==nothing)
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
  clustercov(disallowmissing(convert(Matrix,df[inc,1:size(x,2)])),
             disallowmissing(df[:cid][inc]))
end

end # module
