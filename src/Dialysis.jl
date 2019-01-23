module Dialysis

using DataFrames
using LinearAlgebra: det, Diagonal, UniformScaling
using Distributions: pdf, MvNormal
using Statistics: std, var, mean, cov
import RData
import ShiftedArrays

export panellag, locallinear, loaddata, partiallinearIV,
  errors_gm, objective_gm, polyreg

"""
    panellag(x::Symbol, data::AbstractDataFrame, id::Symbol, t::Symbol,
              lags::Integer=1)

Create lags of variables in panel data.

Inputs:
  - `x` variable to create lag of
  - `data` DataFrame containing `x`, `id`, and `t`
  - `id` cross-section identifier
  - `t` time variable
  - `lags` number of lags. Can be negative, in which cause leads will
     be created

Output:
  - A vector containing lags of data[x]. Will be missing for `id` and
   `t` combinations where the lag is not contained in `data`.

"""
function panellag(x::Symbol, data::AbstractDataFrame, id::Symbol, t::Symbol,
                   lags::Integer=1)
  if (!issorted(data, (id, t)))
    @warn "data is not sorted, panellag() will be more efficient with a sorted DataFrame"
    p = sortperm(data, (id, t))
    df = data[p,:]
  else
    p = nothing
    df = data
  end
  idlag= ShiftedArrays.lag(df[id], lags)
  tlag = ShiftedArrays.lag(df[t], lags)
  xlag = ShiftedArrays.lag(df[x], lags)
  xlag = copy(xlag)
  xlag[ (typeof.(tlag).==Missing) .|
        (tlag .!= df[t].-lags) .|
        (idlag .!= df[id]) ] .= missing
  if (p == nothing)
    return(xlag)
  else
    xout = similar(xlag)
    xout[p] .= xlag
    return(xout)
  end
end

"""

    locallinear(xpred::AbstractMatrix,
                xdata::AbstractMatrix,
                ydata::AbstractMatrix)

Computes local linear regression of ydata on xdata. Returns predicted
y at x=xpred. Uses Scott's rule of thumb for the bandwidth and a
Gaussian kernel. xdata should not include an intercept. 

Inputs:
  - `xpred` x values to compute fitted y
  - `xdata` observed x
  - `ydata` observed y, must have `size(y)[1] == size(xdata)[1]`
  - `bandwidth_multiplier` multiply Scott's rule of thumb bandwidth by
     this number
Output: 
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
  function kernel(dx::AbstractVector) # Gaussian kernel
    pdf(dist, dx)
  end  
  X = hcat(ones(n), xdata)
  w = Array{eltype(xpred), 1}(undef, n)
  for i in 1:size(xpred)[1]
    for j in 1:size(xdata)[1]
      w[j] = sqrt(kernel((xdata[j,:] - xpred[i,:])))
    end 
    #ypred[i,:] = X[i,:]'* ((X'*Diagonal(w)*X) \
    #(X'*Diagonal(w)*ydata))
    ypred[i,:] = X[i,:]'* ((Diagonal(w)*X) \ (Diagonal(w)*ydata))
  end
  return(ypred)
end


"""

      polyreg(xpred::AbstractMatrix,
              xdata::AbstractMatrix,
              ydata::AbstractMatrix; degree=1)

Computes  polynomial regression of ydata on xdata. Returns predicted
y at x=xpred. 

Inputs:
  - `xpred` x values to compute fitted y
  - `xdata` observed x
  - `ydata` observed y, must have `size(y)[1] == size(xdata)[1]`
  - `degree`
Output: 
  - Estimates of `f(xpred)`
"""
function polyreg(xpred::AbstractMatrix,
                 xdata::AbstractMatrix,
                 ydata::AbstractMatrix;
                 degree=1)
  function makepolyx(xdata, degree)
    X = ones(size(xdata,1),1)
    for d in 1:degree
      Xnew = Array{eltype(xdata), 2}(undef, size(xdata,1), size(X,2)*size(xdata,2))
      k = 1
      for c in 1:size(xdata,2)
        for j in 1:size(X,2)
          Xnew[:, k] = X[:,j] .* xdata[:,c]
          k += 1
        end
      end
      X = hcat(X,Xnew)
    end
    return(X)
  end
  X = makepolyx(xdata,degree)
  Xp = makepolyx(xpred,degree)
  ypred = Xp * (X \ ydata)
  return(ypred)
end

"""
    loaddata()

Loads "dialysisFacilityReports.rda". Returns a DataFrame.
"""
function loaddata()
  rdafile=normpath(joinpath(@__DIR__,"..","data","dialysisFacilityReports.rda"))
  dt = RData.load(rdafile,convert=true)
  dt["dialysis"]
end

"""
      partiallinearIV(y::Symbol, q::Symbol, z::Symbol,
                      controls::Array{Symbol,1}, data::DataFrame;
                      npregress<:Function)
 
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


Inputs:
- `y` symbol specificying y variable
- `q`
- `z`
- `controls` list of control variables entering Φ
- `data` DataFrame where all variables are found
- `npregress` function for estimating E[w|x] nonparametrically. Used
   to partial out E[y|controls], E[q|z,controls], and E[q|controls]. Syntax
   should be the same as `locallinear` or `polyreg`

Output:
- `α` estimate of α
- `Φ` estimate of Φ(controls)
"""
function partiallinearIV(y::Symbol, q::Symbol, z::Symbol,
                         controls::Array{Symbol,1}, data::DataFrame;
                         npregress::Function=locallinear)
  # drop missing observations
  vars = [y, q, z, controls...]
  inc = completecases(data[vars])
  Y = disallowmissing(data[y][inc])
  Q = disallowmissing(data[q][inc])
  Z = disallowmissing(data[z][inc])
  X = disallowmissing(convert(Matrix, data[controls][inc,:]))
  XZ = hcat(X,Z)
  qhat = npregress(XZ,XZ,reshape(Q,length(Q),1))
  fits = npregress(X,X,hcat(Y,qhat))
  ey = Y - fits[:,1]
  eq = Q - fits[:,2]
  ez = qhat - fits[:,2]
  α = (ey'*ez)/(eq'*ez)
  #Φi = npregress(X, X, reshape(Y - α*Q, length(Y),1))
  Φi = fits[:,1] - α*fits[:,2]
  
  # put missings back in
  if (sum(inc) != length(inc)) 
    Φ = Array{Union{Missing, eltype(Φi)},1}(undef, length(data[y]))
    Φ .= missing
    Φ[inc] = Φi
  else
    Φ = Φi
  end
  return(α=α, Φ=Φ)
end

"""
    errors_gm(y::Symbol, k::Symbol, l::Symbol, Φ::Symbol,
                   id::Symbol, t::Symbol, data::DataFrame;
                   npregress::Function=polyreg)

Returns functions that given β calculate ω(β) and η(β).

Inputs:
 - `y` Symbol for y variable in data
 - `k` Symbol for k variable in data
 - `l` Symbol for l variable in data
 - `q` Symbol for q variable in data
 - `Φ` Symbol for Φ variable in data
 - `id`  Symbol for id variable in data
 - `t` Symbol for t variable in data
 - `data` DataFrame containing variables
 - `α` estimate of α

Output: 
 - ωfunc(β) computes ω given β for the `data` and α passed in as
   input. `length(ωfunc(β)) == nrow(data)` ωfunc(β) will contain missings
   if the data does.
 - ηfunc(β) computes η fiven β. for the `data` and α passed in as
   input. `length(ηfunc(β)) == nrow(data)` ηfunc(β) will contain
   missings.

"""
function errors_gm(y::Symbol,  k::Symbol, l::Symbol, q::Symbol,
                   Φ::Symbol, id::Symbol, t::Symbol, data::DataFrame,
                   α::Real;
                   npregress::Function=polyreg, degree=1)
  function Ω(β::AbstractVector)
    data[Φ] - data[k]*β[1] - data[l] * β[2];
  end
  df = deepcopy(data)
  function Η(β::AbstractVector)
    df[:ω] = Ω(β);
    df[:ωlag] = panellag(:ω, df, id, t);
    df[:ytilde] = df[y] - α*df[q] -df[k]*β[1] - df[l]*β[2];
    inc = completecases(df[[:ωlag, :ytilde]])
    X = reshape(disallowmissing(df[:ωlag][inc]), sum(inc), 1)
    Y = reshape(disallowmissing(df[:ytilde][inc]), sum(inc),1)
    ηi = Y - npregress(X,X,Y, degree=degree)
    η = Array{Union{Missing, eltype(ηi)}, 1}(undef, nrow(df))
    η .= missing
    η[inc] = ηi
    return(η)
  end
  return(ωfunc=Ω, ηfunc=Η)
end

"""
    objective_gm(k::Symbol, l::Symbol, data::DataFrame,
                 ηfunc::Function; W=I)



"""
function objective_gm(instruments::Array{Symbol,1}, data::DataFrame,
                      ηfunc::Function; W=UniformScaling(1.))
  z = convert(Matrix,data[instruments]);
  function momenti(β::AbstractVector)
    η = ηfunc(β)
    η.*z
  end
  function obj(β::AbstractVector)
    m = momenti(β)
    M = mapslices(x->mean(skipmissing(x)), m, dims=1)
    (M*W*M')[1]
  end
  function cue(β::AbstractVector)
    m = momenti(β)
    gi = m[.!ismissing.(m[:,1]) .& .!ismissing.(m[:,2]),:]
    W = inv(cov(gi))
    M = mean(gi,dims=1)
    (size(gi,1)*M*W*M')[1]
  end
  return(obj=obj, momenti=momenti, cue=cue)
end


end # module



