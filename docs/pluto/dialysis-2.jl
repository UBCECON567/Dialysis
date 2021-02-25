### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ e25c28d8-6a36-11eb-07f3-2fcd5b400618
begin
	using Pkg
	try
		using Dialysis # This assignment itself is in the "Dialysis" package.
	catch
		# if Dialysis hasn't been downloaded, the previous
		# using will cause an error, so we enter this catch

		# download Dialysis package
	    Pkg.develop(PackageSpec(url="https://github.com/UBCECON567/Dialysis"))
	    using Dialysis
	end

	# the packages we will use in this assignment are listed in
	# https://github.com/UBCECON567/Dialysis/docs/Project.toml (and Manifest.toml)
	# Now we activate that this Project.toml environment
	docdir = normpath(joinpath(dirname(Base.pathof(Dialysis)), "..","docs"))
	Pkg.activate(docdir)
	# Now any Pkg.add or Pkg.develop command will modify Dialysis/docs/Project.toml instead of your default project environment (which is stored in something like ~/.julia/environments/v1.5/Project.toml)
	# It's good to have separate Project.toml files for different tasks

	Pkg.develop(PackageSpec(url="https://github.com/UBCECON567/Dialysis"))
	Pkg.instantiate() # download all the packages needed
	using DataFrames, Plots, StatsPlots, StatsBase, Statistics, PlutoUI
	Plots.gr(fmt="png")
end

# ╔═╡ 7c447c7a-6a37-11eb-03cf-6f2c5b11326b
"""
    module Cleaning

Functions used in cleaning Dialysis Facility Reports data.
"""
module Cleaning
	using Dates, DataFrames, StatsBase, Statistics
	export guesstype, converttype, combinefiscalyears, parse, upcase, dayssince
	"""
	    guesstype(x)

	Try to guess at appropriate type for x.
	"""
	function guesstype(x::AbstractArray{T}) where T <:Union{String,Missing}
		if all(occursin.(r"(^\.$)|(^(-|)\d+)$",skipmissing(x)))
			return Int
		elseif all(occursin.(r"(^\.$)|(^(-|\d)\d*(\.|)\d*$)",skipmissing(x)))
			return Float64
		elseif all(occursin.(
					r"(^\.$)|(^\d\d\D{3}\d{4}$|^\d\d/\d\d/\d{4}$)",
					skipmissing(x)))
			return Date
		else
			return String
		end
	end
	guesstype(x) = eltype(x)

	
	parse(t, x) = Base.parse(t, x)
	# we need a parse that "converts" a string to string
	parse(::Type{String}, x::String) = x	
	# a version of parse that works for the date formats in this data
	parse(::Type{Dates.Date}, x::String) = occursin(r"\D{3}",x) ? Date(x, "dduuuyyyyy") : Date(x,"m/d/y")
	
	"""  
	    converttype(x)
    
    Convert `x` from an array of strings to a more appropriate numeric type 
	if possible.
	"""
	function converttype(x::AbstractArray{T}) where T <: Union{Missing, String}
		etype = guesstype(x)
		return([(ismissing(val) || val==".") ? missing : parse(etype,val)
			for val in x])
	end
	converttype(x) = x

	function combinefiscalyears(x::AbstractArray{T}) where T <: Union{Missing,Number}
		if all(ismissing.(x))
			return(missing)
		else
			v = median(skipmissing(x))
			return(isnan(v) ? missing : v)
		end
	end
	function combinefiscalyears(x)
		# use most common value
		if all(ismissing.(x))
			return(missing)
		else
			return(maximum(countmap(skipmissing(x))).first)
		end
	end

	upcase(x) = Base.uppercase(x)
	upcase(m::Missing) = missing	

	function dayssince(year, dates) 
		today = Date(year, 12, 31)
		past = [x.value for x in today .- dates if x.value>=0]
		if length(past)==0
			return(missing)
		else
			return(minimum(past))
		end
	end
	
end

# ╔═╡ 3b1eb36e-6a37-11eb-06ac-ef996c61826d
dialysis, datadic = let 
	using .Cleaning, DataFrames
	dialysis, datadic = Dialysis.loadDFR(recreate=false)
	
	# fix the identifier strings. some years they're listed as ="IDNUMBER", others, they're just IDNUMBER
	dialysis.provfs = replace.(dialysis.provfs, "="=>"")
	dialysis.provfs = replace.(dialysis.provfs,"\""=>"")
	# convert strings to numeric types
	dialysis=mapcols(converttype, dialysis) 

	
	dialysis = combine(groupby(dialysis, [:provfs,:year]),
		names(dialysis) .=> combinefiscalyears .=> names(dialysis))
	sort!(dialysis, [:provfs, :year])
	
	pt = Symbol.(filter(x->occursin.(r"PT$",x),names(dialysis)))
	ft = Symbol.(filter(x->occursin.(r"FT$",x),names(dialysis)))			
	dialysis[!,:labor]=sum.(skipmissing(eachrow(dialysis[!,pt])))*0.5 + 
					   sum.(skipmissing(eachrow(dialysis[!,ft])))
	
	dialysis.hiring = panellag(:labor, dialysis, :provfs, :year, -1) - dialysis.labor
	dialysis.investment = panellag(:totstas_f, dialysis, :provfs, :year, -1) - dialysis.totstas_f
	
	
	dialysis.forprofit = (x->(x=="Unavailable" ? missing : 
		x=="For Profit")).(dialysis.owner_f)
	
	# Chains
	dialysis.fresenius = (x->(ismissing(x) ? false :
			occursin(r"(FRESENIUS|FMC)",x))).(dialysis.chainnam)
	dialysis.davita = (x->(ismissing(x) ? false :
			occursin(r"(DAVITA)",x))).(dialysis.chainnam)
	# could add more
	
	# State inpection rates
	inspect = combine(groupby(dialysis, :provfs), 
		:surveydt_f => x->[unique(skipmissing(x))])
	rename!(inspect, [:provfs, :inspection_dates])
	df=innerjoin(dialysis, inspect, on=:provfs)
	@assert nrow(df)==nrow(dialysis)

	df=transform(df, [:year, :inspection_dates] => (y,d)->dayssince.(y,d))	
	rename!(df, names(df)[end] =>:days_since_inspection)
	df[!,:inspected_this_year] = ((df[!,:days_since_inspection].>=0) .&
		(df[!,:days_since_inspection].<365))
	
	# then take the mean by state
	stateRates = combine(groupby(df, [:state, :year]),
                	:inspected_this_year => 
			(x->mean(skipmissing(x))) => :state_inspection_rate)
	# if no inpections in a state in a year then 
	# mean(skipmissing(x)) will be mean([]) = NaN. 0 makes more sense
	stateRates.state_inspection_rate[isnan.(stateRates.state_inspection_rate)] .= 0
	
	df = innerjoin(df, stateRates, on=[:state, :year])
	@assert nrow(df)==nrow(dialysis)
	
	# competitors
	df[!,:provcity] = upcase.(df[!,:provcity])
	comps = combine(groupby(df,[:provcity,:year]),
    	       		:dy => 
			(x -> length(skipmissing(x).>=0.0)) => 
			:competitors
           )
	comps = comps[.!ismissing.(comps.provcity),:]
 	dialysis = outerjoin(df, comps, on = [:provcity,:year], matchmissing=:equal)	
	@assert nrow(dialysis)==nrow(df)
	
	dialysis, datadic
end;

# ╔═╡ b5170194-7711-11eb-29e3-b54bef191b1e
using FixedEffectModels

# ╔═╡ 70fbf634-7781-11eb-2aca-e33cfac2bc00
using Test

# ╔═╡ 0e285b06-7781-11eb-2594-81071962ac97
let 	
	function sim_ols(n; β = ones(3))
		x = randn(n, length(β))
  		ϵ = randn(n)
  		y = x*β + ϵ
  		return(x,y)
	end
	β = ones(2)
	(x, y) = sim_ols(100; β=β)
	βols = (x'*x) \ (x'*y)

	function gmm_objective(β)
  		gi = (y - x*β) .* x
  		Egi = mean(gi, dims=1) # note that Egi will be 1 × length(β)
  		error("This is incomplete; you must finish it")
  		# It is is likely that the code you will write will return a 1 x 1,
  		# 2 dimensional array. For compatibility with Optim, you need to
  		# return a scalar. If foo is a 1x1 array, write `foo[1]` to return a scalar instead of
  		# 1x1 array
	end

	# minimizer gmm_objective
	using Optim # github page : https://github.com/JuliaNLSolvers/Optim.jl
	# docs : http://julianlsolvers.github.io/Optim.jl/stable/
	res = optimize(gmm_objective,
  				   zeros(size(β)), # initial value
	               BFGS(), # algorithm, see http://julianlsolvers.github.io/Optim.jl/stable/
	               autodiff=:forward)
  	βgmm = res.minimizer
	Test.@test βgmm ≈ βols
	res, βgmm, βols
end

# ╔═╡ c3feba7a-6a36-11eb-0303-7b823e9867b0
md"""

# Reproducing Grieco & McDevitt (2017)

Paul Schrimpf

[UBC ECON567](https://faculty.arts.ubc.ca/pschrimpf/565/565.html)

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)
"""

# ╔═╡ e85bfc08-7789-11eb-3a9d-b7b7a95b74ae
md"""
Download the [notebook file from github](https://raw.githubusercontent.com/UBCECON567/Dialysis/master/docs/pluto/dialysis-2.jl) and open it in Pluto.

!!! tip

    Instead of downloading the notebook manually, you can let Julia download when Pluto starts by entering
    ```julia
    Pluto.run(notebook="https://raw.githubusercontent.com/UBCECON567/Dialysis/master/docs/pluto/dialysis-2.jl")
    ```
    instead of Pluto.run(). Be sure to save the notebook somewhere on your computer after it opens.

"""

# ╔═╡ 1a1313d4-7713-11eb-2fef-af1b32a702c2
md""" 
# Update Dialysis.jl package

I have made some small changes to the Dialysis.jl package (mostly adding some variables that might be useful). Assuming that you used the code below to originally install it, you can update it by running the following.
```julia
try 
	using Revise
catch
	using Pkg
	Pkg.add("Revise")
	using Revise
end
using Dialysis
pkgdir = normpath(joinpath(dirname(Base.pathof(Dialysis)), ".."))
cmd = `cd $pkgdir && git pull origin master`
run(cmd)
```

Note: we only have to update the package this way, because it was installed via `Pkg.develop` instead of `Pkg.add`. With `Pkg.add`, a simple `Pkg.update("Dialysis")` would update it. 

!!! info "Revise.jl"
    The [Revise](https://timholy.github.io/Revise.jl/stable/) package is a very useful tool for writing Julia code. It makes it so that changes to package source code are immediately loaded into a running Julia session. The `git pull origin master` command update the Dialysis.jl code, and Revise makes sure that this update gets loaded into Julia right away. Without Revise.jl, we would have to restart Julia for the changes in Dialysis.jl to have an effect.
"""

# ╔═╡ e4fdebf8-6a36-11eb-30d3-dffd45f2943f
md"""
# Data Cleaning

We begin by taking the data you created in part 1 of the assignment. You can modify the cell below if you wish (e.g. if you prefer different variable names or definitions).
"""

# ╔═╡ 3e436bda-6a5e-11eb-3a3f-c94522e2aba1
describe(dialysis)

# ╔═╡ 0c93c0f8-7710-11eb-000e-0fba119bd871
md"""

# Quality 

Grieco and McDevitt (2017) use the residuals from regressing the infection rate on patient characteristics as a measure of quality. Since the infection rate is a noisy measure of quality, they instrument with the standardized mortality ratio as a second measure of quality. Medicare collects the data we are using in part to create the “Dialysis Facility Compare” website, which is meant to allow consumers to compare quality of dialysis facilities. Browsing around the [Dialysis Facility Compare](https://www.medicare.gov/dialysisfacilitycompare/) or by looking at the first few pages of a [sample Dialysis Facility Report](https://data.cms.gov/Medicare/Sample-Dialysis-Facility-Report-for-Current-Year/82bq-h92z), you will see that there are a number of other variables that Medicare considers indicators of quality. 

!!! question 
    Pick one of these (it may or may not be included in the extract of data we have), and argue for or against using it instead of or in addition to the septic infection rate and standardized mortality ratio.

"""

# ╔═╡ a179f2e6-7710-11eb-2786-3790a6a86369


# ╔═╡ f8a531da-770f-11eb-3564-23e339ffdeb1
md"""
There are a number of Julia packages that can be used for regression. I like [`FixedEffectModels`](https://github.com/FixedEffects/FixedEffectModels.jl). As the name suggests, it is focused on fixed effect models, but also works for regressions without fixed effects. It is written by economists, so it has features that economists use a lot --- convenient ways to get robust and/or clustered standard errors, and good support for IV. The [`GLM`](https://juliastats.org/GLM.jl/stable/) package is a reasonable alternative. 

Usage of FixedEffectModels is fairly similar to R, in that it has a formula interface for constructing `X` matrices from a `DataFrame`. 
"""

# ╔═╡ de6dc514-7711-11eb-2c01-bfd2d49bfe43
sort(datadic)

# ╔═╡ 442c71b8-6a5e-11eb-1dda-8b6b4c760f53
begin 
	dialysis[!,:idcat] = categorical(dialysis[!,:provfs])
	# FixedEffectModels requires clustering and fixed effect variables to
	# be categorical

	qreg = reg(dialysis, @formula(sepi ~ days_since_inspection + age +
	                              sex + vin + ppavf + 
								  clmcntcom +
                                  hgm),
		       Vcov.cluster(:idcat),
		       save=true) # saves residuals 
	dialysis[!,:quality] = -qreg.residuals
	qreg
end

# ╔═╡ 00cae6b8-7731-11eb-258a-f1795edc3190
md""" 
# OLS and Fixed Effects Estimates


!!! question
    Reproduce columns 2,3, 5, and 6 of Table 5. The syntax for fixed effects regression is shown below. Be sure to add the other columns. If you’d like, you could use [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl) to produce tables that look a lot like the ones in the paper.
"""

# ╔═╡ 51460370-7731-11eb-13be-a9c361b59e11
# you may want to use patient_years_hd or patient_years_rom instead
begin
	# -Inf causes as error in reg()
	log_infmiss = x->ifelse(!ismissing(x) && x>0, log(x), missing) 
	
	dialysis[!,:lpy] = log_infmiss.(dialysis[!,:dy]) # or hdy or phd?
	dialysis[!,:logL] = log_infmiss.(dialysis[!,:labor])
	dialysis[!,:logK] = log_infmiss.(dialysis[!,:totstas_f])

	# you may want to restrict sample to match sample that can be 
	# used in control function estimates 
	reg(dialysis, @formula(lpy ~ quality + logK + logL + fe(idcat)),
		Vcov.cluster(:idcat))
end

# ╔═╡ 546b2e76-7732-11eb-0b27-1bc2d7c1e576
md"""
```math
\def\indep{\perp\!\!\!\perp}
\def\Er{\mathrm{E}}
\def\R{\mathbb{R}}
\def\En{{\mathbb{E}_n}}
\def\Pr{\mathrm{P}}
\newcommand{\norm}[1]{\left\Vert {#1} \right\Vert}
\newcommand{\abs}[1]{\left\vert {#1} \right\vert}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\def\inprob{\,{\buildrel p \over \rightarrow}\,}
\def\indist{\,{\buildrel d \over \rightarrow}\,}
```

# Estimation of α

As discussed in section 5 of @grieco2017, the coefficient on quality,
$\alpha$, is estimated from

```math
y_{jt} = \alpha q_{jt} + \Phi(\underbrace{h_{jt}, k_{jt}, l_{jt}, x_{jt}}_{w_{jt}}) +
\epsilon_{jt}
```

with a second noisy measure of quality, $z_{jt}$, used to instrument
for $q_{jt}$. To estimate $\alpha$, first the exogenous variables,
$w$, can be partialed out to give:

```math
y_{jt} - \Er[y_{jt} | w_{jt} ] = \alpha (q_{jt} - \Er[q_{jt}|w_{jt}]) +
\epsilon_{jt}
```

where we used the assumption that $\Er[\epsilon_{jt} | w_{jt} ] = 0$
and the fact that $\Er[\Phi(w) | w] = \Phi(w)$. Under the assumption
that $\Er[\epsilon| z, w] = 0$, we can estimate $\alpha$ based on the
moment condition:

```math
\begin{align*}
0 = & \Er[\epsilon f(z,w) ] \\
0 = & \Er\left[ \left(y_{jt} - \Er[y_{jt} | w_{jt} ] - \alpha
(q_{jt} - \Er[q_{jt}|w_{jt}])\right) f(z_{jt},w_{jt}) \right]
\end{align*}
```

If $Var(\epsilon|z,w)$ is constant, the efficient choice of $f(z,w)$
is

```math
\Er[\frac{\partial \epsilon}{\partial \alpha} |z, w ] = \Er[q| z, w] - \Er[q|w]
```

To estimate $\alpha$, we simply replace these conditional expectations with
regression estimates, and replace the unconditional expectation with a
sample average. Let $\hat{\Er}[y|w]$ denote a nonparmetric estimate of
the regression of $y$ on $w$. Then,

```math
\hat{\alpha} = \frac{\sum_{j,t} (y_{jt} -
\hat{E}[y|w_{jt}])(\hat{E}[q|z_{jt},w_{jt}] - \hat{E}[q|w_{jt}])}
{\sum_{j,t} (q_{jt} - \hat{E}[q|w_{jt}])(\hat{E}[q|z_{jt},w_{jt}] - \hat{E}[q|w_{jt}])}
```

The function `partiallinearIV` in Dialysis.jl will estimate this
model. Also included are two methods for estimating
$\hat{E}[y|w]$. `polyreg` estimates a polynomial series regression,
that is it regresses $y$ on a polynomial of degree $d$ in $w$. To
allow the regression to approximate any function, the degree must
increase with the sample size, but to control variance, the degree must
not increase too quickly. We will not worry too much about the choice
of degree here.

An alternative method (and what @grieco2017 used) is
local linear regression. To estimate $\hat{E}[y|x_{jt}]$, local linear
regression estimates a linear regression of $y$ on $x$, but weights
observations by how close $x_{it}$ is to $x_{jt}$. That is,

```math
\hat{E}[y|x_{jt}] = x_{jt} \hat{\beta}(x_jt)
```

where

```math
\hat{\beta}(x_{jt}) = \argmin_\beta \sum_{i,t} (y_{it} -
x_{it}\beta)^2 k((x_{it} - x_{jt})/h_n)
```

Here $k()$ is some function with its maximum at 0 (and has some other
properties), like $k(x) \propto e^{-x^2}$. The bandwidth, $h_n$,
determines how much weight to place on observations close to vs far
from $x_{jt}$. Similar to the degree in polynomial regression, the
bandwidth must decrease toward 0 with sample size allow local linear
regression to approximate any function, but to control variance the
bandwidth must not decrease too quickly. We will not worry too much
about the choice of bandwidth. Anyway, the function `locallinear` in
Dialysis.jl estimates a local linear regression.
"""

# ╔═╡ 91e11e7a-7780-11eb-033e-efe7415b20ac
md"""
!!! question
    Estimate $\alpha$ using the following code. You may want to modify
    some aspects of it and/or report estimates of $\alpha$ for different
    choices of instrument, nonparametric estimation method, degree or
    bandwidth. Compare your estimate(s) of $\alpha$ with the ones in
    Tables 5 and 6 of @grieco2017.
"""

# ╔═╡ e4a73dac-777f-11eb-20e7-99da9a11f1af
begin 
	# create indicator for observations usable in estimation of α
	inc1 = ((dialysis[!,:dy] .> 0) .& (dialysis[!,:labor] .> 0) .&
    	       (dialysis[!,:totstas_f] .> 0) .&
        	   .!ismissing.(dialysis[!,:quality]) .&
           		.!ismissing.(dialysis[!,:smr]) .&
           		(dialysis[!,:investment].==0) .&
	           	(dialysis[!,:hiring].!=0))
	inc1[ismissing.(inc1)] .= false
	dialysis[!,:inc1] = inc1
	dialysis[!,:lsmr] = log.(dialysis[!,:smr] .+ .01)

	# As degree → ∞ and/or bandwidth → 0, whether we use :std_mortality or
	# some transformation as the instrument should not matter. However,
	# for fixed degree or bandwidth it will have some (hopefully small)
	# impact.

	(α, Φ, αreg, eyqz)=partiallinearIV(:lpy,  # y
    	                     :quality, # q
        	                 :lsmr,   # z
            	             [:hiring, :logL, :logK,
                	         :state_inspection_rate, :competitors], # w
                    	     dialysis[findall(dialysis[!,:inc1]),:];
                        	 npregress=(xp, xd,yd)->polyreg(xp,xd,yd,degree=1),
                         	parts=true
                         	# You may want to change the degree here
                         	#
                         	# You could also change `polyreg`  to
                         	# `locallinear` and `degree` to
                         	# `bandwidthmultiplier`
                         	#
                         	# locallinear will likely take some time to
                         	# compute (≈350 seconds on my computer)
                         	)

	# we will need these later in step 2
	dialysis[!,:Φ] = similar(dialysis[!,:lpy])
	dialysis[:,:Φ] .= missing
	rows = findall(dialysis[!,:inc1])
	dialysis[rows,:Φ] = Φ
	dialysis[!,:ey] = similar(dialysis[!,:lpy])
	dialysis[:,:ey] .= missing
	dialysis[rows,:ey] = eyqz[:,1]
	dialysis[!,:eq] = similar(dialysis[!,:lpy])
	dialysis[:,:eq] .= missing
	dialysis[rows,:eq] = eyqz[:,2]
	dialysis[!,:ez] = similar(dialysis[!,:lpy])
	dialysis[:,:ez] .= missing
	dialysis[rows,:ez] = eyqz[:,3]

	α
end

# ╔═╡ ae691f34-7780-11eb-2196-3b4b000f6bee
md"""
# Brief introduction to GMM

The coefficients on labor and capital are estimated by GMM. The idea
of GMM is as follows. We have a model that implies

```math
\Er[c(y,x;\theta) | z ] = 0
```

where $y$, $x$, and $z$ are observed variables. $c(y,x;\theta)$ is
some known function of the data and some parameters we want to
estimate, $\theta$. Often, $c(y,x;\theta)$ are the residuals from some
equation. For example, for linear IV, we'd have
```math
c(y,x;\theta) = y - x\theta 
```
The conditional moment restriction above implies that

```math
\Er[c(y,x;\theta)f(z) ] = 0
```
for any function $f()$. We can then estimate $\theta$ by replacing the
population expectation with a sample average and finding
$\hat{\theta}$ such that
```math
\En[c(y,x;\hat{\theta})f(z) ] \approx 0
```
The dimension of $f(z)$ should be greater than or equal to the
dimension of $\theta$, so we have at least as many equations as
unknowns. We find this $\hat{\theta}$ by minimizing a quadratic form
of these equations. That is,
```math
\hat{\theta} = \argmin_\theta \En[g_i(\theta)] W_n \En[g_i(\theta)]'
```
were $g_i(\theta) = c(y_i, x_i;\theta)f(z_i)$, and $W_n$ is some
positive definite weighting matrix.
"""

# ╔═╡ e1212ed0-7780-11eb-0606-af5ba16a2316
md"""

!!! question 
    As practice with GMM, use it to estimate a simple regression model,
    ```math
    y = x\beta + \epsilon
    ```
    assuming $\Er[\epsilon|x] = 0$. Test your code on simulated data. The
    following will help you get started.
"""

# ╔═╡ fae052fa-7781-11eb-09de-65aec4c5666a
md"""
# Estimating $\beta$

The model
implies that
```math
\omega_{jt} = \Phi(w_{jt}) - \beta_k k_{jt} - \beta_l l_{jt}
```
and
```math
y_{jt} - \alpha q_{jt} - \beta_k k_{jt} - \beta_l l_{jt} =
g(\omega_{jt-1}) + \eta_{jt}
``` 
The timing and exogeneity assumptions imply that
```math
\Er[\eta_{jt} | k_{jt}, l_{jt}, w_{jt-1}]
```
Given a value of $\beta$, and our above estimates of $\Phi$ and
$\alpha$, we can compute $\omega$ from the equation above, and then estimate
$g()$ and $\eta$ by a nonparametric regression of
$y_{jt} - \alpha q_{jt} - \beta_k k_{jt} - \beta_l l_{jt}$ on
$\omega_{jt-1}$. $\beta$ can then be estimated by finding the value of
$\beta$ that comes closest to satisfying the moment condition
```math
\Er[\eta(\beta)_{jt} k_{jt}] = 0 \text{ and } \Er[\eta(\beta)_{jt} l_{jt}]
= 0
```
To do this, we minimize
```math
Q_n(\beta) = \left( \frac{1}{N} \sum_{j,t} \eta(\beta)_{jt} (k_{jt}, l_{jt}) \right) W_n
\left( \frac{1}{N} \sum_{j,t} \eta(\beta)_{jt} (k_{jt}, l_{jt}) \right)'
```
"""


# ╔═╡ 36b69d3e-7782-11eb-3ff2-873f88e0c02d
md"""
!!! question 
    Write the body of the $Q_n(\beta)$ function below. Use it to estimate
    $\beta$. Compare your results with those of @grieco2017. Optionally,
    explore robustness of your results to changes in the specification.    
"""

# ╔═╡ 574881ac-7782-11eb-06b9-d927413d0fb7
Qn = let
	(ωfunc, ηfunc) = errors_gm(:lpy, :logK, :logL, :quality, :Φ, :provfs, :year,
    	                       dialysis, α; degree=1)
	function Qn(β)
  		η = ηfunc(β) # note that η will be vector of length = size(dialysis,1)
		             # η will include missings 
		error("You must write the body of this function")
	end
end

# ╔═╡ e7c0bff8-7785-11eb-025f-c3a86f0ae20e
begin
	out = []
	PlutoUI.with_terminal() do
		res = optimize(Qn,    # objective
	     	          [0.0, 0.0], # lower bounds, should not be needed, but
    	    	      # may help numerical performance
	                  [1.0, 1.0], # upper bounds
	                  [0.4, 0.2], # initial value
	                  Fminbox(BFGS()),  # algorithm
	                  autodiff=:forward, 
		              Optim.Options(show_trace=true))
		@show res
	    @show res.minimizer
		push!(out,res)
	end
end

# ╔═╡ 98a54b7c-7786-11eb-21f6-21d8e645491b
β̂ = out[1].minimizer

# ╔═╡ Cell order:
# ╟─c3feba7a-6a36-11eb-0303-7b823e9867b0
# ╟─e85bfc08-7789-11eb-3a9d-b7b7a95b74ae
# ╟─1a1313d4-7713-11eb-2fef-af1b32a702c2
# ╠═e25c28d8-6a36-11eb-07f3-2fcd5b400618
# ╟─e4fdebf8-6a36-11eb-30d3-dffd45f2943f
# ╠═7c447c7a-6a37-11eb-03cf-6f2c5b11326b
# ╠═3b1eb36e-6a37-11eb-06ac-ef996c61826d
# ╠═3e436bda-6a5e-11eb-3a3f-c94522e2aba1
# ╟─0c93c0f8-7710-11eb-000e-0fba119bd871
# ╠═a179f2e6-7710-11eb-2786-3790a6a86369
# ╟─f8a531da-770f-11eb-3564-23e339ffdeb1
# ╠═b5170194-7711-11eb-29e3-b54bef191b1e
# ╠═de6dc514-7711-11eb-2c01-bfd2d49bfe43
# ╠═442c71b8-6a5e-11eb-1dda-8b6b4c760f53
# ╟─00cae6b8-7731-11eb-258a-f1795edc3190
# ╠═51460370-7731-11eb-13be-a9c361b59e11
# ╟─546b2e76-7732-11eb-0b27-1bc2d7c1e576
# ╟─91e11e7a-7780-11eb-033e-efe7415b20ac
# ╠═e4a73dac-777f-11eb-20e7-99da9a11f1af
# ╟─ae691f34-7780-11eb-2196-3b4b000f6bee
# ╟─e1212ed0-7780-11eb-0606-af5ba16a2316
# ╠═70fbf634-7781-11eb-2aca-e33cfac2bc00
# ╠═0e285b06-7781-11eb-2594-81071962ac97
# ╟─fae052fa-7781-11eb-09de-65aec4c5666a
# ╟─36b69d3e-7782-11eb-3ff2-873f88e0c02d
# ╠═574881ac-7782-11eb-06b9-d927413d0fb7
# ╠═e7c0bff8-7785-11eb-025f-c3a86f0ae20e
# ╠═98a54b7c-7786-11eb-21f6-21d8e645491b
