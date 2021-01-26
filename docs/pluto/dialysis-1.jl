### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ af8e3f0e-5f73-11eb-1de8-5994ab9a8612
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
end

# ╔═╡ a80227f8-5f77-11eb-1211-95dd2c151877
using DataFrames, # package for storing and interacting with datasets
	Dates

# ╔═╡ 2dda9576-5f90-11eb-29eb-91dec5be5175
using Statistics, StatsBase # for mean, var, etc

# ╔═╡ b443a46e-5fa3-11eb-3e71-dfd0683dc6e9
begin
  using StatsPlots , Plots
  Plots.gr(fmt="png") # default graphics format inside Pluto is svg. An svg with so many points will cause your browser to become unresponsive!
end

# ╔═╡ d5554696-5f6f-11eb-057f-a79641cf483a
md"""

# Reproducing Grieco & McDevitt (2017)

Paul Schrimpf

[UBC ECON567](https://faculty.arts.ubc.ca/pschrimpf/565/565.html)

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)
"""

# ╔═╡ ad1cc4c6-5f72-11eb-35d5-53ff88f1f041
md"""

## Getting Started

### Installing Julia and Pluto

You can install Julia and Pluto on your own computer. To do so, first
download and install Julia from
[https://julialang.org/](https://julialang.org/). I recommend the
choosing the current stable release.

After installing, open Julia. This will give a Julia command
prompt. Enter `]` to switch to package manager mode. The command
prompt should change from a green `julia>` to a blue `(@v1.5) pkg>`. In
package mode, type `add Pluto` and press enter. This install the
[Pluto.jl package](https://github.com/fonsp/Pluto.jl) and its
dependencies. It will take a few minutes. When finished, type `Ctrl+c`
to exit package mode. Now at the green `julia>` prompt, enter

```julia
using Pluto
Pluto.run()
```

This will open the Pluto interface in your browser. If you close Julia
and want to start Pluto again, you only need to repeat this last step.

### Julia Resources

This assignment will try to explain aspects of Julia as
needed. However, if at some point you feel lost, you may want to
consult some of the following resources. Reading the first few
sections of either QuantEcon or Think Julia is recommended.

- [QuantEcon with Julia](https://lectures.quantecon.org/jl/)

- [Think Julia](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html#_colophon)
  A detailed introduction to Julia and programming more
  generally. Long, but recommended, especially if you're new to
  programming.

- From the julia prompt, you can access documentation with
  `?functionname`. Some packages have better documentation than
  others.

- [https://julialang.org/](https://julialang.org/) is the website for
  Julia

- Documentation for core Julia can be found at
  [https://docs.julialang.org/](https://docs.julialang.org/). All
  Julia packages also have a github page. Many of these include
  package specific documentation.

- [Notes on Julia from ECON
  622](https://github.com/ubcecon/ECON622_2020) much of this is part
  of QuantEcon, but not all

- [The Julia Express](https://github.com/bkamins/The-Julia-Express)
  short book with examples of Julia usage

- [discourse.julialang.org](https://discourse.julialang.org/) is a discussion  forum for julia

- [Julia Slack](https://julialang.org/slack/)

"""


# ╔═╡ 7b4ecdee-5f73-11eb-388c-4d6f9719d79b
md"""

## Part I: Loading and exploring the data


### Packages

Like many programming environments (R, Python, etc), Julia relies on
packages for lots of its functionality.The following code will
download and install all the packages required for this
assignment (but the packages will still need to be loaded with `using
...`). Execute the code below. It will take some time.
"""

# ╔═╡ a75918ae-5f73-11eb-3a3e-2f64c0dcc49c
md"""
### Load Data

Now let's get to work. I originally downloaded the data for this
problem set from
[data.cms.gov](https://data.cms.gov/browse?q=dialysis). Here, you will
find zip files for each fiscal year from 2008-2021. As in @grieco2017
the data comes from Dialysis Facility Reports (DFRs) created under
contract to the Centers for Medicare and Medicaid Services
(CMS). However, there are some differences. Most notably, this data
covers 2003-2019, instead of 2004-2008 as in @grieco2017.

The Julia code in
[Dialysis/src/data.jl](https://github.com/UBCECON567/Dialysis/blob/master/src/Dialysis.jl)
downloads and combines the data files. I did my best to include all
the variables (plus more) used by @grieco2017. However, the underlying
data is complicated (there are over 1500 variables each year), so it's
possible I have made mistakes. It might be useful to look at
the documentation included with any of the "Dialysis Facility Report
Data for FY20XX" zip files at [data.cms.gov](https://data.cms.gov/browse?q=dialysis).

The result of the code in `data.jl` is the `dfr.zip` file contained in
the git repository for this assignment. This zip file contains a csv
file with most of the variables used by @grieco2017, as well as some
additional information.
"""

# ╔═╡ b9e1f8de-5f77-11eb-25b8-57e263315ac3
begin
	dialysis, datadic = Dialysis.loadDFR()
	dialysis
end

# ╔═╡ a06995aa-5f78-11eb-3939-f9aca087b12c
md"""
The variable `dialysis` now contains a `DataFrame` with the data. The variable `datadic` is a `Dictionary` with brief descriptions of the columns in `dialysis`.

Use `datadic` as follows.
"""

# ╔═╡ 5b2d6ebe-5f78-11eb-0528-ab36ae696a35
datadic["dis2"]

# ╔═╡ 5ea12c9c-5f79-11eb-34e7-2d7f07854b31
md"""
For more information on any of the variables, look at the documentation included with any of the "Dialysis Facility Report
Data for FY20XX" zip files at [data.cms.gov](https://data.cms.gov/browse?q=dialysis). The FY2008 file might be best, since the `dialysis` dataframe uses the variable names from that year (a handful of variable names change later, but most stay the same).
"""

# ╔═╡ 95ad1f3c-5f79-11eb-36fb-1b384c84317c
md"""
We will begin our analysis with some data cleaning. Then we will create some  exploratory statistics and figures. There are at least two reasons for this. 
First, we want to
check for any anomalies in the data, which may indicate an error in
our code, our understanding of the data, or the data itself. Second,
we should try to see if there are any striking patterns in the data
that deserve extra attention. We can get some information about all
the variables in the data as follows
"""

# ╔═╡ ca038b54-5f79-11eb-0851-9f684e3bb83f
describe(dialysis)

# ╔═╡ cfaabda0-5f79-11eb-17e4-a7cd045681da
md"""

### Data Cleaning

From the above, we can see that the data has some problems. It appears that "." is used to indicate missing. We should replace these with `missing`. Also, the `eltype` of most columns is `String.` We should convert to numeric types where appropriate. 

!!! note "Types"
    Every variable in Julia has a [type](https://docs.julialang.org/en/v1/manual/types/), which determines what information the variable can store. You can check the type of a variable with `typeof(variable)`. The columns of our `dialysis` DataFrame will each be some array-like type that can hold some particular types of elements. For examble, `typeof(dialysis[!,:nursePT])` (or equivalently `typeof(dialysis.nursePT)` should currently be `Array{String, 1}`. This means that right now the nursePT column can only hold strings. Therefore trying to assign an integer to an element of the column like `dialysis[2, :nursePT] = 0` will cause an error. If we want to convert the element type of the column, we have to assign the column to an entirely new array. We will do this below. 

!!! note "Missing"
    [Julia includes a special type and value to represent missing data](https://docs.julialang.org/en/v1/manual/missing/). The element type of arrays that include `missing` will be `Union{Missing, OTHERTYPE}` where `OTHERTYPE` is something like `String` or `Float64`. The `Union` means each element of the array can hold either type `Missing` or `OTHERTYPE`. Some functions will behave reasonably when they encounter `missing` values, but many do not. As a result, we will have to be slightly careful with how we handle `missing` values.

Although not apparent in the `describe(dialysis)` output, it is also worth mentioning the unique format of the data. The data is distributed with one file per fiscal year. Each fiscal year file reports the values of most variables in calendar years 6 to 2 years ago. We need to convert things to have a single calendar year value for each variable.

#### Type Conversion

We begin by converting types. We will use [regular expressions](https://docs.julialang.org/en/v1/manual/strings/#Regular-Expressions) to try identify columns whose strings all look like integers or all look like floating point numbers.
Many programming languages have ways to work with regular expressions. It is worth remembering regular expressions are a useful tool for parsing strings, but beyond that do not worry about the dark art of regular expressions too much.
"""

# ╔═╡ c642b578-5f77-11eb-1346-15a35500e61f
"""
    guesstype(x)

Try to guess at appropriate type for x.
"""
function guesstype(x::AbstractArray{T}) where T <:Union{String,Missing}
	# r"" creates a regular expression
	# regular expressions are useful for matching patterns in strings
	# This regular expression matches strings that are either just "." or begin with - or a digit and are followed by 0 or more additional digits
	#
	# all(array) is true if all elements of the array are true
	#
	# skipmissing(x) creates a iterator over the non-missing elements of x (this iterator will behave like an array of the non-missing elements of x)
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

# ╔═╡ 3d6cf930-5f80-11eb-04a1-0f608e26886b
guesstype(x) = eltype(x)

# ╔═╡ c7192aba-5fe8-11eb-1d50-81cb0f959b4a
md"""
!!! info "Broadcasting"
    It is very common to want to apply a function to each element of an array. We call this broadcasting. To broadcast a function, put a `.` between the function name and `(`. Thus, `occursin.(r"(^\.$)|(^(-|)\d+)$",X)`  produces the same result as
    ```julia
    out = Array{Bool, 1}(undef, length(X))
    for i in 1:length(x)
      out[i] = occursin(r"(^\.$)|(^(-|)\d+)$",X[i])
    end    
    ```

"""

# ╔═╡ 600c6368-5f80-11eb-24b1-c35a333d7164
md"""

!!! note "Multiple Dispatch"
    An important Julia feature for organizing code is multiple dispatch. Multiple dispatch refers to having multiple definitions of functions with the same name, and which version of the function gets used is determined by the types of the function arguments. In the second code block above, we defined a generic `guesstype(x)` for any type of argument `x`. In the first code block, we have a more specialized `guesstype(x)` function for `x` that are `AbstractArray` with element type either `String`, `Missing` or `Union{String, Missing}`. When we call `guesstype(whatever)` below, the most specific version of `guesstype` that fits the type of `whatever` will get used.
"""

# ╔═╡ bcaf264a-5f77-11eb-2bf5-1bd3c16dbce6
guesstype(["1", "3.3", "5"]) # this will call the first definition of guesstype since the argument is an Array of String

# ╔═╡ 46da23f6-5f82-11eb-2c42-dbcf1c09192e
guesstype([12.4, -0.8]) # this will call the second definition of guesstype since the argument is an Array of Float64

# ╔═╡ 65d2d0e8-5f85-11eb-2e4b-b3e561a1a63c
md"""
Again using multiple dispatch, we can create a function to convert the types of the columns of the `dialysis` DataFrame.
"""

# ╔═╡ 81b220c8-5f82-11eb-141a-53ed12752330
converttype(x) = x

# ╔═╡ 1f577cac-5feb-11eb-19c7-2ff4856aee9d
md"""
!!! info "Adding Methods to Existing Functions"
    `Base.parse` is a function included in Julia for converting strings to numeric types.
    We want to use parse to convert the types of our DataFrame columns.
    However, for some columns, we want to leave strings as strings, and for others we want to convert strings to dates.
    The builtin parse function only converts strings to numbers.
    However, we can define additional parse methods and use multiple dispatch to handle these cases. 	

    This approach will make the `converttype` function defined below very short and simple.
"""

# ╔═╡ 3f871682-5f86-11eb-2c50-971aa2d55aec
begin
	# we need a parse that "converts" a string to string
	Base.parse(::Type{String}, x::String) = x
	
	# a version of parse that works for the date formats in this data
	Base.parse(::Type{Dates.Date}, x::String) = occursin(r"\D{3}",x) ? Date(x, "dduuuyyyyy") : Date(x,"m/d/y")
end

# ╔═╡ 34fa745c-5fec-11eb-3c3c-67eba7bffa6e
md"""
!!! info "Ternary Operator"
    In `converttype`, we use the [ternary operator](https://docs.julialang.org/en/v1/base/base/#?:), which is just a concise way to write an if-else statement.
    ```julia
    boolean ? value_if_true : value_if_false
    ```
"""

# ╔═╡ 985c4280-5fec-11eb-362b-21e463e63f8d
md"""
!!! info "Array Comprehension"
    In `converttype`, we use an [array comprehension](https://docs.julialang.org/en/v1/manual/arrays/#man-comprehensions) to create the return value. Comprehensions  are a concise and convenient way to create new arrays from existing ones. 

    [Generator expressions](https://docs.julialang.org/en/v1/manual/arrays/#Generator-Expressions) are a related concept. 
"""

# ╔═╡ 7c756c72-5f83-11eb-28d5-7b5654c51ea3
function converttype(x::AbstractArray{T}) where T <: Union{Missing, String}
	etype = guesstype(x)
	return([(ismissing(val) || val==".") ? missing : parse(etype,val)
		for val in x])
end

# ╔═╡ 8c8cab5a-5f85-11eb-1bb0-e506d437545d
md"""
A quick test.
"""

# ╔═╡ 57324928-5f83-11eb-3e9f-4562c8b03cd4
converttype([".","315", "-35.8"])

# ╔═╡ a3452e58-5f85-11eb-18fb-e5f00173defb
clean1 = let
	clean1 = mapcols(converttype, dialysis) # apply converttype to each column of dialysis
	
	# fix the identifier strings. some years they're listed as ="IDNUMBER", others, they're just IDNUMBER
	clean1.provfs = replace.(clean1.provfs, "="=>"")
	clean1.provfs = replace.(clean1.provfs,"\""=>"")
	clean1
end

# ╔═╡ bff2e0a8-5f86-11eb-24fd-9504f5c47ffb
describe(clean1)

# ╔═╡ f895392e-5f8b-11eb-1d7a-3f9c6c5ce483
md"""
That looks better.
"""

# ╔═╡ 123a49dc-5f8c-11eb-1c59-c5e5715b819f
md"""
### Reshaping

Now, we will deal with the fiscal year/calendar year issues. As mentioned earlier, most variables that vary over time have their values reported for four previous years in each of the fiscal year data files. Thus, for these variables we will have four reports of what should be the same value. The values may not be the same if there are any data entry errors or similar problems. Let's begin by checking for this.
"""

# ╔═╡ 0b1b8522-5f90-11eb-2f9e-91707f735fe6
let
	# the Statistics.var function will give errors with Strings
	numcols = (Symbol(c) for c in names(clean1) if eltype(clean1[!,c]) <: Union{Missing, Number})
	# replace NaN with missing
	function variance(x)
		v=var(skipmissing(x))
		return(isnan(v) ? missing : v)
	end

	# display summary states of withint provfs and year variances
	describe(
		# compute variance by provfs and year
		combine(groupby(clean1, [:provfs, :year]),
			(numcols) .=> variance)
		)
end

# ╔═╡ 2beddacc-5f93-11eb-35a0-cfee0361b2eb
md"""
The median variance is generally 0---most providers report variables consistently across years. However, there are large outliers. As a simple, but perhaps not best solution, we will use the median across fiscal years of each variable.
"""

# ╔═╡ 468e9380-5f96-11eb-1e57-9bf6b185cbd1
clean2=let
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
	clean2 = combine(groupby(clean1, [:provfs,:year]),
		names(clean1) .=> combinefiscalyears .=> names(clean1))
	sort!(clean2, [:provfs, :year])
end

# ╔═╡ 2632c508-5f9c-11eb-149b-edb3f5aee983
md"""

### Defining Needed Variables

Now, let's create the variables we will need.

#### Labor

The labor related variables all end in `FT` (for full-time) or `PT` (for part-time). Create labor as a weighted sum of these variables.

"""

# ╔═╡ 62f7ee18-5f9d-11eb-1b6c-4dabc3f9d787
filter(x->occursin.(r"(F|P)T$",x.first), datadic) # list variables ending with PT or FT

# ╔═╡ 656f7c7e-5f9d-11eb-041d-a903e70f6843
md"""

!!! question
    Modify the code below.
"""

# ╔═╡ c7c1fdee-5f9c-11eb-00bb-bd871c7f7d92
clean2.labor = clean2[!,:nursePT]*0.5 + clean2[!,:nurseFT]*1.0 # + more -- you should modify this

# ╔═╡ ca02b9ee-5f9d-11eb-14f2-b54ef6111837
md"""
#### Hiring

We should create hiring for the control function. There is `panellag` function in `Dialysis.jl` to help doing so.

!!! question
    Should hiring at time $t$ be $labor_{t} - labor_{t-1}$ or $labor_{t+1}-labor_{t}$? In other words, should it be a backward or forward difference? Check in the paper if needed and modify the code below accordingly.
"""

# ╔═╡ 8629935e-5f9e-11eb-0073-7b28899deac5
clean2.hiring = clean2.labor - panellag(:labor,clean2, :provfs, :year, 1)
                # or
                #panellag(:labor,clean2, :provfs, :year, -1) - clean2.labor

# ╔═╡ 2729eb0a-5fa2-11eb-2176-4fcbb5cb1c44
md"""
#### Output

There are a few possible output measures in the data.

CMS observes mortality for most, but possibly not all, dialysis patients. To compute mortality rates, the data reports the number of treated patient-years whose mortality will be observed in column `dy`.

CMS only observes hospitalization for patients whose hospitalization is insured by Medicare. This is a smaller set of patients than those for whom mortality is observed. This number of patient years is in column `hdy`. (If I recall correctly, the output reported by @grieco2017 has summary statistics close to `hdy`).

The column `phd` report patient-months of hemodialysis. My understanding is that this number/12 should be close to `dy`. Since there are other forms of dialysis, `dy` might be larger. On the other hand if there are hemodialysis patients whose mortality is not known, then `phd` could be larger.

There might also be other reasonable variables to measure output that I have missed.


!!! question
    Choose one (or more) measure of output to use in the tables and figures below and any subsequent analysis.
"""

# ╔═╡ d6305b3a-5f9e-11eb-163d-736097666c33
md"""
#### For-profit and chain indicators

!!! question
    Create a boolean forprofit indicator from the `owner_f` column, and Fresenius and Davita chain indicators from `chainnam`
"""

# ╔═╡ 5f82fc80-5f9f-11eb-0670-f57b2e1c02fc
unique(clean2.owner_f)

# ╔═╡ 51012006-5f9f-11eb-1c62-3595a0dbd003
clean2.forprotit = (clean2.owner_f .== "For Profit") # modify if needed

# ╔═╡ 9180dc5c-5f9f-11eb-1d51-cb0516deb7b5
countmap(clean2.chainnam)

# ╔═╡ 7b8cb088-5f9f-11eb-3ec1-056ae31d5400
begin
	f(x) = ismissing(x) ? false : occursin(r"(FRESENIUS|FMC)",x) # improve if needed
	clean2.fresenius = f.(clean2.chainnam)
	# do something similar for davita and any other chains you think are important
end

# ╔═╡ 0b4f51ca-5fa1-11eb-1466-4959a7e056ae
md"""

#### State Inspection Rates

State inspection rates are a bit more complicated to create.
"""

# ╔═╡ 5c8d4f8e-5ff3-11eb-0c55-d1a3795358e3
clean3 = let 
	# compute days since most recent inspection
	inspect = combine(groupby(clean1, :provfs), 
		:surveydt_f => x->[unique(skipmissing(x))])
	rename!(inspect, [:provfs, :inspection_dates])
	df=innerjoin(clean2, inspect, on=:provfs)
	@assert nrow(df)==nrow(clean2)
	function dayssince(year, dates) 
		today = Date(year, 12, 31)
		past = [x.value for x in today .- dates if x.value>=0]
		if length(past)==0
			return(missing)
		else
			return(minimum(past))
		end
	end
	
	df=transform(df, [:year, :inspection_dates] => (y,d)->dayssince.(y,d))	
	rename!(df, names(df)[end] =>:days_since_inspection)
	df[!,:inspected_this_year] = ((df[!,:days_since_inspection].>=0) .&
		(df[!,:days_since_inspection].<365))
	
	# then take the mean by state
	stateRates = combine(groupby(df, [:state, :year]),
                	:inspected_this_year => 
			(x->mean(skipmissing(x))) => :state_inspection_rate)
	df = innerjoin(df, stateRates, on=[:state, :year])
	@assert nrow(df)==nrow(clean2)
	df
end

# ╔═╡ 1d6b90b2-5fa1-11eb-0b52-b36c3642539a
md"""
#### Competitors

Creating the number of competitors in the same city is somewhat
similar. Note that @grieco2017 use the number of competitors in the
same HSA, which would be preferrable. However, this dataset does not
contain information on HSAs. If you are feeling ambitious, you could
try to find data linking city, state to HSA, and use that to calculate
competitors in the same HSA.

"""

# ╔═╡ 00c8ef48-5ff8-11eb-1cf3-f7d391228226
clean4=let 
	df = clean3
	upcase(x) = Base.uppercase(x)
	upcase(m::Missing) = missing
	df[!,:provcity] = upcase.(df[!,:provcity])
	comps = combine(groupby(df,[:provcity,:year]),
    	       		:dy => 
			(x -> length(skipmissing(x).>=0.0)) => 
			:competitors
           )
	comps = comps[.!ismissing.(comps.provcity),:]
 	df = outerjoin(df, comps, on = [:provcity,:year], matchmissing=:equal)	
	@assert nrow(df)==nrow(clean3)
	df
end

# ╔═╡ 28b76e70-5fa1-11eb-31de-d12718c8de03
md"""

### Summary Statistics

!!! question
    Create a table (or multiple tables) similar to Tables 1-3 of @grieco2017.
    Comment on any notable differences. The following code
    will help you get started.
"""

# ╔═╡ a4ae6fb8-5fa1-11eb-1113-a565d047be6d
let
	# at the very least, you will need to change this list
	vars = [:phd, :labor, :hiring]

	# You shouldn't neeed to change this function, but you can if you want
	function summaryTable(df, vars;
    	                  funcs=[mean, std, x->length(collect(x))],
	                      colnames=[:Variable, :Mean, :StdDev, :N])
  		# In case you want to search for information about the syntax used here,
	  	# [XXX for XXX] is called a comprehension
  		# The ... is called the splat operator
  		DataFrame([vars [[f(skipmissing(df[!,v])) for v in vars] for f in funcs]...], colnames)
	end
	summaryTable(clean4, vars)
end

# ╔═╡ 7bfe6fee-5fa3-11eb-3f31-59a77a78f035
md"""

### Figures

!!! question
    Create some figures to explore the data. Try to
    be creative.  Are there any strange patterns or other obvious
    problems with the data?

Here are some examples to get started. You may want to look at the
StatPlots.jl, Plots.jl, or VegaLite.jl github pages for more examples.
"""

# ╔═╡ aaf4b772-5fa3-11eb-298f-87459c41c4f4
begin
    vars = [:dy, :hdy, :phd]
    inc = completecases(clean2[!,vars]) # missings will mess up corrplot
    @df clean3[inc,vars] corrplot(cols(vars))
end

# ╔═╡ eca590a6-5fa3-11eb-383c-095e63136428
function yearPlot(var; df=clean4)
  data = df[completecases(df[!,[:year, var]]),:]
  scatter(data[!,:year], data[!,var], alpha=0.1, legend=:none,
          markersize=3, markerstrokewidth=0.0)
	q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
  yearmeans = combine(groupby(data, :year),
               var => (x->[(mean(skipmissing(x)),
						  quantile(skipmissing(x),q)...)])  => 
		["mean", (x->"q$(Int(100*x))").(q)...])
	sort!(yearmeans,:year)
  @df yearmeans plot!(:year, :mean, colour = ^(:black), linewidth=4)
  fig = @df yearmeans plot!(:year, cols(3:ncol(yearmeans)),
                      colour = ^(:red), alpha=0.4, legend=:none,
                      xlabel="year", ylabel=String(var))
  return(fig)
end

# ╔═╡ 12a3d1a0-5fa4-11eb-11a3-297c335010c7
let 
	fig=yearPlot(:labor)
	plot!(fig, ylim=[0,50]) # adjust y-axis range
end

# ╔═╡ 1eaaaf1e-5fa4-11eb-1338-49f1dd9aa2dc
md"""
The above plot shows a scatter of labor vs year. The black lines are
average labor each year. The red lines are the 0.01, 0.1, 0.25, 0.5,
0.75, 0.9, and 0.99 quantiles conditional on year.
"""

# ╔═╡ c09dd940-5ff9-11eb-0db4-bf9f169c5508
md"""

!!! question
    Please hand in both your modified `dialysis-1.jl` and an html export of it. Use the triangle and circle icon at the top of the page to export it.
"""

# ╔═╡ Cell order:
# ╟─d5554696-5f6f-11eb-057f-a79641cf483a
# ╟─ad1cc4c6-5f72-11eb-35d5-53ff88f1f041
# ╟─7b4ecdee-5f73-11eb-388c-4d6f9719d79b
# ╠═af8e3f0e-5f73-11eb-1de8-5994ab9a8612
# ╟─a75918ae-5f73-11eb-3a3e-2f64c0dcc49c
# ╠═a80227f8-5f77-11eb-1211-95dd2c151877
# ╠═b9e1f8de-5f77-11eb-25b8-57e263315ac3
# ╟─a06995aa-5f78-11eb-3939-f9aca087b12c
# ╠═5b2d6ebe-5f78-11eb-0528-ab36ae696a35
# ╟─5ea12c9c-5f79-11eb-34e7-2d7f07854b31
# ╟─95ad1f3c-5f79-11eb-36fb-1b384c84317c
# ╠═ca038b54-5f79-11eb-0851-9f684e3bb83f
# ╟─cfaabda0-5f79-11eb-17e4-a7cd045681da
# ╠═c642b578-5f77-11eb-1346-15a35500e61f
# ╠═3d6cf930-5f80-11eb-04a1-0f608e26886b
# ╟─c7192aba-5fe8-11eb-1d50-81cb0f959b4a
# ╟─600c6368-5f80-11eb-24b1-c35a333d7164
# ╠═bcaf264a-5f77-11eb-2bf5-1bd3c16dbce6
# ╠═46da23f6-5f82-11eb-2c42-dbcf1c09192e
# ╟─65d2d0e8-5f85-11eb-2e4b-b3e561a1a63c
# ╠═81b220c8-5f82-11eb-141a-53ed12752330
# ╟─1f577cac-5feb-11eb-19c7-2ff4856aee9d
# ╠═3f871682-5f86-11eb-2c50-971aa2d55aec
# ╟─34fa745c-5fec-11eb-3c3c-67eba7bffa6e
# ╟─985c4280-5fec-11eb-362b-21e463e63f8d
# ╠═7c756c72-5f83-11eb-28d5-7b5654c51ea3
# ╟─8c8cab5a-5f85-11eb-1bb0-e506d437545d
# ╠═57324928-5f83-11eb-3e9f-4562c8b03cd4
# ╠═a3452e58-5f85-11eb-18fb-e5f00173defb
# ╠═bff2e0a8-5f86-11eb-24fd-9504f5c47ffb
# ╟─f895392e-5f8b-11eb-1d7a-3f9c6c5ce483
# ╟─123a49dc-5f8c-11eb-1c59-c5e5715b819f
# ╠═2dda9576-5f90-11eb-29eb-91dec5be5175
# ╠═0b1b8522-5f90-11eb-2f9e-91707f735fe6
# ╟─2beddacc-5f93-11eb-35a0-cfee0361b2eb
# ╠═468e9380-5f96-11eb-1e57-9bf6b185cbd1
# ╟─2632c508-5f9c-11eb-149b-edb3f5aee983
# ╠═62f7ee18-5f9d-11eb-1b6c-4dabc3f9d787
# ╟─656f7c7e-5f9d-11eb-041d-a903e70f6843
# ╠═c7c1fdee-5f9c-11eb-00bb-bd871c7f7d92
# ╟─ca02b9ee-5f9d-11eb-14f2-b54ef6111837
# ╠═8629935e-5f9e-11eb-0073-7b28899deac5
# ╟─2729eb0a-5fa2-11eb-2176-4fcbb5cb1c44
# ╟─d6305b3a-5f9e-11eb-163d-736097666c33
# ╠═5f82fc80-5f9f-11eb-0670-f57b2e1c02fc
# ╠═51012006-5f9f-11eb-1c62-3595a0dbd003
# ╠═9180dc5c-5f9f-11eb-1d51-cb0516deb7b5
# ╠═7b8cb088-5f9f-11eb-3ec1-056ae31d5400
# ╟─0b4f51ca-5fa1-11eb-1466-4959a7e056ae
# ╠═5c8d4f8e-5ff3-11eb-0c55-d1a3795358e3
# ╟─1d6b90b2-5fa1-11eb-0b52-b36c3642539a
# ╠═00c8ef48-5ff8-11eb-1cf3-f7d391228226
# ╟─28b76e70-5fa1-11eb-31de-d12718c8de03
# ╠═a4ae6fb8-5fa1-11eb-1113-a565d047be6d
# ╟─7bfe6fee-5fa3-11eb-3f31-59a77a78f035
# ╠═b443a46e-5fa3-11eb-3e71-dfd0683dc6e9
# ╠═aaf4b772-5fa3-11eb-298f-87459c41c4f4
# ╠═eca590a6-5fa3-11eb-383c-095e63136428
# ╠═12a3d1a0-5fa4-11eb-11a3-297c335010c7
# ╟─1eaaaf1e-5fa4-11eb-1338-49f1dd9aa2dc
# ╟─c09dd940-5ff9-11eb-0db4-bf9f169c5508
