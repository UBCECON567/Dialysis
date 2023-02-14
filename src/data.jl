"""
    loaddata_old()

Loads "dialysisFacilityReports.rda". Returns a DataFrame.
"""
function loaddata_old()
  rdafile=normpath(joinpath(dirname(Base.pathof(Dialysis)),"..","data","dialysisFacilityReports.rda"))
  dt = RData.load(rdafile,convert=true)
  dt["dialysis"]
end

function loaddata()
  loadDFR()
end


"""
    downloadDFR(;redownload=false)

Downloads Dialysis Facility Reports. Saves zipfiles in `Dialysis/data/`.
"""
function downloadDFR(;redownload=false)
  # urls obtained by searching data.cms.gov for dialysis facility reports
  datadir = normpath(joinpath(dirname(Base.find_package("Dialysis")),"..","data"))
  println("Goto https://data.cms.gov/quality-of-care/medicare-dialysis-facilities to download data.")
  println("Save the data in $datadir")
  nothing
end

singlevars = Dict(
  # Identifiers
  "fiscal_year" => "",
  "provfs" => "CMS provider identifier",
  "provcity" => "city",
  "provname" => "provider name",
  "state" => "",
  "network" => "facility network number",
  "chainnam" => "chain name",
  "modal_f" => "Modality",
  "owner_f" => "Type of organizational control",

  # stations
  "totstas_f" => "Stations in fiscal year-1",

  # inspection survey
  "surveydt_f" => "last survey date",
  "surveyact_f" => "survey action",
  "surveycc_f" => "compliance condition after last survey",
  "surveycfc_f" => "condition for coverage definiciencies in last survey",
  "surveystd_f" => "standard deficiencies in last survey")

altvarnames = Dict(
  "surveydt_f" => "survey_dt",
  "surveyact_f" => "srvy_prpse_cd",
  "surveycc_f" => "compl_cond",
  "surveycfc_f" => "cfc_f",
  "surveystd_f" => "std_f",
  "provname" => "DFR_provname",
  "owner_f" => "cw_owner_f" #ownership_type
)

yvars = Dict(
  # "${key}y4_f" is for fiscal_year - 2
  # "${key}y3_f" is for fiscal_year - 3
  # "${key}y2_f" is for fiscal_year - 4
  # "${key}y1_f" is for fiscal_year - 5

  # Staff (as of end of year)
  "staff" => "total staff",
  "dietFT" => "renal dieticians full time",
  "dietPT" => "renal dieticians part time",
  "nurseFT" => "nurses full time",
  "nursePT" => "nurses part time",
  "ptcareFT" => "patient care technicians full time",
  "ptcarePT" => "patient care technicians part time",
  "socwkFT" => "social workers full time",
  "socwkPT" => "social workers part time",

  # mortality
  "dy" => "patient-years at risk of mortality",
  "dea" => "patient deaths",
  "exd" => "expected deaths",
  "inf" => "% deaths from infection",
  "smr" => "standardized mortality ratio",

  # hosptilization
  "rdsh" => "number of patients with hospitalization info",
  "hdy" => "years at risk of hospitalization days",
  "hty" => "years at risk of hospital admission",
  "hta" => "number of hospital admissions",
  "shrd" => "standardized hospitalization ratio (days)",
  "shrt" => "standardized hospitalization ratio (admissions)",
  "sepi" => "% hospitalizations for septicemia",
  "srr" => "standardized readmission ratio",

  # lab work
  "hctmean" => "average hemocrit",
  "hct33" => "% patients with Hemocrit>=33 (good)",
  "urr65" => "% patients with Urea reduction ratio>=65 (good)",

  # patient counts
  "phd" => "monthly prevalent hemodialysis patient (in-center & home)",
  "ihd" => "monthly average new patients",

  # access type
  "ppavf" => "% receiving treatment with fistula",
  "ppavg" => "% receiving treatment with graft",
  "ppcath" => "% receiving treatment with catheter",
  "ppfist" => "% with fistula placed",
  "ppcg90" => "% with only catheter for more than 90 days",
  "pifist" => "% new patients with fistula placed",
  "pd2inf100mo" => "PD Catheter infection rate per 100 PD patient-months (note: we want the HD infection rate, not this one",

  # patient characteristics (all for set of patients as of last day of year)
  "pah" => "Number of patients at end of year",
  "ncm" => "Medicare patient at end of year",
  "age" => "Average patient age",
  "age1" => "% patients < 20",
  "age2" => "% patients 20-64",
  "age3" => "% patients >=65",
  "sex" => "% female",
  "rac1" => "% Asian/Pacific Islander",
  "rac2" => "% African American",
  "rac3" => "% Native American",
  "rac4" => "% White",
  "eth1" => "% Hispanic",
  "eth2" => "% Non-Hispanic",
  "dis1" => "% diabetes",
  "dis2" => "% hypertension",
  "dis3" => "% glomerulonephritis",
  "dis4" => "% other/unknown cause",
  "vin" => "Avg years of prior ESRD therapy",
  "nrshome" => "Number of nursing facility patients",
  "modcapd" => "% on CAPD",
  "modccpd" => "% on CPPD",
  "modhd" => "% on HD",
  "modhhd" => "% on home HD",
  "modshd" => "% on in-center self HD",

  # comorbidites (among medicare patients)
  "clmalcom" => "% alcohol dependence",
  "clmanem" => "% anemia",
  "clmcam" => "% cardiac arrest",
  "clmcanm" => "% cancer",
  "clmcdm" => "% Cardiac Dysrythmias",
  "clmchfm" => "% congestive heart failure",
  "clmcopdm" => "% Chronic Obstructive Pulmonary Disease",
  "clmcvdm" => "% Cerebrovascular Disease",
  "clmdiabm" => "% Diabete Type I",
  "clmdrugm" => "% drug dependence",
  "clmgtbm" => "% gastro-instentinal bleeding",
  "clmhepbm" => "% hepatitis B",
  "clmhepothm" => "% hepatitis other",
  "clmhivaidm" => "% AIDS",
  "clmhypthym" => "% Hyperparathyroidism",
  "clminfm" => "% infection comorbidity",
  "clmihdm" => "% ischemic heart disease",
  "clmmim" => "% myocardial infarction",
  "clmpvdm" => "% Peripheral Vascular Disease",
  "clmpnem" => "% pneumonia",
  "clmcntcom" => "average number of comorbidities",

  # characteristics of new patients
  "agem" => "average age of new patients",
  "femm" => "% female among new patients",
  "asianm" => "% Asian among new patients",
  "blackm" => "% Black among new patients",
  "whitem" => "% White among new patients",
  "ethm" => "% Hispanic among new patients",
  "dbprim" => "% diabetes primary cause among new patients",
  "gnprim" => "% GN primary cause among new patients",
  "htprim" => "% hypertension primary cause among new patients",
  "insempy" => "% employer insured among new patients",
  "insmdcdm" => "% medicaid only among new new patients",
  "insmdcrcdm" => "% medicaid & medicare among new patients",
  "insmdcrm" => "% medicare only among new patients",
  "insmdcrom" => "% medicare & other among new patients",
  "insnonem" => "% no insurance among new patients",
  "bmifm" => "median BMI among female new patients",
  "bmimm" => "median BMI among male new patients",
  "cempm" => "% employed or student among new patients",
  "pempm" => "% previously employed or student among new patients",
  "mefavfm" => "% fistula among new patietns",
  "mefcathm" => "% catheter among new patients",
  "mefgraftm" => "% av graft among new patients",
  "hemom" => "number of new hemodialysis patients",
  "hgm" => "average hemoglobin among new patients",
  "salbm" => "average serum albumin among new patients",
  "cream" => "average creatine among new patients",
  "alcom" => "% alcoholic among new patients",
  "ambum" => "% unable to ambulate among new patients",
  "ashdm" => "% atherosclerotic heart disease among new patients",
  "canm" => "% cancer among new patients",
  "chfm" => "% CHF among new patients",
  "copdm" => "% COPD among new patients",
  "cvam" => "% CVD, CVA, TIA among new patients",
  "diabm" => "% diabetes among new patients",
  "drugm" => "% drug dependent among new patients",
  "hxhtm" => "% hypertension history among new patients",
  "othcarm" => "% other cardiac disorder among new patients",
  "pvdm" => "% PVD among new patients",
  "smokm" => "% smoker among new patients",
  "cntcom" => "average number of comorbidities among new patients",
  "nephnom" => "new patients, no prior ESRD care",
  "nephunkmissm" => "new patients, unknown prior care",
  "nephy12m" => "new patients, >12m prior Nephrologist care",
  "nephy612m" => "new patients, 6-12m prior Nephrologist care",
  "nephy6m" => "new patients, <6m prior Nephrologist care",

  # Anemia management
  "CWhdavgHGB" => "average hemoglobin levels (g/dL) of hemodialysis patients",
  "CWpdavgHGB" => "average hemoglobin levels (g/dL) of peritoneal dialysis patients",
  "CWpdesarx" => "% PD patients prescribed ESA",
  "CWhdesarx" => "% HD patients prescribed ESA",
  "strr" => "Standardized transfusion rate",
  "tf" => "Number of transfusions",
  "tfy" => "Patient years at risk of transfusion"

  # Inspections

)


"""
    loadDFR(;recreate=false)

If Dialysis/data/dfr.zip exists, load it from disk. Otherwise,
create Dialysis/data/dfr.zip exists by loading Dialysis Facility
Reports from zipfiles in `Dialysis/data/`.
"""
function loadDFR(;recreate=false)
  datadir = normpath(joinpath(dirname(Base.find_package("Dialysis")),"..","data"))
  dfrfile = joinpath(datadir, "dfr.zip")
  if isfile(dfrfile) && !recreate
    z = ZipFile.Reader(dfrfile)
    csvinzip = filter(x->occursin(".csv",x.name), z.files)
    @show csvinzip
    length(csvinzip)==1 || error("Multiple csv files found in $file")
    println("Reading $csvinzip[1]")
    df = CSV.File(read(csvinzip[1])) |> DataFrame
    close(z)
    return(df, merge(singlevars, yvars))
  end

  downloadDFR()

  files = readdir(datadir,join=true)
  files = files[occursin.(r"\d\d\d\d.zip",files )]
  @show files
  years = [parse(Int64,match(r"(\d\d\d\d)",file).captures[1]) for file in files]
  for y ∈ minimum(years):maximum(years)
    if !(y ∈ years)
      @warn "Data for $y not found"
    end
  end

  ally = DataFrame()
  alls = DataFrame()
  for file in files
    year = parse(Int64,match(r"(\d\d\d\d)",file).captures[1])
    @show file
    z = ZipFile.Reader(file)
    csvinzip = filter(x->occursin(Regex("$year.+csv\$"),x.name), z.files)
    length(csvinzip)==1 || error("Multiple csv files found in $file")
    println("Reading $(csvinzip[1].name)")
    ydf = CSV.File(read(csvinzip[1])) |> DataFrame
    close(z)
    ydf[!,:fiscal_year] .= year
    if (size(ydf,2)>100)
      tmp = singlevars_old(ydf)
      append!(alls, tmp, promote=true)
      append!(ally, yearvars_old(ydf), cols=:union, promote=true)
    else
      tmp = singlevars_new(ydf)
      append!(alls, tmp, promote=true)
      append!(ally, yearvars_new(ydf), cols=:union, promote=true)
    end
  end
  df = outerjoin(alls, ally, on=[:provfs, :fiscal_year])

  let
    z = ZipFile.Writer(dfrfile)
    f = ZipFile.addfile(z, "dfr.csv", method=ZipFile.Deflate)
    df |> CSV.write(f)
    close(z)
  end

  return(df, merge(singlevars, yvars))
end

function yearvars_old(ydf; yvars=yvars)
  year = unique(ydf.fiscal_year)
  @assert length(year)==1
  year = year[1]
  ally=DataFrame()
  for y in 1:4
    println("year = $year, y=$y")
    v = vcat([ names(ydf)[occursin.(Regex("^$(x)y$(y)_f"),
                                    names(ydf))]
               for x in keys(yvars) ]...)
    tmp2 = ydf[!,[:provfs, :fiscal_year, Symbol.(v)...]]
    tmp2[!,:year] .= year - 6 + y
    rename!(x->replace(x, "y$(y)_f" => "") , tmp2)
    append!(ally, tmp2, cols=:union)
  end
  return(ally)
end

"""
parse data from file in old format (2019 or older)
"""
function singlevars_old(ydf; singlevars=singlevars, altvarnames=altvarnames)
  oldv = collect(keys(singlevars))
  v = copy(oldv)
  for i in eachindex(v)
    if !(v[i] in names(ydf))
      @show v[i]
      newv = replace(v[i], "_f" => "_n_f")
      if !(newv in names(ydf))
        if (v[i] in keys(altvarnames))
          av = altvarnames[v[i]]
          if lowercase(av) ∈ lowercase.(names(ydf))
            newv = av
          end
        else
          newv = v[i]
        end
      end
      if !(newv in names(ydf))
        m = findall(lowercase(newv).==lowercase.(names(ydf)))[1]
        newv = names(ydf)[m]
      end
      v[i] = newv
    end
    if !(v[i] ∈ names(ydf))
      @error "Could not find $(v[i]) in data for $year"
    end
  end
  tmp = ydf[!,Symbol.(v)]

  for (o, n) in zip(oldv, v)
    if n in names(tmp)
      rename!(tmp, n=>o)
    end
  end
  return(tmp)
end

"""
parse data from file in new format (2020 or newer)
"""
function yvars_new(df; yvars=yvars)
  @warn "Parsing data from 2020 or newer not implemented"
  return(DataFrame())
end

"""
parse data from file in new format (2020 or newer)
"""
function singlevars_new(df; singlevars=singlevars)
  @warn "Parsing data from 2020 or newer not implemented"
  return(DataFrame())
  oldv = collect(keys(singlevars))
  v = copy(oldv)
  for i in eachindex(v)
    if !(v[i] in names(ydf))
      @show v[i]
      newv = replace(v[i], "_f" => "_n_f")
      if !(newv in names(ydf))
        if (v[i] in keys(altvarnames))
          for av ∈ altvarnames[v[i]]
            if lowercase(av) ∈ lowercase.(names(ydf))
              newv = av
              break
            end
          end
        end
      else
        newv = v[i]
      end
      if !(newv in names(ydf))
        m = findall(lowercase(newv).==lowercase.(names(ydf)))[1]
        newv = names(ydf)[m]
      end
      v[i] = newv
    end
    if !(v[i] ∈ names(ydf))
      @error "Could not find $(v[i]) in data for $year"
    end
  end
  tmp = ydf[!,Symbol.(v)]

  for (o, n) in zip(oldv, v)
    if n in names(tmp)
      rename!(tmp, n=>o)
    end
  end
  return(tmp)
end

################################################################################

module Cleaning

using Dates, DataFrames, StatsBase, Statistics
"""
     guesstype(x)

Try to guess at appropriate type for x.
"""
function guesstype(x::AbstractArray{T}) where T <:Union{S,Missing} where S <: AbstractString
  if all(occursin.(r"(^\.$)|(^(-|)\d+)$",skipmissing(x)))
    return Int
  elseif all(occursin.(r"(^\.$)|(^(-|\d)\d*(\.|)\d*$)",skipmissing(x)))
    return Float64
  elseif all(occursin.(
    r"(^\.$)|(^\d\d\D{3}\d{4}$|^\d\d/\d\d/\d{4}$)",
    skipmissing(x)))
    return Date
  else
    return S
  end
end
guesstype(x) = eltype(x)


parse(t, x) = Base.parse(t, x)
# we need a parse that "converts" a string to string
parse(::Type{S}, x::S) where S <: AbstractString = x
# a version of parse that works for the date formats in this data
parse(::Type{Dates.Date}, x::AbstractString) = occursin(r"\D{3}",x) ? Date(x, "dduuuyyyyy") : Date(x,"m/d/y")

"""
	    converttype(x)

Convert `x` from an array of strings to a more appropriate numeric type
if possible.
"""
function converttype(x::AbstractArray{T}) where T <: Union{Missing, AbstractString}
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

function combinefiscalyears(x::DataFrames.PooledVector)
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

function cleanDFR(dialysis)
  makestring(x) = String(x)
  makestring(x::Int64) = "$x"
  dialysis.provfs = makestring.(dialysis.provfs)
  # fix the identifier strings. some years they're listed as ="IDNUMBER", others, they're just IDNUMBER
  dialysis.provfs = replace.(dialysis.provfs, "="=>"")
  dialysis.provfs = replace.(dialysis.provfs,"\""=>"")
  # convert strings to numeric types
  dialysis=mapcols(Cleaning.converttype, dialysis)

  #gdf = groupby(dialysis,[:provfs,:year])
  #for n ∈ names(dialysis)
  #  combine(gdf, n => Cleaning.combinefiscalyears)
  #end


  dialysis = combine(groupby(dialysis, [:provfs,:year]),
                     names(dialysis) .=> Cleaning.combinefiscalyears .=> names(dialysis))
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

  df=transform(df, [:year, :inspection_dates] => (y,d)->Cleaning.dayssince.(y,d))
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
  df[!,:provcity] = Cleaning.upcase.(df[!,:provcity])
  comps = combine(groupby(df,[:provcity,:year]),
  :dy =>
    (x -> length(skipmissing(x).>=0.0)) =>
    :competitors
  )
  comps = comps[.!ismissing.(comps.provcity),:]
  dialysis = outerjoin(df, comps, on = [:provcity,:year], matchmissing=:equal)
  @assert nrow(dialysis)==nrow(df)

  return(dialysis)
end

end
