"""
    downloadDFR(;redownload=false)

Downloads Dialysis Facility Reports. Saves zipfiles in `Dialysis/data/`.
"""
function downloadDFR(;redownload=false)
  # urls obtained by searching data.cms.gov for dialysis facility reports
  urls = Dict(
    2008=>"https://data.cms.gov/download/tmj4-uh2k/application%2Fzip",
    2009=>"https://data.cms.gov/download/df3w-m5va/application%2Fzip",
    2010=>"https://data.cms.gov/download/pu2m-x27s/application%2Fzip",
    2011=>"https://data.cms.gov/download/tmty-dzj6/application%2Fzip",
    2012=>"https://data.cms.gov/download/gp9x-yv2a/application%2Fzip",
    2013=>"https://data.cms.gov/download/rhn4-qw2b/application%2Fzip",
    2014=>"https://data.cms.gov/download/uz5q-z58c/application%2Fzip",
    2015=>"https://data.cms.gov/download/8hbp-qbht/application%2Fzip",
    2016=>"https://data.cms.gov/download/q4sr-4779/application%2Fzip",
    2017=>"https://data.cms.gov/download/ub2a-kzvr/application%2Fzip",
    2018=>"https://data.cms.gov/download/33dw-bemn/application%2Fzip",
    2019=>"https://data.cms.gov/download/7ha2-yz4a/application%2Fzip",
    2020=>"https://data.cms.gov/download/4y3p-fyx9/application%2Fzip",
    2021=>"https://data.cms.gov/download/wy3d-7buy/application%2Fzip"
  )
  datadir = normpath(joinpath(dirname(Base.find_package("Dialysis")),"..","data"))
  isdir(datadir) || mkdir(datadir)
  for (y, url) in urls
    zfile = joinpath(datadir,"FY$y.zip")
    if !isfile(zfile) || redownload
       download(url,zfile)
    end
  end
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
  "owner_f" => "cw_owner_f")

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
    csvinzip = filter(x->occursin("dfr.csv",x.name), z.files)
    length(csvinzip)==1 || error("Multiple csv files found in $file")
    println("Reading $csvinzip[1]")
    df = CSV.File(read(csvinzip[1])) |> DataFrame
    close(z)
    return(df, merge(singlevars, yvars))
  end

  downloadDFR()
  files = readdir(datadir,join=true)
  files = files[occursin.(r"FY\d+\.zip",files )]

  ally = DataFrame()
  alls = DataFrame()
  for file in files
    year = parse(Int64,match(r"FY(\d+)\.zip",file).captures[1])
    z = ZipFile.Reader(file)
    csvinzip = filter(x->occursin("$year.csv",x.name), z.files)
    length(csvinzip)==1 || error("Multiple csv files found in $file")
    println("Reading $csvinzip[1]")
    ydf = CSV.File(read(csvinzip[1])) |> DataFrame
    close(z)
    ydf[!,:fiscal_year] .= year

    oldv = collect(keys(singlevars))
    v = copy(oldv)
    for i in eachindex(v)
      if !(v[i] in names(ydf))
        newv = replace(v[i], "_f" => "_n_f")
        if !(newv in names(ydf))
          if ((v[i] in keys(altvarnames)) &&
              (lowercase(altvarnames[v[i]]) in lowercase.(names(ydf))))
            newv = altvarnames[v[i]]
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
    end
    tmp = ydf[!,Symbol.(v)]

    for (o, n) in zip(oldv, v)
      if n in names(tmp)
        rename!(tmp, n=>o)
      end
    end
    append!(alls, tmp, promote=true)
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



module API

function getvariable(variable)

  uuids =Dict( 2021 => "e66eaf03-398f-4a8c-9d1b-6aaf25fb6425",
               2020 => "44aedde0-c535-497d-ab27-b415ec12ff5a",
               2019 => "ffb183e0-93ef-4071-b818-eea2e81d6763",
               2018 => "75deb6cc-b53d-4db9-bc23-db5e617b0e14",
               2017 => "1f33b642-3ec0-4477-9808-3d2be0b1f540",
               2016 => "a01f9359-f831-4b85-9df7-791d6d0a58d4",
               2015 => "5348c5a0-fec4-43a3-a571-5a55e5be8617",
               2014 => "eb5d36cd-6898-476a-91bb-06042a95a50f",
               2013 => "10fdee58-4d52-458b-b910-a9ddcc86d8e2",
               2012 => "35e24742-51e5-4a8c-8581-14dc2c5e7c87",
               2011 => "a874d015-be12-417c-9497-c062ec775b1e",
               2010 => "3bd7488a-ef51-445b-8458-a6656ff4e9ce",
               2009 => "a47fd59f-2d80-4b6c-a272-d9b82054683b" )



end

end

# https://data.cms.gov/provider-data/archived-data/dialysis-facilities
