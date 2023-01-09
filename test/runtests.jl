using Dialysis
using Test, DataFrames, Distributions, Statistics, LinearAlgebra, FixedEffectModels


# test panel lag
@testset "panellag" begin
  N = 100
  T = 10
  id = vcat([fill(i,T) for i in 1:N]...)
  t  = repeat(1:T, N)
  x = rand(N*T)
  df = DataFrame(id=id, t=t, x=x)
  for lags in -2:2
    xlag = panellag(:x, df, :id, :t, lags)
    @test length(xlag)==N*T
    @test sum(ismissing.(xlag))==N*abs(lags)
  end
end

@testset "locallinear" begin
  N = 1000
  for d in 1:4
    x = randn(N,d)
    fx = pdf(MvNormal(d,1.0), x')
    y = randn(N,1)
    yhat = locallinear(x,x,y)
    # checking that satisfies theoretical error bounds
    @test sum(abs.(yhat).> fx.^(-0.5)*N^(-2/(d+4)))/N <= 0.05
  end
end

@testset "polyreg" begin
  N = 1000
  d = 3
  x = randn(N,d)
  normx = norm.(eachrow(x))
  #normx = 1.0
  y = randn(N,1)
  # checking that satisfies theoretical error bounds
  for deg in 1:4
    yhat = polyreg(x,x,y, degree=deg)
    terms = binomial(d + deg, deg)
    obj = sum(abs.(yhat) .> 1.96*sqrt(terms/N)*normx)/N
    @test sum(abs.(yhat) .> 1.96*sqrt(terms/N)*normx)/N .< 0.05
    #println("n=$N, d=$d, deg=$deg, terms=$terms, $(mean(yhat)), $(std(yhat)), $obj")
  end

  # compare with reg
  x = randn(N)
  y = randn(N)
  df = DataFrame(y=y, x=x, x2=x.^2, x3=x.^3, x4=x.^4)
  foo=reg(df,@formula(y ~ x + x2 + x3 + x4), save=true)
  yhatreg = y - foo.residuals
  yhat = polyreg(reshape(x,N,1), reshape(x,N,1), reshape(y,N,1), degree=4)
  @test yhat ≈ yhatreg
end

@testset "partiallinear interface" begin
  N = 20
  T = 3
  id = (1:N)*ones(Int,T)'
  t = ones(Int,N)*(1:T)'
  df = DataFrame(id = vec(id), t=vec(t), y = randn(N*T),
                 l = randn(N*T), k=randn(N*T),
                 invest = Vector{Union{Float64, Missing}}(undef,N*T))
  df.invest[:] .= randn(N*T)
  df.invest[df.t.==1] .= missing
  sort!(df, [:id, :t])
  preg = (xp, xd, yd)->Dialysis.polyreg(xp, xd, yd, degree = 2)
  step1, eyex = Dialysis.partiallinear(:y, [:l], [:invest, :k], df,
                                       npregress=preg,clustervar=:id)

  @test eyex isa Array
end
