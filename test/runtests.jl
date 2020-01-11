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
    @test sum(abs.(yhat).> fx.^(-0.5)*N^(-2/(d+4)))/N <= 0.05
  end  
end

@testset "polyreg" begin
  N = 1000
  d = 3
  x = randn(N,d)
  normx = norm.(eachrow(x))
  y = randn(N,1)
  for deg in 1:4
    yhat = polyreg(x,x,y, degree=deg)
    @test sum(abs.(yhat) .> 1.96*sqrt(deg/N)*normx)/N .< 0.15
    println("n=$N, d=$d, deg=$deg, $(mean(yhat)), $(std(yhat))")
  end

  # compare with reg
  x = randn(N)
  y = randn(N)
  df = DataFrame(y=y, x=x, x2=x.^2, x3=x.^3, x4=x.^4)
  foo=reg(df,@formula(y ~ x + x2 + x3 + x4), save=true)
  yhatreg = y - foo.augmentdf.residuals
  yhat = polyreg(reshape(x,N,1), reshape(x,N,1), reshape(y,N,1), degree=4)
  @test yhat ≈ yhatreg
end

#@testset "partiallinear" begin

#end
