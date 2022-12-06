using CSV
using DataFrames
using Pkg
##
using Distributions, Plots, MLJ, MLJLinearModels, Random, OpenML
##
Pkg.activate(temp = true)
Pkg.develop(url = "https://github.com/jbrea/MLCourse")
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

##
df = DataFrame(CSV.File("/Users/sabo4ever/Sabrina/EPFL/machine learning/Project/train.csv"))

##
dropmissing(df)
predictors=select(df,Not(:labels))
idx=std.(eachcol(predictors)) .!= 0
df_clean_const = predictors[:, idx]                  #drop zero std predictors

findall(≈(1), cor(Matrix(df_clean_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)