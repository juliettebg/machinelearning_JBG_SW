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

indices=findall(≈(1), cor(Matrix(df_clean_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)
indices=DataFrame(indices)
indices_ = union([indices[!,1][i][1] for i in 1:length(indices[!,1])])
col_no=df_clean_const[:,indices_] #not right yet, here are all repeated ones.
df_clean = df_clean_const[:, eachcol(df_clean_const).!= eachcol(col_no)]
