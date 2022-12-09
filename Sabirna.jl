using CSV
using DataFrames

using Random
using Pkg
Pkg.activate(temp = true)
Pkg.develop(url = "https://github.com/jbrea/MLCourse")
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Distributions
using MLJ
using MLJLinearModels
using OpenML
using Plots


##
df = DataFrame(CSV.File("/Users/sabo4ever/Sabrina/EPFL/machine learning/Project/train.csv"))
test=DataFrame(CSV.File("/Users/sabo4ever/Sabrina/EPFL/machine learning/Project/test.csv"))
##
dropmissing(df)
predictors=select(df,Not(:labels))
idx=std.(eachcol(predictors)) .!= 0
df_clean_const = predictors[:, idx]                  #drop zero std predictors
using Serialization
serialize("df_after_std.dat", df_clean_const)
indices=findall(≈(1), cor(Matrix(df_clean_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)
indices=DataFrame(indices)
indices_ = union([indices[!,1][i][1] for i in 1:length(indices[!,1])])
col_no=df_clean_const[:,indices_] #not right yet, here are all repeated ones.
df_clean = DataFrame(select(df_clean_const, Not(indices_)))
serialize("cleaned_training_data.dat", df_clean)

st_mach = fit!(machine(Standardizer(), df_clean))
st_data = MLJ.transform(st_mach, df_clean)
serialize("st_training_data.dat", st_data)


using Serialization
st_data = deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/st_training_data.dat")
pre_training=st_data[1:1000,:]
using NearestNeighborModels

using Distances
Pkg.add("MLJModelInterface")
import MLJModelInterface
const MMI = MLJModelInterface
MMI.@mlj_model mutable struct SimpleKNNClassifier <: MMI.Probabilistic
    K::Int = 5 :: (_ > 0)
    metric::Distances.Metric = Euclidean()
end
function MMI.fit(::SimpleKNNClassifier, verbosity, X, y)
    fitresult = (; X = MMI.matrix(X, transpose = true), y)
    fitresult, nothing, nothing
end
function MMI.predict(model::SimpleKNNClassifier, fitresult, Xnew)
    similarities = pairwise(model.metric,
                            fitresult.X, MMI.matrix(Xnew, transpose = true))
    [Distributions.fit(UnivariateFinite, fitresult.y[partialsortperm(col, 1:model.K)])
     for col in eachcol(similarities)]
end
function MMI.predict_mode(model::SimpleKNNClassifier, fitresult, Xnew)
    mode.(predict(model, fitresult, Xnew))
end        

KNNmac= machine(SimpleKNNClassifier(K = 3), st_data, categorical(df.labels))
fit!(KNNmac)
 
mean(predict_mode(KNNmac, select(test, Not(:labels))) .!= test.labels)


self_tuning_model = TunedModel(model = SimpleKNNClassifier(), # the model to be tuned
                                   resampling = CV(nfolds = 5), # how to evaluate
                                   range = range(SimpleKNNClassifier(), :(K),
                                                 values = 3:8), # see below
                                   measure = auc) # evaluation measure
    self_tuning_mach = machine(self_tuning_model, st_data, categorical(df.labels))
	fit!(self_tuning_mach, verbosity = 0)

    
    