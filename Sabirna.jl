
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

indices=findall(≈(1), cor(Matrix(df_clean_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)
indices=DataFrame(indices)
indices_ = union([indices[!,1][i][1] for i in 1:length(indices[!,1])])
col_no=df_clean_const[:,indices_] #not right yet, here are all repeated ones.
df_clean = DataFrame(select(df_clean_const, Not(indices_)))

st_mach = fit!(machine(Standardizer(), df_clean))
st_data = MLJ.transform(st_mach, df_clean)
st_data[!,:labels ]= df.labels
serialize("st_training_data.dat", st_data)


using Serialization
st_data = deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/st_training_data.dat")
st_training=st_data[1:3500,:]
st_validation=st_data[3501:end,:]
df=deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/training.dat")
test=deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/test.dat")

##

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

df_train=df[1:3500,:]
df_test=df[3501:end, :]
KNNmac= machine(SimpleKNNClassifier(K = 8), select(df_train, Not(:labels)), categorical(df_train.labels))
fit!(KNNmac)

KNN_misclassfication_train= mean(predict_mode(KNNmac, select(df_train, Not(:labels))) .!= df_train.labels)
KNN_misclassfication= mean(predict_mode(KNNmac, select(df_test, Not(:labels))) .!= df_test.labels)

##
misclassification_rate()

model1 = SimpleKNNClassifier()
self_tuning_model = TunedModel(model = model1, # the model to be tuned
                                   resampling = CV(nfolds = 5),
                                   tuning=Grid(), 
                                   range=range(model1, :(K),
                                   values = 2:8), # see below
                                   measure = MisclassificationRate()) # evaluation measure
    self_tuning_mach = machine(self_tuning_model, select(st_data, Not(:labels)), categorical(st_data.labels))
	fit!(self_tuning_mach)
    evaluate!(self_tuning_mach, select(st_training, Not(:labels)), categorical(st_training.labels),
          resampling = CV(nfolds = 5), measure = MisclassificationRate())

    MLJ.save("self_tuning_mach.jlso",self_tuning_mach)

    self_tuning_mach=machine(("self_tuning_mach.jlso"))
    
    ##
    Pkg.add("Optim")
    using Optim
    Pkg.add(url = "https://github.com/JuliaAI/MLJLinearModels.jl",
        rev = "a41ee42", preserve = Pkg.PRESERVE_ALL)

    solver = MLJLinearModels.LBFGS(optim_options = Optim.Options(time_limit = 3))
    model_l = LogisticClassifier(penalty = :l2, lambda = 1e-4, solver = solver)
    Mach_Logistic = machine(model_l,
                DataFrame(rand(5000, 10_000), :auto),
                categorical(rand(("a", "b", "c"), 5000)))
    fit!(Mach_Logistic)