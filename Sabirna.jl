
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
using MLCourse
using OpenML
using Plots
import MLCourse


##
df = DataFrame(CSV.File("/Users/sabo4ever/Sabrina/EPFL/machine learning/Project/train.csv"))
test=DataFrame(CSV.File("/Users/sabo4ever/Sabrina/EPFL/machine learning/Project/test.csv"))
##
df=dropmissing(df)
predictors=select(df,Not(:labels))
idx=std.(eachcol(predictors)) .!= 0
df_clean_const = predictors[:, idx]                  #drop zero std predictors

indices=findall(≈(1), cor(Matrix(df_clean_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)
indices=DataFrame(indices)# idices of both with correlation 1
indices_ = union([indices[!,1][i][1] for i in 1:length(indices[!,1])])# indices just of of the correlated
col_no=df_clean_const[:,indices_] #cleaning up
df_clean = DataFrame(select(df_clean_const, Not(indices_))) #w/o correlation

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
using UMAP
umap_proj = umap(Array(select(df, Not(:labels)))', 2, min_dist = .4, n_neighbors = 50);
gr()
umap_plot=scatter(umap_proj[1, :], umap_proj[2, :], legend = false,
        c = vcat([fill(i, 50) for i in 1:3]...), xlabel = "UMAP 1", ylabel = "UMAP 2")
savefig(umap_plot,"umap.png")
##
using TSne
tsne_proj = tsne(Array(select(df, Not(:labels))), 2, 0, 100, 50.0, progress = true);


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
    Pkg.add("Optim", preserve = Pkg.PRESERVE_ALL)
    using Optim
    Pkg.add(url = "https://github.com/JuliaAI/MLJLinearModels.jl", rev = "a41ee42", preserve = Pkg.PRESERVE_ALL)

    solver = MLJLinearModels.LBFGS(optim_options = Optim.Options(time_limit = 100))
    model_l = LogisticClassifier(penalty = :l2, lambda = 1e-4, solver = solver)
    Mach_Logistic = machine(model_l,select(st_training, Not(:labels)), categorical(st_training.labels))
    fit!(Mach_Logistic)

    logistic_misclassfication_train= mean(predict_mode(Mach_Logistic, select(st_training, Not(:labels))) .!= st_training.labels)
    logistic_misclassfication= mean(predict_mode(Mach_Logistic, select(st_validation, Not(:labels))) .!= st_validation.labels)

    MLJ.save("Mach_Logistic.jlso",Mach_Logistic)
   #
   model_reg = model_l |> LassoRegression()
   self_tuning_model1 = TunedModel(model = model_reg,
                                  tuning =  Grid(goal = 100),
                                  resampling = CV(nfolds = 5),
                                  range = [range(model_reg, :(logistic_classifier.lambda),
                                                 lower = 1e-10, upper = 1e-3,
                                                 scale = :log10)],
                                  measure = MisclassificationRate())
   self_tuning_mach1 = machine(self_tuning_model1, select(st_training, Not(:labels)), categorical(st_training.labels))
   fit!(self_tuning_mach1, verbosity = 2)
end;

   
    ##
using MLJFlux
using Flux
nn_mach = machine(NeuralNetworkClassifier(
                        builder = MLJFlux.Short(n_hidden = 128,
                        dropout = .5,
                        σ = relu),
                        optimiser = ADAMW(),
                        batch_size = 128,
                        epochs = 100),
                        select(st_training, Not(:labels)), categorical(st_training.labels))
    fit!(nn_mach, verbosity = 2)
    mean(MLJ.predict_mode(nn_mach, select(st_data, Not(:labels)), categorical(st_data.labels)))
    MLJ.evaluate!(nn_mach, select(st_data, Not(:labels)), categorical(st_data.labels))
 
    MLJ.save("nn_mach.jlso",nn_mach)
    nn_mach=machine(("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/nn_mach.jlso"))
  
##

    nn_misclassfication= mean(predict_mode(nn_mach, select(st_training, Not(:labels))) .!= st_training.labels)
    nn_misclass_validation= mean(predict_mode(nn_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)

    nndeep_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 50, relu), Dense(50, 30, relu), Dense(30, 30, relu), Dense(30,n_out))),
                         batch_size = 32,
                         epochs = 60),
                         select(st_training, Not(:labels)),
                         categorical(st_training.labels))
fit!(nndeep_mach, verbosity = 2)
nn_misclass_validation= mean(predict_mode(nndeep_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)
nndeep_dmisclassfication= mean(predict_mode(nn_mach, select(st_training, Not(:labels))) .!= st_training.labels)
MLJ.save("nndeep_mach.jlso",nndeep_mach)

##
nndeeper_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 30, relu), Dense(30, 50, relu)Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60),
                         select(st_training, Not(:labels)),
                         categorical(st_training.labels))
fit!(nndeep_macher, verbosity = 2)
nndeeper_misclass_validation= mean(predict_mode(nndeep_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)
nndeeper_dmisclassfication= mean(predict_mode(nn_mach, select(st_training, Not(:labels))) .!= st_training.labels)