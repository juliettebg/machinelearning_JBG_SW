
using CSV
using DataFrames

using Random
using Pkg
Pkg.activate(temp = true)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Distributions
using MLJ
using MLJLinearModels
using MLJXGBoostInterface
using MLJMultivariateStatsInterface
using MLCourse
using OpenML
using Plots
using Statistics, StatsPlots

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
test_ = test[:, names(st_training)[begin:end-1]]  
st_test=MLJ.transform(fit!(machine(Standardizer(), test_)), test_)
#replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)
st_test_ = mapcols(col -> replace!(col, NaN=>0), st_test) 
#st_test=DataFrame(mapreduce(permutedims, vcat, st_test), :auto)


##
plot_corr=@df predictors corrplot([:Gm1992 :Gm19938 :Gm37381 :Gm37323],
                     grid = false, fillcolor = cgrad(), size = (700, 600))
                     savefig(plot_corr,"corr.png")
@df predictors corrplot([:Catip :Gm29107],
                     grid = false, fillcolor = cgrad(), size = (700, 600))
##
using UMAP
umap_proj = umap(Array(select(st_training, Not(:labels)))', 2, min_dist = .4, n_neighbors = 50);
gr()
umap_plot=scatter(umap_proj[1, :], umap_proj[2, :], legend = false,
        c = vcat([fill(i, 50) for i in 1:3]...), xlabel = "UMAP 1", ylabel = "UMAP 2")
savefig(umap_plot,"umap2.png")
##
using TSne
tsne_proj = tsne(Array(select(df, Not(:labels))), 2, 0, 2000, 80.0, progress = true);
tsne_plot=scatter(tsne_proj[:, 1], tsne_proj[:, 2], legend = false,
            c = vcat([fill(i, 50) for i in 1:3]...), xlabel = "tSNE 1", ylabel = "tSNE 2")
savefig(tsne_plot,"tsne_80neighbors_2000iter.png")
##
#not much correlation between parameters, clusters do not correctly group labels, not useful
             
components = MLJ.transform(machvis, select(df,Not(:labels)))
components.labels = df.labels
projection = fitted_params(machvis).projection
loadings = projection' .* report(machvis).principalvars
plot(components, x=:x1, y=:x2, color=:labels, mode="markers",
                     Layout(shapes=[line(x0=0, y0=0, x1=loadings[1, i], y1=loadings[2, i])
                                for i in 1:length(df.labels)],
                     annotations=[attr(x=loadings[1, i], y=loadings[2, i], text=names(select(df,Not(:labels)))[i],
                                     xanchor="center", yanchor="bottom") 
                                     for i in 1:length(df.labels)]
                         ))
pca_mach = fit!(machine(PCA(variance_ratio = 1), select(st_training, Not(:labels))),
	            verbosity = 0)
p1 = biplot(pca_mach, score_style = 2, loadings=20)
savefig(p1,"pca20.png")
MLJ.save("pca_mach.jlso",pca_mach)

vars = report(pca_mach).pca.principalvars ./ report(pca_mach).pca.tvar
p1 = plot(vars, label = nothing, yscale = :log10,
          xlabel = "component", ylabel = "proportion of variance explained")
p2 = plot(cumsum(vars),
          label = nothing, xlabel = "component",
          ylabel = "cumulative prop. of variance explained")
p3 = plot(p1, p2, layout = (1, 2), size = (700, 400))

pca_training = MLJ.transform(pca_mach, select(st_training,Not(:labels)))
pca_training.labels = st_training.labels
pca_validation=MLJ.transform(pca_mach, select(st_validation,Not(:labels)))
pca_validation.labels = st_validation.labels

pca_training=deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/pca_training_data.dat")
pca_validation=deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/pca_validation_data.dat")
st_pca_training=pca_training=deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/st_pca_train.dat")
st_pca_validation=deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/st_pca_validation.dat")

st_pca_training= MLJ.transform(fit!(machine(Standardizer(), select(pca_training,Not(:labels)))), select(pca_training,Not(:labels)))
st_pca_validation= MLJ.transform(fit!(machine(Standardizer(), select(pca_validation,Not(:labels)))), select(pca_validation,Not(:labels)))
pca_mach_whole= fit!(machine(PCA(variance_ratio = 1), select(st_data, Not(:labels))),
	            verbosity = 0)

pca_whole = MLJ.transform(pca_mach_whole, select(st_data,Not(:labels)))
pca_whole.labels = st_data.labels    
pca_training_whole=pca_whole[1:3500,:]
pca_validation_whole=pca_whole[3501:end,:]  
serialize("pca_whole.dat", pca_whole)  
MLJ.save("pca_mach_whole.jlso",pca_mach_whole)
pca_whole=deserialize("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/pca_whole.dat")
pca_mach_whole=machine(("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/pca_mach_whole.jlso"))
pca_test=MLJ.transform(pca_mach_whole, st_test_)
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
    
    ##logistic
    Pkg.add("Optim", preserve = Pkg.PRESERVE_ALL)
    using Optim
    Pkg.add(url = "https://github.com/JuliaAI/MLJLinearModels.jl", rev = "a41ee42", preserve = Pkg.PRESERVE_ALL)

    solver = MLJLinearModels.LBFGS(optim_options = Optim.Options(time_limit = 100))
    model_l = LogisticClassifier(penalty = :l2, lambda = 1e-5, solver = solver)
    Mach_Logistic = machine(model_l,select(st_training, Not(:labels)), categorical(st_training.labels))
    fit!(Mach_Logistic)

    logistic_misclassfication_train= mean(predict_mode(Mach_Logistic, select(st_training, Not(:labels))) .!= st_training.labels)
    logistic_misclassfication= mean(predict_mode(Mach_Logistic, select(st_validation, Not(:labels))) .!= st_validation.labels)

    MLJ.save("Mach_Logistic.jlso",Mach_Logistic)

    pca_Mach_Logistic = machine(model_l,select(pca_training, Not(:labels)), categorical(pca_training.labels))
    fit!(pca_Mach_Logistic)
    pca_logistic_misclassfication_train= mean(predict_mode(pca_Mach_Logistic, select(pca_training, Not(:labels))) .!= pca_training.labels)
    pca_logistic_misclassfication= mean(predict_mode(pca_Mach_Logistic, select(pca_validation, Not(:labels))) .!= pca_validation.labels)

    pca_Mach_Logistic_fin = machine(model_l,select(pca_whole, Not(:labels)), categorical(pca_whole.labels))
    fit!(pca_Mach_Logistic_fin)
    pca_logistic_predicition=predict_mode(pca_Mach_Logistic_fin, pca_test)

    ##pca logistic
    pca_self_tuning_model = TunedModel(model = model_l, # the model to be tuned
                                   resampling = CV(nfolds = 5),
                                   tuning=Grid(), 
                                   range= [range(model_l, :(lambda),
                                                  lower = 1e-12, upper = 1e-3,
                                                  scale = :log10)], 
                                   measure = MisclassificationRate()) # evaluation measure
    pca_self_tuning_mach = machine(pca_self_tuning_model, select(pca_training, Not(:labels)), categorical(pca_training.labels))
    
    fit!(pca_self_tuning_mach)
    pca_logistic_misclassfication_train= mean(predict_mode(pca_self_tuning_mach, select(pca_training, Not(:labels))) .!= pca_training.labels)
    pca_logistic_misclassfication= mean(predict_mode(pca_self_tuning_mach, select(pca_validation, Not(:labels))) .!= pca_validation.labels)
    MLJ.save("pca__logistic_self_tuning_mach.jlso",pca_self_tuning_mach)
    #
  

   
    ##neural networks
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
                         epochs = 10),
                         select(st_training, Not(:labels)),
                         categorical(st_training.labels))
fit!(nndeep_mach, verbosity = 2)
nn_misclass_validation= mean(predict_mode(nndeep_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)
nndeep_dmisclassfication= mean(predict_mode(nn_mach, select(st_training, Not(:labels))) .!= st_training.labels)
MLJ.save("nndeep_mach.jlso",nndeep_mach)
nndeep_mach=machine(("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/nndeep_mach.jlso"))

pca_nndeep_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 50, relu), Dense(50, 30, relu), Dense(30, 30, relu), Dense(30,n_out))),
                         batch_size = 32,
                         epochs = 60),
                         select(pca_training, Not(:labels)),
                         categorical(pca_training.labels))
fit!(pca_nndeep_mach, verbosity = 2)
##most promising nn machine so far
nndeeper_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 30, relu), Dense(30, 50, relu),Dense(50,n_out))),
                         batch_size = 32,
                         optimiser = ADAMW(),
                         epochs = 60),
                         select(st_data, Not(:labels)),
                         categorical(st_data.labels))
fit!(nndeeper_mach, verbosity = 2)
nn_predictions=predict_mode(nndeeper_mach, st_test_)
nndeeper_misclass_validation= mean(predict_mode(nndeeper_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)
nndeeper_dmisclassfication= mean(predict_mode(nndeeper_mach, select(st_training, Not(:labels))) .!= st_training.labels)
MLJ.save("nndeeper_mach_wholedata.jlso",nndeeper_mach)
nndeeper_mach=machine(("/Users/sabo4ever/Documents/GitHub/machinelearning_JBG_SW/nndeeper_mach.jlso"))

##
pca_nndeeper_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 30, relu), Dense(30, 50, relu),Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60),
                         select(pca_training, Not(:labels)),
                         categorical(pca_training.labels))
fit!(pca_nndeeper_mach, verbosity = 2)

pca_nndeeper_misclass_validation= mean(predict_mode(pca_nndeeper_mach, select(pca_training, Not(:labels))) .!= pca_training.labels)
pca_nndeeper_dmisclassfication= mean(predict_mode(pca_nndeeper_mach, select(pca_validation, Not(:labels))) .!= pca_validation.labels)
MLJ.save("pca_nndeeper_mach.jlso",nndeeper_mach)
##

nndeeper1_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 50, relu), Dense(50, 50, sigmoid),Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60,),
                         select(st_training, Not(:labels)),
                         categorical(st_training.labels))
fit!(nndeeper1_mach, verbosity = 2)
nndeeper1_misclass_validation= mean(predict_mode(nndeeper1_mach, select(st_training, Not(:labels))) .!= st_training.labels)
nndeeper1_misclassfication= mean(predict_mode(nndeeper1_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)


pca_nndeeper1_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 50, relu), Dense(50, 50, sigmoid),Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60,),
                         select(pca_training, Not(:labels)),
                         categorical(pca_training.labels))
fit!(pca_nndeeper1_mach, verbosity = 2)
nndeeper1_misclass_validation= mean(predict_mode(pca_nndeeper1_mach, select(pca_training, Not(:labels))) .!= pca_training.labels)
nndeeper1_misclassfication= mean(predict_mode(pca_nndeeper1_mach, select(pca_validation, Not(:labels))) .!= pca_validation.labels)

##
nndeeper2_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 50, relu), Dense(50, 50, relu),Dense(50, 50, relu),Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60,),
                         select(st_training, Not(:labels)),
                         categorical(st_training.labels))
fit!(nndeeper2_mach, verbosity = 2)
nndeeper2_misclass_validation= mean(predict_mode(nndeeper2_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)
nndeeper2_misclassfication= mean(predict_mode(nndeeper2_mach, select(st_training, Not(:labels))) .!= st_training.labels)

using CSV
CSV.write("nn_predictions.csv",output, writeheader=true, header=["id", "prediction"])
output=DataFrame(id= [i for i in 1:3093],prediction=nn_predictions)
output1=DataFrame(id= [i for i in 1:3093],prediction=pca_logistic_predicition)
CSV.write("logistic_predictions.csv",output1, writeheader=true, header=["id", "prediction"])
