
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


##loading dataframes
df = DataFrame(CSV.File("train.csv"))
test=DataFrame(CSV.File("test.csv"))
##
df=dropmissing(df) #clean missing values
predictors=select(df,Not(:labels))
idx=std.(eachcol(predictors)) .!= 0 # finding indexes of constant predictor columns
df_clean_const = predictors[:, idx]                  #drop zero std (constant) predictors columns

indices=findall(≈(1), cor(Matrix(df_clean_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)
indices=DataFrame(indices)# idices of both with correlation 1
indices_ = union([indices[!,1][i][1] for i in 1:length(indices[!,1])])# indices just of of the correlated
col_no=df_clean_const[:,indices_] #cleaning up
df_clean = DataFrame(select(df_clean_const, Not(indices_))) #cleaned datadrame w/o correlation

st_mach = fit!(machine(Standardizer(), df_clean)) #normalising the cleaned dataframe
st_data = MLJ.transform(st_mach, df_clean)
st_data[!,:labels ]= df.labels

st_training=st_data[1:3500,:]#split up labelled data into training and validation
st_validation=st_data[3501:end,:]

test_ = test[:, names(st_training)[begin:end-1]]  #keeping the same predictors as the training set
st_test=MLJ.transform(fit!(machine(Standardizer(), test_)), test_) #standardising test data
st_test_ = mapcols(col -> replace!(col, NaN=>0), st_test) # replace all NaN values from standardisaation with 0


##PCA transformations
pca_mach = fit!(machine(PCA(variance_ratio = 1), select(st_training, Not(:labels))),
	            verbosity = 0)#training PCA machine

pca_training = MLJ.transform(pca_mach, select(st_training,Not(:labels)))
pca_training.labels = st_training.labels# transform training set into PCA space
pca_validation=MLJ.transform(pca_mach, select(st_validation,Not(:labels)))
pca_validation.labels = st_validation.labels #mapping validation onto same PCA space

pca_mach_whole= fit!(machine(PCA(variance_ratio = 1), select(st_data, Not(:labels))),
	            verbosity = 0)#train PCA mapping machine for whole labelled dataset

pca_whole = MLJ.transform(pca_mach_whole, select(st_data,Not(:labels))) #transform whole dataset to PCA
pca_whole.labels = st_data.labels
pca_test=MLJ.transform(pca_mach_whole, st_test_)# map test set onto PCA space 

##KNN classifier
using Distances
Pkg.add("MLJModelInterface")
import MLJModelInterface
const MMI = MLJModelInterface
MMI.@mlj_model mutable struct SimpleKNNClassifier <: MMI.Probabilistic
    K::Int = 5 :: (_ > 0)
    metric::Distances.Metric = Euclidean()# defining the simple KNN model which takes less time to train
end
function MMI.fit(::SimpleKNNClassifier, verbosity, X, y)
    fitresult = (; X = MMI.matrix(X, transpose = true), y)
    fitresult, nothing, nothing #define fit function
end
function MMI.predict(model::SimpleKNNClassifier, fitresult, Xnew)   #define predict function
    similarities = pairwise(model.metric,
                            fitresult.X, MMI.matrix(Xnew, transpose = true))
    [Distributions.fit(UnivariateFinite, fitresult.y[partialsortperm(col, 1:model.K)])
     for col in eachcol(similarities)]
end
function MMI.predict_mode(model::SimpleKNNClassifier, fitresult, Xnew)# define predict_mode to output most likely classfication
    mode.(predict(model, fitresult, Xnew))
end        

df_train=df[1:3500,:]# here we resolve back to the original data df as st_data with removed parameter will be worse for KNN using the neighboring relation
df_validation=df[3501:end, :]

KNNmac= machine(SimpleKNNClassifier(K = 8), select(df_train, Not(:labels)), categorical(df_train.labels))
fit!(KNNmac)

KNN_misclassfication_train= mean(predict_mode(KNNmac, select(df_train, Not(:labels))) .!= df_train.labels) #training misclassfication rate
KNN_misclassfication= mean(predict_mode(KNNmac, select(df_validation, Not(:labels))) .!= df_validation.labels)# validation misclassification rate

    
##logistic regression
    Pkg.add("Optim", preserve = Pkg.PRESERVE_ALL)
    using Optim
    Pkg.add(url = "https://github.com/JuliaAI/MLJLinearModels.jl", rev = "a41ee42", preserve = Pkg.PRESERVE_ALL)

    solver = MLJLinearModels.LBFGS(optim_options = Optim.Options(time_limit = 100)) #time limited to 100 seconds which ensures accuracy but also fasst to train
    model_l = LogisticClassifier(penalty = :l2, lambda = 1e-5, solver = solver)# using a L2 penalty.
    Mach_Logistic = machine(model_l,select(st_training, Not(:labels)), categorical(st_training.labels))
    fit!(Mach_Logistic)

    logistic_misclassfication_train= mean(predict_mode(Mach_Logistic, select(st_training, Not(:labels))) .!= st_training.labels)
    logistic_misclassfication= mean(predict_mode(Mach_Logistic, select(st_validation, Not(:labels))) .!= st_validation.labels)
## train the same logistic regression model using PCA transformed data
    pca_Mach_Logistic = machine(model_l,select(pca_training, Not(:labels)), categorical(pca_training.labels))
    fit!(pca_Mach_Logistic)
    pca_logistic_misclassfication_train= mean(predict_mode(pca_Mach_Logistic, select(pca_training, Not(:labels))) .!= pca_training.labels)
    pca_logistic_misclassfication= mean(predict_mode(pca_Mach_Logistic, select(pca_validation, Not(:labels))) .!= pca_validation.labels)
## these are for the final predictions
    pca_Mach_Logistic_fin = machine(model_l,select(pca_whole, Not(:labels)), categorical(pca_whole.labels))
    fit!(pca_Mach_Logistic_fin)
    pca_logistic_predicition=predict_mode(pca_Mach_Logistic_fin, pca_test)
 
    ##pca logistic in a self-tuning machine
    pca_self_tuning_model = TunedModel(model = model_l, # the model to be tuned
                                   resampling = CV(nfolds = 5),
                                   tuning=Grid(), 
                                   range= [range(model_l, :(lambda),
                                                  lower = 1e-12, upper = 1e-3,# tuning the lamda
                                                  scale = :log10)], 
                                   measure = MisclassificationRate()) # evaluation measure
    pca_self_tuning_mach = machine(pca_self_tuning_model, select(pca_training, Not(:labels)), categorical(pca_training.labels))
    
    fit!(pca_self_tuning_mach)
    pca_logistic_misclassfication_train= mean(predict_mode(pca_self_tuning_mach, select(pca_training, Not(:labels))) .!= pca_training.labels)
    pca_logistic_misclassfication= mean(predict_mode(pca_self_tuning_mach, select(pca_validation, Not(:labels))) .!= pca_validation.labels)
    

   
 ##neural networks
using MLJFlux
using Flux
#trial 1: simple neural network with 1 hidden layer of 128 neurons
nn_mach = machine(NeuralNetworkClassifier(
                        builder = MLJFlux.Short(n_hidden = 128, 
                        dropout = .5,
                        σ = relu),
                        optimiser = ADAMW(),#act like gradient descent
                        batch_size = 128,
                        epochs = 100),
                        select(st_training, Not(:labels)), categorical(st_training.labels))
    fit!(nn_mach, verbosity = 2)
   
    nn_misclassfication= mean(predict_mode(nn_mach, select(st_training, Not(:labels))) .!= st_training.labels)
    nn_misclass_validation= mean(predict_mode(nn_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)
## trial2: deeper network with 3 hidden layers
    nndeep_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 50, relu), Dense(50, 30, relu), Dense(30, 30, relu), Dense(30,n_out))),
                         batch_size = 32,
                         epochs = 10),
                         select(st_training, Not(:labels)),
                         categorical(st_training.labels))
fit!(nndeep_mach, verbosity = 2)
nn_misclass_validation= mean(predict_mode(nndeep_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)
nndeep_dmisclassfication= mean(predict_mode(nn_mach, select(st_training, Not(:labels))) .!= st_training.labels)

## trial 3: (used as best non--linear method) 4 hidden layers with more neurons
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


## trial 4.1 using the PCA data on the network above, the results aren't as goood
pca_nndeeper_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 30, relu), Dense(30, 50, relu),Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60),
                         select(pca_training, Not(:labels)),
                         categorical(pca_training.labels))
fit!(pca_nndeeper_mach, verbosity = 2)

pca_nndeeper_misclass_validation= mean(predict_mode(pca_nndeeper_mach, select(pca_training, Not(:labels))) .!= pca_training.labels)
pca_nndeeper_dmisclassfication= mean(predict_mode(pca_nndeeper_mach, select(pca_validation, Not(:labels))) .!= pca_validation.labels)

## trial 5 with same number of neurons and layers but different activation functions, yield poor results
nndeeper1_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 50, relu), Dense(50, 50, sigmoid),Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60,),
                         select(st_training, Not(:labels)),
                         categorical(st_training.labels))
fit!(nndeeper1_mach, verbosity = 2)
nndeeper1_misclass_validation= mean(predict_mode(nndeeper1_mach, select(st_training, Not(:labels))) .!= st_training.labels)
nndeeper1_misclassfication= mean(predict_mode(nndeeper1_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)

## trial 5.1 PCA data on machine above, poorer results
pca_nndeeper1_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 50, relu), Dense(50, 50, sigmoid),Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60,),
                         select(pca_training, Not(:labels)),
                         categorical(pca_training.labels))
fit!(pca_nndeeper1_mach, verbosity = 2)
nndeeper1_misclass_validation= mean(predict_mode(pca_nndeeper1_mach, select(pca_training, Not(:labels))) .!= pca_training.labels)
nndeeper1_misclassfication= mean(predict_mode(pca_nndeeper1_mach, select(pca_validation, Not(:labels))) .!= pca_validation.labels)

## trial 6 with deeper network of 5 layers, maybe overfitting as not as good results as 4 layers
nndeeper2_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 50, relu), Dense(50, 50, relu),Dense(50, 50, relu),Dense(50,n_out))),
                         batch_size = 32,
                         epochs = 60,),
                         select(st_training, Not(:labels)),
                         categorical(st_training.labels))
fit!(nndeeper2_mach, verbosity = 2)
nndeeper2_misclass_validation= mean(predict_mode(nndeeper2_mach, select(st_validation, Not(:labels))) .!= st_validation.labels)
nndeeper2_misclassfication= mean(predict_mode(nndeeper2_mach, select(st_training, Not(:labels))) .!= st_training.labels)

## output files with the best models
using CSV
CSV.write("nn_predictions.csv",output, writeheader=true, header=["id", "prediction"])
output=DataFrame(id= [i for i in 1:3093],prediction=nn_predictions)
output1=DataFrame(id= [i for i in 1:3093],prediction=pca_logistic_predicition)
CSV.write("logistic_predictions.csv",output1, writeheader=true, header=["id", "prediction"])
