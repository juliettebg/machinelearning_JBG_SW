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
#change this path to the path of the folder containing all the files in our git repository
folder = "/Users/juliettebg/Desktop/EPFL/MachineLearning/machinelearning_JBG_SW/"

df = DataFrame(CSV.File(folder * "train.csv"))
test=DataFrame(CSV.File(folder * "test.csv"))

#data cleaning 
df=dropmissing(df)
predictors=select(df,Not(:labels))
idx=std.(eachcol(predictors)) .!= 0
df_clean_const = predictors[:, idx]         #drop predictors with zero std

indices=findall(≈(1), cor(Matrix(df_clean_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)
indices=DataFrame(indices)# idices of both with correlation 1
indices_ = union([indices[!,1][i][1] for i in 1:length(indices[!,1])])# keep indices just of of the correlated
col_no=df_clean_const[:,indices_] #cleaning up: get rid of correlated features
df_clean = DataFrame(select(df_clean_const, Not(indices_))) #w/o correlation

st_mach = fit!(machine(Standardizer(), df_clean))
st_data = MLJ.transform(st_mach, df_clean)
st_data[!,:labels ]= df.labels #standardised data

st_training=st_data[1:3500,:]
st_validation=st_data[3501:end,:]# split labelled data into training and validation

test_ = test[:, names(st_training)[begin:end-1]]  #keep only same predictors as labelled set
st_test=MLJ.transform(fit!(machine(Standardizer(), test_)), test_)#standardise test set
st_test_ = mapcols(col -> replace!(col, NaN=>0), st_test) # there are NaNs in the standardised test set so we replace them by 0

#pca denoising
pca_mach_whole= fit!(machine(PCA(variance_ratio = 1), select(st_data, Not(:labels))),
	            verbosity = 0)
pca_whole = MLJ.transform(pca_mach_whole, select(st_data,Not(:labels)))#labelled data transformed into PCA space
pca_whole.labels = st_data.labels   
pca_test=MLJ.transform(pca_mach_whole, st_test_)#test data projected onto same PCA space

#logistic classifier
Pkg.add("Optim", preserve = Pkg.PRESERVE_ALL)
using Optim
Pkg.add(url = "https://github.com/JuliaAI/MLJLinearModels.jl", rev = "a41ee42", preserve = Pkg.PRESERVE_ALL)

solver = MLJLinearModels.LBFGS(optim_options = Optim.Options(time_limit = 100))
model_l = LogisticClassifier(penalty = :l2, lambda = 1e-5, solver = solver)

#run either the first two lines or the third one
pca_Mach_Logistic_fin = machine(model_l,select(pca_whole, Not(:labels)), categorical(pca_whole.labels))
fit!(pca_Mach_Logistic_fin)
pca_Mach_Logistic_fin = machine((folder * "pca_Mach_Logistic_whole.jlso"))
pca_logistic_predicition=predict_mode(pca_Mach_Logistic_fin, pca_test)

##neural network classifier
using MLJFlux
using Flux
#run either the first two lines or the third one
nndeeper_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, 50, relu), Dense(50, 30, relu), Dense(30, 50, relu),Dense(50,n_out))),
                         batch_size = 32,
                         optimiser = ADAMW(),
                         epochs = 60),
                         select(st_data, Not(:labels)),
                         categorical(st_data.labels))
fit!(nndeeper_mach, verbosity = 2)
nndeeper_mach=machine((folder*"nndeeper_mach_wholedata.jlso"))
nn_predictions=predict_mode(nndeeper_mach, st_test_)

## output files
using CSV
output=DataFrame(id= [i for i in 1:3093],prediction=nn_predictions)
CSV.write("nn_predictions.csv",output, writeheader=true, header=["id", "prediction"])

output1=DataFrame(id= [i for i in 1:3093],prediction=pca_logistic_predicition)
CSV.write("logistic_predictions.csv",output1, writeheader=true, header=["id", "prediction"])
