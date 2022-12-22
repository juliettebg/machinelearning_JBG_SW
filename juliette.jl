#all imports and using necessary to run the code
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, LinearAlgebra, Statistics, Random, CSV, MLJLinearModels
using MLJXGBoostInterface, MLJDecisionTreeInterface, MLJClusteringInterface, Distributions, Distances
using MLJTuning
using PlotlyJS
using StatsPlots
using Plots
using MLJ: fit!
using PlotlyJS: plot
using PlotlyJS: savefig
using Serialization
#this is the cleaned data that is the same as st_data in the Main file
data_cleaned = deserialize("/Users/juliettebg/Desktop/EPFL/MachineLearning/machinelearning_JBG_SW/st_training_data.dat")

# define the train and validation data
training=data_cleaned[1:3500,:]
validation=data_cleaned[3501:end,:]
training = deserialize("/Users/juliettebg/Desktop/EPFL/MachineLearning/pca_training_data.dat")
validation = deserialize("/Users/juliettebg/Desktop/EPFL/MachineLearning/pca_validation_data.dat")

#define inputs that the tree related methods will be able to read
features = select(training, Not(:labels))
labels = training.labels
features = float.(features)
labels   = string.(labels)              

#normal tree
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
mach = machine(tree, features, categorical(labels)) |> fit!
predict(mach, select(validation, Not(:labels)))
confusion_matrix(predict_mode(mach, select(validation, Not(:labels))),  validation.labels)
#test error
misclassification_rate(confusion_matrix(predict_mode(mach, select(validation, Not(:labels))),  validation.labels))
#training error
misclassification_rate(confusion_matrix(predict_mode(mach, select(training, Not(:labels))),  training.labels))

#Random forest
Forest = @load RandomForestClassifier pkg=DecisionTree
forest = Forest(n_trees = 1, sampling_fraction=1)
mach2 = machine(forest, features, categorical(labels)) |> fit!
confusion_matrix(predict_mode(mach2, select(validation, Not(:labels))),  validation.labels)
#test error
misclassification_rate(confusion_matrix(predict_mode(mach2, select(validation, Not(:labels))),  validation.labels))
#training error
misclassification_rate(confusion_matrix(predict_mode(mach2, select(training, Not(:labels))),  training.labels))

#tuned model forest with XGB
xgb = XGBoostClassifier()
#define the ranges to evaluate for each parameter
r6 = range(xgb, :eta, lower = 1e-2, upper = .1, scale = :log);
r7 = range(xgb, :num_round, lower = 50, upper = 500);
r8 = range(xgb, :max_depth, lower = 2, upper = 50);

mach3bis = machine(TunedModel(model = xgb,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                        tuning = Grid(goal = 25),
                     range = [r6, r7, r8]),
              features, categorical(labels))|> fit!

confusion_matrix(predict_mode(mach3bis, select(validation, Not(:labels))),  validation.labels)
#test error
misclassification_rate(confusion_matrix(predict_mode(mach3bis, select(validation, Not(:labels))),  validation.labels))
#training error
misclassification_rate(confusion_matrix(predict_mode(mach3bis, select(training, Not(:labels))),  training.labels))

#Tuned forest
#define the ranges to evaluate for each parameter
r1 = range(forest, :sampling_fraction, lower = 0.1, upper = 1.0);
r2 = range(forest, :n_trees, lower = 10, upper = 500);
r3 = range(forest, :max_depth, lower = 2, upper = 100);

mach3 = machine(TunedModel(model = forest,
                        resampling = CV(nfolds = 6),
                        measure = MisclassificationRate(),
                     range = [r1, r2, r3]),
              features, categorical(labels))|> fit!
#find and output the best model
entry = report(mach3).best_history_entry
entry.model
confusion_matrix(predict_mode(mach3, select(validation, Not(:labels))),  validation.labels)
#test error
misclassification_rate(confusion_matrix(predict_mode(mach3, select(validation, Not(:labels))),  validation.labels))
#training error
misclassification_rate(confusion_matrix(predict_mode(mach3, select(training, Not(:labels))),  training.labels))

#tuned model tree
#define the range to evaluate the parameter
r3 = range(tree, :max_depth, lower = 2, upper = 50);
mach4 = machine(TunedModel(model = tree,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                     range = r3),
              features, categorical(labels))|> fit!
#find and output the best model
entry2 = report(mach4).best_history_entry
entry2.model.max_depth
confusion_matrix(predict_mode(mach4, select(validation, Not(:labels))),  validation.labels)
#test error
misclassification_rate(confusion_matrix(predict_mode(mach4, select(validation, Not(:labels))),  validation.labels))
#training error
misclassification_rate(confusion_matrix(predict_mode(mach4, select(training, Not(:labels))),  training.labels))

#ridge and lasso regularization logistic classification
#prepare the inputs
coerce!(training, :labels => OrderedFactor)
labels_train = training.labels
coerce!(validation, :labels => OrderedFactor)
labels_valid = validation.labels

#ridge regression
ridge = LogisticClassifier(penalty = :l2)
r5 = range(ridge, :lambda, lower = 0, upper = 1);
mach5 = machine(TunedModel(model = ridge,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                     range = r5),
                     select(training, Not(:labels)), labels_train)|> fit!

confusion_matrix(predict_mode(mach5, select(validation, Not(:labels))),  validation.labels)
#test error
misclassification_rate(confusion_matrix(predict_mode(mach5, select(validation, Not(:labels))),  validation.labels))
#training error
misclassification_rate(confusion_matrix(predict_mode(mach5, select(training, Not(:labels))),  training.labels))

#lasso regression
lasso = LogisticClassifier(penalty = :l1)
r6 = range(lasso, :lambda, lower = 0, upper = 1);
mach6 = machine(TunedModel(model = lasso,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                     range = r6),
                     select(training, Not(:labels)), labels_train)|> fit!
confusion_matrix(predict_mode(mach6, select(validation, Not(:labels))),  validation.labels)
#test error
misclassification_rate(confusion_matrix(predict_mode(mach6, select(validation, Not(:labels))),  validation.labels))
#training error
misclassification_rate(confusion_matrix(predict_mode(mach6, select(training, Not(:labels))),  training.labels))



#test with clustering
coerce!(data_cleaned, :labels => Multiclass)

#KMeans clustering
mach7 = machine(KMeans(k = 3), select(data_cleaned, Not(:labels)))
fit!(mach7, verbosity = 0)
pred = predict(mach7)
confusion_matrix(pred, data_cleaned.labels) |> x -> DataFrame(x.mat, x.labels)

#HierarchicalClustering
hc = machine(HierarchicalClustering(k = 3, linkage = :complete, metric = Euclidean()))
pred_hc = predict(hc, select(data_cleaned, Not(:labels)))
confusion_matrix(pred_hc, data_cleaned.labels) |> x -> DataFrame(x.mat, x.labels)

#DBSCAN
dbscan_pred = predict(machine(DBSCAN(min_cluster_size = 10, radius = 1)), select(data_cleaned, Not(:labels)))
confusion_matrix(dbscan_pred, data_cleaned.labels) |> x -> DataFrame(x.mat, x.labels)