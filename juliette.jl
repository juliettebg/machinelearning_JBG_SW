#load the data that has previously been cleaned
using Pkg


Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, LinearAlgebra, Statistics, Random, CSV, MLJLinearModels
using MLJXGBoostInterface, MLJDecisionTreeInterface
using MLJTuning
using PlotlyJS
using StatsPlots
using Plots
using MLJ: fit!
using PlotlyJS: plot
using PlotlyJS: savefig
#import PlutoPlotly as PP
#using PlutoPlotly

using Serialization

data_cleaned = deserialize("/Users/juliettebg/Desktop/EPFL/MachineLearning/machinelearning_JBG_SW/st_training_data.dat")

#visualization

df = data_cleaned
predictors=select(df,Not(:labels))
 test = [:Gm1992, :Gm19938, :Gm37381 ,:Gm37323]

features = Symbol.(names(predictors))
#plotlyJS
plot(df, dimensions=features, color=:labels, kind="splom")
pca= fit!(machine(PCA(), data_cleaned[:, 1:end-1]), verbosity = 0);

#statsplots
@df predictors corrplot([:Gm1992 :Gm19938 :Gm37381 :Gm37323],
                     grid = false, fillcolor = cgrad(), size = (700, 600))



                     
                     

                     
                     machvis = machine(PCA(maxoutdim=2), df[!, features])
                     
                     fit!(machvis)
                     
                     
                     components = MLJ.transform(machvis, df[!, features])
                     
                     components.labels = df.labels
                     
                     projection = fitted_params(machvis).projection
                     
                     loadings = projection' .* report(machvis).principalvars
                     
                     
                     plot(
                     
                         components, x=:x1, y=:x2, color=:labels, mode="markers",
                     
                         Layout(
                     
                             shapes=[
                     
                                 line(x0=0, y0=0, x1=loadings[1, i], y1=loadings[2, i])
                     
                                 for i in 1:length(features)
                     
                             ],
                     
                             annotations=[
                     
                                 attr(
                     
                                     x=loadings[1, i], y=loadings[2, i], text=features[i],
                     
                                     xanchor="center", yanchor="bottom"
                     
                                 )
                     
                                 for i in 1:length(features)
                     
                             ]
                     
                         )
                     
                     )

#3d plot
machvis2 = machine(PCA(maxoutdim=3), df[!, features])

fit!(machvis2)

components = MLJ.transform(machvis2, df[!, features])

components.labels = df.labels

total_var = report(machvis2).tprincipalvar / report(machvis2).tvar


p4 =plot(

    components, x=:x1, y=:x2, z=:x3, color=:labels,

    kind="scatter3d", mode="markers",

    labels=attr(;[Symbol("x", i) => "PC $i" for i in 1:3]...),

    Layout(

        title="Total explained variance: $(round(total_var, digits=2))"

    )

)
savefig(p4, "/Users/juliettebg/Desktop/EPFL/MachineLearning/machinelearning_JBG_SW/visualization_3d.png")
savefig(p4, "/Users/juliettebg/Desktop/EPFL/MachineLearning/machinelearning_JBG_SW/visualization_3d.pdf")
#pca with 20 components for visualization
pca_20 =fit!(machine(PCA(maxoutdim = 20), data_cleaned[:, 1:end-1]), verbosity = 0);



vars = report(pca_20).principalvars ./ report(pca_20).tvar
p1 = plot(vars, label = nothing, yscale = :log10,
          xlabel = "component", ylabel = "proportion of variance explained")
p2 = plot(cumsum(vars),
          label = nothing, xlabel = "component",
          ylabel = "cumulative prop. of variance explained")
p3 = plot(p1, p2, layout = (1, 2), size = (700, 400))
savefig(p3, "/Users/juliettebg/Desktop/EPFL/MachineLearning/machinelearning_JBG_SW/visualization_1.png")
savefig(p3, "/Users/juliettebg/Desktop/EPFL/MachineLearning/machinelearning_JBG_SW/visualization_1.pdf")

gr()
biplot(pca_20)

#train and test data
training=data_cleaned[1:3500,:]
validation=data_cleaned[3501:end,:]



#tree


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
misclassification_rate(confusion_matrix(predict_mode(mach, select(validation, Not(:labels))),  validation.labels))
misclassification_rate(confusion_matrix(predict_mode(mach, select(training, Not(:labels))),  training.labels))

#Random forest

Forest = @load RandomForestClassifier pkg=DecisionTree
forest = Forest(n_trees = 100, sampling_fraction=0.7)

mach2 = machine(forest, features, categorical(labels)) |> fit!


confusion_matrix(predict_mode(mach2, select(validation, Not(:labels))),  validation.labels)

misclassification_rate(confusion_matrix(predict_mode(mach2, select(validation, Not(:labels))),  validation.labels))
misclassification_rate(confusion_matrix(predict_mode(mach2, select(training, Not(:labels))),  training.labels))
#tuned model forest
xgb = XGBoostClassifier()
r1 = range(forest, :sampling_fraction, lower = 0.1, upper = 1.0);
r2 = range(forest, :n_trees, lower = 10, upper = 100);
r3 = range(forest, :max_depth, lower = 2, upper = 50);

mach3 = machine(TunedModel(model = forest,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                     range = [r1, r2, r3]),
              features, categorical(labels))|> fit!

entry = report(mach3).best_history_entry
entry.model
confusion_matrix(predict_mode(mach3, select(validation, Not(:labels))),  validation.labels)
misclassification_rate(confusion_matrix(predict_mode(mach3, select(validation, Not(:labels))),  validation.labels))
misclassification_rate(confusion_matrix(predict_mode(mach3, select(training, Not(:labels))),  training.labels))

#tuned model tree
mach4 = machine(TunedModel(model = tree,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                     range = r3),
              features, categorical(labels))|> fit!
entry2 = report(mach4).best_history_entry
entry2.model.max_depth
confusion_matrix(predict_mode(mach4, select(validation, Not(:labels))),  validation.labels)
misclassification_rate(confusion_matrix(predict_mode(mach4, select(validation, Not(:labels))),  validation.labels))
misclassification_rate(confusion_matrix(predict_mode(mach4, select(training, Not(:labels))),  training.labels))



#logistic and logistic regression

coerce!(training, :labels => OrderedFactor)
labels_train = training.labels
#labels_train = string.(labels_train)
coerce!(validation, :labels => OrderedFactor)
labels_valid = validation.labels
#labels_valid = string.(labels_valid)
#ridge regression
logistic_fit = fit!(machine(LogisticClassifier(penalty = :l2, lambda = 1e-5),
                        select(training, Not(:labels)),
                        labels_train), verbosity = 0)
confusion_matrix(predict_mode(logistic_fit, select(validation, Not(:labels))),  validation.labels)
misclassification_rate(confusion_matrix(predict_mode(logistic_fit, select(validation, Not(:labels))),  validation.labels))
misclassification_rate(confusion_matrix(predict_mode(logistic_fit, select(training, Not(:labels))),  training.labels))



#PCA classification
pca_class =fit!(machine(PCA(), training[:, 1:end-1]), verbosity = 0)
#genes = pixel, cell = faces, label = name
cells = training
KAT5  = training[training.labels .== "KAT5", :]
df[df.A .> 500, :]
