
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

## plotting correlation between randomly picked predictors
plot_corr=@df predictors corrplot([:Gm1992 :Gm19938 :Gm37381 :Gm37323],
                     grid = false, fillcolor = cgrad(), size = (700, 600))
                     savefig(plot_corr,"corr.png")
##UMAP plot
using UMAP
umap_proj = umap(Array(select(st_training, Not(:labels)))', 2, min_dist = .4, n_neighbors = 50); #projecting onto 2D space
gr()
umap_plot=scatter(umap_proj[1, :], umap_proj[2, :], legend = false,
        c = vcat([fill(i, 50) for i in 1:3]...), xlabel = "UMAP 1", ylabel = "UMAP 2")
savefig(umap_plot,"umap2.png")


## TSNE plot
using TSne
tsne_proj = tsne(Array(select(df, Not(:labels))), 2, 0, 2000, 80.0, progress = true);#projecting onto 2D space
tsne_plot=scatter(tsne_proj[:, 1], tsne_proj[:, 2], legend = false,
            c = vcat([fill(i, 50) for i in 1:3]...), xlabel = "tSNE 1", ylabel = "tSNE 2")
savefig(tsne_plot,"tsne_80neighbors_2000iter.png")
##
#not much correlation between parameters, clusters do not correctly group labels, not useful

##PCA Plots
pca_mach = fit!(machine(PCA(variance_ratio = 1), select(st_training, Not(:labels))),
	            verbosity = 0)## training pca machine to map data to the PCs
p0 = biplot(pca_mach, score_style = 2, loadings=20)## biplot of the first 20 loadings with highest variance
savefig(p0,"pca20.png")


vars = report(pca_mach).pca.principalvars ./ report(pca_mach).pca.tvar
p1 = plot(vars, label = nothing, yscale = :log10,
          xlabel = "component", ylabel = "proportion of variance explained")
p2 = plot(cumsum(vars),
          label = nothing, xlabel = "component",
          ylabel = "cumulative prop. of variance explained")
p3 = plot(p1, p2, layout = (1, 2), size = (700, 400))## combining plots together