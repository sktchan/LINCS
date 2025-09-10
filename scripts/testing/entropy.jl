# just for calculating entropy of ranks vs exp

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

# using Infiltrator
using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra
CUDA.device!(0)

start_time = now()

# data = load("data/lincs_untrt_data.jld2")["filtered_data"] # untrt only
data = load("data/lincs_trt_untrt_data.jld2")["filtered_data"] # trt and untrt data

### tokenization (row 1 is ranks of gene 1 in each sample)

function sort_gene(expr)
    n, m = size(expr)
    data_ranked = Matrix{Int}(undef, size(expr)) # faster than fill(-1, size(expr))
    sorted_ind_col = Vector{Int}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        sortperm!(sorted_ind_col, unsorted_expr_col, rev=true)
            # rev=true -> data[1, :] = index (into gene.expr) of highest expression value in experiment/column 1
        for i in 1:n
            data_ranked[i, j] = sorted_ind_col[i]
        end
    end
    return data_ranked
end

@time X = sort_gene(data.expr) # lookup table of indices from highest rank to lowest rank gene, 978 x 100425

n_features = size(X, 1) + 1
n_classes = size(X, 1)\

#######################################################################################################################################

### calculating entropy per row (ranks)

function calculate_entropy(row::Vector{Int})
    n = length(row)
    if n == 0
        return 0.0
    end
    counts_dict = counts(row)
    probabilities = values(counts_dict) ./ n
    entropy = -sum(p * log2(p) for p in probabilities)
    return entropy
end

entropies = [calculate_entropy(row) for row in eachrow(X)]
ranks = 1:size(X, 1)

Plots.plot(ranks, entropies,
    seriestype = :scatter,
    title = "entropy by rank",
    xlabel = "gene rank (1 = highest exp)",
    ylabel = "entropy",
    legend = false,
)

# entropy_plot = Plots.plot(
#     ranks, 
#     entropies;
#     xlabel="gene rank (1=highest exp)",
#     ylabel="entropy",
#     title="entropy by rank",
#     legend=false,
#     lw=2
# )

save_dir = "/home/golem/scratch/chans/lincs/plots/untrt/masked_rankings"
savefig(joinpath(save_dir, "entropy.png"))