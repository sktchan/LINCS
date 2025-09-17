#= 
ONLY FOR GENERATING ENTROPY GRAPHS
=#

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
n_classes = size(X, 1)

#######################################################################################################################################

### calculating entropy per row (ranks)

function calculate_entropy(row)
    n = length(row)
    if n == 0
        return 0.0
    end
    counts_dict = countmap(row)
    probabilities = values(counts_dict) ./ n
    entropy = -sum(p * log2(p) for p in probabilities)
    return entropy
end

entropies = Float64[] 
for row in eachrow(X)
    e = calculate_entropy(row)
    push!(entropies, e)
end

ranks = 1:size(X, 1)

begin
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1],
        xlabel="gene rank (1 = highest exp)", 
        ylabel="entropy", 
        title="entropy by rank"
    )
    scatter!(ax, ranks, entropies, label="entropy", alpha=0.5)
    display(fig)
end

save_dir = "/home/golem/scratch/chans/lincs/plots/trt_and_untrt/entropies"
save(joinpath(save_dir, "rank_entropy_trt.png"), fig)

#######################################################################################################################################

### calculating entropy per row (gene expression)

eX = data.expr

# what is the probability that it is in the same bin?
function calculate_binned_entropy(row; min_val=0.0, max_val=15.1, bin_width=0.1)
    n = length(row)
    if n == 0
        return 0.0
    end
    num_bins = Int(ceil((max_val - min_val) / bin_width))
    bin_counts = zeros(Int, num_bins)

    for x in row
        if x >= min_val && x < max_val
            bin_index = floor(Int, (x - min_val) / bin_width) + 1
            bin_counts[bin_index] += 1
        end
    end

    non_zero_counts = filter(c -> c > 0, bin_counts)
    if isempty(non_zero_counts)
        return 0.0
    end
    probabilities = non_zero_counts ./ n
    entropy = -sum(p * log2(p) for p in probabilities)
    
    return entropy
end

min = minimum(eX)
max = maximum(eX)
ex_entropies = Float64[]
for gene_row in eachrow(eX)
    e = calculate_binned_entropy(gene_row; min_val=min, max_val=max, bin_width=0.1)
    push!(ex_entropies, e)
end

gene_indices = 1:size(eX, 1)
gene_symbols = data.gene.gene_symbol

begin
    fig_ex = Figure(size = (800, 600))
    ax_ex = Axis(fig_ex[1, 1],
        xlabel="gene index", 
        ylabel="entropy", 
        title="entropy by gene"
        # xticks = (gene_indices, gene_symbols),
        # xticklabelrotation = pi/2,
        # xticklabelsize = 2
    )
    scatter!(ax_ex, gene_indices, ex_entropies, label="entropy", alpha=0.5)
    display(fig_ex)
end

save(joinpath(save_dir, "gene_exp_entropy_trt_0.1.png"), fig_ex)

#######################################################################################################################################

### calculating by entropy by cell line/column (gene expression)

min_expr, max_expr = extrema(eX)
sample_expr_entropies = Float64[]
for sample_col in eachcol(eX)
    push!(sample_expr_entropies, calculate_binned_entropy(sample_col; min_val=min_expr, max_val=max_expr, bin_width=0.01))
end

begin
    fig = Figure(size = (1600, 800))

    ax = Axis(fig[1, 1],
        xlabel = "cell line",
        ylabel = "entropy",
        title = "entropy by cell line (gene expression data)",
        xticklabelsize = 6, 
        xticklabelrotation = pi/3
    )
    cell_lines = string.(data.inst.cell_iname)
    unique_cells = unique(cell_lines)
    cell_indices = [findfirst(==(cell), unique_cells) for cell in cell_lines]
    
    boxplot!(ax, cell_indices, sample_expr_entropies, width = 1, markersize = 2)
    ax.xticks = (1:length(unique_cells), unique_cells)
    xlims!(ax, 0.5, length(unique_cells) + 0.5)
    display(fig)
end
save(joinpath(save_dir, "cellline_entropy_0.01.png"), fig)

#######################################################################################################################################

### calculating standard deviation per row (gene expression) 

eX = data.expr

gene_std_devs = Float64[] 

for gene_row in eachrow(eX) 
    s = std(gene_row)
    push!(gene_std_devs, s)
end

gene_indices = 1:size(eX, 1)

begin 
    fig_ex_std = Figure(size = (800, 600)) 
    ax_ex_std = Axis(fig_ex_std[1, 1], 
        xlabel="gene index",
        ylabel = "standard deviation", 
        title= "standard deviation in expression by gene")

    scatter!(ax_ex_std, gene_indices, gene_std_devs, alpha = 0.5) 
    display(fig_ex_std) 
end

save_dir = "/home/golem/scratch/chans/lincs/plots/trt_and_untrt/infographs"
save(joinpath(save_dir, "gene_exp_std_dev_trt.png"), fig_ex_std)


#######################################################################################################################################

### organizing by highest --> lowest expression

gene_means = [mean(row) for row in eachrow(eX)]
gene_std_devs = [std(row) for row in eachrow(eX)]
gene_indices = 1:size(eX, 1)

begin
    fig_mean_expr = Figure(size = (800, 600))
    ax_mean_expr = Axis(fig_mean_expr[1, 1],
        xlabel = "gene index",
        ylabel = "mean expression level",
        title = "expression level by gene")

    scatter!(ax_mean_expr, gene_indices, gene_means, alpha = 0.5)
    display(fig_mean_expr)
end
save(joinpath(save_dir, "gene_mean_exp_trt.png"), fig_mean_expr)

sorted_indices_by_mean = sortperm(gene_means, rev=true)
jldsave("/home/golem/scratch/chans/lincs/plots/trt_and_untrt/infographs/sorted_gene_indices_by_exp.jld2"; sorted_indices_by_mean)

begin
    fig_mean_sorted = Figure(size = (800, 600))
    ax_mean_sorted = Axis(fig_mean_sorted[1, 1],
        xlabel = "gene index (sorted)",
        ylabel = "mean expression level",
        title = "expression level by sorted gene")

    scatter!(ax_mean_sorted, gene_indices, gene_means[sorted_indices_by_mean], alpha = 0.5)
    display(fig_mean_sorted)
end
save(joinpath(save_dir, "sorted_gene_mean_exp_trt.png"), fig_mean_sorted)

begin
    fig_std_sorted = Figure(size = (800, 600))
    ax_std_sorted = Axis(fig_std_sorted[1, 1],
        xlabel = "gene index (sorted)",
        ylabel = "standard deviation",
        title = "standard deviation by sorted gene")

    scatter!(ax_std_sorted, gene_indices, gene_std_devs[sorted_indices_by_mean], alpha = 0.5)
    display(fig_std_sorted)
end

save(joinpath(save_dir, "sorted_gene_std_dev_trt.png"), fig_std_sorted)