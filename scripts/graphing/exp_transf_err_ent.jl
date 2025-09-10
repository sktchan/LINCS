using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

# using Infiltrator
using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra
CUDA.device!(1)
# GC.gc()
CUDA.reclaim()

start_time = now()

# data = load("data/lincs_untrt_data.jld2")["filtered_data"] # untrt only
data = load("data/lincs_trt_untrt_data.jld2")["filtered_data"] # trt and untrt data ### REMEMBER TO NOT SAVE PREDSTRUES.CSV IF RUNNING THIS ONE

@time X = data.expr #!# use raw expression values!!!

n_genes = size(X, 1)
n_classes = 1 #!# n_classes is 1 for regression (n_features is not vocabulary size)

#######################################################################################################################################

# so we can use GPU or CPU :D
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32}, CuArray{Float32, 3}}

### positional encoder

struct PosEnc
    pe_matrix::CuArray{Float32,2}
end

#!# uses n_genes as max_len directly
function PosEnc(embed_dim::Int, max_len::Int) # max_len is number of genes
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        if mod(i, 2) == 1
            pe_matrix[i,pos] = sin(angle) # odd indices
        else
            pe_matrix[i,pos] = cos(angle) # even indices
        end
    end
    return PosEnc(cu(pe_matrix))
end

Flux.@functor PosEnc

function (pe::PosEnc)(input::Float32Matrix3DType)
    seq_len = size(input,2)
    return input .+ pe.pe_matrix[:,1:seq_len] # adds positional encoding to input embeddings
end

### building transformer section

struct Transf
    mha::Flux.MultiHeadAttention
    att_dropout::Flux.Dropout
    att_norm::Flux.LayerNorm # this is the normalization aspect
    mlp::Flux.Chain
    mlp_norm::Flux.LayerNorm
end

function Transf(
    embed_dim::Int, 
    hidden_dim::Int; 
    n_heads::Int, 
    dropout_prob::Float64
    )

    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim, 
                                    nheads=n_heads, 
                                    dropout_prob=dropout_prob
                                    )

    att_dropout = Flux.Dropout(dropout_prob)
    
    att_norm = Flux.LayerNorm(embed_dim)
    
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim),
        Flux.Dropout(dropout_prob)
        )
    mlp_norm = Flux.LayerNorm(embed_dim)

    return Transf(mha, att_dropout, att_norm, mlp, mlp_norm)
end

Flux.@functor Transf

function (tf::Transf)(input::Float32Matrix3DType) # input shape: embed_dim × seq_len × batch_size
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1] # outputs a tuple (a, b)
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)

    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size) # dense layers expect 2D inputs
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    
    tf_output = residualed + mlp_out_reshaped
    return tf_output
end

#!# full model for raw value regression

struct Model
    projection::Flux.Dense #!# replace embedding w/ dense layer for cont's input
    pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    classifier::Flux.Chain
end

function Model(;
    seq_len::Int, #!# changed from input_size
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int, #!# 1 for regression
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )

    #!# project the single raw expression value to the embedding dimension
    projection = Flux.Dense(1 => embed_dim)

    pos_encoder = PosEnc(embed_dim, seq_len)

    pos_dropout = Flux.Dropout(dropout_prob)

    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
        )

    #!# classifier preds a singular cont's val
    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => 1, softplus) #!# 1 value returned
        )

    return Model(projection, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@functor Model

#!# fwd pass for raw float inputs
function (model::Model)(input::Float32Matrix2DType)
    seq_len, batch_size = size(input)

    #!# reshape for dense projection: (seq_len, batch_size) -> (1, seq_len * batch_size)
    input_reshaped = reshape(input, 1, :)
    #!# output is (embed_dim, seq_len * batch_size) -> (embed_dim, seq_len, batch_size)
    embedded = reshape(model.projection(input_reshaped), :, seq_len, batch_size)
    
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    
    regression_output = model.classifier(transformed)
    return regression_output
end


#######################################################################################################################################

### splitting data

function split_data(X, test_ratio::Float64, y=nothing)
    n = size(X, 2)
    indices = shuffle(1:n)

    test_size = floor(Int, n * test_ratio)
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]

    X_train = X[:, train_indices]
    X_test = X[:, test_indices]

    if y === nothing
        return X_train, X_test
    else
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, y_train, X_test, y_test
    end
end

X_train, X_test = split_data(X, 0.2)

### masking for raw expression values

const MASK_VALUE = -1.0f0 #!# float for the mask value in the input
mask_ratio=0.1

function mask_input(X::Matrix{Float32}; mask_ratio=mask_ratio)
    X_masked = copy(X)
    mask_labels = fill(NaN32, size(X)) #!# NaN to ignore positions in the loss calculation

    for j in 1:size(X,2) # per column
        num_masked = ceil(Int, size(X,1) * mask_ratio)
        mask_positions = randperm(size(X,1))[1:num_masked]

        for pos in mask_positions
            mask_labels[pos, j] = X[pos, j] 
            X_masked[pos, j] = MASK_VALUE  
        end
    end
    return X_masked, mask_labels
end


X_train_masked, y_train_masked = mask_input(X_train)
X_test_masked, y_test_masked = mask_input(X_test)

#######################################################################################################################################

### training

# n_genes, n_samples = size(X) # n_genes already defined
n_samples = size(X, 2)
batch_size = 64
n_epochs = 5
embed_dim = 64
hidden_dim = 128
n_heads = 1
n_layers = 4
drop_prob = 0.05
lr = 0.001

model = Model(
    seq_len=n_genes,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes, # n_classes is 1
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)

#!# loss is now mse for regression on masked values

function loss(model::Model, x, y, mode::String)
    preds = model(x)  # (1, seq_len, batch_size)
    preds_flat = vec(preds)
    y_flat = vec(y)

    mask = .!isnan.(y_flat)

    if sum(mask) == 0
        return 0.0f0
    end
    
    preds_masked = preds_flat[mask]
    y_masked = y_flat[mask]
    
    regression_loss = Flux.mse(preds_masked, y_masked)

    if mode == "train"
        return regression_loss
    end
    if mode == "test"
        return regression_loss, preds_masked, y_masked
    end
end

train_losses = Float32[]
test_losses = Float32[]
#!# accuracy doesn't work for regression, removed it

# Profile.Allocs.@profile sample_rate=1 begin
for epoch in ProgressBar(1:n_epochs)

    epoch_losses = Float32[]

    # # dynamic masking here (optional, kept as is)
    # X_train_masked = copy(X_train)
    # y_train_masked = mask_input_dyn(X_train_masked)

    for start_idx in 1:batch_size:size(X_train_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train_masked, 2))
        x_gpu = gpu(X_train_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_train_masked[:, start_idx:end_idx])
        
        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, y_gpu, "train")
        end
        Flux.update!(opt, model, grads[1])
        loss_val = loss(model, x_gpu, y_gpu, "train")
        push!(epoch_losses, loss_val)
        # println("hi")
    end

    # println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    push!(train_losses, mean(epoch_losses))

    test_epoch_losses = Float32[]
    
    for start_idx in 1:batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])

        test_loss_val, _, _ = loss(model, x_gpu, y_gpu, "test")
        push!(test_epoch_losses, test_loss_val)

    end

    push!(test_losses, mean(test_epoch_losses))
end
# end

println("starting eval metrics")
### evaluation metrics

# mk dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", "untrt", "masked_expression", timestamp)
mkpath(save_dir)

# loss plot
fig_loss = Figure(size = (800, 600))
ax_loss = Axis(fig_loss[1, 1], 
    xlabel="epoch", 
    ylabel="loss (mse)", 
    title="train vs. test loss"
)
lines!(ax_loss, 1:n_epochs, train_losses, label="train loss", linewidth=2)
lines!(ax_loss, 1:n_epochs, test_losses, label="test loss", linewidth=2)
axislegend(ax_loss, position=:rt)

save(joinpath(save_dir, "loss.png"), fig_loss)

#!# collect all predictions and true values from the test set
all_preds = Float32[]
all_trues = Float32[]
test_epoch_losses = Float32[]
all_gene_indices = Int[]

for start_idx in 1:batch_size:size(X_test_masked, 2)
    end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
    x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
    y_gpu = gpu(y_test_masked[:, start_idx:end_idx])
    test_loss_val, preds_masked, y_masked = loss(model, x_gpu, y_gpu, "test")

    # for calcualting predicted error per gene
    y_cpu_batch = cpu(y_gpu)
    masked_indices_cartesian = findall(!isnan, y_cpu_batch)
    original_gene_indices = [idx[1] for idx in masked_indices_cartesian]
    append!(all_gene_indices, original_gene_indices)

    push!(test_epoch_losses, test_loss_val)
    append!(all_preds, cpu(preds_masked))
    append!(all_trues, cpu(y_masked))
end

correlation = cor(all_trues, all_preds)

### old boxplot

# min_val = minimum(all_trues)
# max_val = maximum(all_trues)

# # define bins
# bin_edges = min_val:1.0:max_val
# bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

# # prep boxplot data
# grouped_preds = Float32[]
# grouped_trues_midpts = Float64[]

# for i in 1:length(bin_edges)-1
#     indices = findall(x -> bin_edges[i] <= x < bin_edges[i+1], all_trues)
#     if !isempty(indices)
#         preds_in_bin = all_preds[indices]
#         midpt = bin_midpts[i]
#         append!(grouped_preds, preds_in_bin)
#         append!(grouped_trues_midpts, fill(midpt, length(preds_in_bin)))
#     end
# end

# # create the boxplot figure and axis
# begin
# fig_box = Figure(size = (800, 600))
# ax_box = Axis(fig_box[1, 1],
#     xlabel="true expression val",
#     ylabel="predicted expression val",
#     title="predicted vs. true expression"
# )

# # plot the boxplot data
# boxplot!(ax_box, grouped_trues_midpts, grouped_preds, range=0, whiskerlinewidth=0)
# # ablines!(ax_box, 0, 1, color=:black, linestyle=:dash, linewidth=1)
# fig_box
# end
# save(joinpath(save_dir, "boxplot.png"), fig_box)

#######################################################################################################################################

### just boxplot

min_val = minimum(all_trues)
max_val = maximum(all_trues)

# define bins
bin_edges = min_val:1.0:max_val
bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

stats = Dict()
x_outliers = Float32[]
y_outliers = Float32[]


for (i, midpt) in enumerate(bin_midpts) # for each bin
    ind = findall(x -> bin_edges[i] <= x < bin_edges[i+1], all_trues)
    bin_preds = all_preds[ind]

    q10 = quantile(bin_preds, 0.10)
    q25 = quantile(bin_preds, 0.25)
    q50 = quantile(bin_preds, 0.50)
    q75 = quantile(bin_preds, 0.75)
    q90 = quantile(bin_preds, 0.90)

    stats[midpt] = (q10=q10, q25=q25, q50=q50, q75=q75, q90=q90)
    outlier_ind = findall(y -> y < q10 || y > q90, bin_preds)
    append!(x_outliers, fill(midpt, length(outlier_ind)))
    append!(y_outliers, bin_preds[outlier_ind])
end

midpts_plot = collect(keys(stats))
q10s = [s.q10 for s in values(stats)]
q25s = [s.q25 for s in values(stats)]
q50s = [s.q50 for s in values(stats)]
q75s = [s.q75 for s in values(stats)]
q90s = [s.q90 for s in values(stats)]

fig_box = Figure(size = (800, 600))
ax_box = Axis(fig_box[1, 1],
    xlabel="true expression val",
    ylabel="predicted expression val",
    title="predicted vs. true expression",
)

scatter!(ax_box, x_outliers, y_outliers, markersize = 5, alpha = 0.5)

# between 10th quartile and 25th quartile (upper)
rangebars!(ax_box, midpts_plot, q10s, q25s,
    color = :black,
    whiskerwidth = 0.5
)

# between 75th quartile nad 90th quartile (lower)
rangebars!(ax_box, midpts_plot, q75s, q90s,
    color = :black,
    whiskerwidth = 0.5
)

boxplot!(ax_box, grouped_trues_midpts, grouped_preds,
    width = 0.5,
    range = false,
    whiskerlinewidth = 0,
    show_outliers = false
)

display(fig_box)
save(joinpath(save_dir, "new_boxplot.png"), fig_box)

#######################################################################################################################################

### boxplot and histogram combined

min_val = minimum(all_trues)
max_val = maximum(all_trues)

# define bins
bin_edges = min_val:1.0:max_val
bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

stats = Dict()
x_outliers = Float32[]
y_outliers = Float32[]
grouped_preds = Float32[]
grouped_trues_midpts = Float64[]

for (i, midpt) in enumerate(bin_midpts) # for each bin
    ind = findall(x -> bin_edges[i] <= x < bin_edges[i+1], all_trues)
    bin_preds = all_preds[ind]
    append!(grouped_preds, bin_preds)
    append!(grouped_trues_midpts, fill(midpt, length(bin_preds)))

    q10 = quantile(bin_preds, 0.10)
    q25 = quantile(bin_preds, 0.25)
    q50 = quantile(bin_preds, 0.50)
    q75 = quantile(bin_preds, 0.75)
    q90 = quantile(bin_preds, 0.90)

    stats[midpt] = (q10=q10, q25=q25, q50=q50, q75=q75, q90=q90)
    outlier_ind = findall(y -> y < q10 || y > q90, bin_preds)
    append!(x_outliers, fill(midpt, length(outlier_ind)))
    append!(y_outliers, bin_preds[outlier_ind])
end

midpts_plot = collect(keys(stats))
q10s = [s.q10 for s in values(stats)]
q25s = [s.q25 for s in values(stats)]
q50s = [s.q50 for s in values(stats)]
q75s = [s.q75 for s in values(stats)]
q90s = [s.q90 for s in values(stats)] 

begin
    # setup
    fig = Figure(size = (800, 800))
    ax_box = Axis(fig[1, 1],
        xlabel="",
        ylabel="predicted expression value",
        title="predicted vs. true expression values"
    )
    ax_hist = Axis(fig[2, 1],
        xlabel="true expression value",
        ylabel="count",
        title="distribution of true expression values",
        xticks = 0:5:15,
    )
    linkxaxes!(ax_box, ax_hist)

    # boxplot
    scatter!(ax_box, x_outliers, y_outliers, markersize = 5, alpha = 0.5)

    # between 10th quartile and 25th quartile (upper)
    rangebars!(ax_box, midpts_plot, q10s, q25s,
        color = :black,
        whiskerwidth = 0.5
    )

    # between 75th quartile nad 90th quartile (lower)
    rangebars!(ax_box, midpts_plot, q75s, q90s,
        color = :black,
        whiskerwidth = 0.5
    )

    boxplot!(ax_box, grouped_trues_midpts, grouped_preds,
        range = false,
        whiskerlinewidth = 0,
        show_outliers = false
    )

    # histogram
    hist!(ax_hist, all_trues, bins = bin_edges,
        strokecolor = :black,
        strokewidth = 1
    )

    rowgap!(fig.layout, 1, 10) # gap between plots

    display(fig)
end

save(joinpath(save_dir, "box_hist.png"), fig)

#######################################################################################################################################

# plot hexbin
fig_hex = Figure(size = (800, 600))
ax_hex = Axis(fig_hex[1, 1],
    xlabel="true expression val",
    ylabel="predicted expression val",
    title="predicted vs. true expression density",
    aspect=DataAspect() 
)

hexplot = hexbin!(ax_hex, all_trues, all_preds)
Colorbar(fig_hex[1, 2], hexplot, label="point count")
# ablines!(ax_hex, 0, 1, color=:red, linestyle=:dash, linewidth=2)
save(joinpath(save_dir, "hexbin.png"), fig_hex)

### error by gene

absolute_errors = abs.(all_trues .- all_preds)

df_gene_errors = DataFrame(
    gene_index = all_gene_indices,
    absolute_error = absolute_errors
)
CSV.write(joinpath(save_dir, "gene_vs_error.csv"), df_gene_errors)

begin
    fig_gene_error_scatter = Figure(size = (800, 600))
    ax_gene_error_scatter = Axis(fig_gene_error_scatter[1, 1],
        title = "prediction error by gene",
        xlabel = "gene index",
        ylabel = "absolute prediction error"
    )
    
    scatter!(ax_gene_error_scatter, all_gene_indices, absolute_errors, markersize=4, alpha=0.3)
    display(fig_gene_error_scatter)
end

save(joinpath(save_dir, "gene_vs_error_scatter.png"), fig_gene_error_scatter)

#######################################################################################################################################

### checking if predicting average
gene_averages_train = vec(mean(X_train, dims=2)) |> cpu
masked_indices = findall(!isnan, y_test_masked)
gene_indices_for_masked_values = getindex.(masked_indices, 1)
baseline_preds = gene_averages_train[gene_indices_for_masked_values]
mse_model = mean((all_trues .- all_preds).^2)
mse_baseline = mean((all_trues .- baseline_preds).^2)

# lims = (floor(min_val), ceil(max_val))

fig_baseline_hex = Figure(size = (800, 600))
ax_baseline_hex = Axis(fig_baseline_hex[1, 1],
    xlabel="true expression val",
    ylabel="gene average val",
    title="predicting the average vs. true expression density",
    aspect=DataAspect()
    # limits=(lims, lims)
)

hexplot_baseline = hexbin!(ax_baseline_hex, all_trues, baseline_preds)
Colorbar(fig_baseline_hex[1, 2], hexplot_baseline, label="point count")
save(joinpath(save_dir, "avg_hexbin.png"), fig_baseline_hex)

#######################################################################################################################################

# log data
df_losses = DataFrame(
    epoch = 1:n_epochs,
    train_loss = train_losses,
    test_loss = test_losses
)
CSV.write(joinpath(save_dir, "losses.csv"), df_losses)

# log pred/true
df_preds = DataFrame(
    all_preds = all_preds,
    all_trues = all_trues
)
CSV.write(joinpath(save_dir, "predstrues.csv"), df_preds)

end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000) # 60 * 1000
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

# log params used
params_txt = joinpath(save_dir, "params.txt")
open(params_txt, "w") do io
    println(io, "PARAMETERS:")
    println(io, "########### this was on kraken")
    println(io, "dataset = untrt")
    println(io, "masking_ratio = $mask_ratio")
    println(io, "mask_value = $MASK_VALUE")
    println(io, "NO DYNAMIC MASKING")
    println(io, "batch_size = $batch_size")
    println(io, "n_epochs = $n_epochs")
    println(io, "embed_dim = $embed_dim")
    println(io, "hidden_dim = $hidden_dim")
    println(io, "n_heads = $n_heads")
    println(io, "n_layers = $n_layers")
    println(io, "learning_rate = $lr")
    println(io, "dropout_probability = $drop_prob")
    println(io, "ADDITIONAL NOTES: testing out error by gene plot on trt, lowered params cuz time crunch")
    println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    println(io, "correlation = $correlation"),
    println(io, "mse model = $mse_model"),
    println(io, "mse baseline = $mse_baseline")
end