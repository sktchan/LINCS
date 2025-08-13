#=
THIS FILE CALCULATES ERROR INSTEAD OF ACCURACY
=#

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

# using Infiltrator
using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(3)

start_time = now()

data = load("data/lincs_untrt_data.jld2")["filtered_data"] # untrt only
# data = load("data/lincs_trt_untrt_data.jld2")["filtered_data"] # trt and untrt data

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

# so we can use GPU or CPU :D
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32, 3}, CuArray{Float32, 3}}

### positional encoder

struct PosEnc
    pe_matrix::CuArray{Float32,2}
end

function PosEnc(embed_dim::Int, max_len::Int) # max_len is usually maximum length of sequence but here it is just len(genes)
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

### full model as << ranked data --> token embedding --> position embedding --> transformer --> classifier head >>

struct Model
    embedding::Flux.Embedding
    pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    classifier::Flux.Chain
end

function Model(;
    input_size::Int,
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int,
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )

    embedding = Flux.Embedding(input_size => embed_dim)

    pos_encoder = PosEnc(embed_dim, input_size)

    pos_dropout = Flux.Dropout(dropout_prob)

    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
        )

    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => n_classes)
        )

    return Model(embedding, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@functor Model

function (model::Model)(input::IntMatrix2DType)
    embedded = model.embedding(input)
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    # pooled = dropdims(mean(transformed; dims=2), dims=2)
    logits_output = model.classifier(transformed)
    return logits_output
end


#######################################################################################################################################

### splitting data

function split_data(X, test_ratio::Float64, y=nothing) # masking doesn't need y!
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

### masking data

const MASK_ID = (n_classes + 1)
mask_ratio=0.1

function mask_input(X::Matrix{Int64}; mask_ratio=mask_ratio)
    X_masked = copy(X) # or view()??
    mask_labels = fill((-100), size(X)) # -100 = ignore, this is not masked

    for j in 1:size(X,2) # per column
        num_masked = ceil(Int, size(X,1) * mask_ratio)
        mask_positions = randperm(size(X,1))[1:num_masked]

        for pos in mask_positions
            mask_labels[pos, j] = X[pos, j] # original label
            X_masked[pos, j] = MASK_ID # mask label
        end
    end
    return X_masked, mask_labels
end

X_train_masked, y_train_masked = mask_input(X_train)
X_test_masked, y_test_masked = mask_input(X_test)

#######################################################################################################################################

### HERE WE GIVE BOTH TRAIN/TEST SAME MASK FOR DEBUGGING :( (same gene/# across columns, diff rows/ranks)

# function mask_input(X::Matrix{Int64}; mask_ratio=0.1)
#     X_masked = copy(X)
#     mask_labels = fill(-100, size(X))
#     all_values = unique(X)
#     all_values = setdiff(all_values, MASK_ID)
#     num_masked_values = ceil(Int, length(all_values) * mask_ratio)
#     masked_values = randperm(length(all_values))[1:num_masked_values]
#     masked_values = all_values[masked_values]
#     for j in 1:size(X, 2)
#         for i in 1:size(X, 1)
#             if X[i, j] in masked_values
#                 mask_labels[i, j] = X[i, j]
#                 X_masked[i, j] = MASK_ID
#             end
#         end
#     end
#     return X_masked, mask_labels
# end


# X_masked_full, y_masked_full = mask_input(X, mask_ratio=mask_ratio)

# # split
# test_ratio = 0.2
# n_total = size(X, 2)
# indices = shuffle(1:n_total)
# test_size = floor(Int, n_total * test_ratio)
# test_indices = indices[1:test_size]
# train_indices = indices[test_size+1:end]

# # now we got the same mask! yay.
# X_train_masked = X_masked_full[:, train_indices]
# y_train_masked = y_masked_full[:, train_indices]

# X_test_masked = X_masked_full[:, test_indices]
# y_test_masked = y_masked_full[:, test_indices]

#######################################################################################################################################

### training

n_genes, n_samples = size(X)
batch_size = 128
n_epochs = 10
embed_dim = 64
hidden_dim = 128
n_heads = 1
n_layers = 4
drop_prob = 0.05
lr = 0.001

model = Model(
    input_size=n_features,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes,
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)

#=
loss: cross-entropy between the model’s predicted distribution and the true token at each masked position
compute the loss by iterating over masked positions OR by using a mask in the loss function
=#

function loss(model::Model, x, y, mode::String)
    logits = model(x)  # (n_classes, seq_len, batch_size)
    logits_flat = reshape(logits, size(logits, 1), :) # (n_classes, seq_len*batch_size)
    y_flat = vec(y) # (seq_len*batch_size) column vec

    mask = y_flat .!= -100 # bit vec, where sum = n_masked
    logits_masked = logits_flat[:, mask] # (n_classes, n_masked)
    y_masked = y_flat[mask] # (n_masked) column vec
    y_oh = Flux.onehotbatch(y_masked, 1:n_classes) # (n_classes, n_masked)

    if mode == "train"
        return Flux.logitcrossentropy(logits_masked, y_oh) 
    end
    if mode == "test"
        return Flux.logitcrossentropy(logits_masked, y_oh), logits_masked, y_masked
    end
end

train_losses = Float32[]
test_losses = Float32[]

test_rank_errors = Float32[]

all_preds = Int[]
all_trues = Int[]

for epoch in ProgressBar(1:n_epochs)

    epoch_losses = Float32[]

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
    end

    push!(train_losses, mean(epoch_losses))

    test_epoch_losses = Float32[]
    epoch_rank_errors = Int[]

    for start_idx in 1:batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])

        test_loss_val, logits_masked, y_masked = loss(model, x_gpu, y_gpu, "test")
        push!(test_epoch_losses, test_loss_val)

        if isempty(y_masked) continue end

        logits_cpu = cpu(logits_masked)
        y_cpu = cpu(y_masked)

        # error in rank pred vs rank true calculated here
        # similar to regression like mse/rmse BUT this kinda makes more sense cuz predicted gene id - actual gene id isn't significant (right?)
        for r in 1:length(y_cpu) 
            true_gene_id = y_cpu[r] # actual mask value
            prediction_logits = logits_cpu[:, r] # probabilities for theh first mask
            ranked_gene_ids = sortperm(prediction_logits, rev=true) # list of most likely genes from highest to lowest likely prediction
            predicted_rank = findfirst(isequal(true_gene_id), ranked_gene_ids) # the rank that the true gene was ranked at
            
            if !isnothing(predicted_rank)
                push!(epoch_rank_errors, predicted_rank - 1) # error is pred rank - 1
            end
        end

        if epoch == n_epochs
            predicted_ranks = Flux.onecold(logits_masked)
            append!(all_preds, cpu(predicted_ranks))
            append!(all_trues, y_cpu)
        end
    end
    push!(test_losses, mean(test_epoch_losses))
    if !isempty(epoch_rank_errors)
        push!(test_rank_errors, mean(epoch_rank_errors))
    else
        push!(test_rank_errors, NaN32)
    end
end

end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)


### evaluation metrics

# mk dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", "untrt", "masked_rankings", timestamp)
mkpath(save_dir)

# loss plot
Plots.plot(1:n_epochs, train_losses; label="training loss",
     xlabel="epoch", ylabel="loss", title="training vs validation loss", lw=2)
Plots.plot!(1:n_epochs, test_losses;  label="test loss", lw=2)
savefig(joinpath(save_dir, "trainval_loss.png"))

# error
acc_plot = Plots.plot(1:n_epochs, test_rank_errors; label="test",
    xlabel="epoch", ylabel="mean error", title="mean rank errors", lw=2)
savefig(joinpath(save_dir, "error.png"))

#!# scatter plot instead of accuracy to visualize :D

# all_preds = Int[]
# all_trues = Int[]

# for start_idx in 1:batch_size:size(X_test_masked, 2)
#     end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
#     x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
#     y_gpu = gpu(y_test_masked[:, start_idx:end_idx])
#     _, logits_masked, y_masked = loss(model, x_gpu, y_gpu, "test")

#     if isempty(y_masked) 
#         continue 
#     end

#     predicted_ranks = Flux.onecold(logits_masked)

#     append!(all_preds, cpu(predicted_ranks))
#     append!(all_trues, cpu(y_masked))
# end

# Plots.scatter(all_trues, all_preds,
#         label="predictions",
#         xlabel="true rank",
#         ylabel="predicted rank",
#         title="pred vs. true ranks",
#         alpha=0.3,
#         aspect_ratio=:equal,
#         markersize=2,
#         xlims=(0, 1000))

# min_val = 1
# max_val = n_classes

# Plots.plot!([min_val, max_val], [min_val, max_val],
#       label="y=x",
#       linestyle=:dash,
#       lw=2)

# correlation = cor(all_trues, all_preds)
# savefig(joinpath(save_dir, "prediction_scatter_plot.png"))

#!# heatmap instead of scatter plot to visualize!

freq_matrix = zeros(Int, n_classes, n_classes)
for i in 1:length(all_trues)
    true_id = all_trues[i]
    pred_id = all_preds[i]
    if 1 <= true_id <= n_classes && 1 <= pred_id <= n_classes
        freq_matrix[true_id, pred_id] += 1
    end
end

log_freq_matrix = log1p.(freq_matrix)

fig = Figure(size = (800, 800))
ax = Axis(fig[1, 1],
          title = "predicted vs. true",
          xlabel = "true gene id",
          ylabel = "predicted gene id",
          aspect = AxisAspect(1))

h = CairoMakie.heatmap!(ax, 1:n_classes, 1:n_classes, log_freq_matrix, colormap = :viridis)
CairoMakie.Colorbar(fig[1, 2], h) # log(1+count)

save(joinpath(save_dir, "prediction_heatmap.png"), fig)

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

# log params used
params_txt = joinpath(save_dir, "params.txt")
open(params_txt, "w") do io
    println(io, "PARAMETERS:")
    println(io, "########### this was on smaug")
    println(io, "dataset = untrt")
    println(io, "masking_ratio = $mask_ratio")
    println(io, "NO DYNAMIC MASKING")
    println(io, "batch_size = $batch_size")
    println(io, "n_epochs = $n_epochs")
    println(io, "embed_dim = $embed_dim")
    println(io, "hidden_dim = $hidden_dim")
    println(io, "n_heads = $n_heads")
    println(io, "n_layers = $n_layers")
    println(io, "learning_rate = $lr")
    println(io, "dropout_probability = $drop_prob")
    println(io, "ADDITIONAL NOTES: rerunning heatmap without y=x line and 10ep")
    println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    # println(io, "correlation = $correlation")
end
