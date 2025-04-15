using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(1)

# using jld2 is way faster for loading/reading than csv
data = load("data/lincs_filtered_data.jld2")["filtered_data"] # untrt only

### tokenization

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

@time ranked_data = sort_gene(data.expr) # lookup table of indices from highest rank to lowest rank gene

### prev: encoding labels (cell lines) from string -> int

unique_cell_lines = unique(data.inst.cell_iname)
label_dict = Dict(name => i for (i, name) in enumerate(unique_cell_lines))
integer_labels = [label_dict[name] for name in data.inst.cell_iname]

num_classes = length(unique_cell_lines)
input_size = size(ranked_data, 1)

X = ranked_data # 978 x 100425
y = integer_labels # 100425 x 1

# TODO: fix all variable naming, they overlap and are super confusing

#######################################################################################################################################

# so we can use GPU or CPU :D
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32, 3}, CuArray{Float32, 3}}

### positional encoder # TODO: do more research into this part

struct PosEnc
    pe::CuArray{Float32,2}
end

function PosEnc(embed_dim::Int, max_len::Int) # max_len is usually maximum length of sequence but here it is just len(genes)
    pe = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        if mod(i, 2) == 1
            pe[i,pos] = sin(angle) # odd indices
        else
            pe[i,pos] = cos(angle) # even indices
        end
    end
    return PosEnc(cu(pe))
end

Flux.@functor PosEnc

function (p::PosEnc)(x::Float32Matrix3DType)
    seq_len = size(x,2)
    return x .+ p.pe[:,1:seq_len] # adds positional encoding to input embeddings
end

### building transformer section

struct Transf
    mhsa::Flux.MultiHeadAttention
    att_drop::Flux.Dropout
    norm_1::Flux.LayerNorm # this is the normalization aspect
    mlp::Flux.Chain
    norm_2::Flux.LayerNorm
end

function Transf(
    embed_dim::Int, 
    hidden_dim::Int; 
    n_heads::Int, 
    dropout_prob::Float64
    )

    mhsa = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim, 
                                    nheads=n_heads, 
                                    dropout_prob=dropout_prob
                                    )
    att_drop = Flux.Dropout(drop_prob)
    norm1 = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim),
        Flux.Dropout(dropout_prob)
        )
    norm2 = Flux.LayerNorm(embed_dim)

    return Transf(mhsa, att_drop, norm1, mlp, norm2)
end

Flux.@functor Transf

function (tf::Transf)(input::Float32Matrix3DType) # input shape: embed_dim × seq_len × batch_size
    norm_out1 = tf.norm_1(input)
    att_full = tf.mhsa(norm_out1, norm_out1, norm_out1)
    att_out = att_full[1]
    att_out = tf.att_drop(att_out)
    res_out = input + att_out
    norm_out2 = tf.norm_2(res_out)

    embed_dim, seq_len, batch_size = size(norm_out2)
    reshaped = reshape(norm_out2, embed_dim, seq_len * batch_size) # dense layers expect 2D inputs
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    
    output = res_out + mlp_out_reshaped
    return output
end

### full model as << ranked data --> token embedding --> position embedding --> transformer --> output >>

struct Model
    embedding::Flux.Embedding
    posencoder::PosEnc
    pos_drop::Flux.Dropout
    transf::Flux.Chain
    output_head::Flux.Chain # default classifier
end

function Model(;
    input_size::Int,
    embed_size::Int,
    n_layers::Int,
    n_classes::Int,
    n_heads::Int,
    hidden_size::Int,
    dropout_prob::Float64
    )

    # input_head = Flux.Chain(Flux.Dense(input_size => embed_size))
    embed = Flux.Embedding(input_size => embed_size)
    pos_encoder = PosEnc(embed_size, input_size)
    pos_drop = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain(
        [Transf(embed_size, hidden_size; n_heads, dropout_prob) 
        for _ in 1:n_layers]...
        )
    output_head = Flux.Chain(
        Flux.Dense(embed_size => embed_size, gelu),
        Flux.LayerNorm(embed_size),
        Flux.Dense(embed_size => n_classes)
        )

    return Model(embed, pos_encoder, pos_drop, transformer, output_head)
end

Flux.@functor Model

function (model::Model)(x::IntMatrix2DType)
    embedded = model.embedding(x)
    encoded = model.posencoder(embedded)
    encoded = model.pos_drop(encoded)
    transformed = model.transf(encoded)
    pooled = dropdims(mean(transformed; dims=2), dims=2)
    logits = model.output_head(pooled)
    return logits
end

#######################################################################################################################################

### splitting data

function split_data(X::Matrix{Int64}, y::Vector{Int64}, test_ratio::Float64)
    n = size(X, 2)
    indices = shuffle(1:n) 

    test_size = floor(Int, n * test_ratio) 
    train_size = n - test_size

    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]

    X_train, y_train = X[:, train_indices], y[train_indices]
    X_test, y_test = X[:, test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test
end

X_train, y_train, X_test, y_test = split_data(X, y, 0.2)

### training

n_genes, n_samples = size(X)
batch_size = 8
n_epochs = 1 # TODO: currently at 6-7mins per epoch
embed_dim = 64
hidden_dim = 128
n_heads = 4
n_layers = 1
drop_prob = 0.1
lr = 0.001

model = Model(
    input_size=input_size,
    embed_size=embed_dim,
    n_layers=n_layers,
    n_classes=num_classes,
    n_heads=n_heads,
    hidden_size=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)
function loss(model, x, y)
    logits = model(x)  # (num_classes, batch_size)
    y_oh = Flux.onehotbatch(y, 1:num_classes)  # convert to one-hot since target labels are of (batch_size,) to match logits
    return Flux.logitcrossentropy(logits, y_oh)
end
# loss = logitcrossentropy(model(x), y) + α * sum(p -> sum(abs2, p), params(model)) # L2 regularization
train_dataloader = Flux.DataLoader((X_train, y_train), batchsize=batch_size)
test_dataloader = Flux.DataLoader((X_test, y_test), batchsize=batch_size)

train_losses = Float32[]
test_losses = Float32[]
test_accuracies = Float32[]
for epoch in ProgressBar(1:n_epochs)
    Flux.trainmode!(model)
    epoch_losses = Float32[]
    for (x, y) in train_dataloader # x dimensions = 978 * batch_size. y dimensions = 1 * batch_size
        x_gpu, y_gpu = gpu(x), gpu(y)

        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, y_gpu)
        end
        Flux.update!(opt, model, grads[1])
        push!(epoch_losses, loss_val)
    end
    push!(train_losses, mean(epoch_losses))
    # println("epoch $epoch, train loss: $(train_losses[end])")
    
    Flux.testmode!(model)
    test_epoch_losses = Float32[]
    tp = 0
    totals = 0
    for (x, y) in test_dataloader
        x_gpu, y_gpu = gpu(x), gpu(y)

        test_loss_val = loss(model, x_gpu, y_gpu)
        push!(test_epoch_losses, test_loss_val)

        logits = model(x_gpu)
        preds = Flux.onecold(logits) |> cpu # convert logits to predicted class labels
        y_cpu = cpu(y_gpu)

        tp += sum(preds .== y_cpu) # true positives
        totals += length(y_cpu) # total samples
    end

    test_acc = tp / totals
    push!(test_losses, mean(test_epoch_losses))
    push!(test_accuracies, test_acc)

end

### evaluation metrics

# loss plot
Plots.plot(1:n_epochs, train_losses, label="training loss", xlabel="epoch", ylabel="loss", 
     title="training vs validation loss", lw=2)
Plots.plot!(1:n_epochs, test_losses, label="test loss", lw=2)
Plots.savefig("data/plots/transf_ranked_untrt/trainval_loss.png")