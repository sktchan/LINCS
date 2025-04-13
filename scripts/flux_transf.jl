using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(0)

data = load("data/lincs_filtered_data.jld2")["filtered_data"] # using jld2 is way faster for loading/reading than csv

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

# TODO: DO I NEED TO NORMALIZE INDICES??? NO, RIGHT???

@time ranked_data = sort_gene(data.expr) # lookup table of indices from highest rank to lowest rank gene

### prev: encoding labels (cell lines) from string -> int

unique_cell_lines = unique(data.inst.cell_iname)
label_dict = Dict(name => i for (i, name) in enumerate(unique_cell_lines))
integer_labels = [label_dict[name] for name in data.inst.cell_iname]

X = ranked_data # 978 x 100425
y = integer_labels # 100425 x 1

num_classes = length(unique_cell_lines)
input_size = size(X, 1)

#######################################################################################################################################

### positional encoder # TODO: ADD THIS PART IN 

struct PosEncoder
end

function PosEncoder()
end

function (pe::PosEncoder)(x::Float32Matrix3DType)
end

### building transformer section

const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}} # so we can use GPU or CPU :D
const Float32Matrix3DType = Union{Array{Float32, 3}, CuArray{Float32, 3}}

struct Transf
    mhsa::Flux.MultiHeadAttention
    norm_1::Flux.LayerNorm
    mlp::Flux.Chain
    norm_2::Flux.LayerNorm
    # dropout::Flux.Dropout
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
    norm1 = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim)
        )
    norm2 = Flux.LayerNorm(embed_dim)
    # dropout = Flux.Dropout(dropout_prob)

    return Transf(mhsa, norm1, mlp, norm2)
end

Flux.@functor Transf

function (tf::Transf)(input::Float32Matrix2DType) # input is (embed_dim x n_genes x batch)
    # TODO: ADD THIS PART IN 
end

### full model as embedding --> transf --> output

struct Model
    embedding::Flux.Embedding
    # posencoder::PositionalEncoding
    transf::Flux.Chain
    output_head::Flux.Chain # TODO: SHOULD I BE USING POOLING INSTEAD???
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
    # pos_encoder = PositionalEncoding(embed_size, 1)
    transformer = Flux.Chain(
        [Transf(embed_size, hidden_size; n_heads, dropout_prob) 
        for _ in 1:n_layers]
        )
    output_head = Flux.Chain(
        Flux.Dense(embed_size => embed_size, gelu),
        Flux.LayerNorm(embed_size),
        Flux.Dense(embed_size => n_classes)
        )

    return Model(embed, transformer, output_head)
end

Flux.@functor Model

function (model::Model)(x::Float32Matrix2DType) # x is a matrix of gene indices: n_genes × batch_size
    # TODO: ADD THIS PART IN 
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
batch_size = 32
n_epochs = 10
embed_dim = 128
hidden_dim = 256
n_heads = 8
n_layers = 3
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