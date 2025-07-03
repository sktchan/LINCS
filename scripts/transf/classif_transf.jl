# TODO: look into pos encoding
# TODO: on untrt only: 10epochs = 2.5hrs
# TODO: on trt+untrt: 10epochs = 46hrs
# FIXME: there's an issue somwhere in Transf.mha (NNlib.dot_product_attention) due to a 58g size intermediate object when you use a batch size of 4096. look into later!

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(0)

# using jld2 is way faster for loading/reading than csv
# data = load("data/lincs_untrt_data.jld2")["filtered_data"] # untrt only
data = load("data/lincs_trt_untrt_data.jld2")["filtered_data"] # trt and untrt data

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

n_classes = length(unique_cell_lines)
n_features = size(ranked_data, 1)

X = ranked_data # 978 x 100425
y = integer_labels # 100425 x 1

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

    att_dropout = Flux.Dropout(drop_prob)
    
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

    # input_head = Flux.Chain(Flux.Dense(input_size => embed_dim))

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
    pooled = dropdims(mean(transformed; dims=2), dims=2)
    logits_output = model.classifier(pooled)
    return logits_output
end

# embedded = model.embedding(x)
# encoded = model.pos_encoder(embedded)
# encoded_dropped = model.pos_dropout(encoded)
# transformed = model.transformer(encoded_dropped)
# pooled = dropdims(mean(transformed; dims=2), dims=2)
# logits_output = model.classifier(pooled)


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
batch_size = 64
n_epochs = 10
embed_dim = 32
hidden_dim = 64
n_heads = 4
n_layers = 1
drop_prob = 0.1
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

# trr = Flux.trainables(model)
# zzz = [prod(size(t)) for t in trr]

opt = Flux.setup(Adam(lr), model)
function loss(model, x, y)
    logits = model(x)  # (n_classes, batch_size)
    y_oh = Flux.onehotbatch(y, 1:n_classes)  # convert to one-hot since target labels are of (batch_size,) to match logits
    return Flux.logitcrossentropy(logits, y_oh)
end
# loss = logitcrossentropy(model(x), y) + α * sum(p -> sum(abs2, p), params(model)) # L2 regularization
train_dataloader = Flux.DataLoader((X_train, y_train), batchsize=batch_size)
test_dataloader = Flux.DataLoader((X_test, y_test), batchsize=batch_size)

# (x, y) = first(train_dataloader)
# x_gpu, y_gpu = gpu(x), gpu(y)
# loss_val, grads = Flux.withgradient(model) do m
#     loss(m, x_gpu, y_gpu)
# end

# ll = loss(model, x_gpu, y_gpu)
# logits = model(x) 

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
Plots.savefig("plots/trt_and_untrt/transformer/trainval_loss.png")

df = DataFrame(test_accuracy = test_accuracies)
CSV.write("plots/trt_and_untrt/transformer/test_accuracies.csv", df)