using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(0)

# @time df = CSV.read("data/all_cellline_geneexpr.csv", DataFrame) # trt and untrt
@time df = CSV.read("data/cellline_geneexpr.csv", DataFrame) # untrt only

### prev: ranking gene expr level

df_ranked = copy(df)
gene_columns = names(df)[2:end]
n_exp = nrow(df_ranked)

for col_name in gene_columns
    expression_values = df_ranked[!, col_name]
    ranks = ordinalrank(-expression_values)
    df_ranked[!, col_name] = ranks ./ n_exp
end

### prev: encoding labels (cell lines) from string -> int

df.cell_line = categorical(df.cell_line) 
df.cell_line_encoded = levelcode.(df.cell_line)

X = Float32.(Matrix(df_ranked[:, 2:end-1])') # flux requires format (features × samples)
y = df.cell_line_encoded

num_classes = length(levels(df.cell_line))
input_size = size(X, 1)

# y_oh = Float32.(Flux.onehotbatch(y, 1:num_classes)) # i don't think you have to one-hot encode them?

### building transformer section

const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}} # so we can use GPU or CPU :D

struct Transf
    mhsa::Flux.MultiHeadAttention
    norm_1::Flux.LayerNorm
    mlp::Flux.Chain
    norm_2::Flux.LayerNorm
    # dropout::Flux.Dropout
end

function Transf(
    embed_dim::Int64, 
    hidden_dim::Int64; 
    n_heads::Int64, 
    dropout_prob::Float64
    )

    mhsa = Flux.MultiHeadAttention(embed_dim, nheads=n_heads, dropout_prob=dropout_prob)
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

function (tf::Transf)(input::Float32Matrix2DType) # input is features (978) x batch
    features = size(input, 1)
    batch = size(input, 2)
    input_3d = reshape(input, features, 1, batch) # (features x 1 x batch) for mhsa input

    normed = tf.norm_1(input_3d)
    sa_out = tf.mhsa(normed, normed, normed) # OG paper states norm after sa, but norm before sa is more common?

    # FIXME: why is it outputting a 3D and 4D tensor?
    print("type: ", typeof(sa_out)) # output: type: Tuple{CuArray{Float32, 3, CUDA.DeviceMemory}, CuArray{Float32, 4, CUDA.DeviceMemory}}
    print("size of 1: ", size(sa_out[1])) # output: (128, 1, 32)
    print("size of 2: ", size(sa_out[2])) # output: (1, 1, 8, 32)

    sa_out_2d = reshape(sa_out, features, batch) # (features x batch) for mlp input
    x = input + sa_out_2d
    mlp_out = tf.mlp(tf.norm_2(x))
    x = x + mlp_out

    return x
end

### encoding ?? not sure if this is needed...

# struct PosEnc
#     pe::Array{Float32}
# end

# function gen_posenc(embed_dim::Int, max_len::Int) # actually idk how to implement this really
# end

# encoder = Flux.Chain(
#     Dense(input_size => embed_dim),
#     Dropout(0.1)
#     )

### need to define as input --> encoder --> transf --> output i think??? so this is the full model i think

struct Model
    input_head::Flux.Chain
    # pos encoder here
    transf::Flux.Chain
    output_head::Flux.Chain
end

function Model(;
    input_size::Int64,
    embed_size::Int64,
    n_layers::Int64,
    n_classes::Int64,
    n_heads::Int64,
    hidden_size::Int64,
    dropout_prob::Float64
    )

    input_head = Flux.Chain(Flux.Dense(input_size => embed_size))
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

    return Model(input_head, transformer, output_head)
end

Flux.@functor Model

function (model::Model)(x::Float32Matrix2DType)
    x = model.input_head(x)
    # x = model.pos_encoder(x)
    x = model.transf(x)
    x = model.output_head(x)
end

### splitting data

function split_data(X::Matrix{Float32}, y::Vector{Int64}, test_ratio::Float64)
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
# TODO: crossentropy vs. mae...?
loss(model, x, y) = Flux.logitcrossentropy(model(x), y)
# loss = logitcrossentropy(model(x), y) + α * sum(p -> sum(abs2, p), params(model)) # L2 regularization
train_dataloader = Flux.DataLoader((X_train, y_train), batchsize=batch_size)

train_losses = Float32[]
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
    push!(train_losses, epoch_losses[end])
    println("epoch $epoch, train loss: $(train_losses[end])")
end