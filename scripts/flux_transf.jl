using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressMeter, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(0)

@time df = CSV.read("data/all_cellline_geneexpr.csv", DataFrame) # trt and untrt

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

### building model

const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}

struct Transf
    mhsa::Flux.MultiHeadAttention
    norm_1::Flux.LayerNorm
    mlp::Flux.Chain
    norm_2::Flux.LayerNorm
    dropout::Flux.Dropout
end

function gen_transf(embed_dim::Int, hidden_dim::Int; n_heads=8, dropout_prob=0.1)

    mhsa = Flux.MultiHeadAttention(embed_dim, nheads=n_heads, dropout_prob=dropout_prob)
    norm1 = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim)
    )
    norm2 = Flux.LayerNorm(embed_dim)
    dropout = Flux.Dropout(dropout_prob)

    Transf(mhsa, norm1, mlp, norm2, dropout)

end

Flux.@functor Transf

function (model::Transf)(input::Float32Matrix2DType)
    sa_out = model.mhsa(model.norm_1(input))
    x = x + model.dropout(sa_out)
    mlp_out = model.mlp(model.norm_2(x))
    x = x + model.dropout(mlp_out)
    return x
end

### encoding ?? not sure if this is needed...

# struct PosEnc
#     pe::Array{Float32}
# end

# function gen_posenc(embed_dim::Int, max_len::Int) # actually idk how to implement this really
# end

### if doing encoding, need to implement transf: input --> encoder --> transformer sa layer --> mlp layer as one struct + fxn

    # insert here

### splitting data

test_ratio = 0.1
n = size(X, 2)

indices = shuffle(1:n) 

test_size = floor(Int, n * test_ratio) 
train_size = n - test_size

test_indices = indices[1:test_size]
train_indices = indices[test_size+1:end]

X_train, y_train = X[:, train_indices], y[train_indices] |> gpu
X_test, y_test = X[:, test_indices], y[test_indices] |> gpu

### training

batch_size = 32
n_epochs = 10
embed_dim = 256
hidden_dim = 1024
n_heads = 8
n_layers = 6
α = 0.001
opt = Adam(3e-4)
train_dataloader = Flux.DataLoader((X_train, y_train), batchsize=batch_size)

# TODO: idk how to define the model, not sure if need to define another struct + fxn + fxn for multiple transf blocks? or if this is sufficient...

# model = Chain(
#     Dense(input_size => embed_dim),
#     Transf(),
#     Dense(embed_dim => num_classes)
# ) |> gpu

# TODO: crossentropy vs. mae?

# loss = logitcrossentropy(model(x), y) + α * sum(p -> sum(abs2, p), params(model))

for epoch in 1:n_epochs

  for (x, y) in train_dataloader # x dimensions = 978 * batch_size. y dimensions = 1 * batch_size
    
    x, y = x |> gpu, y |> gpu
    
    loss, grads = Flux.withgradient(model) do m
      Flux.mae(m(x), y) + α * sum(p -> sum(abs2, p), ps) # L2 regularization
    end
  
    Flux.update!(opt, model, grads[1])

    train_loss =  Flux.mae(model(X_train), Y_train)
    test_loss = Flux.mae(model(X_test), Y_test) 
    
    @info "epoch: $epoch - train loss: $train_loss, test loss: $test_loss"
    
    end 
end