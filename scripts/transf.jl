using CSV, DataFrames, Statistics, Flux, CUDA, Random, CategoricalArrays, ProgressBars
using Transformers, Transformers.Layers, Transformers.TextEncoders
CUDA.device!(1)

@time df = CSV.read("data/cellline_geneexpr.csv", DataFrame)
matrix = Float32.(Matrix(df[:, 2:end])')

df.cell_line = categorical(df.cell_line) 
df.cell_line_encoded = levelcode.(df.cell_line)

X = Float32.(Matrix(df[:, 2:end-1])') # flux requires format (features × samples)
y = df.cell_line_encoded

num_classes = length(levels(df.cell_line))
input_size = size(X, 1)
cell_lines = unique(df.cell_line) 

y_oh = Flux.onehotbatch(y, 1:num_classes)

test_ratio = 0.1
n = size(X, 2)

indices = shuffle(1:n)
test_size = floor(Int, n * test_ratio)
train_size = n - test_size

test_indices = indices[1:test_size]
train_indices = indices[test_size+1:end]

X_train, y_train = X[:, train_indices], y_oh[:, train_indices]
X_test, y_test = X[:, test_indices], y_oh[:, test_indices]

batch_size = 256
train_data = Flux.DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
test_data = Flux.DataLoader((X_test, y_test), batchsize=batch_size)


###############################################################################################


### baseline transformer, no token embeddings
hidden_size = 512
head_num = 8
head_size = hidden_size ÷ head_num
ffn_dim = 4 * hidden_size
num_layers = 2

input_proj = Dense(input_size => hidden_size) |> gpu
encoder = Transformer(TransformerBlock, num_layers, head_num, hidden_size, head_size, ffn_dim) |> gpu

classifier = Chain(
    LayerNorm(hidden_size),
    Dense(hidden_size => hidden_size ÷ 2, gelu),
    Dense(hidden_size ÷ 2 => num_classes)
) |> gpu

function model(X)
    X = reshape(X, size(X, 1), 1, size(X, 2)) # (batch × genes → batch × 1 × genes)
    # X = reshape(X, size(X, 1), size(X, 2), 1) ????????

    proj = input_proj(X)
    proj = permutedims(proj, (2, 1, 3)) # to (seq_len × batch × hidden_size)
    encoded = encoder(proj).hidden_state

    pooled = mean(encoded, dims=1)
    pooled = dropdims(pooled, dims=1)
    
    return classifier(pooled)
end |> gpu

# training
loss_fn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y))

lr = 3e-4
opt = Adam(lr)
params = Flux.params(input_proj, encoder, classifier)

train_losses = Float32[]
test_losses = Float32[]
function train!(model, train_data, test_data, epochs=10)
    for epoch in 1:epochs
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0
        
        for (X, y) in ProgressBar(train_data)
            
            grads = gradient(params) do
                ŷ = model(X)
                loss = loss_fn(ŷ, y)
                return loss
            end
            
            Flux.update!(opt, params, grads)
            
            ŷ = model(X)
            train_loss += loss_fn(ŷ, y)
            train_acc += accuracy(ŷ, y)
            num_batches += 1
        end
        
        avg_train_loss = train_loss / num_batches
        avg_train_acc = train_acc / num_batches
        
        test_loss, test_acc = evaluate(model, test_data, num_classes)
        
        @info "Epoch $epoch" train_loss=avg_train_loss train_acc=avg_train_acc test_loss=test_loss test_acc=test_acc
    end
end

function evaluate(model, data, num_classes)
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for (X, y) in data
        y_onehot = Flux.onehotbatch(y, 1:num_classes)
        ŷ = model(X)
        
        total_loss += loss_fn(ŷ, y_onehot)
        total_acc += accuracy(ŷ, y_onehot)
        num_batches += 1
    end
    
    return (total_loss / num_batches, total_acc / num_batches)
end

# add per epoch here
train!(model, train_data, test_data, 10)


###############################################################################################


### w/ token embeddings
num_genes = 978
embedding_dim = 32

gene_embedding = Embed(hidden_size, len(num_genes)) |> gpu
pos_embed = SinCosPositionEmbed(num_genes, embedding_dim) |> gpu
encoder = Transformer(TransformerBlock, 2, head_num, hidden_size, hidden_size÷head_num, 4hidden_size) |> gpu
classifier = Chain(Dense(hidden_size => num_classes), softmax) |> gpu

function model(X)
    gene_ids = 1:num_genes
    emb = gene_embedding(gene_ids)  # (num_genes × embedding_dim)
    scaled = X * emb'  # (batch × num_genes) × (num_genes × embedding_dim)

    scaled += pos_embed(gene_ids)

    encoded = encoder(permutedims(scaled, (2, 1, 3))).hidden_state
    pooled = mean(encoded, dims=1)
    return classifier(dropdims(pooled, dims=1))
end


###############################################################################################


### w/ binning
labels = ["<unk>", "<s>", "</s>", "low", "medium", "high"] # low/medium/high exp level bins
startsym = "<s>"
endsym = "</s>"
unksym = "<unk>"

textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym)

# since range is 0-15, with median/mean being around 8.4
low_threshold = 6.5
medium_threshold = 9.5

function encode_expression_data(expression_data)
    token_sequence = []
    for expr_value in expression_data
        if expr_value < low_threshold
            push!(token_sequence, "low")
        elseif expr_value < medium_threshold
            push!(token_sequence, "medium")
        else
            push!(token_sequence, "high")
        end
    end
    return encode(textenc, join(token_sequence, ' '))
end

nhead = 8
dimhead = 64
dimhidden = 512
dimffn = 2048
num_layers = 4
num_classes = unique(df.cell_line)

exp_embed = encode_expression_data(matrix)


###############################################################################################


### autoencoder


###############################################################################################


### variational graph autoencoder / VAE ??
