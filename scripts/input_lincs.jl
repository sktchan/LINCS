using LincsProject, DataFrames, CSV, Dates, JSON

# load in beta data via terminal if not present
#=
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_all_n3026460x12328.gctx
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt
=#

### load in lincs dataset (this is the one filtered for Lea's project)
@time lincs_data = LincsProject.Lincs("data/", "level3_beta_all_n3026460x12328.gctx", "data/lincs_data.jld2")

#=
struct Lincs
    expr::Matrix{Float32}
    gene::DataFrame
    compound::DataFrame
    inst::DataFrame ## Converted to identifiers only, use inst_si to convert back
end
=#

o = LincsProject.create_filter(lincs_data, Dict(
    :qc_pass => [Symbol("1")], 
    :pert_type => [:ctl_untrt, :ctl_vehicle] # , = or
    ))

# creating df of cell line x gene expression 
filtered_lincs = lincs_data[o]
filtered_expr = lincs_data.expr[:, o]
df = DataFrame(transpose(filtered_expr), :auto)
insertcols!(df, 1, :cell_line => filtered_lincs.cell_iname)
col_names = ["cell_line"; lincs_data.gene.gene_symbol]
rename!(df, col_names)
CSV.write("data/cellline_geneexpr.csv", df)


####################################################################################################################


### MLP

using Flux, Random, OneHotArrays, CategoricalArrays, ProgressMeter, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(0)

@time df = CSV.read("data/cellline_geneexpr.csv", DataFrame)

# encoding cell line to numerical vals
df.cell_line = categorical(df.cell_line) 
df.cell_line_encoded = levelcode.(df.cell_line)

X = Float32.(Matrix(df[:, 2:end-1])') # flux requires format (features × samples)
y = df.cell_line_encoded

num_classes = length(levels(df.cell_line))
input_size = size(X, 1)

y_oh = Flux.onehotbatch(y, 1:num_classes)
# X_mean = mean(X, dims=2)
# X_std = std(X, dims=2)
# X_norm = (X .- X_mean) ./ (X_std .+ 1e-6) 
# NO DIVIDING BY STD DEV (causes values to be too great after subtracting the mean)
# is this in all gene expr values?

model = Chain(
    Dense(input_size, 256, relu),
    Dense(256, num_classes)
) |> gpu

n_epochs = 100
n_batches = 128
loss(model, x, y) = Flux.logitcrossentropy(model(x), y)
opt = Flux.setup(Adam(0.001), model)

# if val

val_ratio = 0.2 
test_ratio = 0.1 
n = size(X, 2)

indices = shuffle(1:n) 

val_size = floor(Int, n * val_ratio)
test_size = floor(Int, n * test_ratio) 
train_size = n - val_size - test_size

test_indices = indices[1:test_size]
val_indices = indices[test_size+1:test_size+val_size]
train_indices = indices[test_size+val_size+1:end]

X_train, y_train = X[:, train_indices], y_oh[:, train_indices]
X_val, y_val = X[:, val_indices], y_oh[:, val_indices]
X_test, y_test = X[:, test_indices], y_oh[:, test_indices]

train_data = Flux.DataLoader((X_train, y_train), batchsize=n_batches, shuffle=true)
val_data = Flux.DataLoader((X_val, y_val), batchsize=n_batches, shuffle=true)
test_data = Flux.DataLoader((X_test, y_test), batchsize=n_batches, shuffle=true) 


train_losses = Float32[] 
val_losses = Float32[]
# always try to define the type of array! (if needs to be ultra-fast, use zeroes(type, dim) and reassign a[i] = i rather than push!(a, i))
# @code_warntype f(x) will show you where the type instability is

@showprogress for epoch in 1:n_epochs
    Flux.trainmode!(model)
    epoch_losses = Float64[]
    
    for (x, y) in train_data
        x_gpu, y_gpu = gpu(x), gpu(y)
        loss_val, grads = Flux.withgradient(model) do m
            # loss(model, x_gpu, y_gpu)
            Flux.logitcrossentropy(m(x_gpu), y_gpu)
        end
        Flux.update!(opt, model, grads[1])
        push!(epoch_losses, loss_val)
    end
    push!(train_losses, mean(epoch_losses))
    
    Flux.testmode!(model)
    val_epoch_losses = Float64[]
    
    for (x, y) in val_data
        x_gpu, y_gpu = gpu(x), gpu(y)
        # val_loss = loss(model, x_gpu, y_gpu)
        val_loss = Flux.logitcrossentropy(model(x_gpu), y_gpu)
        push!(val_epoch_losses, val_loss)
    end
    
    push!(val_losses, mean(val_epoch_losses))
    
    println("epoch $epoch, train loss: $(train_losses[end]), val loss: $(val_losses[end])")
end


### evaluation metrics

# loss plot
Plots.plot(1:n_epochs, train_losses, label="training loss", xlabel="epoch", ylabel="loss", 
     title="training vs validation loss", lw=2)
Plots.plot!(1:n_epochs, val_losses, label="validation Loss", lw=2)
Plots.savefig("data/trainval_loss.png")

# compute on test set - for conf matrix + heatmap
output = model(gpu(X_test))
maxes = maximum(output, dims=1)
pred_test = Int.(output .== maxes)'
true_test = Int.(y_oh[:, test_indices]')

matrix = cpu(pred_test)'true_test # conf matrix
accuracy = tr(matrix) / sum(matrix) * 100

f = CairoMakie.Figure(size=(800, 800))
ax = CairoMakie.Axis(
    f[1, 1], 
    xlabel="cell line (n=$num_classes)", 
    ylabel="cell line (n=$num_classes)",
    title="accuracy: $accuracy%")
conf_matrix = CairoMakie.heatmap!(ax, log10.(matrix .+ 1))
CairoMakie.Colorbar(f[1, 2], conf_matrix, label="log10(count)")
CairoMakie.save("data/conf_matrix.png", f)
f








####################################################################################################################
####################################################################################################################
####################################################################################################################


# if filtering cell lines w/ <10 samples

min_samples_per_class = 10
class_counts = [count(==(i), y) for i in 1:num_classes]
valid_classes = findall(class_counts .>= min_samples_per_class)

valid_samples = [y[i] in valid_classes for i in 1:length(y)]
X_filt = X[:, valid_samples]
y_filt = y[valid_samples]

num_filtered_classes = length(valid_classes)

y_mapping = Dict(valid_classes[i] => i for i in 1:length(valid_classes))
y_remapped = [y_mapping[y_filtered[i]] for i in 1:length(y_filtered)]

y_filt_oh = Flux.onehotbatch(y_remapped, 1:num_filtered_classes)
X_filt_mean = mean(X_filt, dims=2)
X_filt_std = std(X_filt, dims=2)
X_filt_norm = (X_filt .- X_filt_mean) ./ (X_filt_std .+ 1e-6)

model = Chain(
    Dense(input_size, 1024, relu),
    BatchNorm(1024),
    Dropout(0.3),
    Dense(1024, 512, relu),
    BatchNorm(512),
    Dropout(0.3),
    Dense(512, 256, relu),
    BatchNorm(256),
    Dropout(0.3),
    Dense(256, num_classes),
    softmax
) |> gpu

val_ratio = 0.2
n_filt = size(X_filt, 2)
indices_filt = shuffle(1:n_filt)
val_filt_size = floor(Int, n_filt * val_ratio)

val_filt_indices = indices_filt[1:val_filt_size]
train_filt_indices = indices_filt[val_filt_size+1:end]

X_filt_train = X_filt_norm[:, train_filt_indices]
y_filt_train = y_filt_oh[:, train_filt_indices]
X_filt_val = X_filt_norm[:, val_filt_indices]
y_filt_val = y_filt_oh[:, val_filt_indices]

n_epochs = 100
n_batches = 64
loss(x, y) = Flux.logitcrossentropy(model(x), y)
opt = Flux.setup(Adam(0.0001), model)

train_filt_data = Flux.DataLoader((X_filt_train, y_filt_train), batchsize=n_batches, shuffle=true)
val_filt_data = Flux.DataLoader((X_filt_val, y_filt_val), batchsize=n_batches, shuffle=false)

train_filt_losses = []
val_filt_losses = []

@showprogress for epoch in 1:n_epochs
    Flux.trainmode!(model)
    epoch_losses = Float64[]
    for (x, y) in train_filt_data
        x_gpu, y_gpu = gpu(x), gpu(y)
        loss_val, grads = Flux.withgradient(model) do m
            loss(x_gpu, y_gpu)
        end
        Flux.update!(opt, model, grads[1])
        push!(epoch_losses, loss_val)
    end
    push!(train_filt_losses, mean(epoch_losses))
    Flux.testmode!(model)
    val_epoch_losses = Float64[]
    for (x, y) in val_filt_data
        x_gpu, y_gpu = gpu(x), gpu(y)
        val_loss = loss(x_gpu, y_gpu)
        push!(val_epoch_losses, val_loss)
    end
    push!(val_filt_losses, mean(val_epoch_losses))
    if epoch % 10 == 0
        println("epoch $epoch, train loss: $(train_filt_losses[end]), val loss: $(val_filt_losses[end])")
    end
end








