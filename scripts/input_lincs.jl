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

### filter for untreated cell lines
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

### on all cell line profiles (trt + untrt)
x = LincsProject.create_filter(lincs_data, Dict(
    :qc_pass => [Symbol("1")], 
    :pert_type => [:ctl_untrt, :ctl_vehicle, :trt_cp] # , = or
    ))

filtered_lincs_all = lincs_data[x]
filtered_expr_all = lincs_data.expr[:, x]
df = DataFrame(transpose(filtered_expr_all), :auto)
insertcols!(df, 1, :cell_line => filtered_lincs_all.cell_iname)
col_names = ["cell_line"; lincs_data.gene.gene_symbol]
rename!(df, col_names)
CSV.write("data/all_cellline_geneexpr.csv", df)

####################################################################################################################

#TODO: just run the below code to ensure accuracy/loss is ok when using both untrt and trt cell lines
    #also make sure that it saves into /home/golem/scratch/chans/lincs/data/plots/trt_and_untrt

### MLP

using Flux, Random, OneHotArrays, CategoricalArrays, ProgressMeter, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(0)

# @time df = CSV.read("data/cellline_geneexpr.csv", DataFrame)
@time df = CSV.read("data/all_cellline_geneexpr.csv", DataFrame)

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
Plots.savefig("data/plots/trt_and_untrt/trainval_loss.png")

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
CairoMakie.save("data/plots/trt_and_untrt/conf_matrix.png", f)
f

# hierarchical clustering
using Distances, Clustering, StatsPlots

data = Matrix(df[:, 2:end-1])

cell_line_names = df.cell_line
unique_cell_lines = unique(cell_line_names)
mean_profiles = zeros(size(data, 2), length(unique_cell_lines))

for (i, cell_line) in enumerate(unique_cell_lines)
    indices = findall(x -> x == cell_line, cell_line_names)
    mean_profiles[:, i] = mean(data[indices, :], dims=1)
end

distances = pairwise(Euclidean(), mean_profiles)

result = hclust(distances, linkage=:average)

dend2 = StatsPlots.plot(result, 
                xrotation=90, 
                xticks=(1:length(unique_cell_lines), unique_cell_lines[result.order]),
                title="Cell Line Clustering",
                size=(2400, 1000),
                tickfont=font(8),
                titlefont=font(24))  
StatsPlots.savefig("data/plots/trt_and_untrt/dendrogram.png")

hier_order = unique_cell_lines[result.order]

# heatmap sorted

order_mapping = Dict(name => i for (i, name) in enumerate(hier_order))
reorder_indices = [findfirst(==(name), unique_cell_lines) for name in hier_order]
sorted_matrix = matrix[reorder_indices, reorder_indices]

f = CairoMakie.Figure(size=(1600, 1600))
ax = CairoMakie.Axis(
    f[1, 1], 
    xlabel="cell line (n=$num_classes)", 
    ylabel="cell line (n=$num_classes)",
    title="accuracy: $accuracy%",
    xticks=(1:length(hier_order), hier_order),
    yticks=(1:length(hier_order), hier_order),
    xticklabelrotation=90,
    xticklabelsize=8,
    yticklabelsize=8,
    )
hm = CairoMakie.heatmap!(ax, log10.(sorted_matrix .+ 1))
CairoMakie.Colorbar(f[1, 2], hm)
CairoMakie.save("data/plots/trt_and_untrt/sorted_conf_matrix.png", f)


####################################################################################################################
####################################################################################################################
####################################################################################################################


