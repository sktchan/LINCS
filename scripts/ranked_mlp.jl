using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, Plots, CairoMakie, LinearAlgebra
CUDA.device!(0)

# data = load("data/lincs_untrt_data.jld2")["filtered_data"] 
data = load("data/lincs_trt_untrt_data.jld2")["filtered_data"] # trt and untrt data

### ranking encodings across columns

ranked_data = data.expr 
n_genes, n_samples = size(ranked_data)

for gene_idx in 1:n_genes
    gene_values = ranked_data[gene_idx, :]
    ranked_data[gene_idx, :] .= ordinalrank(-gene_values) ./ n_samples
end

### prev: encoding labels (cell lines) from string -> int

unique_cell_lines = unique(data.inst.cell_iname)
label_dict = Dict(name => i for (i, name) in enumerate(unique_cell_lines))
integer_labels = [label_dict[name] for name in data.inst.cell_iname]

n_classes = length(unique_cell_lines)
n_features = size(ranked_data, 1)

X = ranked_data # 978 x 100425
y = integer_labels # 100425 x 1

y_oh = Float32.(Flux.onehotbatch(y, 1:n_classes))

# if val

function split_data(X, y_oh; val_ratio=0.2, test_ratio=0.1)
    n = size(X, 2)
    indices = shuffle(1:n)
    
    val_size = floor(Int, n * val_ratio)
    test_size = floor(Int, n * test_ratio)

    test_indices  = indices[1:test_size]
    val_indices   = indices[test_size+1 : test_size+val_size]
    train_indices = indices[test_size+val_size+1 : end]
    
    X_train, y_train = X[:, train_indices], y_oh[:, train_indices]
    X_val,   y_val   = X[:, val_indices], y_oh[:, val_indices]
    X_test,  y_test  = X[:, test_indices], y_oh[:, test_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
end

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y_oh)

### training

model = Chain(
    Dense(n_features, 256, relu; init=Flux.glorot_uniform),
    Dropout(0.3),
    Dense(256, 128, relu; init=Flux.glorot_uniform),
    Dropout(0.3),
    Dense(128, n_classes; init=Flux.glorot_uniform)
) |> gpu

n_epochs = 100
batch_size = 128
loss(model, x, y) = Flux.logitcrossentropy(model(x), y)
opt = Flux.setup(Adam(0.0001), model)

train_data = Flux.DataLoader((X_train, y_train), batchsize=n_batches, shuffle=true)
val_data = Flux.DataLoader((X_val, y_val), batchsize=n_batches)
test_data = Flux.DataLoader((X_test, y_test), batchsize=n_batches, shuffle=true) 

train_losses = Float32[] 
val_losses = Float32[]

# for epoch in ProgressBar(1:n_epochs)
#     Flux.trainmode!(model)
#     epoch_losses = Float64[]
    
#     for (x, y) in train_data
#         x_gpu, y_gpu = gpu(x), gpu(y)
#         loss_val, grads = Flux.withgradient(model) do m
#             # loss(model, x_gpu, y_gpu)
#             Flux.logitcrossentropy(m(x_gpu), y_gpu)
#         end
#         Flux.update!(opt, model, grads[1])
#         loss_val = Flux.logitcrossentropy(model(x_gpu), y_gpu)
#         push!(epoch_losses, loss_val)
#     end
#     push!(train_losses, mean(epoch_losses))
    
#     Flux.testmode!(model)
#     val_epoch_losses = Float64[]
    
#     for (x, y) in val_data
#         x_gpu, y_gpu = gpu(x), gpu(y)
#         # val_loss = loss(model, x_gpu, y_gpu)
#         val_loss = Flux.logitcrossentropy(model(x_gpu), y_gpu)
#         push!(val_epoch_losses, val_loss)
#     end
    
#     push!(val_losses, mean(val_epoch_losses))
# end

for epoch in ProgressBar(1:n_epochs)

    epoch_losses = Float32[]

    for start_idx in 1:batch_size:size(X_train, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train, 2))
        x_gpu = gpu(X_train[:, start_idx:end_idx])
        y_gpu = gpu(y_train[:, start_idx:end_idx])
        
        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, y_gpu)
        end
        Flux.update!(opt, model, grads[1])
        loss_val = loss(model, x_gpu, y_gpu)
        push!(epoch_losses, loss_val)
    end

    push!(train_losses, mean(epoch_losses))
    
    test_epoch_losses = Float32[]
    
    for start_idx in 1:batch_size:size(X_val, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_val, 2))
        x_gpu = gpu(X_val[:, start_idx:end_idx])
        y_gpu = gpu(y_val[:, start_idx:end_idx])

        test_loss_val = loss(model, x_gpu, y_gpu)
        push!(test_epoch_losses, test_loss_val)
    end
    push!(val_losses, mean(test_epoch_losses))

end


### evaluation metrics

# loss plot
Plots.plot(1:n_epochs, train_losses, label="training loss", xlabel="epoch", ylabel="loss", 
     title="training vs validation loss", lw=2)
Plots.plot!(1:n_epochs, val_losses, label="validation loss", lw=2)
Plots.savefig("plots/trt_and_untrt/ranked_mlp/trainval_loss.png")

# accuracy
X_test_gpu = gpu(X_test)
y_test_gpu = gpu(y_test)
test_output = model(X_test_gpu)

test_output_cpu = cpu(test_output)
y_test_cpu = cpu(y_test_gpu)
pred_labels = map(i -> argmax(test_output_cpu[:, i]), 1:size(test_output_cpu, 2))
true_labels = map(i -> argmax(y_test_cpu[:, i]), 1:size(y_test_cpu, 2))
acc = mean(pred_labels .== true_labels)

# plotting ROC/AUC
function roc_curve(scores, labels)
    n_pos = count(labels) # positives  (= TP + FN)
    n_neg = length(labels) - n_pos # negatives  (= FP + TN)

    # sort descending by score
    order = sortperm(scores; rev = true)
    tp = 0; fp = 0
    tpr = Float64[]; fpr = Float64[]

    current_score = Inf
    for idx in order
        s = scores[idx]
        if s != current_score
            push!(tpr, tp / n_pos)
            push!(fpr, fp / n_neg)
            current_score = s
        end
        if labels[idx]
            tp += 1
        else
            fp += 1
        end
    end
    push!(tpr, tp / n_pos)
    push!(fpr, fp / n_neg)
    return fpr, tpr
end

auc_trapz(x, y) = sum(diff(x) .* (y[1:end-1] .+ y[2:end])) / 2
proba = Flux.softmax(test_output_cpu; dims = 1)
true_int = [argmax(y_test_cpu[:, i]) for i in 1:size(y_test_cpu, 2)]
fpr_cls = Vector{Vector{Float64}}(undef, n_classes)
tpr_cls = similar(fpr_cls)
auc_cls = zeros(Float64, n_classes)

for c in 1:n_classes
    scores_c = vec(proba[c, :])
    labels_c = true_int .== c
    fpr_c, tpr_c = roc_curve(scores_c, labels_c)
    auc_c = auc_trapz(fpr_c, tpr_c)

    fpr_cls[c] = fpr_c
    tpr_cls[c] = tpr_c
    auc_cls[c] = auc_c
end

macro_auc = mean(auc_cls)
scores_flat = Float64[]
labels_flat = Bool[]

for (j, gt) in enumerate(true_int), c in 1:n_classes
    push!(scores_flat, proba[c, j])
    push!(labels_flat, gt == c)
end

fpr2, tpr2 = roc_curve(scores_flat, labels_flat)
micro_auc = auc_trapz(fpr2, tpr2)

# import exp MLP info in
using DelimitedFiles, Plots; gr()
roc1 = readdlm("roc1.csv", ',')
fpr1 = roc1[:,1]; tpr1 = roc1[:,2]

# since we can't fit it on the plot
function cutoff(fpr, tpr, maxpts=2000)
    if length(fpr) > maxpts
    step = ceil(Int, length(tpr) / maxpts)
    fpr_plot = fpr[1:step:end]
    tpr_plot = fpr[1:step:end]
    else
        fpr_plot = fpr
        tpr_plot = tpr
    end
    return fpr_plot, tpr_plot
end
exp_fpr, exp_tpr = cutoff(fpr1, tpr1)
rank_fpr, rank_tpr = cutoff(fpr2, tpr2)

# plotting
plt = Plots.plot(title  = "Cell Line Classification",
           xlabel = "False positive rate",
           ylabel = "True positive rate",
           size   = (800,400))

Plots.plot!(plt, exp_fpr, exp_tpr, lw = 3,
      label = "Raw Expression AUC = 0.999")
Plots.plot!(plt, rank_fpr, rank_tpr, lw = 3,
      label = "Ranked AUC = 0.998")
Plots.plot!(plt, [0,1], [0,1], linestyle = :dash, color = :black, label = false)
savefig(plt, "plots/untrt/roc_auc/roc_final.png")  



# # compute on test set - for conf matrix + heatmap
# output = model(gpu(X_test))
# maxes = maximum(output, dims=1)
# pred_test = Int.(output .== maxes)'

# all_sum = sum(cpu_pt, dims=1)[1,:]
# sort(all_sum)
# # true_test = Int.(y_oh[:, test_indices]')
# true_test = y_test

# matrix = cpu(pred_test)'true_test # conf matrix # FIXME: not working due to dimension issues in matrmult. fix if time :P
# accuracy = tr(matrix) / sum(matrix) * 100

# f = CairoMakie.Figure(size=(800, 800))
# ax = CairoMakie.Axis(
#     f[1, 1], 
#     xlabel="cell line (n=$n_classes)", 
#     ylabel="cell line (n=$n_classes)",
#     title="accuracy: $accuracy%")
# conf_matrix = CairoMakie.heatmap!(ax, log10.(matrix .+ 1))
# CairoMakie.Colorbar(f[1, 2], conf_matrix, label="log10(count)")
# CairoMakie.save("data/plots/trt_and_untrt/conf_matrix.png", f)
# f

# # hierarchical clustering
# using Distances, Clustering, StatsPlots

# data = Matrix(df[:, 2:end-1])

# cell_line_names = df.cell_line
# unique_cell_lines = unique(cell_line_names)
# mean_profiles = zeros(size(data, 2), length(unique_cell_lines))

# for (i, cell_line) in enumerate(unique_cell_lines)
#     indices = findall(x -> x == cell_line, cell_line_names)
#     mean_profiles[:, i] = mean(data[indices, :], dims=1)
# end

# distances = pairwise(Euclidean(), mean_profiles)

# result = hclust(distances, linkage=:average)

# dend2 = StatsPlots.plot(result, 
#                 xrotation=90, 
#                 xticks=(1:length(unique_cell_lines), unique_cell_lines[result.order]),
#                 title="Cell Line Clustering",
#                 size=(2400, 1000),
#                 tickfont=font(8),
#                 titlefont=font(24))  
# StatsPlots.savefig("data/plots/trt_and_untrt/dendrogram.png")

# hier_order = unique_cell_lines[result.order]

# # heatmap sorted

# order_mapping = Dict(name => i for (i, name) in enumerate(hier_order))
# reorder_indices = [findfirst(==(name), unique_cell_lines) for name in hier_order]
# sorted_matrix = matrix[reorder_indices, reorder_indices]

# f = CairoMakie.Figure(size=(1600, 1600))
# ax = CairoMakie.Axis(
#     f[1, 1], 
#     xlabel="cell line (n=$n_classes)", 
#     ylabel="cell line (n=$n_classes)",
#     title="accuracy: $accuracy%",
#     xticks=(1:length(hier_order), hier_order),
#     yticks=(1:length(hier_order), hier_order),
#     xticklabelrotation=90,
#     xticklabelsize=8,
#     yticklabelsize=8,
#     )
# hm = CairoMakie.heatmap!(ax, log10.(sorted_matrix .+ 1))
# CairoMakie.Colorbar(f[1, 2], hm)
# CairoMakie.save("data/plots/trt_and_untrt/sorted_conf_matrix.png", f)


########################################################################################################################