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

# result_dir = "/home/golem/scratch/chans/lincs/output"
# if !isdir(result_dir)
#     mkpath(result_dir)
# end


### MLP

using Flux, Random, OneHotArrays, CategoricalArrays, ProgressMeter, CUDA
CUDA.device!(1)

@time df = CSV.read("data/cellline_geneexpr.csv", DataFrame)

# encoding cell line to numerical vals
df.cell_line = categorical(df.cell_line) 
df.cell_line_encoded = levelcode.(df.cell_line)

X = Float32.(Matrix(df[:, 2:end-1])') # flux requires format (features × samples)
y = df.cell_line_encoded

num_classes = length(levels(df.cell_line))
input_size = size(X, 1)

model = Chain(
    Dense(input_size, 128, relu),
    Dense(128, 64, relu),
    Dense(64, num_classes),
    softmax
) |> gpu

n_epochs = 100
n_batches = 1
loss(x, y) = Flux.logitcrossentropy(model(x), y)
opt = Flux.setup(Adam(), model)

y_oh = Flux.onehotbatch(y, 1:num_classes)
train_data = Flux.DataLoader((X, y_oh), batchsize=n_batches, shuffle=true) |> gpu

losses = []
@showprogress for epoch in 1:n_epochs
    loss_val = 0.0
    for (x, y) in train_data
        loss_val, grads = Flux.withgradient(model) do m
            loss(x, y)
        end
        Flux.update!(opt, model, grads[1])
    end
    push!(losses, loss_val)
    if epoch % 10 == 0
        println("Epoch $epoch, Loss: $(losses[end])")
    end
end