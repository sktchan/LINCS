using LincsProject, DataFrames, CSV, Dates, JSON

# import lea's processing code
push!(LOAD_PATH, "/src")
using .CompoundEmbeddings
using .GeneEmbeddings
using .DataCoupling
using .DataSplit

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

o = LincsProject.create_filter(lincs_data, Dict(
    :qc_pass => [Symbol("1")], 
    :pert_type => [:ctl_untrt, :ctl_vehicle] # , = or
    ))


# get untreated gene expression profiles (978 landmark genes) that passed quality control for various cell lines. (lea's project)
#= Returns a df of 979 columns: cell line, untreated expression profile (978 genes).
If several untrt profiles are available for a given cell line, they are all stored in the returned df. =#
untrt_profiles = get_untreated_profiles(lincs_data)
CSV.write("data/untrt_profiles.csv", untrt_profiles, header=true)

result_dir = "/home/golem/scratch/chans/lincs/output"
if !isdir(result_dir)
    mkpath(result_dir)
end


# ### convert genes into tokens
# gene_names = names(untrt_profiles)[2:end]
# gene_token_dict = Dict(gene => i for (i, gene) in enumerate(gene_names))
# output_path = "/home/golem/scratch/chans/lincs/data/gene_token_dict.json"
# open(output_path, "w") do f
#     JSON.print(f, gene_token_dict)
# end



### ranking to do later

