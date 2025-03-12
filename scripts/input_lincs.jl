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

# get untreated gene expression profiles (978 landmark genes) that passed quality control for various cell lines. (lea's project)
#= Returns a df of 979 columns: cell line, untreated expression profile (978 genes).
If several untrt profiles are available for a given cell line, they are all stored in the returned df. =#
untrt_profiles = get_untreated_profiles(lincs_data)
CSV.write("data/untrt_profiles.csv", untrt_profiles, header=true)

result_dir = "/home/golem/scratch/chans/lincs/output"
if !isdir(result_dir)
    mkpath(result_dir)
end


### convert genes into tokens
gene_names = names(untrt_profiles)[2:end]
gene_token_dict = Dict(gene => i for (i, gene) in enumerate(gene_names))
output_path = "/home/golem/scratch/chans/lincs/data/gene_token_dict.json"
open(output_path, "w") do f
    JSON.print(f, gene_token_dict)
end

### ranking?

### convert structure of input to geneformer input




### load in geneformer <!< (PYCALL ERROR - USE OUTPUT_GF.PY INSTEAD) >!>
using PyCall # must have ENV["PYTHON"] = "/u/chans/anaconda3/envs/venv/bin/python" then PyCall rebuilt!!!
geneformer = pyimport("geneformer") # use python -m pip list to check pkgs installed in venv

current_date = Dates.now()
datestamp = "$(Dates.year(current_date))$(lpad(Dates.month(current_date), 2, '0'))$(lpad(Dates.day(current_date), 2, '0'))_$(lpad(Dates.hour(current_date), 2, '0'))$(lpad(Dates.minute(current_date), 2, '0'))"

output_prefix = "gene_class_test"
output_dir = "/home/golem/scratch/chans/lincs/output/$(datestamp)"
if !isdir(output_dir)
    mkpath(output_dir)
end

classifier = geneformer.Classifier(classifier="gene",
                                    gene_class_dict = gene_token_dict,
                                    max_ncells = 10_000,
                                    freeze_layers = 4,
                                    num_crossval_splits = 5,
                                    forward_batch_size=200,
                                    nproc=16)

classifier.prepare_data(input_data_file="/path/to/gc-30M_sample50k.dataset",
                        output_directory=output_dir,
                        output_prefix=output_prefix)

all_metrics = classifier.validate(model_directory="/home/golem/scratch/chans/lincs/Geneformer/geneformer",
                                    prepared_input_data_file="$(output_dir)/$(output_prefix)_labeled.dataset",
                                    id_class_dict_file="$(output_dir)/$(output_prefix)_id_class_dict.pkl",
                                    output_directory=output_dir,
                                    output_prefix=output_prefix)