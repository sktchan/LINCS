#=
in julia REPL, check:
registry status
if LabRegistry not in list, do:
registry add git@github.com:lemieux-lab/LabRegistry.git
if error, do:
ENV["JULIA_PKG_USE_CLI_GIT"] = true
=#


using LincsProject, DataFrames, CSV

# import lea's code
push!(LOAD_PATH, "/src")
# import CompoundEmbeddings as CE
# import GeneEmbeddings as GE
# import DataCoupling as DC
# import DataSplit as DS
using .CompoundEmbeddings
using .GeneEmbeddings
using .DataCoupling
using .DataSplit


### from carl, if not using LincsProject package:
# include("/home/golem/scratch/munozc/DDPM/LINCS_data_explorer/SETUP/Lincs.jl")
# using .Lincs: Data
# @time data = Data("/home/golem/scratch/munozc/DDPM/Data/out_20M_4.h5")


### load in beta data via terminal
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
untrt_profiles = get_untreated_profiles(lincs_data)
CSV.write("data/untrt_profiles.csv", untrt_profiles, header=true)

result_dir = "/home/golem/scratch/chans/lincs/results"
if !isdir(result_dir)
    mkpath(result_dir)
end

# LINCS data filtering criteria:
qc = Symbol("1")
ref_cl = Symbol("MCF7")
perturbation = Symbol("trt_cp")
exposure_time = Symbol("24 h")
dose =  Symbol("10 uM")
p_max = 1

# remove space between value and unit: 
t = replace(String(exposure_time), r"\s+" => "") 
d = replace(String(dose), r"\s+" => "")

run_id = "ref_cl_$(ref_cl)_time_$(t)_dose_$(d)"
run_dir = result_dir * "/" * run_id


# get landmark gene signatures (no need for gf project)
frogs_path = "/home/golem/scratch/chans/lincs/FRoGS"
frogs_gene_embedding_file = frogs_path * "/gene_vec_go_256.csv"
frogs_gene_to_id_file = frogs_path * "/term2gene_id.csv"
@time gene_to_gene_embedding_dict = GE.create_gene_to_gene_embedding_dict(frogs_gene_to_id_file, frogs_gene_embedding_file)
# NB: some landmark genes are lost (no FRoGS gene embedding available).

# filtering criteria:
ref_criteria = Dict{Symbol, Symbol}(:qc_pass => qc, 
                                    :cell_iname => ref_cl, 
                                    :pert_type => perturbation, 
                                    :pert_itime => exposure_time, 
                                    :pert_idose => dose)

target_criteria = Dict{Symbol, Symbol}(:qc_pass => qc, 
                                        :pert_type => perturbation, 
                                        :pert_itime => exposure_time, 
                                        :pert_idose => dose)
# get all target cell lines: 
target_cell_lines = DC.get_target_cell_lines(lincs_data, ref_criteria, target_criteria)

# create data and splitted_data folders in run_dir:
if !isdir(run_dir * "/data")
    mkpath(run_dir * "/data")
end
splitted_data_dir = run_dir * "/splitted_data"
if !isdir(splitted_data_dir)
    mkpath(splitted_data_dir)
end

# decide which data to concatenate to create the input of the model:
include_vec = Vector{Symbol}()
push!(include_vec, :compound_embeddings)  # push if include flag == true
push!(include_vec, :delta_ref_profiles)
push!(include_vec, :target_neutral_profile)
push!(include_vec, :target_gene_embeddings)

for target_cl in target_cell_lines

    # prepare the data:
    data_file = run_dir * "/data/data_ref_$(ref_cl)_target_$(target_cl)_dose_$(d)_time_$(t).h5"
    ## if data file does not exist, create it:
    if !isfile(data_file)
        tmp_target_criteria = copy(target_criteria)
        tmp_target_criteria[:cell_iname] = target_cl
        data = create_data_struct(lincs_data, 
                                gene_to_gene_embedding_dict,
                                ref_criteria, 
                                tmp_target_criteria, 
                                p_max)
        save_data_to_hdf5(data_file, data)
    ## if data file exists, load it:
    else 
        data = DC.load_data_from_hdf5(data_file)
    end

    # split data between train and test:
    if length(data.compounds) > 1  # We need at least two compounds to split the data
        splitted_data_file = splitted_data_dir * "/splitted_data_ref_$(ref_cl)_target_$(target_cl)_dose_$(d)_time_$(t).h5"
        if !isfile(splitted_data_file)
            splitted_data = DS.SplittedData(data, include_vec) # out of memory error here from concatenation
            save_splitted_data_to_hdf5(splitted_data_file, splitted_data)
        end
    end

end

# load splitted data:
splitted_data_file = splitted_data_dir * "/splitted_data_ref_$(ref_cl)_target_PC3_dose_$(d)_time_$(t).h5"
splitted_data = DS.load_splitted_data_from_hdf5(splitted_data_file)



### load in geneformer
using PyCall # must have ENV["PYTHON"] = "/u/chans/anaconda3/envs/venv/bin/python" then PyCall rebuilt!!!
geneformer = pyimport("geneformer") # use python -m pip list to check pkgs installed in venv
classifier = geneformer.Classifier

