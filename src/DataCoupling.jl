module DataCoupling

using DataFrames, ElasticArrays, NamedArrays
using HDF5
using Random, Statistics

using LincsProject
# include("CompoundEmbeddings.jl")
# push!(LOAD_PATH, "src")
using CompoundEmbeddings

export Data
export create_data_struct
export save_data_to_hdf5, load_data_from_hdf5
export get_target_cell_lines


struct Data
    ref_cl::Symbol
    target_cl::Symbol
    compounds::Vector{Symbol}                 # n (nb of experiments)
    compound_embeddings::Matrix{Float32}      # 2048 * n
    delta_ref_profiles::Matrix{Float32}       # 978 * n (treated profile - average neutral profile for each experiment)
    target_neutral_profile::Matrix{Float32}   # 978 * 1. Average profile
    target_genes::Vector{String}              # length = g (nb of genes which have a FRoGS gene embedding)       
    target_gene_embeddings::Matrix{Float32}   # 256 * g
    target_gene_expressions::Vector{Float32}  # length = n * g --> g expressions for experiment 1, followed by g expressions for experiment 2, etc.
end


function Base.size(data::Data, include_vec::Vector{Symbol})
    m = 0
    for symb in include_vec
        m += size(getfield(data, symb), 1)
    end
    n = length(data.target_gene_expressions)
    return m, n
end


function save_data_to_hdf5(filename::String, data::Data)
    for field in fieldnames(Data)
        field_value = getfield(data, field)  
        if isa(field_value, Symbol) || isa(field_value, Vector{Symbol})
            h5write(filename, String(field), String.(field_value))
        else
            h5write(filename, String(field), field_value)
        end
    end
end


function load_data_from_hdf5(filename::String)::Data
    data_dict = Dict{Symbol, Any}()

    # Read the data from the HDF5 file
    for field in fieldnames(Data)
        field_value = HDF5.h5read(filename, String(field))
        data_dict[field] = field_value
    end

    # Return a new instance of Data_for_gene_expr_prediction
    return Data(Symbol(data_dict[:ref_cl]),
                Symbol(data_dict[:target_cl]),
                Symbol.(data_dict[:compounds]), 
                data_dict[:compound_embeddings], 
                data_dict[:delta_ref_profiles],
                data_dict[:target_neutral_profile],
                data_dict[:target_genes], 
                data_dict[:target_gene_embeddings], 
                data_dict[:target_gene_expressions])
end


function create_filter(lm_data::Lincs, criteria::Dict{Symbol, Symbol})
    # Returns a bit vector: 1 for experiments that satisfy all the criteria, else 0.
    filters = []
    for (k, v) in criteria
        f = lm_data[k, v]
        push!(filters, f)
    end
    return reduce(.&, filters)
end


function get_unique_filtered_compounds(lm_data::Lincs, criteria::Dict{Symbol, Symbol})
    bv = create_filter(lm_data, criteria) 
    filtered_experiments = lm_data[bv]
    cps = unique(filtered_experiments[:, :pert_id])
    return cps
end


function get_target_cell_lines(lm_data::Lincs, ref_criteria::Dict{Symbol, Symbol}, target_criteria::Dict{Symbol, Symbol})
    
    # Get cell lines with an available neutral profile:
    bv_neutral = lm_data.inst[:pert_type, :ctl_untrt]
    cl_neutral = unique(lm_data.inst[bv_neutral].cell_iname)
   
    # Filter experiments matching the target_criteria::
    bv1 = create_filter(lm_data, target_criteria)

    # Filter experiments with compounds also used to treat the reference cell line:
    ref_cps = get_unique_filtered_compounds(lm_data, ref_criteria)
    bv2 = lm_data[:pert_id, ref_cps]

    # Combine the two bit vectors:
    bv = bv1 .& bv2
    cl_treatment = unique(lm_data.inst[bv].cell_iname)

    # Get the intersection, and remove ref_cl:
    target_cell_lines = setdiff(intersect(cl_neutral, cl_treatment), [ref_criteria[:cell_iname]])
    
    return target_cell_lines
end


function get_average_neutral_profile(lm_data::Lincs, cell_line::Symbol)
    # Returns the average neutral profile of the cell line.
    neutral_criteria = Dict{Symbol, Symbol}(:qc_pass => Symbol("1"), :pert_type => Symbol("ctl_untrt"), :cell_iname => cell_line)
    neutral_filter = create_filter(lm_data, neutral_criteria)     
    neutral_profiles = lm_data.expr[:, neutral_filter] 
    return mean(neutral_profiles, dims=2)  # Matrix{Float32} of size 978 * 1
end


function pair_ref_and_target_profiles(lm_data::Lincs, 
                                        compound::Symbol,                  
                                        ref_filter::BitVector,
                                        target_filter::BitVector, 
                                        p_max::Int64)
    #= p is the number of pairs (ref, target).
    This function returns:
    - ref_prof_matrix: 978 * p Matrix{Float32} containing the gene expression profiles in the reference cell line 
        for the p selected experiments matching with ref_criteria AND :pert_id == compound;
    - target_prof_matrix: 978 * p Matrix{Float32} containing the gene expression profiles in the target cell line 
        for the p selected experiments matching with target_criteria AND :pert_id == compound.
    =#

    ref_cp_filter = (ref_filter .& lm_data[:pert_id, compound])        # Bit vector --> 1 for experiments matching with ref_criteria AND :pert_id == cp
    target_cp_filter = (target_filter .& lm_data[:pert_id, compound])  # Bit vector --> 1 for experiments matching with target_criteria AND :pert_id == cp

    ref_indices = findall(x -> x == 1, ref_cp_filter)                  # Indices of experiments matching with ref_criteria AND :pert_id == cp
    target_indices = findall(x -> x == 1, target_cp_filter)            # Indices of experiments matching with target_criteria AND :pert_id == cp

    p = min(min(length(ref_indices), length(target_indices)), p_max)
  
    ref_indices_picked = shuffle(ref_indices)[1:p]
    target_indices_picked = shuffle(target_indices)[1:p]
      
    ref_prof_matrix = lm_data.expr[:, ref_indices_picked] 
    target_prof_matrix = lm_data.expr[:, target_indices_picked]  

    return ref_prof_matrix, target_prof_matrix
end


function create_data_struct(lm_data::Lincs, 
                            gene_to_gene_embedding_dict::Dict{String, Vector{Float64}},
                            ref_criteria::Dict{Symbol, Symbol}, 
                            target_criteria::Dict{Symbol, Symbol},
                            p_max::Int64)
    #= p_max is the max number of pairs we keep for each experimental condition (1 compound x 1 dose x 1 exposure time). =#

    ref_cl = ref_criteria[:cell_iname]
    target_cl = target_criteria[:cell_iname]

    # Arrays to fill:
    compound_vec = Vector{Symbol}()
    compound_embeddings = ElasticArray{Float32}(undef, 2048, 0)   # 2048 is the fingerprint length
    delta_ref_profiles = ElasticArray{Float32}(undef, 978, 0)     # 978 is the number of landmark genes
    target_gene_expressions = ElasticArray{Float32}(undef, 1, 0)    

    # Neutral profiles:
    ref_neutral_profile = get_average_neutral_profile(lm_data, ref_criteria[:cell_iname])        # 978 * 1 Matrix{Float32}
    target_neutral_profile = get_average_neutral_profile(lm_data, target_criteria[:cell_iname])  # 978 * 1 Matrix{Float32}

    # The target genes are the landmark genes that have a FRoGS gene embedding:
    target_genes = intersect(lm_data.gene.gene_symbol, keys(gene_to_gene_embedding_dict))
    target_gene_embeddings = hcat([Float32.(gene_to_gene_embedding_dict[g]) for g in target_genes]...)

    ref_filter = create_filter(lm_data, ref_criteria) 
    target_filter = create_filter(lm_data, target_criteria) 

    # Compounds shared by ref_experiments and target_experiments:
    ref_compounds = get_unique_filtered_compounds(lm_data, ref_criteria)
    target_compounds = get_unique_filtered_compounds(lm_data, target_criteria)
    compounds = intersect(ref_compounds, target_compounds)

    for cp in shuffle(compounds)  # Random order of compounds

        # Get the canonical SMILES associated to the current compound:
        cp_bv = lm_data.compound[:pert_id, cp] 
        smiles = lm_data.compound[cp_bv, :canonical_smiles][1]

        # Get cp fingerprint:
        fp = get_rdkit_fingerprint(cp, smiles)
        #fp = get_morgan_fingerprint(cp, smiles)

        # Some compounds do not have a fingerprint.
        if fp != nothing                                  

            # Convert fp into Vector{Int32}:
            fp_str_vec = String.(split(fp, ""))
            fp_vec = parse.(Float32, fp_str_vec)

            # Get reference and target data for the current compound:
            ref_prof_matrix, target_prof_matrix = pair_ref_and_target_profiles(lm_data, cp, ref_filter, target_filter, p_max)

            # For each picked experiment:
            for j in 1:size(ref_prof_matrix, 2)             

                ref_prof = ref_prof_matrix[:, j]  # Vector{Float32} of length 978
                delta_ref_prof = ref_prof - vec(ref_neutral_profile)

                push!(compound_vec, cp)
                append!(compound_embeddings, fp_vec)
                append!(delta_ref_profiles, delta_ref_prof)

                # For each landmark gene
                for i in 1:size(target_prof_matrix, 1)        

                    # gene i becomes the target gene.
                    target_gene = lm_data.gene[i, :gene_symbol]

                    # Some genes are not associated to a FRoGS gene embedding.
                    if haskey(gene_to_gene_embedding_dict, target_gene)
                        target_gene_expr = target_prof_matrix[i, j]
                        append!(target_gene_expressions, target_gene_expr)
                    end
                end
            end
        end
    end

    return Data(ref_cl,
                target_cl,
                compound_vec, 
                Matrix(compound_embeddings), 
                Matrix(delta_ref_profiles), 
                target_neutral_profile,
                target_genes,
                target_gene_embeddings,
                vec(target_gene_expressions))
end 


end