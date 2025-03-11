module DataSplit

using Statistics
using CUDA
using HDF5

using LincsProject
using DataCoupling

export SplittedData   
export save_splitted_data_to_hdf5, load_splitted_data_from_hdf5


struct SplittedData
    include_vec::Vector{Symbol}
    X_train::Matrix{Float32}
    Y_train::Matrix{Float32}
    X_test::Matrix{Float32}
    Y_test::Matrix{Float32}
end


function save_splitted_data_to_hdf5(filename::String, data::SplittedData)
    for field in fieldnames(SplittedData)
        field_value = getfield(data, field)  
        if isa(field_value, Vector{Symbol})
            h5write(filename, String(field), String.(field_value))
        else
            h5write(filename, String(field), field_value)
        end
    end
end


function load_splitted_data_from_hdf5(filename::String)::SplittedData
    data_dict = Dict{Symbol, Any}()

    # Read the data from the HDF5 file
    for field in fieldnames(SplittedData)
        field_value = HDF5.h5read(filename, String(field))
        data_dict[field] = field_value
    end

    # Return a new instance of Data_for_gene_expr_prediction
    return SplittedData(Symbol.(data_dict[:include_vec]), 
                        data_dict[:X_train], 
                        data_dict[:Y_train],
                        data_dict[:X_test],
                        data_dict[:Y_test])
end


function reformat_mat(mat::Matrix{Float32}, nb_expe::Int64, g::Int64)
    #= 
    nb_expe = number of experiments.
    g = number of target genes per experiment.

    There are three possibilities: 
    - size(mat) = (978, 1) (if mat = ref_neutral_profile, or target_neutral_profile), OR
    - size(mat, 2) = nb experiments (if mat = compound_embeddings, or ref_profiles), OR
    - size(mat, 2) = g (if mat = target_gene_embeddings)    
    =#

    if size(mat, 2) == 1
        # The neutral profile is identical for all the predictions: 
        Y_length = nb_expe * g
        new_mat = repeat(mat, 1, Y_length)  # Matrix{Float32} of size (978, Y_length)
        
    elseif size(mat, 2) == nb_expe
        # Each column is repeated g times: 
        # Example for g = 3: col1 col1 col1 col2 col2 col2 col3 col3 col3 ...
        new_mat = Matrix{Float32}(undef, size(mat, 1), size(mat, 2) * g)
        cur = 1
        for col in eachcol(mat)
            repeated_col = reshape(repeat(col, g), size(mat, 1), g)
            new_mat[:, cur:(cur + g - 1)] = repeated_col
            cur += g
        end

    elseif size(mat, 2) == g
        # Each column is repeated nb_expe times --> the repeated columns are interleaved: 
        # Example for nb_expe = 2: col1 col2 col3 col1 col2 col3
        new_mat = repeat(mat, 1, nb_expe)

    else
        error("The matrix has an unexpected number of columns.")
    end

    return new_mat
end


function concatenate_inputs(data::Data, include_vec::Vector{Symbol})
  
    m, n = size(data, include_vec)
    input = Matrix{Float32}(undef, m, n)

    # Find where to put mat in input:
    cur = 1
    function insert_mat(input, cur, mat)
        tmp = size(mat, 1)
        input[cur:(cur + tmp - 1), 1:n] = mat
        return cur + tmp
    end

    nb_expe = length(data.compounds)
    g = length(data.target_genes)
   
    for symb in include_vec
        mat = getfield(data, symb)
        if size(mat, 2) != n 
            mat = reformat_mat(mat, nb_expe, g)
        end
        cur = insert_mat(input, cur, mat)
    end

    if isempty(input)
        error("The input is empty.")
    end

    return input
end


function remove_mean(Y_train::Vector{Float32}, Y_test::Vector{Float32}, g::Int64)

    # Vectors of length (n * g) --> matrices of size (g, n), with n = n_train or n_test:
    reshaped_Y_train = reshape(Y_train, g, length(Y_train) ÷ g) 
    reshaped_Y_test = reshape(Y_test, g, length(Y_test) ÷ g)

    # Compute the mean per gene on the training set:
    mean_per_gene = mean(reshaped_Y_train, dims=2)

    # Remove the mean per gene from the training and test set:
    reshaped_Y_train = reshaped_Y_train .- mean_per_gene
    reshaped_Y_test = reshaped_Y_test .- mean_per_gene

    # Reshape the matrices into vectors of length (n * g):
    ## g expressions for experiment 1, followed by g expressions for experiment 2, etc.
    Y_train = reshape(reshaped_Y_train, 1, length(Y_train))
    Y_test = reshape(reshaped_Y_test, 1, length(Y_test))

    return Y_train, Y_test
end


function SplittedData(data::Data, include_vec::Vector{Symbol}) 
    #= 
    The order of the (unique) compounds is random across data.
    Pairs of experiments associated to the same compound are successive. 
    There is no compound both in train and test set. 
    =#

    # Get cursor to split the experiments between train and test: 
    unique_compounds = unique(data.compounds)
    cur = Int64(floor(length(unique_compounds)* 0.8))
    unique_train_cps = unique_compounds[1:cur]
    # If p_max != 1, then the same compound can be associated to different experiments.
    train_cps = filter(x -> x in unique_train_cps, data.compounds)
    expe_cur = findfirst(x -> x == train_cps[end], data.compounds)  # Find the index of the last compound in train_cps

    # Get cursor to split target gene expressions (Y) between train and test, and use it to create train_bv:
    y_cur = expe_cur * length(data.target_genes)
    train_bv = BitVector([x <= y_cur ? 1 : 0 for x in 1:length(data.target_gene_expressions)])

    # Create input: 
    input = concatenate_inputs(data, include_vec) 

    # Train:  
    X_train = input[:, train_bv] 
    Y_train = data.target_gene_expressions[train_bv]

    # Test:
    test_bv = .!train_bv
    X_test = input[:, test_bv]
    Y_test = data.target_gene_expressions[test_bv]

    # Remove mean: 
    g = length(data.target_genes)
    Y_train, Y_test = remove_mean(Y_train, Y_test, g)  
    #= Y_train and Y_test are not vectors anymore, but matrices of size (1, n) 
    --> Works for the loss computation between model(X_train) and Y_train during training. =# 
    
    return SplittedData(include_vec, X_train, Y_train, X_test, Y_test) 

end


end