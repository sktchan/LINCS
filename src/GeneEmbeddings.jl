module GeneEmbeddings

using CSV, DataFrames

using LincsProject

export create_gene_to_gene_embedding_dict
export create_gene_embedding_df, create_gene_to_id_dict 


function create_gene_embedding_df(frogs_gene_embedding_file::String)::DataFrame
    #= 
    Returns a df containing the FRoGS gene embeddings.
    The first col is "gene_id". The following columns contain the coordinates in the embedding space (256 dims). 
    =#
    df = CSV.File(frogs_gene_embedding_file, header=false) |> DataFrame
    column_names = ["gene_id"] 
    append!(column_names, ["dim$(n)" for n in 1:256])  
    rename!(df, column_names)  
    return df
end


function create_gene_to_id_dict(frogs_gene_to_id_file::String)::Dict{String, Int64}
    # Returns a dict where the keys are genes (gene symbols), and the values are gene ids.
    df = CSV.File(frogs_gene_to_id_file, header=true) |> DataFrame    
    return Dict(zip(string.(df.Symbol), df.gene_id))              # Convert df.Symbol from String15 to String
end


function create_gene_to_gene_embedding_dict(frogs_gene_to_id_file::String, frogs_gene_embedding_file::String)::Dict{String, Vector{Float64}}
    # Returns a dict where the keys are genes (gene symbols), and the values are gene embeddings.
    df = create_gene_embedding_df(frogs_gene_embedding_file)
    gene_to_id_dict = create_gene_to_id_dict(frogs_gene_to_id_file) 
    gene_symbols = collect(keys(gene_to_id_dict))      
    embeddings = zeros(Float64, length(gene_symbols), 256)        # 256 is the number of dimensions of the FRoGS gene embedding 
    for (i, gene) in enumerate(gene_symbols)
        gene_id = gene_to_id_dict[gene]
        gene_embed = filter(row -> row[:gene_id] == gene_id, df)  # Contains the gene_id in first position
        # Some genes do not have a FRoGS gene embedding.
        if size(gene_embed, 1) > 0
            embeddings[i, :] .= collect(gene_embed[1, 2:end])
        end
    end
    return Dict(gene_symbols[i] => embeddings[i, :] for i in 1:length(gene_symbols))
end


end