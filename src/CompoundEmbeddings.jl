module CompoundEmbeddings

using RDKitMinimalLib

export get_morgan_fingerprint, get_rdkit_fingerprint


function get_morgan_fingerprint(compound::Symbol, smiles::String)
    # Returns Morgan (ECFP like)
    mol = get_mol(smiles)   
    if mol == nothing
        println("SMILES non available for compound: $compound ($smiles)")
        fp = nothing
    else
        fp = get_morgan_fp(mol)  
    end
    return fp
end


function get_rdkit_fingerprint(compound::Symbol, smiles::String)
    # Returns RDKit fingerprint. 
    mol = get_mol(smiles)   
    if mol == nothing
        println("SMILES non available for compound: $compound ($smiles)")
        fp = nothing
    else
        fp = get_rdkit_fp(mol)  
    end
    return fp
end


end