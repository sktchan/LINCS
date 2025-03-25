using CSV, DataFrames, Statistics, Flux, Transformers, CUDA, Transformers.Layers, Transformers.TextEncoders
enable_gpu(CUDA.functional())

@time df = CSV.read("data/cellline_geneexpr.csv", DataFrame)
matrix = Float32.(Matrix(df[:, 2:end-1])')

labels = ["<unk>", "<s>", "</s>", "low", "medium", "high"] # low/medium/high exp level bins
startsym = "<s>"
endsym = "</s>"
unksym = "<unk>"

textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym)

# since range is 0-15, with median/mean being around 8.4
low_threshold = 6.5
medium_threshold = 9.5

function encode_expression_data(expression_data)
    token_sequence = []
    for expr_value in expression_data
        if expr_value < low_threshold
            push!(token_sequence, "low")
        elseif expr_value < medium_threshold
            push!(token_sequence, "medium")
        else
            push!(token_sequence, "high")
        end
    end
    return encode(textenc, join(token_sequence, ' '))
end

nhead = 8
dimhead = 64
dimhidden = 512
dimffn = 2048
num_layers = 4
num_classes = unique(df.cell_line)

exp_embed = encode_expression_data(matrix)

#TODO: encoding exp matrix per row or is the above ok?
#TODO: do i need to define the positional encoder as well?
#TODO: define transformer encoder/decoder
#TODO: input into model