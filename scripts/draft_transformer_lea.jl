module Neural_network

using CSV
using CUDA
using Distributions
using Flux
using GLM
using LinearAlgebra
using Optimisers
using Statistics
using StatsBase
using Zygote


export create_MLP, create_model_with_self_attention, custom_train!, evaluate_model


const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}


function create_MLP(input_dim::Int64, 
                    hidden_layers::Vector{Int64}, 
                    output_dim::Int64, 
                    activation_f, # Not a string, but a function from Flux
                    dropout_arr::Union{Vector{Float32}, Nothing}=nothing)
  layers = [Dense(i => j, activation_f) for (i, j) in zip([input_dim; hidden_layers[1:end-1]], hidden_layers)]
  if dropout_arr !== nothing
    dropout_layers = [Dropout(d) for d in dropout_arr]
    layers = [layer for pair in zip(layers[1:end], dropout_layers) for layer in pair]  # Interleave layers and dropout
  end
  return Chain(layers..., Dense(hidden_layers[end] => output_dim, identity))
end


######################## Model with self attention ########################

########## Self-attention ##########

struct SA
  q_proj::Float32Matrix2DType
  k_proj::Float32Matrix2DType
  v_proj::Float32Matrix2DType
end

function create_self_attention(dq::Int64, embedding_dim::Int64) 
  q_proj = randn(Float32, dq, embedding_dim)
  k_proj = randn(Float32, dq, embedding_dim) # dk = dq
  v_proj = randn(Float32, embedding_dim, embedding_dim)
  return SA(q_proj, k_proj, v_proj) 
end

(Flux.@functor SA)
function (sa::SA)(input::Float32Matrix2DType) # input dimensions = 978 * batch size
  #= CAUTION: Usually, the input of a self-attention layer is an embedding, but in our case the embedding_dim HAS TO BE 1. 
  If not, this function will not work --> the use of hcat will be problematic. 
  TODO: Code a model taking as input a multi-dimensional input (embedding_dim * 978). =#
  transformed_input = [
    begin
        profile = input[:, i] # profile dimension = (978,). Transposed profile dimensions = 1 * 978
        Q = sa.q_proj * profile' # Q dimensions = dq * 978
        K = sa.k_proj * profile' # K dimensions = dq * 978
        V = sa.v_proj * profile' # V dimensions = embedding_dim * 978
        attention_scores = Q' * K # attention_scores dimensions = 978 * 978
        attention_weights = softmax((attention_scores / sqrt(size(sa.q_proj, 1))), dims=2) # attention_weights dimensions = 978 * 978
        vec(V * attention_weights) # Returns the transformed profile for this iteration. Transformed profile dimension = embedding_dim * 978
    end
    for i in 1:size(input, 2) # For each profile in batch
  ] 
  #= transformed_input dimensions = embedding_dim * 978 * batch_size. 
  Using hcat allows to get ride of the embedding_dim. The first dimension becomes embedding_dim * 978, but embedding_dim = 1, 
  so finally, our first dimension is 978 (expected model's input).
  Use reduce with hcat to concatenate without in-place modification, then send to GPU: =#
  return reduce(hcat, transformed_input) |> gpu # Returned transformed_input dimensions = 978 * batch_size. 
end

########## Self-attention > MLP ##########

struct ModelWithSelfAttention
  sa::SA
  mlp::Chain
end

function create_model_with_self_attention(embedding_dim::Int64, 
                                          dq::Int64, 
                                          input_dim::Int64,
                                          hidden_layers::Vector{Int64},
                                          output_dim::Int64, 
                                          activation_f, # Not a string, but a function from Flux
                                          dropout_arr::Union{Vector{Float32}, Nothing}=nothing)                   
  sa = create_self_attention(dq, embedding_dim)
  mlp = create_MLP(input_dim, hidden_layers, output_dim, activation_f, dropout_arr) 
  return ModelWithSelfAttention(sa, mlp)
end

(Flux.@functor ModelWithSelfAttention)
function (model::ModelWithSelfAttention)(input::Float32Matrix2DType) # input dimensions = 978 * batch_size
  transformed_input = model.sa(input)
  return model.mlp(transformed_input)
end


################################################################


function custom_train!(X_train::CuArray{Float32}, 
                      Y_train::CuArray{Float32}, 
                      X_test::CuArray{Float32}, 
                      Y_test::CuArray{Float32}, 
                      batch_size::Int64,
                      nb_epochs::Int64,
                      model,
                      opt_state,   # Object which encodes the optimiser and its state (created by Flux.setup(...))
                      α::Float32,  # Parameter for L2 regularization
                      out::String) # CSV file to register losses, Pearson and Spearman coefficients during training

  csvfile = open(out, "w")
  write(csvfile, "epoch,loss_train,loss_test,r_train,r_test,spearman_train,spearman_test\n")

  # Load training data mini-batches:
  train_data_loader = Flux.DataLoader((X_train, Y_train), batchsize=batch_size)

  ps = Flux.params(model) 

  for epoch in 1:nb_epochs
    
    # Training on each mini-batch:
    if std_dev_noise === nothing 
      for (x, y) in train_data_loader # x dimensions = 978 * batch_size. y dimensions = 1 * batch_size
        loss, grads = Flux.withgradient(model) do m
          Flux.mae(m(x), y) + α * sum(p -> sum(abs2, p), ps) # L2 regularization
        end
        Flux.update!(opt_state, model, grads[1])
      end
    else
        for (x, y) in train_data_loader # x dimensions = 978 * batch_size. y dimensions = 1 * batch_size
          loss, grads = Flux.withgradient(model) do m
            Flux.mae(m(x), y) + α * sum(p -> sum(abs2, p), ps) # L2 regularization
          end
          Flux.update!(opt_state, model, grads[1])
        end
    end
  
    
    # Losses for the current epoch:
    loss_train =  Flux.mae(model(X_train), Y_train)
    loss_test = Flux.mae(model(X_test), Y_test) 

    # Every 10 epochs, compute Pearson and Spearman correlation coefs. between predictions made by trained model and truth:
    if epoch % 10 == 0
      Ŷ_train, r_train, spearman_train = evaluate_model(X_train, Y_train, model) 
      Ŷ_test, r_test, spearman_test = evaluate_model(X_test, Y_test, model) 
      # Save results in the CSV file:
      results = epoch, loss_train, loss_test, r_train, r_test, spearman_train, spearman_test
      write(csvfile, join(results, ","), "\n")   
    end
    
  end 
  
  close(csvfile)
end


function evaluate_model(x_gpu::CuArray{Float32}, y_gpu::CuArray{Float32}, model::Any)
  #= Returns on CPU:
  - ŷ: predictions made by the trained model;
  - r: Pearson correlation coef. between the predictions and the truth;
  - spearman: Spearman correlation coef. between the predictions and the truth. 
  The function cor which computes the r works on the CPU only. =#
  testmode!(model) # Deactivate possible dropout
  ŷ = model(x_gpu) |> cpu 
  y = y_gpu |> cpu
  r = round(cor(vec(y), vec(ŷ)), digits=2) 
  spearman = round(corspearman(vec(y), vec(ŷ)), digits=2) 
  return ŷ, r, spearman
end


end

