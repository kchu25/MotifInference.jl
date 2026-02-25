
mutable struct training_and_rendering_config
    ###### training configuration fields ######
    datapath::String                     # path to data file (a jld2 file that contains a seq2exp-data object)
    model_creator::Function              # function to create the model
    seed::Union{Nothing, Int}            # random seed for reproducibility
    max_training_epochs::Int             # maximum epochs for processor training
    max_processor_epochs::Int            # maximum epochs for processor training
    predict_position::Union{Symbol, Int} # which output position to predict (:all or integer index)
    patience::Int                        # patience for triggering early stopping (number of epochs with no improvement)
    seq_type::Symbol                     # :dna, :rna, or :protein
    type::Symbol                         # :conv or :mut  --> this will affect training and rendering
    normalization_method::Symbol                # :identity, :zscore, or :log, etc.
    loss_spec::NamedTuple{(:loss, :agg), Tuple{Function, Function}}  # loss function and aggregation method for training
    ###### motif inference fields ######
    scale_back::Bool                     # whether to scale back the normalization for Banzhaf index calculations
    top_and_bot_counts::Int              # number of top and bottom significant motifs to render
    activation_thresh::Float64           # percentile threshold for considering a motif "active" in a sequence
    motif_sizes::Vector{Int}             # list of multi-motif sizes to consider, e.g. [2], or [2,3]; always starts from 2, i.e. pairs
    count_threshold::Int                 # 
    Q_threshold::Float64                 # Q-value threshold for significance filtering
    ###### rendering configuration fields ######
    dpi::Int                             # resolution for rendering plots
    save_path::String                    # path to save the models and rendered output files; a folder
    title_string::String                 # title for the rendered output
end


# Keep the struct as-is, add external constructor
function training_and_rendering_config(
    datapath, model_creator, save_path, title_string;
    seq_type=:dna,
    seed=nothing,
    type=:conv,
    normalization_method=:zscore,    
    loss_spec=loss_specs[:mse],
    max_training_epochs=40,
    max_processor_epochs=30,
    predict_position=:all,
    patience=10,
    scale_back=true,
    motif_sizes=[2, 3],
    activation_thresh=0.8,
    top_and_bot_counts=8,
    count_threshold=25,
    Q_threshold=1e-25,
    dpi=60)
    return training_and_rendering_config(        
        datapath, model_creator, seed, max_training_epochs, 
        max_processor_epochs, predict_position, patience, seq_type, type, 
        normalization_method, loss_spec, scale_back, top_and_bot_counts, activation_thresh, motif_sizes, count_threshold, Q_threshold, 
        dpi, save_path, title_string
    )
end