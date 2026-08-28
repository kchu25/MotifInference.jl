
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
    normalization_method::Symbol                # :identity, :zscore, :zscore_wt, :log, etc.
    wt_reference::Union{Nothing, Real, AbstractVector{<:Real}}  # value the labels are centred on when normalization_method=:zscore_wt.
                                                                # nothing for every other method. For mutagenesis this is the WILD-TYPE
                                                                # measurement, so the model's structural f(wild type)=0 lands on it.
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
    normalization_note::String           # human-readable sentence describing how the labels were
                                         # transformed, for the rendered page. Empty when the caller
                                         # named the method explicitly (nothing was decided here).
                                         # TRAILING FIELD: added after the other 22, so every
                                         # pre-existing positional construction still works.
    output_subdir::String                # sub-folder of `save_path` for the RUN-SPECIFIC outputs:
                                         # the motif caches, the rendering folders and top_motifs.csv.
                                         # "" (the default) means write them straight into
                                         # `save_path`, which is byte-for-byte the old layout.
                                         # Multi-run sets it to "run_1", "run_2", ... so several
                                         # seeds can share one `models/`, `json/` and results CSV.
                                         # See `output_path`. TRAILING FIELD, same reason as above.
end

"""
    output_path(trc) -> String

Where this run's own outputs go: `save_path/output_subdir`, or just `save_path`
when `output_subdir` is empty (`joinpath(p, "") == p`).

The split exists because the model artifacts are already seed-keyed
(`models/model_<seed>.jld2`, `models/processor_<seed>_pp_<i>.jld2`) and the
tuning artifacts (`json/trial_seed_*.json`, `results_*.csv`) are shared, so
several runs can live under one `save_path` while only these four collide:

  * `motifs_cache_output_<i>.jld2`
  * `cache/motifs_size_*.arrow`   (model-dependent — they take `m` and `processor`)
  * `renderings_<feature>/`
  * `top_motifs.csv`

Those four, and only those four, are addressed through this function.
"""
output_path(trc) = isempty(trc.output_subdir) ? trc.save_path :
                   joinpath(trc.save_path, trc.output_subdir)

# Legacy 22-argument constructor: anything that built a config before
# `normalization_note` existed keeps working, with an empty note.
training_and_rendering_config(datapath, model_creator, seed, max_training_epochs,
    max_processor_epochs, predict_position, patience, seq_type, type,
    normalization_method, wt_reference, loss_spec, scale_back, top_and_bot_counts,
    activation_thresh, motif_sizes, count_threshold, Q_threshold, dpi, save_path,
    title_string) =
    training_and_rendering_config(datapath, model_creator, seed, max_training_epochs,
        max_processor_epochs, predict_position, patience, seq_type, type,
        normalization_method, wt_reference, loss_spec, scale_back, top_and_bot_counts,
        activation_thresh, motif_sizes, count_threshold, Q_threshold, dpi, save_path,
        title_string, "", "")

# Legacy 22-argument constructor: anything that built a config after
# `normalization_note` but before `output_subdir` keeps working, writing its
# outputs straight into `save_path`.
training_and_rendering_config(datapath, model_creator, seed, max_training_epochs,
    max_processor_epochs, predict_position, patience, seq_type, type,
    normalization_method, wt_reference, loss_spec, scale_back, top_and_bot_counts,
    activation_thresh, motif_sizes, count_threshold, Q_threshold, dpi, save_path,
    title_string, normalization_note) =
    training_and_rendering_config(datapath, model_creator, seed, max_training_epochs,
        max_processor_epochs, predict_position, patience, seq_type, type,
        normalization_method, wt_reference, loss_spec, scale_back, top_and_bot_counts,
        activation_thresh, motif_sizes, count_threshold, Q_threshold, dpi, save_path,
        title_string, normalization_note, "")

# Keep the struct as-is, add external constructor
function training_and_rendering_config(
    datapath, model_creator, save_path, title_string;
    seq_type=:dna,
    seed=nothing,
    type=:conv,
    normalization_method=:auto,   # :auto reads the dataset and picks; see label_transform.jl.
                                  # Pass an explicit method to override and to get the old default.
    wt_reference=nothing,
    loss_spec=loss_specs[:mse],
    max_training_epochs=40,
    max_processor_epochs=60,
    predict_position=:all,
    patience=10,
    scale_back=true,
    motif_sizes=[2, 3],
    activation_thresh=0.8,
    top_and_bot_counts=8, # temporary for now
    # top_and_bot_counts=500, # temporary for now
    count_threshold=25,
    Q_threshold=1e-25,
    dpi=60,
    output_subdir="")   # "" => the old single-run layout; multi-run sets "run_i"
    return training_and_rendering_config(        
        datapath, model_creator, seed, max_training_epochs, 
        max_processor_epochs, predict_position, patience, seq_type, type, 
        normalization_method, wt_reference, loss_spec, scale_back, top_and_bot_counts, activation_thresh, motif_sizes, count_threshold, Q_threshold, 
        dpi, save_path, title_string, "", output_subdir
    )
end