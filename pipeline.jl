
function setup_results_folder(saved_folder; parent_saved_folder="../RESULTS")
    result_save_folder = joinpath(parent_saved_folder, saved_folder)
    mkpath(result_save_folder)
    return result_save_folder
end

# ─────────────────────────────────────────────────────────────────────────────
# TRC construction
# ─────────────────────────────────────────────────────────────────────────────

# const datasets_processed_folder = "/home/shane/Desktop/academia/data/SEQ2EXP/DATASETS_PROCESSED"
const datasets_processed_folder = "//home/kchu25/Desktop/work/code/cur_proj/DATASETS_PROCESSED"

"""
    make_trc(f; datasets_folder=datasets_processed_folder, results_parent="../RESULTS")

Construct a training_and_rendering_config from a dataset entry (NamedTuple).
"""
function make_trc(f; 
        datasets_folder=datasets_processed_folder, 
        results_parent="../RESULTS")
    datapath = joinpath(datasets_folder, f.name, f.file)
    save_path = setup_results_folder(f.name; parent_saved_folder=results_parent)
    
    MotifInference.training_and_rendering_config(
        datapath, f.model_creator, save_path, "n/a";
        seq_type=f.seq_type, seed=f.seed, type=f.type,
        motif_sizes=f.motif_sizes, normalization=f.normalization,
        activation_thresh=f.activation_thresh
    )
end

"""
    make_trc(datapath::String; kwargs...)

Construct a trc directly from a .jld2 file path. 
Infers the dataset name from the filename and uses sensible defaults.

# Keyword Arguments
- `seq_type = :dna`
- `type = :conv`
- `normalization = :zscore`
- `seed = nothing` (triggers tuning)
- `motif_sizes = [2, 3]`
- `activation_thresh = 0.9`
- `multioutput = false`
- `results_parent = "../RESULTS"`

# Examples
    trc = make_trc("/path/to/mydata.jld2")
    trc = make_trc("/path/to/mydata.jld2"; seq_type=:rna, seed=42)
    run_pipeline(trc)
"""
function make_trc(datapath::String;
        seq_type::Symbol = :dna,
        type::Symbol = :conv,
        normalization::Symbol = :zscore,
        seed::Union{Int, Nothing} = nothing,
        motif_sizes::Vector{Int} = [2, 3],
        activation_thresh::Float64 = 0.9,
        multioutput::Bool = false,
        results_parent::String = "../RESULTS")
    
    name = splitext(basename(datapath))[1]  # "mydata.jld2" → "mydata"
    save_path = setup_results_folder(name; parent_saved_folder=results_parent)
    model_creator = resolve_model_creator(; seq_type, type, multioutput)
    
    MotifInference.training_and_rendering_config(
        datapath, model_creator, save_path, "n/a";
        seq_type, seed, type, motif_sizes, normalization, activation_thresh
    )
end

"""
    load_data(trc) -> data

Load raw data from disk and wrap it as a OnehotSEQ2EXP_Dataset.
Creates save_path directory if needed.
"""
function load_data(trc)
    mkpath(trc.save_path)
    @load trc.datapath raw_data
    return MotifInference.OnehotSEQ2EXP_Dataset(raw_data)
end

"""
    tune_if_needed!(trc, data; tune_max_epochs=15, tune_n_trials=2, tune_patience=10)

Run hyperparameter tuning if `trc.seed` is not set. Updates `trc.seed` in-place.
"""
function tune_if_needed!(trc, data; tune_max_epochs=15, tune_n_trials=2, tune_patience=10)
    if isnothing(trc.seed)
        results, best_model, best_info = 
            MotifInference.perform_hyperparameter_tuning(data, trc; 
                trial_number_start=1, max_epochs=tune_max_epochs, 
                n_trials=tune_n_trials, patience=tune_patience)
        trc.seed = best_info.seed
        @info "Tuning complete. Best seed: $(trc.seed)"
    end
end

"""
    process_output(data, m, train_stats, dl_train, dl_test, trc, output_index)

Process a single output index: compute contributions, tag significance, 
compute interactions, and return all results.

Returns: (contributions_df_filtered, dfs, interaction_summaries_str, 
          interaction_summaries, render_folder_name)
"""
function process_output(data, m, train_stats, dl_train, dl_test, trc, output_index)
    processor, pts_train, pts = MotifInference.obtain_processor(m, dl_train, dl_test, trc; 
        predict_position=output_index)

    contributions_df_filtered, contributions_df_filtered_singletons, dfs = 
        MotifInference.load_or_save_raw_motifs(data, m, processor, train_stats, trc; 
        output_index=output_index)

    # Tag significant singletons
    significant_fils = contributions_df_filtered_singletons.filter_index |> Set
    contributions_df_filtered.significant = [idx in significant_fils for idx in contributions_df_filtered.filter_index]

    # Interaction summaries
    interaction_summaries_str, interaction_summaries = 
        MotifInference.BanzhafInference.obtain_interaction_results(
            contributions_df_filtered, dfs)

    # Render folder name
    render_folder_name = "renderings_" * begin
        if !isnothing(data.raw_data.feature_names)
            data.raw_data.feature_names[output_index]
        else
            "$(output_index)"
        end
    end

    return contributions_df_filtered, dfs, pts_train, pts, interaction_summaries_str, 
           interaction_summaries, render_folder_name
end

"""
    render_html(data, m, trc, contributions_df_filtered, dfs, pts, 
                split_indices, interaction_summaries_str, render_folder_name)

Generate the HTML motif visualization pages.
"""
function render_html(data, m, trc, contributions_df_filtered, dfs, pts, 
                     split_indices, interaction_summaries_str, render_folder_name)
    MotifInference.GlyphEctoplasm.plot_motifs_conv_case(data, m, trc.motif_sizes, 
        contributions_df_filtered, dfs, pts, split_indices; 
        interaction_summaries=interaction_summaries_str,    
        dpi=trc.dpi, 
        save_path=joinpath(trc.save_path, render_folder_name), 
        page_title=trc.title_string, 
        rna=trc.seq_type == :rna)
end

"""
    run_pipeline(trc; tune_max_epochs=15, tune_n_trials=2, tune_patience=10, output_indices=nothing)

Run the full motif inference pipeline: load data, optionally tune, train, 
and process all (or selected) output indices with HTML rendering.

# Keyword Arguments
- `tune_max_epochs=15`: max epochs for hyperparameter tuning
- `tune_n_trials=2`: number of tuning trials  
- `tune_patience=10`: early stopping patience
- `output_indices=nothing`: which outputs to process (default: all if `trc.predict_position == :all`)
"""
function run_pipeline(trc; tune_max_epochs=25, tune_n_trials=25, tune_patience=5, output_indices=nothing)
    data = load_data(trc)
    tune_if_needed!(trc, data; tune_max_epochs, tune_n_trials, tune_patience)

    m, train_stats, dl_train, dl_test, split_indices = 
        MotifInference.obtain_trained_model_and_splited_datasets(data, trc)

    # Determine which outputs to process
    indices = if !isnothing(output_indices)
        output_indices
    elseif trc.predict_position == :all
        if !isnothing(data.raw_data.feature_names)
            @assert data.Y_dim == length(data.raw_data.feature_names)
        end
        1:data.Y_dim
    else
        [1]
    end

    for output_index in indices
        @info "Processing output $output_index / $(length(indices))..."

        contributions_df_filtered, dfs, pts_train, pts, interaction_summaries_str, 
            interaction_summaries, render_folder_name = 
            process_output(data, m, train_stats, dl_train, dl_test, trc, output_index)

        render_html(data, m, trc, contributions_df_filtered, dfs, pts, 
                    split_indices, interaction_summaries_str, render_folder_name)

        @info "Output $output_index done → $(render_folder_name)"
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run from dataset entry, file path, or batch
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_pipeline(dataset_entry; kwargs...)

Run pipeline from a dataset entry (as returned by `load_datasets()`).

# Examples
    ds = load_datasets("ecoli")
    run_pipeline(ds[1])
    run_pipeline(ds[1]; output_indices=1:5)
"""
function run_pipeline(f::NamedTuple; kwargs...)
    trc = make_trc(f)
    run_pipeline(trc; kwargs...)
end

"""
    run_pipeline(datapath; seq_type=:dna, seed=nothing, ...)

Run pipeline directly from a .jld2 file path. 
No pre-registration needed — just point to your data.

All keyword arguments from `make_trc` (seq_type, normalization, seed, etc.) 
and `run_pipeline` (output_indices, tune_* params) are accepted.

# Examples
    # Simplest — just a path (will auto-tune since no seed)
    run_pipeline("/path/to/mydata.jld2")

    # With overrides
    run_pipeline("/path/to/mydata.jld2"; seq_type=:rna, seed=42)

    # RNA with specific outputs
    run_pipeline("/path/to/mydata.jld2"; seq_type=:rna, seed=42, output_indices=1:5)
"""
function run_pipeline(datapath::String; 
        # make_trc kwargs
        seq_type::Symbol = :dna,
        type::Symbol = :conv,
        normalization::Symbol = :zscore,
        seed::Union{Int, Nothing} = nothing,
        motif_sizes::Vector{Int} = [2, 3],
        activation_thresh::Float64 = 0.9,
        multioutput::Bool = false,
        results_parent::String = "../RESULTS",
        # run_pipeline kwargs
        kwargs...)
    trc = make_trc(datapath; seq_type, type, normalization, seed, 
                   motif_sizes, activation_thresh, multioutput, results_parent)
    run_pipeline(trc; kwargs...)
end

"""
    run_all(datasets; kwargs...)

Run pipeline for multiple datasets sequentially.

# Examples
    run_all(load_datasets())                              # everything
    run_all(load_datasets("ecoli", "yeast"))              # specific ones
    run_all(load_datasets(); output_indices=1:3)          # first 3 outputs each
"""
function run_all(datasets; kwargs...)
    for (i, f) in enumerate(datasets)
        @info "═══ Dataset $i/$(length(datasets)): $(f.name) ═══"
        run_pipeline(f; kwargs...)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Usage examples (uncomment to run)
# ─────────────────────────────────────────────────────────────────────────────

# From a .jld2 file path (simplest):
#   run_pipeline("/path/to/mydata.jld2")
#   run_pipeline("/path/to/mydata.jld2"; seq_type=:rna, seed=42)

# From a registered dataset:
#   run_pipeline(load_datasets("ecoli")[1])
#   run_pipeline(load_datasets("ecoli")[1]; output_indices=1:5)

# Batch:
#   run_all(load_datasets("ecoli", "yeast"))
#   run_all(load_datasets())