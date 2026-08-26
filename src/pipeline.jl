
function setup_results_folder(save_folder_name; save_root=".")
    result_save_folder = joinpath(save_root, save_folder_name)
    mkpath(result_save_folder)
    return result_save_folder
end

# ─────────────────────────────────────────────────────────────────────────────
# TRC construction
# ─────────────────────────────────────────────────────────────────────────────

# const datasets_processed_folder = "/home/shane/Desktop/academia/data/SEQ2EXP/DATASETS_PROCESSED"
const datasets_processed_folder = "/home/kchu25/Desktop/work/code/cur_proj/DATASETS_PROCESSED"

"""
    make_trc(f; datasets_folder=datasets_processed_folder, save_root=".", save_folder_name=nothing)

Construct a training_and_rendering_config from a dataset entry (NamedTuple).
`save_folder_name` defaults to `f.name`.
"""
function make_trc(f, datasets_folder=datasets_processed_folder;
    save_root=".",
    save_folder_name=nothing)
    datapath = joinpath(datasets_folder, f.name, f.file)
    folder = something(save_folder_name, f.name)
    save_path = setup_results_folder(folder; save_root)

    MotifInference.training_and_rendering_config(
        datapath, f.model_creator, save_path, "n/a";
        seq_type=f.seq_type, seed=f.seed, type=f.type,
        motif_sizes=f.motif_sizes, normalization_method=f.normalization_method,
        wt_reference=get(f, :wt_reference, nothing),
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
- `normalization_method = :zscore`   (`:zscore_wt` centres on `wt_reference` instead of the label mean)
- `wt_reference = nothing`           value the labels are centred on when `normalization_method=:zscore_wt`
- `seed = nothing` (triggers tuning)
- `motif_sizes = [2, 3, 4, 5]`
- `activation_thresh = 0.9`
- `multioutput = false`
- `conv_bottleneck = false` (DNA/RNA *convolution* runs only: squeeze the inference-code
  conv layer, like the amino-acid bottleneck model; needs sequences ≳ 25nt. Ignored when
  `type = :mut`, which already selects a bottleneck mutagenesis model)
- `save_root = "."` (current working directory)
- `save_folder_name = nothing` (defaults to the file's basename without extension)

# Examples
    trc = make_trc("/path/to/mydata.jld2")
    trc = make_trc("/path/to/mydata.jld2"; seq_type=:rna, seed=42)
    trc = make_trc("/path/to/mydata.jld2"; save_root="/custom", save_folder_name="run01")
    run_method(trc)
"""
function make_trc(datapath::String;
    seq_type::Symbol=:dna,
    type::Symbol=:conv,
    normalization_method::Symbol=:zscore,
    wt_reference::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    seed::Union{Int,Nothing}=nothing,
    motif_sizes::Vector{Int}=[2, 3, 4, 5],
    activation_thresh::Float64=0.9,
    multioutput::Bool=false,
    conv_bottleneck::Bool=false,
    save_root::String=".",
    save_folder_name::Union{String,Nothing}=nothing)

    name = something(save_folder_name, splitext(basename(datapath))[1])
    save_path = setup_results_folder(name; save_root)
    model_creator = resolve_model_creator(; seq_type, type, multioutput, conv_bottleneck)

    MotifInference.training_and_rendering_config(
        datapath, model_creator, save_path, "n/a";
        seq_type, seed, type, motif_sizes, normalization_method, wt_reference, activation_thresh
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
    sanitize_path_component(s)

Turn an arbitrary feature/label name into a single filesystem- and URL-safe path
component. Feature names carry units like `ddG_ML_float (kcal/mol)`; the `/` would
otherwise split the rendering folder into a nested subfolder, and spaces/parens make
the paths awkward to serve and to reference from HTML.

Keeps `[A-Za-z0-9_-]`, maps everything else to `_`, collapses runs, trims, and caps
the length. Returns `""` if nothing survives.
"""
function sanitize_path_component(s::AbstractString; maxlen::Int=100)
    cleaned = replace(String(s), r"[^A-Za-z0-9_-]+" => "_")
    cleaned = strip(cleaned, ['_'])
    length(cleaned) > maxlen && (cleaned = rstrip(cleaned[1:maxlen], '_'))
    return cleaned
end

"""
    process_output(data, m, train_stats, dl_train, dl_test, trc, output_index)

Process a single output index: compute contributions, tag significance, 
compute interactions, and return all results.

Returns: (contributions_df_filtered, dfs, interaction_summaries_str, 
          interaction_summaries, render_folder_name)
"""
function process_output(data, m, train_stats, dl_train, dl_test, trc, output_index)
    processor, pts_all, pts_test = MotifInference.obtain_processor(m, dl_train, dl_test, trc; predict_position=output_index)

    contributions_df_filtered, contributions_df_filtered_singletons, dfs, scale_back_function =
        MotifInference.load_or_save_raw_motifs(data, m, processor, train_stats, trc; output_index=output_index)

    # Apply scale_back_function to all fields and reconstruct (NamedTuples are immutable)
    # note: the function has taken account into the output index for multi-output cases, so we can apply it directly without worrying about the dimensions
    if trc.scale_back
        pts_all = (; predictions=scale_back_function.(pts_all.predictions),
            labels=scale_back_function.(pts_all.labels),
            proc_prod=scale_back_function.(pts_all.proc_prod))
        pts_test = (; predictions=scale_back_function.(pts_test.predictions),
            labels=scale_back_function.(pts_test.labels),
            proc_prod=scale_back_function.(pts_test.proc_prod))
    end

    # Tag significant singletons
    significant_fils = contributions_df_filtered_singletons.filter_index |> Set
    contributions_df_filtered.significant = [idx in significant_fils for idx in contributions_df_filtered.filter_index]

    # Interaction summaries
    interaction_summaries_str, interaction_summaries =
        MotifInference.BanzhafInference.obtain_interaction_results(
            contributions_df_filtered, dfs)

    # Render folder name (sanitized: feature names may contain `/`, spaces, parens)
    render_folder_name = "renderings_" * begin
        name = if !isnothing(data.raw_data.feature_names)
            sanitize_path_component(data.raw_data.feature_names[output_index])
        else
            ""
        end
        isempty(name) ? "$(output_index)" : name
    end

    return contributions_df_filtered, dfs, pts_all, pts_test, interaction_summaries_str, interaction_summaries, render_folder_name
end

"""
    render_html(data, m, trc, contributions_df_filtered, dfs, pts, 
                all_indices, interaction_summaries_str, render_folder_name)

Generate the HTML motif visualization pages.

The top-movers ranking behind `index.html` is also dumped to
`<save_path>/top_motifs.csv` — one file per dataset, at the result-folder root
rather than inside a rendering folder, so a whole run can be compared with a
single `run_N/*/top_motifs.csv` glob. Multi-output runs append into that one
file; `feature_label` distinguishes the rows.

`report_location_z=true` adds a `location_z` column to that CSV, immediately
after `nnd_pvalue` — the motif's location z-score, i.e. how far its carriers'
mean label sits from the population mean in standard errors. It answers a
different question than the NND p-value beside it (displaced, not merely
coherent) and is closed-form, so it costs nothing. Off by default: the column is
absent unless asked for, so an existing 18-column consumer is unaffected. Keep
it consistent across a multi-output run — those writes append into one file, and
a mixed setting puts the header and the later rows out of step.

`points_only=true` writes only `<rendering>/indicator_points.csv` — the point
cloud behind the indicator (yy-KDE) plots — and skips every figure and page.
Use it to backfill that file into an existing result folder without paying for
a full re-render.
"""
function render_html(data, m, trc, contributions_df_filtered, dfs, pts_all, pts_test, all_indices, interaction_summaries_str, render_folder_name; sensitivity_analysis=false, dataset_name=nothing, protein_name=nothing, feature_label=nothing, append_top_motifs=false, points_only=false, report_location_z=false)
    save_path = joinpath(trc.save_path, render_folder_name)
    top_movers_csv = joinpath(trc.save_path, "top_motifs.csv")

    if trc.type == :mut
        MotifInference.GlyphEctoplasm.plot_motifs_mut_case(data, m, contributions_df_filtered, dfs;
            pts=pts_all,
            pts_test=pts_test,
            all_indices,
            interaction_summaries=interaction_summaries_str,
            reduction_on_ref=true,
            dpi=trc.dpi,
            save_path=save_path,
            page_title=trc.title_string,
            protein_name=protein_name,
            use_rna=trc.seq_type == :rna,
            top_movers_csv=top_movers_csv,
            top_movers_csv_append=append_top_motifs,
            top_movers_label=feature_label,
            report_location_z=report_location_z,
            points_only=points_only,
            )
    else
        MotifInference.GlyphEctoplasm.plot_motifs_conv_case(data, m, trc.motif_sizes, contributions_df_filtered, dfs, pts_all, pts_test, all_indices; interaction_summaries=interaction_summaries_str,
            dpi=trc.dpi,
            save_path=save_path,
            page_title=trc.title_string,
            rna=trc.seq_type == :rna,
            sensitivity_analysis=sensitivity_analysis,
            dataset_name=dataset_name,
            top_movers_csv=top_movers_csv,
            top_movers_csv_append=append_top_motifs,
            top_movers_label=feature_label,
            report_location_z=report_location_z,
            points_only=points_only,
            )
    end
end

"""
    run_method(trc; tune_max_epochs=25, tune_n_trials=25, tune_patience=5, output_indices=nothing)

Run the full motif inference pipeline: load data, optionally tune, train,
and process all (or selected) output indices with HTML rendering.

Tuning runs only when `trc.seed === nothing`; the `tune_*` arguments have no
effect once a seed is set.

# Keyword Arguments
- `tune_max_epochs=25`: max epochs per hyperparameter-tuning trial
- `tune_n_trials=25`: number of tuning trials (candidate models); lower this for a faster run
- `tune_patience=5`: early stopping patience during tuning
- `output_indices=nothing`: which outputs to process (default: all if `trc.predict_position == :all`)
- `points_only=false`: write only `indicator_points.csv` per rendering folder and
  skip all figures/pages. Intended for backfilling an existing result folder:
  with the cached model, processor and motifs `.jld2` in place, the run reduces
  to a forward pass plus one CSV write. Pass `trc.seed` explicitly so the run
  reuses the cached model instead of re-tuning.
"""
function run_method(trc; tune_max_epochs=25, tune_n_trials=25, tune_patience=5, output_indices=nothing, sensitivity_analysis=false, dataset_name=nothing, protein_name=nothing, non_overlapping_sparsify=false, sparse_unpool_size=nothing, sparse_unpool_alpha=nothing, bottleneck_filters=nothing, bottleneck_height=nothing, points_only=false, report_location_z=false)
    data = load_data(trc)
    run_method(trc, data; tune_max_epochs, tune_n_trials, tune_patience,
        output_indices, sensitivity_analysis, dataset_name, protein_name,
        non_overlapping_sparsify, sparse_unpool_size, sparse_unpool_alpha,
        bottleneck_filters, bottleneck_height, points_only, report_location_z)
end

"""
    run_method(trc, data; ...)

Run the pipeline with an already-loaded `data` (a `OnehotSEQ2EXP_Dataset`),
bypassing `load_data`. Used by the in-memory `run_method(strings, labels; ...)`
entry point so no .jld2 file is required.
"""
function run_method(trc, data; tune_max_epochs=25, tune_n_trials=25, tune_patience=5, output_indices=nothing, sensitivity_analysis=false, dataset_name=nothing, protein_name=nothing, non_overlapping_sparsify=false, sparse_unpool_size=nothing, sparse_unpool_alpha=nothing, bottleneck_filters=nothing, bottleneck_height=nothing, points_only=false, report_location_z=false)
    # Optional: make the inference-code layer's receptive fields non-overlapping.
    # Wrap the model creator so every model built for this run (tuning + final)
    # enables the sparse max-unpool op. Off by default.
    if non_overlapping_sparsify
        base_creator = trc.model_creator
        trc.model_creator = (args...; kw...) ->
            base_creator(args...; use_sparse_unpool=true, sparse_unpool_size=sparse_unpool_size,
                         sparse_unpool_alpha=sparse_unpool_alpha, kw...)
    end

    # Optional: override the bottleneck squeeze dimensions (filter count and height).
    # Only affects bottleneck model creators; each unset value keeps its hardcoded
    # default (BOTTLENECK_FILTERS=50, BOTTLENECK_HEIGHT=3). Wrap so every model built
    # for this run (tuning + final) uses the overrides.
    if !isnothing(bottleneck_filters) || !isnothing(bottleneck_height)
        base_creator = trc.model_creator
        bn_kw = Pair{Symbol,Any}[]
        isnothing(bottleneck_filters) || push!(bn_kw, :bottleneck_filters => bottleneck_filters)
        isnothing(bottleneck_height)  || push!(bn_kw, :bottleneck_height  => bottleneck_height)
        trc.model_creator = (args...; kw...) -> base_creator(args...; bn_kw..., kw...)
    end

    tune_if_needed!(trc, data; tune_max_epochs, tune_n_trials, tune_patience)

    m, train_stats, dl_train_eval, dl_test_eval, split_indices =
        MotifInference.obtain_trained_model_and_splited_datasets(data, trc)

    # use all of the data points for motif indicator plot
    all_indices = vcat(split_indices.train, split_indices.test)

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

    for (loop_pos, output_index) in enumerate(indices)
        @info "Processing output $output_index / $(length(indices))..."

        contributions_df_filtered, dfs, pts_all, pts_test, interaction_summaries_str, interaction_summaries, render_folder_name =
            process_output(data, m, train_stats, dl_train_eval, dl_test_eval, trc, output_index)

        # update trc.motif_sizes to exclude empty trailing dataframes
        last_nonempty_idx = 0
        for i in length(dfs):-1:1
            if !isempty(dfs[i])
                last_nonempty_idx = i
                break
            end
        end
        if last_nonempty_idx > 0
            trc.motif_sizes = trc.motif_sizes[1:last_nonempty_idx]
            dfs = dfs[1:last_nonempty_idx]
            @debug "Updated motif sizes to $(trc.motif_sizes) and trimmed dfs based on non-empty dataframes"
        else
            # all dataframes are empty
            trc.motif_sizes = Int[]
            dfs = []
            @warn "No motifs found for output $output_index"
        end

        # Raw (unsanitized) label for the CSV; the first output truncates
        # top_motifs.csv, the rest append into it.
        feature_label = isnothing(data.raw_data.feature_names) ? nothing :
                        data.raw_data.feature_names[output_index]

        render_html(data, m, trc, contributions_df_filtered, dfs, pts_all, pts_test, all_indices, interaction_summaries_str, render_folder_name; sensitivity_analysis, dataset_name, protein_name, feature_label, append_top_motifs=(loop_pos > 1), points_only, report_location_z)

        @info "Output $output_index done → $(render_folder_name)"
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run from dataset entry, file path, or batch
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_method(dataset_entry; kwargs...)

Run pipeline from a dataset entry (as returned by `load_datasets()`).

# Examples
    ds = load_datasets("ecoli")
    run_method(ds[1])
    run_method(ds[1]; output_indices=1:5)
"""
function run_method(f::NamedTuple;
    datasets_parent_folder=datasets_processed_folder,
    save_root=".",
    save_folder_name=nothing,
    kwargs...)
    trc = make_trc(f, datasets_parent_folder; save_root, save_folder_name)
    run_method(trc; dataset_name=f.name, kwargs...)
end

"""
    read_seq_label_csv(path) -> (strings, labels)

Parse a two-column CSV of `<sequence>, <scalar-label>` into a `Vector{String}`
of sequences and a `Vector{Float64}` of labels. A header row is auto-detected
(and skipped) when the second field of the first line is not a number. Blank
lines are ignored. No external CSV dependency is used — the format is trivial.
"""
function read_seq_label_csv(path::String)
    isfile(path) || throw(ArgumentError("CSV file not found: $path"))
    strings = String[]
    labels = Float64[]
    for (lineno, raw) in enumerate(readlines(path))
        line = strip(raw)
        isempty(line) && continue
        parts = split(line, ',')
        length(parts) ≥ 2 ||
            throw(ArgumentError("Line $lineno of $path is not `<sequence>, <label>`: \"$line\""))
        seq = String(strip(parts[1]))
        lab = tryparse(Float64, strip(parts[2]))
        if isnothing(lab)
            # first line with a non-numeric label is treated as a header; otherwise it's an error
            (lineno == 1 && isempty(strings)) && continue
            throw(ArgumentError("Line $lineno of $path has a non-numeric label: \"$(parts[2])\""))
        end
        push!(strings, seq)
        push!(labels, lab)
    end
    isempty(strings) && throw(ArgumentError("No (sequence, label) rows parsed from $path"))
    return strings, labels
end

"""
    announce_bottleneck_selection(; seq_type, type, conv_bottleneck, bottleneck_filters, bottleneck_height)

Make the bottleneck selection explicit at run time. When a bottleneck model is
selected (see [`model_uses_bottleneck`](@ref)), log the squeeze dimensions in
effect (falling back to the hardcoded `BOTTLENECK_FILTERS` / `BOTTLENECK_HEIGHT`
defaults). When `bottleneck_filters` / `bottleneck_height` were passed but the
selected model is *not* a bottleneck model, warn that the overrides are ignored
so they never silently no-op.
"""
function announce_bottleneck_selection(; seq_type, type, conv_bottleneck,
                                       bottleneck_filters, bottleneck_height)
    if model_uses_bottleneck(; seq_type, type, conv_bottleneck)
        @info "Loading bottleneck model" seq_type type conv_bottleneck bottleneck_filters=something(bottleneck_filters, VeryBasicCNN2.BOTTLENECK_FILTERS) bottleneck_height=something(bottleneck_height, VeryBasicCNN2.BOTTLENECK_HEIGHT)
    elseif !isnothing(bottleneck_filters) || !isnothing(bottleneck_height)
        @warn "bottleneck_filters/bottleneck_height were set but the selected model is not a bottleneck model — these overrides are ignored. For a DNA/RNA convolution run pass conv_bottleneck=true; mutation runs (type=:mut) are bottleneck by default for protein and for DNA/RNA alike." seq_type type conv_bottleneck
    end
end

"""
    run_method(datapath; seq_type=:dna, seed=nothing, ...)

Run pipeline directly from a file path. The format is chosen by extension:
- `.csv` — a two-column `<sequence>, <scalar-label>` file (parsed in memory; no
  .jld2 needed). A header row is auto-detected. Currently scalar labels only.
- anything else — a `.jld2` file containing a `SEQ2EXP_Dataset` as `raw_data`.

No pre-registration needed — just point to your data.

All keyword arguments from `make_trc` (seq_type, normalization, seed, etc.)
and `run_method` (output_indices, tune_* params) are accepted.

# Examples
    # Simplest — just a path (will auto-tune since no seed)
    run_method("/path/to/mydata.jld2")
    run_method("/path/to/mydata.csv")

    # With overrides
    run_method("/path/to/mydata.jld2"; seq_type=:rna, seed=42)
    run_method("/path/to/mydata.csv"; seq_type=:protein, type=:mut, GET_CONSENSUS=true)

    # RNA with specific outputs
    run_method("/path/to/mydata.jld2"; seq_type=:rna, seed=42, output_indices=1:5)

    # RNA/DNA mutagenesis — needs a dataset built with GET_CONSENSUS=true, so the
    # mutation encoding (X_mut) exists; without a consensus the run silently falls
    # back to plain one-hot.
    run_method("/path/to/rna_dms.jld2"; seq_type=:rna, type=:mut, activation_thresh=0.8)
"""
function run_method(datapath::String;
    # make_trc kwargs
    seq_type::Symbol=:dna,
    type::Symbol=:conv,
    normalization_method::Symbol=:zscore,
    wt_reference::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    seed::Union{Int,Nothing}=nothing,
    motif_sizes::Vector{Int}=[2, 3, 4, 5],
    activation_thresh::Float64=0.9,
    multioutput::Bool=false,
    conv_bottleneck::Bool=false,
    bottleneck_filters=nothing,
    bottleneck_height=nothing,
    save_root::String=".",
    save_folder_name::Union{String,Nothing}=nothing,
    # run_method kwargs
    kwargs...)
    if endswith(lowercase(datapath), ".csv")
        strings, labels = read_seq_label_csv(datapath)
        return run_method(strings, labels;
            seq_type, type, normalization_method, wt_reference, seed, motif_sizes,
            activation_thresh, multioutput, conv_bottleneck,
            bottleneck_filters, bottleneck_height, save_root, save_folder_name,
            name=splitext(basename(datapath))[1], kwargs...)
    end
    announce_bottleneck_selection(; seq_type, type, conv_bottleneck,
        bottleneck_filters, bottleneck_height)
    trc = make_trc(datapath; seq_type, type, normalization_method, wt_reference, seed,
        motif_sizes, activation_thresh, multioutput, conv_bottleneck, save_root, save_folder_name)
    run_method(trc; bottleneck_filters, bottleneck_height, kwargs...)
end

"""
    run_method(strings, labels; feature_names=nothing, GET_CONSENSUS=false, ...)

Run the pipeline directly from in-memory `strings` and `labels`, with no .jld2
file on disk. A `SEQ2EXP_Dataset` is constructed internally (equivalent to
`SEQ2EXP_Dataset(strings, labels; ...)`), so this is the in-memory counterpart
to `run_method(datapath::String; ...)`.

All keyword arguments from `make_trc`/`run_method` are accepted, plus:
- `feature_names = nothing`: column names for multi-output `labels`
- `GET_CONSENSUS = false`: compute a consensus sequence (e.g. amino-acid mutagenesis)
- `name = "inmemory"`: used for the results folder when `save_folder_name` is unset

# Examples
    # nucleotides
    run_method(strings, labels)
    run_method(strings, labels; seq_type=:rna, seed=42, output_indices=1:5)

    # amino acids for mutagenesis
    run_method(strings, labels; GET_CONSENSUS=true, seq_type=:protein, type=:mut)

    # nucleotides for mutagenesis (GET_CONSENSUS=true is what turns on the mutation encoding)
    run_method(strings, labels; GET_CONSENSUS=true, seq_type=:rna, type=:mut)
"""
function run_method(strings::Vector{String}, labels::Union{AbstractVector,AbstractMatrix};
    feature_names::Union{Vector{String},Nothing}=nothing,
    GET_CONSENSUS::Bool=false,
    # make_trc kwargs
    seq_type::Symbol=:dna,
    type::Symbol=:conv,
    normalization_method::Symbol=:zscore,
    wt_reference::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    seed::Union{Int,Nothing}=nothing,
    motif_sizes::Vector{Int}=[2, 3, 4, 5],
    activation_thresh::Float64=0.9,
    multioutput::Bool=false,
    conv_bottleneck::Bool=false,
    bottleneck_filters=nothing,
    bottleneck_height=nothing,
    save_root::String=".",
    save_folder_name::Union{String,Nothing}=nothing,
    name::String="inmemory",
    # run_method kwargs
    kwargs...)

    raw_data = SEQ2EXP_Dataset(strings, labels, feature_names; GET_CONSENSUS)
    data = MotifInference.OnehotSEQ2EXP_Dataset(raw_data)

    folder = something(save_folder_name, name)
    save_path = setup_results_folder(folder; save_root)
    announce_bottleneck_selection(; seq_type, type, conv_bottleneck,
        bottleneck_filters, bottleneck_height)
    model_creator = resolve_model_creator(; seq_type, type, multioutput, conv_bottleneck)

    # datapath is unused here (data is already in memory), so pass an empty placeholder
    trc = MotifInference.training_and_rendering_config(
        "", model_creator, save_path, "n/a";
        seq_type, seed, type, motif_sizes, normalization_method, wt_reference, activation_thresh)

    run_method(trc, data; bottleneck_filters, bottleneck_height, kwargs...)
end

"""
    run_all(datasets; kwargs...)

Run pipeline for multiple datasets sequentially.

# Examples
    run_all(load_datasets())                              # everything
    run_all(load_datasets("ecoli", "yeast"))              # specific ones
    run_all(load_datasets(); output_indices=1:3)          # first 3 outputs each
"""
function run_all(datasets; datasets_parent_folder=datasets_processed_folder, kwargs...)
    for (i, f) in enumerate(datasets)
        @info "═══ Dataset $i/$(length(datasets)): $(f.name) ═══"
        run_method(f; datasets_parent_folder, kwargs...)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Usage examples (uncomment to run)
# ─────────────────────────────────────────────────────────────────────────────

# From a .jld2 file path (simplest):
#   run_method("/path/to/mydata.jld2")
#   run_method("/path/to/mydata.jld2"; seq_type=:rna, seed=42)

# From a registered dataset:
#   run_method(load_datasets("ecoli")[1])
#   run_method(load_datasets("ecoli")[1]; output_indices=1:5)

# Batch:
#   run_all(load_datasets("ecoli", "yeast"))
#   run_all(load_datasets())