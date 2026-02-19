
function obtain_trained_model_and_splited_datasets(data, trc)
    # train the model using train + validation data
    m, train_stats, dl_train_eval, dl_test_eval, split_indices = 
            train_and_evaluate_model(data, trc.model_creator, trc.save_path, trc.seed; 
            max_epochs=trc.max_training_epochs, patience=trc.patience,);
    return m, train_stats, dl_train_eval, dl_test_eval, split_indices
end

function obtain_processor(m, dl_train, dl_test, trc; 
    predict_position=1, scale_back=true, train_stats=nothing)

    processor, _, pts_all, pts_test = train_and_evaluate_processor!(
        m, dl_train, dl_test, trc.save_path, trc.seed, VeryBasicCNN2.proc_wrap; 
        predict_position=predict_position, max_epochs=trc.max_processor_epochs);

    return processor, pts_all, pts_test
end

"""
Obtain motifs from the model and save as dataframes for rendering
"""
function obtain_motifs(data, m, processor, train_stats, trc; predict_position=1)
    # obtain the contributions and configurations
    contribs_filtered, contributions_df_filtered, ec, ac, mdc, bc = 
        BanzhafInference.obtain_contribs_filtered_and_configs(
            data, m, processor, train_stats; 
            scale_back=trc.scale_back, 
            activation_thresh=trc.activation_thresh, 
            predict_position=predict_position);

    # sample random coalitions (background) for significance testing
    random_coalitions = BanzhafInference.compute_random_coalition_banzhafs_all_datapoints(
      contributions_df_filtered, ac, bc; seed=trc.seed, verbose=false
    );

    mutegenesis = trc.type == :mut

    # singletons motifs
    df_significant, contributions_df_filtered_singletons = 
        BanzhafInference.single_motifs_and_significance_filtering!(
            ac, ec, contribs_filtered, contributions_df_filtered, random_coalitions; 
                mutegenesis = mutegenesis, top_and_bot_counts=trc.top_and_bot_counts); 
    # multi-motifs
    dfs, df_significants = BanzhafInference.obtain_multi_motifs_and_banzhafs(
        contributions_df_filtered, mdc, ec, ac, random_coalitions;
        seed=trc.seed, motif_sizes=trc.motif_sizes, mutegenesis=mutegenesis, 
        COUNT_THRESHOLD=trc.count_threshold, Q_THRESHOLD=trc.Q_threshold, 
        top_and_bot_counts=trc.top_and_bot_counts
        )

    return contributions_df_filtered, 
           contributions_df_filtered_singletons, 
           dfs
end


"""
    load_or_save_raw_motifs(data, m, processor, train_stats, trc; output_index=1)

Obtain or load raw motifs to avoid recomputation. Caches results in JLD2 format.
"""
function load_or_save_raw_motifs(data, m, processor, train_stats, trc; output_index=1)
    cache_file = joinpath(trc.save_path, "motifs_cache_output_$(output_index).jld2")
    
    if isfile(cache_file)
        println("Loading cached motifs from: $cache_file")
        @load cache_file contributions_df_filtered contributions_df_filtered_singletons dfs
    else
        println("Computing motifs (this may take a while)...")
        contributions_df_filtered, contributions_df_filtered_singletons, dfs = 
            obtain_motifs(data, m, processor, train_stats, trc; 
                predict_position=output_index)
        
        println("Saving motifs cache to: $cache_file")
        @save cache_file contributions_df_filtered contributions_df_filtered_singletons dfs
    end

    # band-aid solution for now
    is_multioutput = (trc.predict_position == :all) && (data.Y_dim > 1)
    scale_back_function = BanzhafInference._make_scale_back_function(trc.scale_back, train_stats, is_multioutput, output_index)
    
    return contributions_df_filtered, contributions_df_filtered_singletons, dfs, scale_back_function
end
