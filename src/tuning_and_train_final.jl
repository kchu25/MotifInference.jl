
function perform_hyperparameter_tuning(data, trc; 
    trial_number_start=1, max_epochs=40, n_trials=50, patience=5)
    results, best_model, best_info = AutoComputationalGraphTuning.tune_hyperparameters(data, 
        trc.model_creator;
        trial_number_start=trial_number_start, 
        n_trials=n_trials, 
        normalization_method=trc.normalization_method,
        wt_reference=trc.wt_reference,
        save_folder=trc.save_path, 
        max_epochs=max_epochs, 
        patience=patience,
        loss_spec=trc.loss_spec,
        print_every=100);

    # Every tuning trial returned `nothing`, so no model was ever built. Left
    # alone this surfaces several stages downstream as
    # `FieldError: type Nothing has no field seed` at `best_info.seed`, which
    # says nothing about the cause.
    #
    # The dominant cause is a sequence shorter than the receptive field. Note it
    # is the TRIMMED length that matters: `trim_common_ends` strips every
    # position that is identical across the library, so a 93-residue protein
    # whose assay mutates only 4 sites encodes to 6 positions and cannot host an
    # 8-wide region. `create_model` returns `nothing` in that case and the
    # tuning loop silently skips the trial.
    isnothing(best_info) && _report_no_viable_architecture(data, trc, n_trials)

    return results, best_model, best_info
end

"""
    _report_no_viable_architecture(data, trc, n_trials)

Raise a diagnostic error when hyperparameter tuning produced no model at all.
Always throws.
"""
function _report_no_viable_architecture(data, trc, n_trials)
    L        = try data.X_dim[2] catch; nothing end
    full_len = try length(first(data.raw_data.strings)) catch; nothing end
    offset   = try data.prefix_offset catch; nothing end
    bh       = VeryBasicCNN2.BOTTLENECK_HEIGHT

    io = IOBuffer()
    println(io, "Hyperparameter tuning produced no usable model in $n_trials trial(s).")
    println(io)
    if L !== nothing
        println(io, "  encoded (trimmed) length : $L")
        full_len === nothing || println(io, "  untrimmed length         : $full_len")
        offset === nothing   || println(io, "  prefix trimmed away      : $offset")
        println(io, "  default region width     : $bh   (bottleneck_height)")
        println(io)
        if L < bh
            println(io, "The encoded length ($L) is smaller than the region width ($bh), so no")
            println(io, "model can be built: a region cannot be wider than the sequence it reads.")
            println(io)
            println(io, "`trim_common_ends` removes every position that is identical across the")
            println(io, "whole library, so what matters is the TRIMMED length, not the sequence")
            println(io, "length. An assay that mutates only a handful of sites encodes to a very")
            println(io, "short tensor even when the protein is long.")
            println(io)
            println(io, "Fix: pass a smaller region width, e.g.")
            println(io, "    run_method(...; bottleneck_height=$(max(2, min(L, bh ÷ 2))))")
            println(io, "or build the model creator with `bottleneck_height=` set below $L.")
        else
            println(io, "The encoded length clears the default region width, so the cause is")
            println(io, "something else. Check the trial JSONs under:")
            println(io, "    $(trc.save_path)/json/")
            println(io, "and re-run with more trials to see whether any architecture is viable.")
        end
    else
        println(io, "Could not inspect the dataset dimensions to diagnose further.")
    end
    error(String(take!(io)))
end

function train_and_evaluate_model(data, trc; 
    max_epochs=40, patience=10, print_every=100)

    model_folder = "$(trc.save_path)/models/"
    model_path = joinpath(model_folder, "model_$(trc.seed).jld2")
    
    config_json = AutoComputationalGraphTuning.load_trial_config(
        "$(trc.save_path)/json/trial_seed_$(trc.seed).json")
    
    m=nothing
    if isfile(model_path)
        println("Loading existing model from $model_path")
        @load model_path model_cpu stats train_stats
        m = model2gpu(model_cpu)
        
        # Recreate dataloaders
        _, _, _, _, _, split_indices = 
            AutoComputationalGraphTuning.train_final_model_from_config(
                data, trc.model_creator, config_json, trc;                 
                max_epochs=0, patience=10, print_every=100)
    else
        println("Training new model...")
        m, stats, train_stats, _, _, split_indices = 
            AutoComputationalGraphTuning.train_final_model_from_config(
                data, trc.model_creator, config_json, trc; 
                max_epochs=max_epochs, patience=patience, print_every=print_every)
        
        mkpath(model_folder)
        model_cpu = model2cpu(m)
        @save model_path model_cpu stats train_stats split_indices
    end
    
    m.training[] = false  # Ensure evaluation mode

    # band-aid for now
    setup, batch_size = AutoComputationalGraphTuning._prepare_final_model_setup(data, trc.model_creator; 
    seed = config_json.seed, 
    randomize_batchsize = config_json.randomize_batchsize, 
    normalization_method = trc.normalization_method,
    wt_reference = trc.wt_reference)
    dl_train_eval, dl_test_eval = AutoComputationalGraphTuning._create_eval_dataloaders(setup, batch_size)

    return m, train_stats, dl_train_eval, dl_test_eval, split_indices
end

function train_and_evaluate_processor!(m, dl_train, dl_test, save_where, seed, pw; 
    predict_position=1, max_epochs=5)
    model_folder = "$save_where/models/"
    processor_path = joinpath(model_folder, "processor_$(seed)_pp_$(predict_position).jld2")
    m.training[] = false  # Set to evaluation mode

    processor=nothing
    if isfile(processor_path)
        println("Loading existing processor from $processor_path")
        @load processor_path processor_cpu
        processor = processor2gpu(processor_cpu);
    else
        println("Training new processor...")
        processor, _ = AutoComputationalGraphTuning.train_code_processor(
            m, dl_train, pw; seed=seed, predict_position=predict_position, max_epochs=max_epochs)
        
        mkpath(model_folder)
        processor_cpu = processor2cpu(processor)
        @save processor_path processor_cpu
    end
    
    eval!(processor)

    _, pts_train = AutoComputationalGraphTuning.evaluate_processor(
        m, processor, dl_train, "Train"; predict_position=predict_position)
    
    proc_stats, pts_test = AutoComputationalGraphTuning.evaluate_processor(
        m, processor, dl_test, "Test"; predict_position=predict_position)
    
    pts_all = merge_pts(pts_train, pts_test)

     # TODO: Sanity check: Ensure that the number of predictions matches the number of labels

    return processor, proc_stats, pts_all, pts_test
end

"""
    merge_pts(pts_train, pts_test)

Concatenate `pts_train` and `pts_test` into a single object with the same
fields (`predictions`, `labels`, `proc_prod`), each formed by `vcat`-ing the
train and test counterparts.
"""
function merge_pts(pts_train, pts_test)
    return (;
        predictions = vcat(pts_train.predictions, pts_test.predictions),
        labels      = vcat(pts_train.labels,      pts_test.labels),
        proc_prod   = vcat(pts_train.proc_prod,   pts_test.proc_prod),
    )
end