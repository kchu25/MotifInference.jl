
function perform_hyperparameter_tuning(data, trc; 
    trial_number_start=1, max_epochs=40, n_trials=50, patience=5)
    results, best_model, best_info = AutoComputationalGraphTuning.tune_hyperparameters(data, 
        trc.model_creator;
        trial_number_start=trial_number_start, 
        n_trials=n_trials, 
        normalization_method=trc.normalization,
        save_folder=trc.save_path, 
        max_epochs=max_epochs, 
        patience=patience,
        print_every=100);
    return results, best_model, best_info
end

function train_and_evaluate_model(data, model_creator, save_where, seed; 
    max_epochs=40, patience=10, print_every=100)

    model_folder = "$save_where/models/"
    model_path = joinpath(model_folder, "model_$seed.jld2")
    
    config_json = AutoComputationalGraphTuning.load_trial_config(
        "$save_where/json/trial_seed_$seed.json")
    
    m=nothing
    if isfile(model_path)
        println("Loading existing model from $model_path")
        @load model_path model_cpu stats train_stats
        m = model2gpu(model_cpu)
        
        # Recreate dataloaders
        _, _, _, _, _, split_indices = 
            AutoComputationalGraphTuning.train_final_model_from_config(
                data, model_creator, config_json; 
                max_epochs=0, patience=10, print_every=100)
    else
        println("Training new model...")
        m, stats, train_stats, _, _, split_indices = 
            AutoComputationalGraphTuning.train_final_model_from_config(
                data, model_creator, config_json; 
                max_epochs=max_epochs, patience=patience, print_every=print_every)
        
        mkpath(model_folder)
        model_cpu = model2cpu(m)
        @save model_path model_cpu stats train_stats split_indices
    end
    
    m.training[] = false  # Ensure evaluation mode

    setup, batch_size = AutoComputationalGraphTuning._prepare_final_model_setup(data, model_creator; config_json.seed, config_json.randomize_batchsize)
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