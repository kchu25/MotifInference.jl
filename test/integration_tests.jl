using Test

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
Run a single output through the pipeline and collect all the pieces
needed for validation, without doing any rendering.
"""
function collect_test_artifacts(trc; output_index=1)
    data = load_data(trc)
    tune_if_needed!(trc, data)

    m, train_stats, dl_train, dl_test, split_indices =
        MotifInference.obtain_trained_model_and_splited_datasets(data, trc)

    # obtain_processor now returns (processor, pts_all) where pts_all = merge_pts(pts_train, pts_test)
    processor, pts_all =
        MotifInference.obtain_processor(m, dl_train, dl_test, trc;
            predict_position=output_index)

    return (;
        pts_all,
        split_indices,
        dl_train_eval = dl_train,
        dl_test_eval  = dl_test,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "pts / split_indices sanity checks" begin

    # ── change these to point at a real dataset before running ───────────────
    trc = make_trc("/path/to/your/data.jld2"; seed=42)
    # ─────────────────────────────────────────────────────────────────────────

    art = collect_test_artifacts(trc; output_index=1)
    (; pts_all, split_indices, dl_train_eval, dl_test_eval) = art

    n_train = length(split_indices.train)
    n_test  = length(split_indices.test)
    n_total = n_train + n_test

    # ── 1. pts_all prediction-count == train + test ───────────────────────────
    @testset "prediction length == total split size" begin
        @test length(pts_all.predictions) == n_total
        @test length(pts_all.labels)      == n_total
    end

    # ── 2. train / test indices are disjoint ──────────────────────────────────
    @testset "train and test indices are disjoint" begin
        common = intersect(split_indices.train, split_indices.test)
        @test isempty(common)
    end

    # ── 3. pts_all labels agree with concatenated dataloader labels ───────────
    @testset "pts_all labels match vcat of dataloader labels" begin
        # dl.data[2] is the label matrix/vector; train comes first (mirrors merge_pts)
        combined_labels = vcat(dl_train_eval.data[2], dl_test_eval.data[2])

        @info "Label comparison (pts_all vs vcat(dl_train, dl_test)):" hcat(pts_all.labels, combined_labels)

        @test pts_all.labels == combined_labels
    end
end
