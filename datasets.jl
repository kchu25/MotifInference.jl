# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────
#
# This file defines the registered datasets. The functional utilities
# (resolve_model_creator, dataset constructor, load_datasets) are defined in
# dataset_utils.jl, which is included by MotifInference.jl before this file.
#
# Usage:
#   datasets = load_datasets()                    # all active datasets
#   datasets = load_datasets("ecoli", "yeast")    # specific ones by name
#   run_method(datasets[1])                     # run one
#
# To add a new dataset, just add an entry to DATASETS below.
# ─────────────────────────────────────────────────────────────────────────────

# const DATASETS = [
#     # ——— Promoter / expression ———
#     dataset(name="yeast",  file="yeast.jld2",  seed=nothing, motif_sizes=[2,3,4,5]),
#     dataset(name="ecoli",  file="ecoli.jld2",  seed=nothing, motif_sizes=[2,3,4,5], ),

#     # ——— RNA binding ———
#     # dataset(name="RNAcompeteMINIMAL", file="RNAcompete.jld2", 
#             # seq_type=:rna, normalization_method=:log, seed=52,             
#             # activation_thresh=0.95, multioutput=true),

#     # ——— 5' UTR ———
#     dataset(name="utr_yeast", file="utr_yeast.jld2",  seq_type=:rna, seed=nothing, normalization_method=:identity, motif_sizes=[2,3,4,5]),
#     dataset(name="utr_human", file="utr_human.jld2",  seq_type=:rna, seed=25, motif_sizes=[2,3,4,5]),

#     # ——— Gene therapy 5' UTR ———
#     # dataset(name="utr_5p_gene_therapy",          file="utr_5p.jld2",        seq_type=:rna, seed=13),
#     # dataset(name="utr_5p_gene_therapy/MUSCLE",   file="utr_5p_MUSCLE.jld2", seq_type=:rna, seed=52),
#     # dataset(name="utr_5p_gene_therapy_hek_screen_data", file="utr_5p_screen.jld2", seq_type=:rna, seed=40),

#     # ——— Plant promoters ———
#     dataset(name="jores_leaf",   file="jores_leaf.jld2",  motif_sizes=[2,3,4,5], seed=nothing),
#     dataset(name="jores_proto",  file="jores_proto.jld2", motif_sizes=[2,3,4,5], seed=nothing),

#     # ——— Splicing ———
#     dataset(name="rosenberg2015_5p", normalization_method=:identity, 
#     model_creator= VeryBasicCNN2.create_model_nucleotides_fixed_pool_stride_sigmoid, seq_type=:rna,
#     loss_spec = loss_specs[:binary_cross_entropy], file="rosenberg2015_A5SS.jld2", motif_sizes=[2,3,4,5], seed=nothing),

# #     dataset(name="rosenberg2015_3p", normalization_method=:identity, 
# #     model_creator= VeryBasicCNN2.create_model_nucleotides_fixed_pool_stride_sigmoid, seq_type=:rna,
# #     loss_spec = loss_specs[:binary_cross_entropy], 
# #     file="rosenberg2015_A3SS.jld2", motif_sizes=[2,3,4,5], seed=nothing),

#     dataset(name="splirent", file="splirent.jld2", seed=nothing, multioutput=true, 
#     model_creator=VeryBasicCNN2.
#     create_model_nucleotides_fixed_pool_stride_multioutputs_sigmoid, seq_type=:rna, motif_sizes=[2,3,4,5], normalization_method=:identity, loss_spec = loss_specs[:binary_cross_entropy]),

#     # ——— 3' UTR / mRNA degradation ———
# #     dataset(name="utr_mrna_degradation", file="utr_degradation.jld2", 
#         #     seq_type=:rna, seed=9, multioutput=true),
# ]

# before
const DATASETS = [
    # ——— Promoter / expression ———
    dataset(name="yeast",  file="yeast.jld2",  seed=13, motif_sizes=[2,3,4,5]),
    dataset(name="ecoli",  file="ecoli.jld2",  seed=2, motif_sizes=[2,3,4,5], ),

    # ——— RNA binding ———
    # dataset(name="RNAcompeteMINIMAL", file="RNAcompete.jld2", 
            # seq_type=:rna, normalization_method=:log, seed=52,             
            # activation_thresh=0.95, multioutput=true),

    # ——— 5' UTR ———
    dataset(name="utr_yeast", file="utr_yeast.jld2",  seq_type=:rna, seed=13, normalization_method=:identity, motif_sizes=[2,3,4,5]),
    # dataset(name="utr_yeast", file="utr_yeast.jld2",  seq_type=:rna, seed=15, normalization_method=:log, motif_sizes=[2,3,4,5]),
    dataset(name="utr_human", file="utr_human.jld2",  seq_type=:rna, seed=63, motif_sizes=[2,3,4,5]),

    # ——— Gene therapy 5' UTR ———
    # dataset(name="utr_5p_gene_therapy",          file="utr_5p.jld2",        seq_type=:rna, seed=13),
    # dataset(name="utr_5p_gene_therapy/MUSCLE",   file="utr_5p_MUSCLE.jld2", seq_type=:rna, seed=52),
    # dataset(name="utr_5p_gene_therapy_hek_screen_data", file="utr_5p_screen.jld2", seq_type=:rna, seed=40),

    # ——— Plant promoters ———
    dataset(name="jores_leaf",   file="jores_leaf.jld2",  motif_sizes=[2,3,4,5], seed=43),
    dataset(name="jores_proto",  file="jores_proto.jld2", motif_sizes=[2,3,4,5], seed=39),

    # ——— Splicing ———
    dataset(name="rosenberg2015_5p", normalization_method=:identity, 
    model_creator= VeryBasicCNN2.create_model_nucleotides_fixed_pool_stride_sigmoid, seq_type=:rna,
    loss_spec = loss_specs[:binary_cross_entropy], file="rosenberg2015_A5SS.jld2", motif_sizes=[2,3,4,5], seed=15),

#     dataset(name="rosenberg2015_3p", normalization_method=:identity, 
#     model_creator= VeryBasicCNN2.create_model_nucleotides_fixed_pool_stride_sigmoid, seq_type=:rna,
#     loss_spec = loss_specs[:binary_cross_entropy], 
#     file="rosenberg2015_A3SS.jld2", motif_sizes=[2,3,4,5], seed=nothing),

    dataset(name="splirent", file="splirent.jld2", seed=6, multioutput=true, 
    model_creator=VeryBasicCNN2.
    create_model_nucleotides_fixed_pool_stride_multioutputs_sigmoid, seq_type=:rna, motif_sizes=[2,3,4,5], normalization_method=:identity, loss_spec = loss_specs[:binary_cross_entropy]),

    # ——— 3' UTR / mRNA degradation ———
#     dataset(name="utr_mrna_degradation", file="utr_degradation.jld2", 
        #     seq_type=:rna, seed=9, multioutput=true),
]




const DATASETS_MUT = [
    dataset(name="LAC_REPRESSION", file="LacI_repression.jld2", 
            seq_type=:protein, type=:mut, seed=50, activation_thresh=0.95),
]

const DATASETS_DEBUG = [
    dataset(name="debug_1", file="debug_1.jld2", seed=1, activation_thresh=0.9, motif_sizes=[2,3,4], normalization_method=:minmax),
    dataset(name="debug_2", file="debug_2.jld2", seed=nothing, activation_thresh=0.9),
]

const DATASETS_subsampled = [
        dataset(name="utr_yeast_subsample", file="utr_yeast_subsample.jld2", seq_type=:rna, seed=17, normalization_method=:identity, 
        motif_sizes=[2,3,4]),

        dataset(name="rosenberg2015_3p_subsample", file="rosenberg2015_A3SS_subsample.jld2", seq_type=:rna, seed=nothing, normalization_method=:identity, motif_sizes=[2,3,4,5], 
        model_creator=VeryBasicCNN2.create_model_nucleotides_fixed_pool_stride_sigmoid,
        loss_spec = loss_specs[:binary_cross_entropy]),

        dataset(name="splirent_subsample", file="splirent_subsample.jld2", 
        model_creator=VeryBasicCNN2.
        create_model_nucleotides_fixed_pool_stride_multioutputs_sigmoid,
        seed=nothing, normalization_method=:identity, multioutput=true, motif_sizes=[2,3,4], loss_spec = loss_specs[:binary_cross_entropy])
]